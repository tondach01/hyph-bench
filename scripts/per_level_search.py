#!/usr/bin/env python3
"""Final per-level GP search for PATGEN weights and thresholds.

Each level searches the ordered weight ratios 1/5, 1/4, 1/3, 1/2, and 1..30.
Ratios are converted to integer PATGEN parameters: 1/n becomes good_wt=n,
bad_wt=1; integer n becomes good_wt=1, bad_wt=n. Each level independently
searches threshold 1..42 by default. Dataset splitting groups normalized word
identities before a seeded hash-ranked 8/1/1 split. Source priorities, when
present, are expanded in training only.
"""

import argparse
import csv
import json
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .gp_optimizer import GPOptimizer
from .objectives import get_objective
from .dataset_split import create_clean_split
from .optimize_validation import (
    evaluate_parameter_set,
    f17_score,
    failed_evaluation_result,
)
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)

# The dated runner uses the extended space; the historical GPopt8 launcher pins
# its original space through PATGEN_OPT_WEIGHT_SPACE=legacy for reproducibility.
_WEIGHT_SPACES = {
    "extended": (5, 4, 3, 2),
    "legacy": (3, 2),
}
_weight_space = os.environ.get("PATGEN_OPT_WEIGHT_SPACE", "extended")
if _weight_space not in _WEIGHT_SPACES:
    raise ValueError(
        f"unknown PATGEN_OPT_WEIGHT_SPACE={_weight_space!r}; "
        f"expected one of {tuple(_WEIGHT_SPACES)}"
    )
FRACTIONAL_DENOMINATORS = _WEIGHT_SPACES[_weight_space]
WEIGHT_LABELS = tuple(f"1/{value}" for value in FRACTIONAL_DENOMINATORS) + tuple(
    str(value) for value in range(1, 31)
)


def decode_weight(code: int) -> Tuple[int, int]:
    """Return (good_wt, bad_wt) for an ordinal GP weight code."""
    if 0 <= code < len(FRACTIONAL_DENOMINATORS):
        return FRACTIONAL_DENOMINATORS[code], 1
    if code < len(WEIGHT_LABELS):
        return 1, code - len(FRACTIONAL_DENOMINATORS) + 1
    raise ValueError(f"weight code out of range: {code}")


def decode_params(params: Tuple[int, ...], n_levels: int) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]
]:
    if len(params) != 2 * n_levels:
        raise ValueError(f"expected {2 * n_levels} parameters, got {len(params)}")
    decoded = tuple(decode_weight(code) for code in params[:n_levels])
    good_weights = tuple(pair[0] for pair in decoded)
    bad_weights = tuple(pair[1] for pair in decoded)
    thresholds = params[n_levels:]
    return good_weights, bad_weights, thresholds


def profile_labels(params: Tuple[int, ...], n_levels: int) -> Tuple[str, ...]:
    return tuple(WEIGHT_LABELS[code] for code in params[:n_levels])


def history_fieldnames(n_levels: int) -> List[str]:
    return (
        ["observation"]
        + [f"weight_{i}" for i in range(1, n_levels + 1)]
        + [f"weight_code_{i}" for i in range(1, n_levels + 1)]
        + [f"good_wt_{i}" for i in range(1, n_levels + 1)]
        + [f"bad_wt_{i}" for i in range(1, n_levels + 1)]
        + [f"threshold_{i}" for i in range(1, n_levels + 1)]
        + [
            "validation_good",
            "validation_bad",
            "validation_missed",
            "validation_f17",
            "objective_score",
            "train_good",
            "train_bad",
            "train_missed",
            "n_patterns",
            "trie_nodes",
        ]
    )


def history_row(
    observation: int,
    params: Tuple[int, ...],
    results: Dict[str, int],
    score: float,
    n_levels: int,
) -> Dict[str, object]:
    good_weights, bad_weights, thresholds = decode_params(params, n_levels)
    labels = profile_labels(params, n_levels)
    return {
        "observation": observation,
        **{f"weight_{i + 1}": labels[i] for i in range(n_levels)},
        **{f"weight_code_{i + 1}": params[i] for i in range(n_levels)},
        **{f"good_wt_{i + 1}": good_weights[i] for i in range(n_levels)},
        **{f"bad_wt_{i + 1}": bad_weights[i] for i in range(n_levels)},
        **{f"threshold_{i + 1}": thresholds[i] for i in range(n_levels)},
        "validation_good": results["good"],
        "validation_bad": results["bad"],
        "validation_missed": results["missed"],
        "validation_f17": results.get("validation_f17", 0.0),
        "objective_score": score,
        "train_good": results["train_good"],
        "train_bad": results["train_bad"],
        "train_missed": results["train_missed"],
        "n_patterns": results["n_patterns"],
        "trie_nodes": results["trie_nodes"],
    }


def write_history(path: str, rows: List[Dict[str, object]], n_levels: int) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fieldnames(n_levels))
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-level GP search over weight ratios and thresholds"
    )
    parser.add_argument("--lang", required=True)
    parser.add_argument("--wordlist")
    parser.add_argument("--translate")
    parser.add_argument("--patgen", default="patgen")
    parser.add_argument("--profile")
    parser.add_argument("--output-dir", default="results/gpopt260828")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ucb-kappa", type=float, default=2.5)
    parser.add_argument("--min-threshold", type=int, default=1)
    parser.add_argument("--max-threshold", type=int, default=42)
    parser.add_argument("--objective", choices=["f17", "f17_trie"], default="f17_trie")
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--final-exploitation", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export-final-patterns", action="store_true")
    args = parser.parse_args()

    if args.min_threshold < 1 or args.max_threshold < args.min_threshold:
        parser.error("threshold range must be nonempty and start at 1 or higher")
    if bool(args.wordlist) != bool(args.translate):
        parser.error("--wordlist and --translate must be provided together")

    if args.wordlist:
        wordlist_path = os.path.abspath(args.wordlist)
        translate_path = os.path.abspath(args.translate)
    else:
        wordlist_path, translate_path = find_dataset(args.lang)
    lang_dir = os.path.join(args.output_dir, args.lang)
    os.makedirs(lang_dir, exist_ok=True)
    split = create_clean_split(
        wordlist_path, os.path.join(lang_dir, "splits"), seed=args.seed
    )


    trie_normalizer = None
    if args.objective == "f17_trie":
        trie_normalizer, fixed = resolve_trie_normalizer(
            args, split["unique"], "scripts.per_level_search", dataset=args.lang
        )
        if fixed:
            warn_fixed_trie_normalizer(
                "scripts.per_level_search", trie_normalizer, "search"
            )

    objective = (
        get_objective(
            args.objective,
            beta=args.beta,
            trie_weight=args.trie_weight,
            trie_normalizer=trie_normalizer,
        )
        if args.objective == "f17_trie"
        else get_objective(args.objective, beta=args.beta)
    )
    pat_ranges = parse_profile(args.profile) if args.profile else DEFAULT_PAT_RANGES
    n_levels = len(pat_ranges)
    bounds = (
        [(0, len(WEIGHT_LABELS) - 1)] * n_levels
        + [(args.min_threshold, args.max_threshold)] * n_levels
    )

    state_path = os.path.join(lang_dir, "final_state.pkl")
    history_path = os.path.join(lang_dir, "final_history.csv")
    patterns_path = os.path.join(lang_dir, "final_patterns.pat")
    selected_path = os.path.join(lang_dir, "selected_profile.json")
    config_path = os.path.join(lang_dir, "run_config.json")

    config = {
        "command": shlex.join(sys.argv),
        "dataset": args.lang,
        "seed": args.seed,
        "patgen": os.path.abspath(args.patgen),
        "objective": objective.name,
        "trie_weight": args.trie_weight,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "weight_values": list(WEIGHT_LABELS),
        "threshold_range": [args.min_threshold, args.max_threshold],
        "pattern_ranges": pat_ranges,
        "split_counts": {
            key: split[f"{key}_count"] for key in ("train", "validation", "test")
        },
        "split_type_counts": {
            key: split[f"{key}_type_count"]
            for key in ("train", "validation", "test")
        },
        "split_method": split["split_method"],
        "weighted_training": split["weighted_training"] == "true",
        "weighted_source": split["weighted_source"] or None,
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    optimizer = GPOptimizer(
        objective, seed=args.seed, bounds=bounds, min_samples_for_gp=args.batch_size
    )
    history_rows: List[Dict[str, object]] = []
    observation = 0
    started = time.time()

    print(f"Objective: {objective.name}")
    print(f"Dataset: {wordlist_path}")
    print(f"Split counts: train={split['train_count']}, validation={split['validation_count']}, test={split['test_count']}")
    print(f"Pattern ranges: {pat_ranges}")
    print(f"Search space ({2 * n_levels}-D): per-level weight in {WEIGHT_LABELS}; per-level threshold in [{args.min_threshold},{args.max_threshold}]")
    print(f"Budget: {args.iterations} iterations x batch {args.batch_size} = {args.iterations * args.batch_size} evaluations")

    def evaluate_batch(pool: ProcessPoolExecutor, candidates: List[Tuple[int, ...]], tag: str) -> None:
        nonlocal observation
        with pool as executor:
            futures = {}
            for index, params in enumerate(candidates):
                good_weights, bad_weights, thresholds = decode_params(params, n_levels)
                futures[executor.submit(
                    evaluate_parameter_set,
                    args.patgen,
                    split["train"],
                    split["validation"],
                    translate_path,
                    bad_weights + thresholds,
                    pat_ranges,
                    good_weights,
                    args.verbose,
                    f"{tag}_{index}",
                )] = params

            for future in as_completed(futures):
                params = futures[future]
                try:
                    _, results = future.result()
                except Exception as exc:
                    print(f"  worker failed: params={params}: {exc!r}; retrying inline")
                    good_weights, bad_weights, thresholds = decode_params(params, n_levels)
                    try:
                        _, results = evaluate_parameter_set(
                            args.patgen, split["train"], split["validation"],
                            translate_path, bad_weights + thresholds, pat_ranges,
                            good_weights, args.verbose, "retry",
                        )
                    except Exception as retry_exc:
                        print(f"  retry failed: params={params}: {retry_exc!r}")
                        results = failed_evaluation_result(split["validation"])
                score = optimizer.update(
                    params, results["good"], results["bad"], results["missed"],
                    n_patterns=results["n_patterns"], trie_nodes=results["trie_nodes"],
                )
                observation += 1
                results["validation_f17"] = f17_score(
                    results["good"], results["bad"], results["missed"]
                )
                history_rows.append(history_row(observation, params, results, score, n_levels))
                print(f"  Tested: weights={profile_labels(params, n_levels)}, thresholds={params[n_levels:]}")
                print(f"  validation: F_1/7={results['validation_f17']:.6f}, good={results['good']}, bad={results['bad']}, missed={results['missed']}, trie_nodes={results['trie_nodes']}, objective={score:.6f}")

    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}\nIteration {iteration + 1}/{args.iterations}")
        candidates = optimizer.suggest_batch(args.batch_size, ucb_kappa=args.ucb_kappa)
        evaluate_batch(ProcessPoolExecutor(max_workers=args.batch_size), candidates, f"it{iteration}")
        best = optimizer.best_so_far()
        print(f"  Best so far: weights={profile_labels(tuple(best['params']), n_levels)}, thresholds={tuple(best['params'])[n_levels:]}, score={best['score']:.6f}, elapsed={time.time() - started:.1f}s")
        optimizer.save(state_path)
        write_history(history_path, history_rows, n_levels)

    if args.final_exploitation:
        print(f"\n{'=' * 60}\nFinal exploitation: {args.final_exploitation} evaluations")
        candidates = optimizer.exploit_best(n=args.final_exploitation)
        evaluate_batch(ProcessPoolExecutor(max_workers=args.final_exploitation), candidates, "final")
        optimizer.save(state_path)
        write_history(history_path, history_rows, n_levels)

    best = optimizer.best_so_far()
    selected_params = tuple(best["params"])
    export_path = patterns_path if args.export_final_patterns else ""

    def run_test(params: Tuple[int, ...]) -> Optional[Dict[str, int]]:
        good_weights, bad_weights, thresholds = decode_params(params, n_levels)
        try:
            _, result = evaluate_parameter_set(
                args.patgen, split["train"], split["test"], translate_path,
                bad_weights + thresholds, pat_ranges, good_weights, args.verbose,
                "test", export_patterns_path=export_path,
            )
            return result
        except Exception as exc:
            print(f"  final test failed for {params}: {exc!r}")
            return None

    test_results = run_test(selected_params)
    if test_results is None:
        for candidate in sorted(optimizer.results, key=lambda item: item["score"], reverse=True):
            params = tuple(candidate["params"])
            if candidate["score"] > 0 and candidate["trie_nodes"] > 0:
                test_results = run_test(params)
                if test_results is not None:
                    selected_params = params
                    print(f"  used fallback profile: {params}")
                    break
    if test_results is None:
        raise RuntimeError("no profile evaluated successfully on the held-out test split")

    good_weights, bad_weights, thresholds = decode_params(selected_params, n_levels)
    selected = {
        "weight_ratios": profile_labels(selected_params, n_levels),
        "good_weights": good_weights,
        "bad_weights": bad_weights,
        "thresholds": thresholds,
        "validation_objective": best["score"],
        "held_out_test": test_results,
        "held_out_test_f17": f17_score(test_results["good"], test_results["bad"], test_results["missed"]),
    }
    with open(selected_path, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
        handle.write("\n")

    print(f"\n{'=' * 60}\nFinal per-level search complete")
    print(f"Selected weights: {selected['weight_ratios']}")
    print(f"PATGEN good_wt: {good_weights}; bad_wt: {bad_weights}; thresholds: {thresholds}")
    print(f"Held-out test: {test_results}")
    print(f"History: {history_path}")
    print(f"Selected profile: {selected_path}")


if __name__ == "__main__":
    main()
