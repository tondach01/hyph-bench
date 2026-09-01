#!/usr/bin/env python3
"""Budget-matched Random/TPE baselines in the full per-level (8-D) search space.

Reruns the paper's HPO ablation (old restricted 5-parameter space, Table
tab:app:hpo-baselines-full) in the SAME search space and protocol as the
camera-ready per-level GP search (scripts.per_level_search / results/gpopt260828):

  * 8-D space: per-level weight ratio in {1/5,1/4,1/3,1/2,1..30} (ordinal code
    0..33) and per-level threshold in [1,42], four levels;
  * same deterministic grouped-hash train/validation/test split;
  * same objective (f17_trie, proportional trie normalizer |D|, weight 0.0005);
  * same budget: 30 rounds x batch 5 = 150 search evaluations, plus 3 extra
    sampler-driven evaluations mirroring the GP run's 3 final-exploitation
    evaluations (153 total per method);
  * same selection rule: best validation objective, then one held-out test run.

The GP itself is NOT rerun here; compare against the recorded gpopt260828
selected_profile.json files, which used the identical protocol and seed.

Usage:
    PATGEN_BIN=/home/dev/patgen-10x uv run python -m scripts.per_level_hpo_baselines \
        --lang cssk/cshyphen --method tpe --patgen "$PATGEN_BIN" \
        --output-dir results/hpo_baselines_8d
"""

import argparse
import json
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .objectives import get_objective
from .dataset_split import create_clean_split
from .optimize_validation import (
    evaluate_parameter_set,
    f17_score,
    failed_evaluation_result,
)
from .per_level_search import (
    WEIGHT_LABELS,
    decode_params,
    history_row,
    profile_labels,
    write_history,
)
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random/TPE baselines over the per-level 8-D search space"
    )
    parser.add_argument("--lang", required=True)
    parser.add_argument("--method", required=True, choices=["random", "tpe"])
    parser.add_argument("--patgen", default="patgen")
    parser.add_argument("--profile")
    parser.add_argument("--output-dir", default="results/hpo_baselines_8d")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--final-extra", type=int, default=3,
                        help="Extra sampler-driven evaluations after the main "
                             "rounds, matching the GP run's 3 final-exploitation "
                             "evaluations (total budget parity at 153).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tpe-startup", type=int, default=10)
    parser.add_argument("--min-threshold", type=int, default=1)
    parser.add_argument("--max-threshold", type=int, default=42)
    parser.add_argument("--objective", choices=["f17", "f17_trie"], default="f17_trie")
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    wordlist_path, translate_path = find_dataset(args.lang)
    lang_dir = os.path.join(args.output_dir, args.method, args.lang)
    os.makedirs(lang_dir, exist_ok=True)
    split = create_clean_split(
        wordlist_path, os.path.join(lang_dir, "splits"), seed=args.seed
    )


    trie_normalizer = None
    if args.objective == "f17_trie":
        trie_normalizer, fixed = resolve_trie_normalizer(
            args, split["unique"], "scripts.per_level_hpo_baselines", dataset=args.lang
        )
        if fixed:
            warn_fixed_trie_normalizer(
                "scripts.per_level_hpo_baselines", trie_normalizer, "search"
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

    history_path = os.path.join(lang_dir, "final_history.csv")
    selected_path = os.path.join(lang_dir, "selected_profile.json")
    config_path = os.path.join(lang_dir, "run_config.json")

    config = {
        "command": shlex.join(sys.argv),
        "dataset": args.lang,
        "method": args.method,
        "seed": args.seed,
        "patgen": os.path.abspath(args.patgen),
        "objective": objective.name,
        "trie_weight": args.trie_weight,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "final_extra": args.final_extra,
        "tpe_startup": args.tpe_startup if args.method == "tpe" else None,
        "weight_values": list(WEIGHT_LABELS),
        "threshold_range": [args.min_threshold, args.max_threshold],
        "pattern_ranges": pat_ranges,
        "split_counts": {key: split[f"{key}_count"] for key in ("train", "validation", "test")},
        "split_type_counts": {
            key: split[f"{key}_type_count"]
            for key in ("train", "validation", "test")
        },
        "split_method": split["split_method"],
        "weighted_training": split["weighted_training"] == "true",
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    if args.method == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        startup = max(1, min(args.tpe_startup, args.iterations * args.batch_size))
        sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    history_rows: List[Dict[str, object]] = []
    observation = 0
    best_score = float("-inf")
    best_params: Tuple[int, ...] = ()
    started = time.time()

    print(f"Method: {args.method}")
    print(f"Objective: {objective.name}")
    print(f"Dataset: {wordlist_path}")
    print(f"Split counts: train={split['train_count']}, validation={split['validation_count']}, test={split['test_count']}")
    print(f"Pattern ranges: {pat_ranges}")
    print(f"Search space ({2 * n_levels}-D): per-level weight in {WEIGHT_LABELS}; per-level threshold in [{args.min_threshold},{args.max_threshold}]")
    print(f"Budget: {args.iterations} rounds x batch {args.batch_size} + {args.final_extra} extra = {args.iterations * args.batch_size + args.final_extra} evaluations")

    def ask_batch(count: int) -> List[Tuple[object, Tuple[int, ...]]]:
        asked = []
        for _ in range(count):
            trial = study.ask()
            params = tuple(
                trial.suggest_int(f"p{j}", lo, hi) for j, (lo, hi) in enumerate(bounds)
            )
            asked.append((trial, params))
        return asked

    def evaluate_batch(asked: List[Tuple[object, Tuple[int, ...]]], tag: str) -> None:
        nonlocal observation, best_score, best_params
        with ProcessPoolExecutor(max_workers=len(asked)) as executor:
            futures = {}
            for index, (trial, params) in enumerate(asked):
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
                    f"{args.method}_{tag}_{index}",
                )] = (trial, params)

            for future in as_completed(futures):
                trial, params = futures[future]
                try:
                    _, results = future.result()
                except Exception as exc:
                    print(f"  worker failed: params={params}: {exc!r}; retrying inline")
                    good_weights, bad_weights, thresholds = decode_params(params, n_levels)
                    try:
                        _, results = evaluate_parameter_set(
                            args.patgen, split["train"], split["validation"],
                            translate_path, bad_weights + thresholds, pat_ranges,
                            good_weights, args.verbose, f"{args.method}_retry",
                        )
                    except Exception as retry_exc:
                        print(f"  retry failed: params={params}: {retry_exc!r}")
                        results = failed_evaluation_result(split["validation"])
                score = objective.score(
                    good=results["good"], bad=results["bad"], missed=results["missed"],
                    n_patterns=results["n_patterns"], trie_nodes=results["trie_nodes"],
                    f17cv=0.0,
                )
                study.tell(trial, score)
                if score > best_score:
                    best_score, best_params = score, params
                observation += 1
                results["validation_f17"] = f17_score(
                    results["good"], results["bad"], results["missed"]
                )
                history_rows.append(history_row(observation, params, results, score, n_levels))
                print(f"  Tested: weights={profile_labels(params, n_levels)}, thresholds={params[n_levels:]}")
                print(f"  validation: F_1/7={results['validation_f17']:.6f}, good={results['good']}, bad={results['bad']}, missed={results['missed']}, trie_nodes={results['trie_nodes']}, objective={score:.6f}")

    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}\nIteration {iteration + 1}/{args.iterations}")
        evaluate_batch(ask_batch(args.batch_size), f"it{iteration}")
        print(f"  Best so far: weights={profile_labels(best_params, n_levels)}, thresholds={best_params[n_levels:]}, score={best_score:.6f}, elapsed={time.time() - started:.1f}s")
        write_history(history_path, history_rows, n_levels)

    if args.final_extra:
        print(f"\n{'=' * 60}\nExtra sampler evaluations: {args.final_extra}")
        evaluate_batch(ask_batch(args.final_extra), "final")
        write_history(history_path, history_rows, n_levels)

    selected_params = best_params
    good_weights, bad_weights, thresholds = decode_params(selected_params, n_levels)
    try:
        _, test_results = evaluate_parameter_set(
            args.patgen, split["train"], split["test"], translate_path,
            bad_weights + thresholds, pat_ranges, good_weights, args.verbose,
            f"{args.method}_test",
        )
    except Exception as exc:
        raise RuntimeError(
            f"held-out test evaluation failed for {selected_params}"
        ) from exc

    selected = {
        "method": args.method,
        "weight_ratios": profile_labels(selected_params, n_levels),
        "good_weights": good_weights,
        "bad_weights": bad_weights,
        "thresholds": thresholds,
        "validation_objective": best_score,
        "held_out_test": test_results,
        "held_out_test_f17": f17_score(
            test_results["good"], test_results["bad"], test_results["missed"]
        ),
    }
    with open(selected_path, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
        handle.write("\n")

    print(f"\n{'=' * 60}\n{args.method} per-level baseline complete")
    print(f"Selected weights: {selected['weight_ratios']}")
    print(f"PATGEN good_wt: {good_weights}; bad_wt: {bad_weights}; thresholds: {thresholds}")
    print(f"Held-out test: {test_results}")
    print(f"History: {history_path}")
    print(f"Selected profile: {selected_path}")


if __name__ == "__main__":
    main()
