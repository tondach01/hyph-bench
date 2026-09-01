#!/usr/bin/env python3
"""Optimize shared PATGEN good-weight and threshold parameters.

The search uses the same deterministic 8/1/1 train, validation, and held-out
test protocol as ``scripts.optimize_validation`` while expanding the parameter
vector to::

    [bad_wt_1, ..., bad_wt_N, threshold, good_wt]

The bad weights remain level-specific. The threshold and good weight are shared
across levels. Default bounds are ``[1, 30]`` for each bad weight, ``[1, 2]``
for the threshold, and ``[1, 5]`` for the good weight.

The default experiment uses 30 iterations with batches of five, UCB kappa 2.5,
seed 42, and the dataset-normalized ``f17_trie`` objective. Results are written
under ``results/shared_parameter_search``.

Usage::

    uv run python -m scripts.optimize_shared_parameters \
        --lang cssk/cshyphen \
        --patgen /path/to/patgen \
        --export-final-patterns
"""

import argparse
import csv
import os
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

def history_fieldnames(n_levels: int) -> List[str]:
    return (
        ["observation"]
        + [f"param_{i}" for i in range(1, n_levels + 3)]
        + ["validation_good", "validation_bad", "validation_missed",
           "validation_f17", "objective_score",
           "train_good", "train_bad", "train_missed", "n_patterns", "trie_nodes"]
    )


def safe_name(name: str) -> str:
    return name.replace(os.sep, "_").replace("/", "_")


def write_history_csv(path: str, rows: List[Dict[str, object]],
                      n_levels: int) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fieldnames(n_levels))
        writer.writeheader()
        writer.writerows(rows)


def history_row(observation: int, params: Tuple[int, ...], results: Dict[str, int],
                score: float, n_levels: int) -> Dict[str, object]:
    return {
        "observation": observation,
        **{f"param_{i}": params[i - 1] for i in range(1, n_levels + 3)},
        "validation_good": results["good"],
        "validation_bad": results["bad"],
        "validation_missed": results["missed"],
        "validation_f17": results.get("validation_f17", results.get("f17", 0.0)),
        "objective_score": score,
        "train_good": results["train_good"],
        "train_bad": results["train_bad"],
        "train_missed": results["train_missed"],
        "n_patterns": results["n_patterns"],
        "trie_nodes": results["trie_nodes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize shared PATGEN good-weight and threshold parameters"
    )
    parser.add_argument("--lang", required=True, help="Language/dataset id, e.g. cssk/cshyphen")
    parser.add_argument("--wordlist", help="Override wordlist path")
    parser.add_argument("--translate", help="Override translate path")
    parser.add_argument("--patgen", default="patgen", help="Path to patgen binary")
    parser.add_argument("--profile", help="Profile file for pat_start/pat_finish")
    parser.add_argument("--output-dir", default="results/shared_parameter_search")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ucb-kappa", type=float, default=2.5)
    parser.add_argument("--max-bad-weight", type=int, default=30)
    parser.add_argument("--min-threshold", type=int, default=1)
    parser.add_argument("--max-threshold", type=int, default=2)
    parser.add_argument("--min-good-weight", type=int, default=1)
    parser.add_argument("--max-good-weight", type=int, default=5)
    parser.add_argument("--objective", choices=["f17", "f17_trie"], default="f17_trie")
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--final-exploitation", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export-final-patterns", action="store_true")
    args = parser.parse_args()

    if args.wordlist and args.translate:
        wordlist_path = os.path.abspath(args.wordlist)
        translate_path = os.path.abspath(args.translate)
    else:
        wordlist_path, translate_path = find_dataset(args.lang)
    lang_dir = os.path.join(args.output_dir, args.lang)
    split = create_clean_split(
        wordlist_path, os.path.join(lang_dir, "splits"), seed=args.seed
    )


    trie_normalizer = None
    if args.objective == "f17_trie":
        trie_normalizer, fixed = resolve_trie_normalizer(
            args,
            split["unique"],
            "scripts.optimize_shared_parameters",
            dataset=args.lang,
        )
        if fixed:
            warn_fixed_trie_normalizer(
                "scripts.optimize_shared_parameters", trie_normalizer, "search"
            )

    pat_ranges = parse_profile(args.profile) if args.profile else DEFAULT_PAT_RANGES
    objective = (
        get_objective(
            args.objective, beta=args.beta, trie_weight=args.trie_weight,
            trie_normalizer=trie_normalizer,
        )
        if args.objective == "f17_trie"
        else get_objective(args.objective, beta=args.beta)
    )


    state_path = os.path.join(lang_dir, "wider_state.pkl")
    history_path = os.path.join(lang_dir, "wider_history.csv")
    final_patterns_path = os.path.join(lang_dir, "wider_final.pat")
    os.makedirs(lang_dir, exist_ok=True)

    n_levels = len(pat_ranges)
    # Search space: n_levels bad_weights + shared threshold + good_weight.
    bounds = (
        [(1, args.max_bad_weight)] * n_levels
        + [(args.min_threshold, args.max_threshold)]
        + [(args.min_good_weight, args.max_good_weight)]
    )
    min_samples = args.batch_size
    optimizer = GPOptimizer(objective, seed=args.seed, bounds=bounds,
                            min_samples_for_gp=min_samples)

    print(f"Objective: {objective.name}")
    print(f"Dataset: {wordlist_path}")
    print(f"Split counts: train={split['train_count']}, validation={split['validation_count']}, "
          f"test={split['test_count']}")
    print(f"Pattern ranges: {pat_ranges}")
    print(f"Search space ({n_levels + 2}-D, {n_levels} levels): bad_wt each "
          f"(1,{args.max_bad_weight}), threshold ({args.min_threshold},{args.max_threshold}), "
          f"good_wt ({args.min_good_weight},{args.max_good_weight})")
    print(f"Budget: {args.iterations} iterations x batch {args.batch_size} "
          f"= {args.iterations * args.batch_size} patgen evaluations")

    history_rows: List[Dict[str, object]] = []
    observation = 0
    start_time = time.time()

    def eval_in_worker(worker_pool, candidates: List[Tuple[int, ...]],
                       tag_prefix: str) -> None:
        nonlocal observation
        with worker_pool as executor:
            futures = {
                executor.submit(
                    evaluate_parameter_set,
                    args.patgen,
                    split["train"],
                    split["validation"],
                    translate_path,
                    params[: n_levels + 1],  # bad_weights + shared threshold
                    pat_ranges,
                    params[n_levels + 1],    # good_weight
                    args.verbose,
                    f"{tag_prefix}_{i}",
                ): params
                for i, params in enumerate(candidates)
            }
            for future in as_completed(futures):
                p6 = futures[future]
                try:
                    _, results = future.result()
                except Exception as exc:
                    print(f"  worker failed: params={p6}: {exc!r}; retrying inline")
                    try:
                        _, results = evaluate_parameter_set(
                            args.patgen, split["train"], split["validation"],
                            translate_path, p6[: n_levels + 1], pat_ranges,
                            p6[n_levels + 1], args.verbose, "retry",
                        )
                        print(f"  retry OK: params={p6}")
                    except Exception as exc2:
                        print(f"  retry failed: params={p6}: {exc2!r}")
                        results = failed_evaluation_result(split["validation"])
                score = optimizer.update(
                    p6, results["good"], results["bad"], results["missed"],
                    n_patterns=results["n_patterns"],
                    trie_nodes=results["trie_nodes"],
                )
                observation += 1
                results["validation_f17"] = f17_score(
                    results["good"], results["bad"], results["missed"])
                history_rows.append(history_row(observation, p6, results, score, n_levels))
                print(f"  Tested: params={p6}")
                f17 = results["validation_f17"]
                print(f"  validation: F_1/7={f17:.6f}, good={results['good']}, "
                      f"bad={results['bad']}, missed={results['missed']}, "
                      f"trie_nodes={results['trie_nodes']}, objective={score:.6f}")

    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}\nIteration {iteration + 1}/{args.iterations}")
        suggestions = optimizer.suggest_batch(args.batch_size, ucb_kappa=args.ucb_kappa)
        eval_in_worker(
            ProcessPoolExecutor(max_workers=args.batch_size),
            suggestions, f"it{iteration}",
        )
        best = optimizer.best_so_far()
        print(f"  Best so far: params={best['params']}, score={best['score']:.6f}, "
              f"elapsed={time.time() - start_time:.1f}s")
        optimizer.save(state_path)
        write_history_csv(history_path, history_rows, n_levels)

    if args.final_exploitation > 0:
        print(f"\n{'=' * 60}\nFinal exploitation: {args.final_exploitation} evaluations")
        best_params = optimizer.exploit_best(n=args.final_exploitation)
        eval_in_worker(
            ProcessPoolExecutor(max_workers=args.final_exploitation),
            best_params, "final",
        )
        optimizer.save(state_path)
        write_history_csv(history_path, history_rows, n_levels)

    best = optimizer.best_so_far()
    export_path = final_patterns_path if args.export_final_patterns else ""
    test_results = None
    eval_params = tuple(best["params"])

    def run_test(params: Tuple[int, ...]) -> Optional[Dict[str, int]]:
        try:
            _, res = evaluate_parameter_set(
                args.patgen, split["train"], split["test"], translate_path,
                params[: n_levels + 1], pat_ranges, params[n_levels + 1],
                args.verbose, "test", export_patterns_path=export_path,
            )
            return res
        except Exception as exc:
            print(f"  final test eval failed for {params}: {exc!r}")
            return None

    test_results = run_test(eval_params)
    if test_results is None:
        print("  falling back to best non-failed profile")
        for cand in sorted(optimizer.results, key=lambda r: r.get("score", 0.0),
                           reverse=True):
            if cand["score"] > 0 and cand["trie_nodes"] > 0:
                eval_params = tuple(cand["params"])
                test_results = run_test(eval_params)
                if test_results is not None:
                    print(f"  used fallback profile: {eval_params}")
                    break
    if test_results is None:
        raise RuntimeError("no profile evaluated successfully on the held-out test split")

    test_f17 = f17_score(
        test_results["good"], test_results["bad"], test_results["missed"])
    print(f"\n{'=' * 60}\nGPoptval4-wider complete")
    print(f"Best parameters ({n_levels} bad_wt + thr + good_wt): {eval_params}")
    print(f"  validation-selected: F_1/7={best['score']:.6f} (objective)")
    print(f"  held-out test: good={test_results['good']}, bad={test_results['bad']}, "
          f"missed={test_results['missed']}, trie_nodes={test_results['trie_nodes']}, "
          f"n_patterns={test_results['n_patterns']}")
    print(f"State saved to: {state_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
