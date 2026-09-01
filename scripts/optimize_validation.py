#!/usr/bin/env python3
"""Validation-set GP optimization for PATGEN parameters.

Candidates train on a deterministic 8/10 split, are selected on a 1/10
validation split, and are reported on a held-out 1/10 test split. Word
identities are grouped before a seeded hash-ranked split so no surface form can
cross partitions. Weighted sources repeat priorities in training only.
"""

import argparse
import csv
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .dataset_split import create_clean_split
from .gp_optimizer import GPOptimizer
from .hyphenator.hyphenator import Hyphenator
from .hyperparameters.score import PatgenScorer
from .hyperparameters.sample import Sample
from .objectives import get_objective
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)


def f17_score(good: int, bad: int, missed: int, beta: float = 1 / 7) -> float:
    if good == 0:
        return 0.0
    precision = good / (good + bad) if good + bad > 0 else 0.0
    recall = good / (good + missed) if good + missed > 0 else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def precision(good: int, bad: int) -> float:
    return good / (good + bad) if good + bad > 0 else 0.0


def recall(good: int, missed: int) -> float:
    return good / (good + missed) if good + missed > 0 else 0.0


def count_hyphen_points(wordlist_path: str) -> int:
    total = 0
    with open(wordlist_path, encoding="utf-8") as wordlist:
        for line in wordlist:
            total += line.count("-")
    return total


def failed_evaluation_result(eval_path: str) -> Dict[str, int]:
    return {
        "good": 0,
        "bad": 0,
        "missed": count_hyphen_points(eval_path),
        "train_good": 0,
        "train_bad": 0,
        "train_missed": 0,
        "n_patterns": 0,
        "trie_nodes": 0,
    }


def safe_name(name: str) -> str:
    return name.replace(os.sep, "_").replace("/", "_")




def train_patgen_multilevel(
    scorer: PatgenScorer,
    params: Tuple[int, ...],
    pat_ranges: List[Tuple[int, int]],
    good_weight: int | Tuple[int, ...],
) -> Tuple[str, Dict[str, int]]:
    n_levels = len(pat_ranges)
    if isinstance(good_weight, int):
        good_weights = (good_weight,) * n_levels
    else:
        if len(good_weight) != n_levels:
            raise ValueError(
                f"expected {n_levels} good weights, got {len(good_weight)}"
            )
        good_weights = good_weight
    if len(params) == 2 * n_levels:
        # bad_1..N + per-level thresholds thr_1..N
        bad_weights = params[:n_levels]
        thresholds = params[n_levels : 2 * n_levels]
    elif len(params) == n_levels + 1:
        # bad_1..N + one threshold shared by all levels
        bad_weights = params[:n_levels]
        thresholds = (params[n_levels],) * n_levels
    else:
        bad_weights = params
        thresholds = (1,) * n_levels

    prev_id = 0
    total_patterns = 0
    final_stats = None
    n_levels = min(len(bad_weights), len(pat_ranges))

    for level in range(1, n_levels + 1):
        pat_start, pat_finish = pat_ranges[level - 1]
        sample = Sample(
            {
                "level": level,
                "prev": prev_id,
                "pat_start": pat_start,
                "pat_finish": pat_finish,
                "good_weight": good_weights[level - 1],
                "bad_weight": bad_weights[level - 1],
                "threshold": thresholds[level - 1],
            }
        )
        scorer.score(sample)
        prev_id = sample.run_id
        total_patterns += sample.stats.get("level_patterns", 0)
        final_stats = sample.stats

    if final_stats is None:
        raise RuntimeError("patgen produced no levels")

    pattern_path = os.path.join(scorer.temp_dir, f"{prev_id}.pat")
    return pattern_path, {
        "train_good": final_stats["tp"],
        "train_bad": final_stats["fp"],
        "train_missed": final_stats["fn"],
        "n_patterns": total_patterns,
        "trie_nodes": final_stats["trie_nodes"],
    }


def evaluate_patterns(wordlist_path: str, pattern_path: str, translate_path: str) -> Dict[str, int]:
    hyphenator = Hyphenator(pattern_path, hyphenation_mark="-", translate_file=translate_path)
    good, bad, missed = 0, 0, 0

    with open(wordlist_path, encoding="utf-8") as wordlist:
        for correct in wordlist:
            correct = correct.strip()
            hyphenated = hyphenator.hyphenate(correct)
            i_corr, i_hyph = 0, 0
            while i_corr < len(correct) and i_hyph < len(hyphenated):
                if correct[i_corr] == "-" and hyphenated[i_hyph] == "-":
                    good += 1
                    i_corr += 1
                    i_hyph += 1
                elif hyphenated[i_hyph] == "-":
                    bad += 1
                    i_hyph += 1
                elif correct[i_corr] == "-":
                    missed += 1
                    i_corr += 1
                else:
                    i_corr += 1
                    i_hyph += 1

    return {"good": good, "bad": bad, "missed": missed}


def evaluate_parameter_set(
    patgen_path: str,
    train_path: str,
    eval_path: str,
    translate_path: str,
    params: Tuple[int, ...],
    pat_ranges: List[Tuple[int, int]],
    good_weight: int | Tuple[int, ...],
    verbose: bool,
    worker_id: str,
    export_patterns_path: str = "",
) -> Tuple[Tuple[int, ...], Dict[str, int]]:
    scorer = PatgenScorer(
        patgen_path,
        train_path,
        translate_path,
        verbose=verbose,
        tmp_suffix=f"_gpoptval4_{worker_id}",
    )
    try:
        pattern_path, train_results = train_patgen_multilevel(
            scorer, params, pat_ranges, good_weight
        )
        eval_results = evaluate_patterns(eval_path, pattern_path, translate_path)
        if export_patterns_path:
            os.makedirs(os.path.dirname(export_patterns_path), exist_ok=True)
            shutil.copyfile(pattern_path, export_patterns_path)
        return params, {**eval_results, **train_results}
    finally:
        scorer.clean()


def write_history_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "observation",
        "param_1",
        "param_2",
        "param_3",
        "param_4",
        "param_5",
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_history_csv(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def history_row(observation: int, params: Tuple[int, ...], results: Dict[str, int], score: float) -> Dict[str, object]:
    row = {
        "observation": observation,
        "validation_good": results["good"],
        "validation_bad": results["bad"],
        "validation_missed": results["missed"],
        "validation_f17": f17_score(results["good"], results["bad"], results["missed"]),
        "objective_score": score,
        "train_good": results["train_good"],
        "train_bad": results["train_bad"],
        "train_missed": results["train_missed"],
        "n_patterns": results["n_patterns"],
        "trie_nodes": results["trie_nodes"],
    }
    for i, value in enumerate(params, start=1):
        row[f"param_{i}"] = value
    return row


def format_result_line(label: str, results: Dict[str, int], score: float = None) -> str:
    f17 = f17_score(results["good"], results["bad"], results["missed"])
    parts = [
        f"F_1/7={f17:.6f}",
        f"precision={precision(results['good'], results['bad']):.6f}",
        f"recall={recall(results['good'], results['missed']):.6f}",
        f"good={results['good']}",
        f"bad={results['bad']}",
        f"missed={results['missed']}",
        f"trie_nodes={results['trie_nodes']}",
        f"patterns={results['n_patterns']}",
    ]
    if score is not None:
        parts.append(f"objective={score:.6f}")
    return f"  {label}: " + ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPoptval4 validation-set optimizer")
    parser.add_argument("--lang", required=True, help="Language/dataset id, e.g. uk/wiktionary")
    parser.add_argument("--wordlist", help="Override wordlist path")
    parser.add_argument("--translate", help="Override translate path")
    parser.add_argument("--patgen", default="patgen", help="Path to patgen binary")
    parser.add_argument("--profile", help="Profile file for pat_start/pat_finish")
    parser.add_argument("--output-dir", default="results/gpoptval4", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ucb-kappa", type=float, default=2.5)
    parser.add_argument("--good-weight", type=int, default=3)
    parser.add_argument("--max-bad-weight", type=int, default=30)
    parser.add_argument("--max-threshold", type=int, default=1)
    parser.add_argument("--objective", choices=["f17", "f17_trie"], default="f17_trie")
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--final-exploitation", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export-final-patterns", action="store_true")
    args = parser.parse_args()

    if args.wordlist and args.translate:
        wordlist_path = os.path.abspath(args.wordlist)
        translate_path = os.path.abspath(args.translate)
    else:
        wordlist_path, translate_path = find_dataset(args.lang)

    lang_dir = os.path.join(args.output_dir, args.lang)
    split_dir = os.path.join(lang_dir, "splits")
    split = create_clean_split(wordlist_path, split_dir, seed=args.seed)

    fixed_trie_normalizer = False
    trie_normalizer = None
    if args.objective == "f17_trie":
        trie_normalizer, fixed_trie_normalizer = resolve_trie_normalizer(
            args,
            split["unique"],
            "scripts.optimize_validation",
            dataset=args.lang,
        )

    pat_ranges = parse_profile(args.profile) if args.profile else DEFAULT_PAT_RANGES
    objective = get_objective(
        args.objective,
        beta=args.beta,
        trie_weight=args.trie_weight,
        trie_normalizer=trie_normalizer,
    ) if args.objective == "f17_trie" else get_objective(args.objective, beta=args.beta)

    state_path = os.path.join(lang_dir, "gpoptval4_state.pkl")
    history_path = os.path.join(lang_dir, "gpoptval4_history.csv")
    final_patterns_path = os.path.join(lang_dir, "gpoptval4_final.pat")
    os.makedirs(lang_dir, exist_ok=True)

    bounds = [(1, args.max_bad_weight)] * 4 + [(1, args.max_threshold)]
    min_samples = args.batch_size
    if args.resume and os.path.exists(state_path):
        optimizer = GPOptimizer.load(state_path, objective)
        optimizer.min_samples_for_gp = min_samples
        optimizer.bounds = bounds
        print(f"Resumed from {len(optimizer.X)} observations")
    else:
        optimizer = GPOptimizer(
            objective,
            seed=args.seed,
            bounds=bounds,
            min_samples_for_gp=min_samples,
        )
        print("Starting fresh optimization")

    print(f"Objective: {objective.name}")
    print(f"Dataset: {wordlist_path}")
    print(f"Translate: {translate_path}")
    print(f"Split counts: train={split['train_count']}, validation={split['validation_count']}, test={split['test_count']}")
    print(f"Pattern ranges: {pat_ranges}")
    print(f"Batch semantics: {args.batch_size} suggestions per GP iteration, evaluated in parallel")
    print("Exploration: no coarse-grid phase; the initial batch is random, then UCB handles exploration")

    history_rows = read_history_csv(history_path) if args.resume else []
    observation = len(optimizer.results)
    start_time = time.time()

    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        suggestions = optimizer.suggest_batch(args.batch_size, ucb_kappa=args.ucb_kappa)

        with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {
                executor.submit(
                    evaluate_parameter_set,
                    args.patgen,
                    split["train"],
                    split["validation"],
                    translate_path,
                    params,
                    pat_ranges,
                    args.good_weight,
                    args.verbose,
                    f"{iteration}_{i}",
                ): params
                for i, params in enumerate(suggestions)
            }
            for future in as_completed(futures):
                params = futures[future]
                try:
                    params, results = future.result()
                except Exception as exc:
                    results = failed_evaluation_result(split["validation"])
                    print(f"  Failed: params={params}: {exc!r}")
                score = optimizer.update(
                    params,
                    results["good"],
                    results["bad"],
                    results["missed"],
                    n_patterns=results["n_patterns"],
                    trie_nodes=results["trie_nodes"],
                )
                observation += 1
                history_rows.append(history_row(observation, params, results, score))
                print(f"  Tested: params={params}")
                print(format_result_line("validation", results, score))

        best = optimizer.best_so_far()
        optimizer.save(state_path)
        write_history_csv(history_path, history_rows)
        elapsed = time.time() - start_time
        print(f"  Best so far: params={best['params']}, score={best['score']:.6f}, elapsed={elapsed:.1f}s")

    if args.final_exploitation > 0:
        print(f"\n{'=' * 60}")
        print(f"Final exploitation phase: {args.final_exploitation} predicted-best evaluations")
        best_params_to_test = optimizer.exploit_best(n=args.final_exploitation)
        with ProcessPoolExecutor(max_workers=args.final_exploitation) as executor:
            futures = {
                executor.submit(
                    evaluate_parameter_set,
                    args.patgen,
                    split["train"],
                    split["validation"],
                    translate_path,
                    params,
                    pat_ranges,
                    args.good_weight,
                    args.verbose,
                    f"final_{i}",
                ): params
                for i, params in enumerate(best_params_to_test)
            }
            for future in as_completed(futures):
                params = futures[future]
                try:
                    params, results = future.result()
                except Exception as exc:
                    results = failed_evaluation_result(split["validation"])
                    print(f"  Failed: params={params}: {exc!r}")
                score = optimizer.update(
                    params,
                    results["good"],
                    results["bad"],
                    results["missed"],
                    n_patterns=results["n_patterns"],
                    trie_nodes=results["trie_nodes"],
                )
                observation += 1
                history_rows.append(history_row(observation, params, results, score))
                print(f"  Tested: params={params}")
                print(format_result_line("validation", results, score))
        optimizer.save(state_path)
        write_history_csv(history_path, history_rows)

    best = optimizer.best_so_far()
    export_path = final_patterns_path if args.export_final_patterns else ""
    _, test_results = evaluate_parameter_set(
        args.patgen,
        split["train"],
        split["test"],
        translate_path,
        tuple(best["params"]),
        pat_ranges,
        args.good_weight,
        args.verbose,
        "test",
        export_patterns_path=export_path,
    )

    print(f"\n{'=' * 60}")
    print("GPoptval4 complete")
    print(f"Best parameters: {tuple(best['params'])}")
    print(format_result_line("validation-selected", best, best["score"]))
    print(format_result_line("held-out test", test_results))
    print(f"State saved to: {state_path}")
    print(f"History saved to: {history_path}")
    if export_path:
        print(f"Final train-only patterns saved to: {export_path}")
    if fixed_trie_normalizer:
        warn_fixed_trie_normalizer(
            "scripts.optimize_validation",
            trie_normalizer,
            "END WARNING",
        )


if __name__ == "__main__":
    main()
