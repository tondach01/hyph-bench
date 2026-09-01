#!/usr/bin/env python3
"""
Threshold ablation for patgen parameter optimization (GPoptval4 protocol).

Ablates two decisions in the paper's main experiments:
  1. threshold fixed at 1  vs. tunable threshold
  2. one threshold shared across all levels vs. per-level thresholds

Three threshold modes (same grouped-hash 8/1/1 validation protocol, objective,
budget, seed, and bounds on bad_weight as scripts.optimize_validation):

  fixed1    bad_1..4 in (1, max_bad), threshold == 1                (paper arm)
  shared    bad_1..4 in (1, max_bad), one threshold in (min_thr, max_thr)
  perlayer  bad_1..4 in (1, max_bad), thr_1..4 each in (min_thr, max_thr)

Every arm gets an identical budget of patgen evaluations:
iterations * batch_size + final_exploitation (default 30*5 + 3 = 153).
GP spends the final 3 on exploitation (as in GPoptval4); TPE/Random spend
them on ordinary sampler rounds. Selection is always best-so-far on the
validation split; the selected parameters are finally evaluated once on the
held-out test split.

TPE mirrors scripts.compare_hpo_methods: optuna TPESampler with
n_startup_trials=10, one suggest_int per dimension, same seed.

Param layout passed to train_patgen_multilevel (threshold via len):
  5 values -> bad_1..4 + shared threshold; 8 values -> bad_1..4 + thr_1..4.

Usage:
  uv run python -m scripts.threshold_ablation --lang cssk/cshyphen \
      --threshold-mode perlayer --method gp --patgen ~/patgen-10x --resume
"""

import argparse
import csv
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .gp_optimizer import GPOptimizer
from .objectives import ObjectiveFunction, get_objective
from .dataset_split import create_clean_split
from .optimize_validation import (
    evaluate_parameter_set,
    f17_score,
    failed_evaluation_result,
    format_result_line,
    precision,
    recall,
)
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)

MAX_PARAMS = 8  # bad_1..4 + thr_1..4
HISTORY_FIELDNAMES = (
    ["observation"]
    + [f"param_{i}" for i in range(1, MAX_PARAMS + 1)]
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


def bounds_for_mode(args) -> List[Tuple[int, int]]:
    """Search bounds implied by the threshold mode; fixed1 reproduces GPoptval4."""
    bad = (1, args.max_bad_weight)
    if args.threshold_mode == "fixed1":
        return [bad] * 4 + [(1, 1)]
    if args.threshold_mode == "shared":
        return [bad] * 4 + [(args.min_threshold, args.max_threshold)]
    if args.threshold_mode == "perlayer":
        return [bad] * 4 + [(args.min_threshold, args.max_threshold)] * 4
    raise ValueError(f"Unknown threshold mode: {args.threshold_mode}")


def history_row(observation: int, params: Tuple[int, ...],
                results: Dict[str, int], score: float) -> Dict[str, object]:
    row: Dict[str, object] = {
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
    for i in range(1, MAX_PARAMS + 1):
        row[f"param_{i}"] = params[i - 1] if i <= len(params) else ""
    return row


def write_history_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_history_csv(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Search methods
# --------------------------------------------------------------------------- #
class GpSearch:
    """GPOptimizer wrapper with the GPoptval4 iteration/exploitation protocol."""

    method = "gp"

    def __init__(self, objective: ObjectiveFunction, bounds, args):
        self.opt = GPOptimizer(
            objective,
            seed=args.seed,
            bounds=bounds,
            min_samples_for_gp=args.batch_size,
        )
        self.args = args
        self.state_path = ""

    def suggest(self, n: int) -> List[Tuple[int, ...]]:
        return list(self.opt.suggest_batch(n, ucb_kappa=self.args.ucb_kappa))

    def tell(self, params: Tuple[int, ...], results: Dict[str, int]) -> float:
        return self.opt.update(
            params,
            results["good"],
            results["bad"],
            results["missed"],
            n_patterns=results["n_patterns"],
            trie_nodes=results["trie_nodes"],
        )

    def n_observed(self) -> int:
        return len(self.opt.results)

    def best(self) -> Optional[Dict]:
        return self.opt.best_so_far()

    def exploitation_candidates(self, n: int) -> List[Tuple[int, ...]]:
        return list(self.opt.exploit_best(n=n))

    def save(self, path: str) -> None:
        self.opt.save(path)

    @classmethod
    def restore(cls, path: str, objective: ObjectiveFunction, bounds, args) -> "GpSearch":
        inst = cls(objective, bounds, args)
        inst.opt = GPOptimizer.load(path, objective)
        inst.opt.min_samples_for_gp = args.batch_size
        inst.opt.bounds = bounds
        inst.opt.n_dims = len(bounds)
        return inst


class OptunaSearch:
    """ask/tell driver matching compare_hpo_methods (TPE / Random via optuna)."""

    def __init__(self, method: str, objective: ObjectiveFunction, bounds, args):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.method = method
        self.objective = objective
        self.bounds = bounds
        self.args = args
        self.results: List[Dict] = []
        if method == "random":
            sampler = optuna.samplers.RandomSampler(seed=args.seed)
        else:
            startup = max(1, min(args.tpe_startup, args.iterations))
            sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=startup)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

    def suggest(self, n: int) -> List[Tuple[int, ...]]:
        self._pending = [self.study.ask() for _ in range(n)]
        return [
            tuple(t.suggest_int(f"p{j}", lo, hi) for j, (lo, hi) in enumerate(self.bounds))
            for t in self._pending
        ]

    def tell(self, params: Tuple[int, ...], results: Dict[str, int]) -> float:
        score = self.objective.score(
            results["good"],
            results["bad"],
            results["missed"],
            n_patterns=results["n_patterns"],
            trie_nodes=results["trie_nodes"],
        )
        trial = self._pending.pop(0)
        self.study.tell(trial, score)
        self.results.append({"params": params, "score": score})
        return score

    def n_observed(self) -> int:
        return len(self.results)

    def best(self) -> Optional[Dict]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r["score"])

    def exploitation_candidates(self, n: int) -> List[Tuple[int, ...]]:
        return []  # no separate exploitation phase; budget spent on sampler rounds

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "method": self.method,
                    "study": self.study,
                    "results": self.results,
                    "bounds": self.bounds,
                },
                handle,
            )

    @classmethod
    def restore(cls, path: str, objective: ObjectiveFunction, bounds, args) -> "OptunaSearch":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        inst = cls(state["method"], objective, bounds, args)
        inst.study = state["study"]
        inst.results = state["results"]
        return inst


def make_search(method: str, objective: ObjectiveFunction, bounds, args, state_path: str):
    if method == "random":
        method_key = "random"
    elif method == "tpe":
        method_key = "tpe"
    else:
        method_key = "gp"
    if args.resume and os.path.exists(state_path):
        cls = GpSearch if method_key == "gp" else OptunaSearch
        search = cls.restore(state_path, objective, bounds, args)
        print(f"Resumed from {search.n_observed()} observations in {state_path}")
        return search
    if method_key == "gp":
        return GpSearch(objective, bounds, args)
    return OptunaSearch(method_key, objective, bounds, args)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold ablation (fixed1 / shared / perlayer) with GP, TPE, or Random",
    )
    parser.add_argument("--lang", required=True, help="Language/dataset id, e.g. cssk/cshyphen")
    parser.add_argument("--wordlist", help="Override wordlist path")
    parser.add_argument("--translate", help="Override translate path")
    parser.add_argument("--patgen", default="patgen", help="Path to patgen binary")
    parser.add_argument("--profile", help="Profile file for pat_start/pat_finish")
    parser.add_argument("--output-dir", default="results/threshold_ablation")
    parser.add_argument("--threshold-mode", choices=["fixed1", "shared", "perlayer"], required=True)
    parser.add_argument("--method", choices=["gp", "tpe", "random"], default="gp")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Suggestions per round (budget semantics, unchanged by --workers)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel patgen evaluations (default: batch-size); affects only wall time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ucb-kappa", type=float, default=2.5)
    parser.add_argument("--good-weight", type=int, default=3)
    parser.add_argument("--max-bad-weight", type=int, default=30)
    parser.add_argument("--min-threshold", type=int, default=1)
    parser.add_argument("--max-threshold", type=int, default=5)
    parser.add_argument("--objective", choices=["f17", "f17_trie"], default="f17_trie")
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--final-exploitation", type=int, default=3,
                        help="Evaluations GP spends on exploitation; other methods spend them on rounds")
    parser.add_argument("--tpe-startup", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export-final-patterns", action="store_true")
    args = parser.parse_args()

    if args.wordlist and args.translate:
        wordlist_path = os.path.abspath(args.wordlist)
        translate_path = os.path.abspath(args.translate)
    else:
        wordlist_path, translate_path = find_dataset(args.lang)
    run_dir = os.path.join(
        args.output_dir, args.lang, f"{args.threshold_mode}_{args.method}"
    )
    os.makedirs(run_dir, exist_ok=True)
    split = create_clean_split(
        wordlist_path, os.path.join(run_dir, "splits"), seed=args.seed
    )


    trie_normalizer = None
    fixed_trie_normalizer = False
    if args.objective == "f17_trie":
        trie_normalizer, fixed_trie_normalizer = resolve_trie_normalizer(
            args, split["unique"], "scripts.threshold_ablation", dataset=args.lang,
        )

    pat_ranges = parse_profile(args.profile) if args.profile else DEFAULT_PAT_RANGES
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

    bounds = bounds_for_mode(args)
    n_dims = len(bounds)
    budget = args.iterations * args.batch_size + args.final_exploitation
    workers = args.workers or args.batch_size


    state_path = os.path.join(run_dir, "state.pkl")
    history_path = os.path.join(run_dir, "history.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    final_patterns_path = os.path.join(run_dir, "final.pat")

    search = make_search(args.method, objective, bounds, args, state_path)

    print(f"Objective: {objective.name}")
    print(f"Dataset: {wordlist_path}")
    print(f"Translate: {translate_path}")
    print(f"Method: {search.method} | Threshold mode: {args.threshold_mode}")
    print(f"Search space ({n_dims}-D): {bounds}")
    print(f"Split counts: train={split['train_count']}, "
          f"validation={split['validation_count']}, test={split['test_count']}")
    print(f"Pattern ranges: {pat_ranges}")
    print(f"Budget: {args.iterations} rounds x {args.batch_size} + "
          f"{args.final_exploitation} final = {budget} patgen evaluations; "
          f"{workers} parallel workers")

    history_rows = read_history_csv(history_path) if args.resume else []
    if args.resume and len(history_rows) != search.n_observed():
        raise RuntimeError(
            f"History ({len(history_rows)} rows) and state "
            f"({search.n_observed()} observations) disagree; refusing to resume."
        )

    observation = len(history_rows)
    start_time = time.time()

    def run_candidates(candidates: List[Tuple[int, ...]], tag_prefix: str) -> None:
        nonlocal observation
        with ProcessPoolExecutor(max_workers=workers) as executor:
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
                    f"{tag_prefix}_{i}",
                ): params
                for i, params in enumerate(candidates)
            }
            # Optuna ask/tell must be fed back in suggestion order; buffer results.
            buffered: Dict[Tuple[int, ...], Dict[str, int]] = {}
            for future in as_completed(futures):
                params = futures[future]
                try:
                    _, results = future.result()
                except Exception as exc:
                    results = failed_evaluation_result(split["validation"])
                    print(f"  Failed: params={params}: {exc!r}")
                buffered[params] = results
        for params in candidates:
            results = buffered[params]
            score = search.tell(params, results)
            observation += 1
            history_rows.append(history_row(observation, params, results, score))
            print(f"  Tested: params={params}")
            print(format_result_line("validation", results, score))

    # Main search rounds.
    while observation < args.iterations * args.batch_size:
        round_no = observation // args.batch_size
        print(f"\n{'=' * 60}\nRound {round_no + 1}/{args.iterations}")
        n_suggest = min(args.batch_size, args.iterations * args.batch_size - observation)
        run_candidates(search.suggest(n_suggest), f"r{round_no}")
        search.save(state_path)
        write_history_csv(history_path, history_rows)
        best = search.best()
        elapsed = time.time() - start_time
        print(f"  Best so far: params={best['params']}, score={best['score']:.6f}, "
              f"elapsed={elapsed:.1f}s")

    # Final budget: GP exploitation, sampler rounds for TPE/Random.
    remaining = budget - observation
    if remaining > 0:
        print(f"\n{'=' * 60}\nFinal phase: {remaining} evaluations")
        if search.method == "gp":
            candidates = search.exploitation_candidates(remaining)
        else:
            candidates = search.suggest(remaining)
        run_candidates(candidates, "final")
        search.save(state_path)
        write_history_csv(history_path, history_rows)

    best = search.best()
    best_params = tuple(best["params"])
    # Metrics of the selected params, reconstructed from history so that the
    # reporting path is identical for GP and optuna-backed searches.
    best_metrics = buffered_best(history_rows, best_params)

    export_path = final_patterns_path if args.export_final_patterns else ""
    _, test_results = evaluate_parameter_set(
        args.patgen,
        split["train"],
        split["test"],
        translate_path,
        best_params,
        pat_ranges,
        args.good_weight,
        args.verbose,
        "test",
        export_patterns_path=export_path,
    )
    test_f17 = f17_score(test_results["good"], test_results["bad"], test_results["missed"])
    wall_time = time.time() - start_time

    summary = {
        "lang": args.lang,
        "method": search.method,
        "threshold_mode": args.threshold_mode,
        "objective": args.objective,
        "seed": args.seed,
        "budget": budget,
        "n_evaluations": observation,
        "bounds": bounds,
        "best_params": list(best_params),
        "thresholds": threshold_summary(args.threshold_mode, best_params),
        "best_objective": best["score"],
        "validation": {
            "f17": f17_score(best_metrics["good"], best_metrics["bad"], best_metrics["missed"]),
            "precision": precision(best_metrics["good"], best_metrics["bad"]),
            "recall": recall(best_metrics["good"], best_metrics["missed"]),
            "good": best_metrics["good"],
            "bad": best_metrics["bad"],
            "missed": best_metrics["missed"],
            "trie_nodes": best_metrics["trie_nodes"],
            "n_patterns": best_metrics["n_patterns"],
        },
        "test": {
            "f17": test_f17,
            "precision": precision(test_results["good"], test_results["bad"]),
            "recall": recall(test_results["good"], test_results["missed"]),
            "good": test_results["good"],
            "bad": test_results["bad"],
            "missed": test_results["missed"],
            "trie_nodes": test_results["trie_nodes"],
            "n_patterns": test_results["n_patterns"],
        },
        "wall_time_sec": wall_time,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n{'=' * 60}\nThreshold ablation run complete")
    print(f"Best parameters ({'b1 b2 b3 b4 thr' if n_dims == 5 else 'b1..b4 thr1..thr4'}): "
          f"{best_params}")
    print(f"Thresholds: {summary['thresholds']}")
    print(format_result_line("validation-selected", best_metrics, best["score"]))
    print(format_result_line("held-out test", test_results))
    print(f"Summary: {summary_path}")
    print(f"History: {history_path}")
    print(f"Wall time: {wall_time:.1f}s")
    if args.export_final_patterns:
        print(f"Final train-only patterns saved to: {final_patterns_path}")
    if fixed_trie_normalizer:
        warn_fixed_trie_normalizer("scripts.threshold_ablation", trie_normalizer, "END WARNING")


def buffered_best(history_rows: List[Dict[str, object]], best_params: Tuple[int, ...]) -> Dict[str, int]:
    """Recover the recorded metrics of the best parameter set from history."""
    for row in history_rows:
        row_params = tuple(
            int(row[f"param_{i}"]) for i in range(1, len(best_params) + 1)
        )
        if row_params == best_params:
            return {
                "good": int(row["validation_good"]),
                "bad": int(row["validation_bad"]),
                "missed": int(row["validation_missed"]),
                "train_good": int(row["train_good"]),
                "train_bad": int(row["train_bad"]),
                "train_missed": int(row["train_missed"]),
                "n_patterns": int(row["n_patterns"]),
                "trie_nodes": int(row["trie_nodes"]),
            }
    raise RuntimeError(f"Best params {best_params} not found in history")


def threshold_summary(mode: str, params: Tuple[int, ...]) -> Dict[str, object]:
    if mode == "fixed1":
        return {"mode": mode, "shared": 1, "per_level": [1, 1, 1, 1]}
    if mode == "shared":
        return {"mode": mode, "shared": params[4], "per_level": [params[4]] * 4}
    return {"mode": mode, "shared": None, "per_level": list(params[4:8])}


if __name__ == "__main__":
    main()
