#!/usr/bin/env python3
"""
Budget-matched HPO baseline comparison: GP vs Random Search vs TPE.

Every method is given an IDENTICAL evaluation budget: ``--iterations`` search
rounds, each with ``--batch-size`` candidate objective evaluations. All methods
optimize the SAME objective over the SAME 5-integer search space (4
``bad_weight`` values + 1 shared ``threshold``). This isolates the effect of the
*search strategy* from the effect of running *any* optimizer at all, which is
exactly the reviewer request: show that the gain over hand-tuned profiles comes
from optimization in general, not specifically from the Gaussian Process.

Methods:
  gp      project's ``GPOptimizer`` (sklearn GP + UCB) - the paper's method
  random  Optuna ``RandomSampler``
  tpe     Optuna ``TPESampler`` (Tree-structured Parzen Estimator)

Each candidate is evaluated with n-fold cross-validation (``--nfold``) through
the shared patgen scoring harness, then scored with the chosen objective
(default ``f17_trie``). Hand-tuned baseline profiles can be evaluated for
reference rows via ``--baseline-profile name=path``.

Usage:
    python -m scripts.compare_hpo_methods --datasets de/wortliste \
        --methods gp random tpe --objective f17_trie --iterations 30 \
        --good-weight 3 --max-bad-weight 30 --max-threshold 1 --ucb-kappa 2.5 \
        --trie-weight 0.0005 --nfold 10 \
        --baseline-profile wortliste=profiles/wortliste.in \
        --output-dir results/hpo_baselines
"""

import argparse
import csv
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .objectives import get_objective, ObjectiveFunction
from .gp_optimizer import GPOptimizer
from .hyperparameters.score import PatgenScorer
from .hyperparameters import combine, sample, metaheuristic
from .train_test import NFoldCrossValidator
from .cross_validate import run_cross_validation
from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)
from .dataset_split import resolve_word_entries
from .optimize import (run_patgen_multilevel, run_parallel_patgen,
                       run_parallel_cross_validation)


def evaluate_params(params: Tuple[int, ...], ctx: dict) -> dict:
    """
    Search-time evaluation of one parameter set: a single full-data patgen run
    (4 levels), returning good/bad/missed/trie_nodes/n_patterns. This is the
    signal the optimizer sees, exactly as the paper's GP search does -- fast and
    identical for every method. Generalization (k-fold CV) is measured only on
    each method's final winner, in evaluate_cv().

    Deterministic for a given dataset, so results are cached and shared across
    methods (a re-asked point still costs the asking method one unit of its
    budget, but is not recomputed).
    """
    key = tuple(int(x) for x in params)
    cache = ctx["cache"]
    if key in cache:
        return cache[key]

    scorer: PatgenScorer = ctx["scorer"]
    scorer.reset()
    res = run_patgen_multilevel(scorer, key, ctx["pat_ranges"],
                                good_weight=ctx["good_weight"])
    cache[key] = res
    return res


def evaluate_cv(params: Tuple[int, ...], ctx: dict) -> dict:
    """k-fold cross-validation of a final parameter set (generalization measure)."""
    res, _ = run_cross_validation(
        ctx["wl"], ctx["tr"], ctx["lang"], [int(x) for x in params], ctx["pat_ranges"],
        nfold=ctx["nfold"], good_weight=ctx["good_weight"], verbose=ctx["verbose"],
    )
    res.setdefault("n_patterns", 0)
    return res


def evaluate_batch(cands: List[Tuple[int, ...]], ctx: dict) -> List[dict]:
    """
    Evaluate a batch of parameter sets, returning results aligned to ``cands``.
    Cache hits are served immediately; cache misses are run in parallel worker
    processes via the project's tested run_parallel_patgen (each worker builds
    its own scorer + temp dir). Counting is unchanged: every candidate the
    optimizer asks for still costs one unit of its budget.
    """
    cache = ctx["cache"]
    results: List[Optional[dict]] = [None] * len(cands)
    misses: Dict[Tuple[int, ...], List[int]] = {}
    for idx, params in enumerate(cands):
        key = tuple(int(x) for x in params)
        if key in cache:
            results[idx] = cache[key]
        else:
            misses.setdefault(key, []).append(idx)

    if misses:
        if ctx["batch_size"] > 1 and len(misses) > 1:
            with ProcessPoolExecutor(max_workers=min(ctx["batch_size"], len(misses))) as ex:
                futs = {
                    ex.submit(run_parallel_patgen, "patgen", ctx["wl"], ctx["tr"],
                              key, ctx["pat_ranges"], ctx["good_weight"],
                              ctx["verbose"], wi): key
                    for wi, key in enumerate(misses)
                }
                for fut in as_completed(futs):
                    _, res = fut.result()
                    cache[futs[fut]] = res
        else:
            scorer: PatgenScorer = ctx["scorer"]
            for key in misses:
                scorer.reset()
                cache[key] = run_patgen_multilevel(
                    scorer, key, ctx["pat_ranges"], good_weight=ctx["good_weight"])
        for key, idxs in misses.items():
            for idx in idxs:
                results[idx] = cache[key]
    return results


def score_result(objective: ObjectiveFunction, res: dict) -> float:
    return objective.score(
        good=res["good"], bad=res["bad"], missed=res["missed"],
        trie_nodes=res.get("trie_nodes", 0), n_patterns=res.get("n_patterns", 0),
        f17cv=res.get("f_17", 0.0),
    )


def f17_of(res: dict) -> float:
    """F_{1/7} computed from good/bad/missed (independent of the trie penalty)."""
    good, bad, missed = res["good"], res["bad"], res["missed"]
    if good == 0:
        return 0.0
    precision = good / (good + bad) if (good + bad) > 0 else 0.0
    recall = good / (good + missed) if (good + missed) > 0 else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    beta_sq = (1 / 7) ** 2
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def run_method(method: str, ctx: dict) -> dict:
    """Run one HPO method for exactly ``ctx['iterations']`` search rounds."""
    objective: ObjectiveFunction = ctx["objective"]
    bounds: List[Tuple[int, int]] = ctx["bounds"]
    iters: int = ctx["iterations"]
    seed: int = ctx["seed"]

    history: List[dict] = []
    best: Optional[dict] = None

    def record(eval_idx: int, params, res: dict, score: float) -> None:
        nonlocal best
        rec = {
            "method": method,
            "eval": eval_idx,
            "params": tuple(int(x) for x in params),
            "good": res["good"], "bad": res["bad"], "missed": res["missed"],
            "trie_nodes": res.get("trie_nodes", 0),
            "n_patterns": res.get("n_patterns", 0),
            "f_17": f17_of(res),
            "score": score,
        }
        if best is None or score > best["score"]:
            best = rec
        row = dict(rec)
        row["best_score_so_far"] = best["score"]
        row["best_f17_so_far"] = best["f_17"]
        history.append(row)

    batch_size = max(1, ctx.get("batch_size", 1))

    if method == "gp":
        opt = GPOptimizer(objective, seed=seed, bounds=bounds,
                          min_samples_for_gp=min(5, iters))
    elif method in ("random", "tpe"):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if method == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            startup = max(1, min(ctx["tpe_startup"], iters))
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup)
        study = optuna.create_study(direction="maximize", sampler=sampler)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Paper convention: `iters` optimization rounds (= model re-plans / GP fits),
    # each proposing `batch_size` candidates that are evaluated in parallel and
    # fed back before the next round. Total objective evaluations (the matched
    # budget) = iters * batch_size, identical for every method.
    total_budget = iters * batch_size
    eval_idx = 0
    for _round in range(iters):
        if method == "gp":
            cands = list(opt.suggest_batch(batch_size, ucb_kappa=ctx["ucb_kappa"]))
            trials = [None] * len(cands)
        else:
            trials = [study.ask() for _ in range(batch_size)]
            cands = [tuple(t.suggest_int(f"p{j}", lo, hi)
                           for j, (lo, hi) in enumerate(bounds)) for t in trials]

        batch_res = evaluate_batch(cands, ctx)

        for params, res, trial in zip(cands, batch_res, trials):
            if method == "gp":
                score = opt.update(
                    params, good=res["good"], bad=res["bad"], missed=res["missed"],
                    n_patterns=res.get("n_patterns", 0),
                    trie_nodes=res.get("trie_nodes", 0), f_17=res.get("f_17", 0.0),
                )
            else:
                score = score_result(objective, res)
                study.tell(trial, score)
            record(eval_idx, params, res, score)
            _log_eval(method, eval_idx, total_budget, params, res, score, best)
            eval_idx += 1

    assert len(history) == total_budget, (
        f"budget mismatch for {method}: {len(history)} != {total_budget}")
    return {"history": history, "best": best}


def _log_eval(method, i, iters, params, res, score, best):
    print(f"  [{method}] {i + 1}/{iters} params={tuple(int(x) for x in params)} "
          f"good={res['good']:.0f} bad={res['bad']:.0f} missed={res['missed']:.0f} "
          f"trie={res.get('trie_nodes', 0):.0f} f17={f17_of(res):.5f} "
          f"score={score:.5f} | best_score={best['score']:.5f}")


# --------------------------------------------------------------------------- #
# Hand-tuned baseline profile evaluation
# --------------------------------------------------------------------------- #
def cross_validate_profile(profile_path: str, wl_path: str, tr_path: str,
                           nfold: int, verbose: bool = False) -> dict:
    """N-fold CV of an existing hand-tuned profile file (full per-level profile)."""
    run_id = uuid.uuid4().hex
    scorer = PatgenScorer("patgen", "", tr_path, verbose=verbose, tmp_suffix=run_id)
    sampler = sample.FileSampler(profile_path)
    meta = metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = combine.SimpleCombiner(meta, verbose=verbose)
    validator = NFoldCrossValidator(combiner, tr_path, nfold, tmp_suffix=run_id)
    validator.validate(wl_path, verbose=verbose)
    scorer.clean()
    lang_name = os.path.basename(os.path.dirname(wl_path))
    ds_name = os.path.basename(wl_path).replace("_dis.wlh", "").replace(".wlh", "")
    res, _ = validator.report(lang=lang_name, name=ds_name,
                              profile=os.path.basename(profile_path), tabular=True)
    res.setdefault("n_patterns", 0)
    return res


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_history_csv(path: str, history: List[dict]) -> None:
    cols = ["method", "eval", "param_1", "param_2", "param_3", "param_4", "param_5",
            "good", "bad", "missed", "trie_nodes", "n_patterns", "f_17", "score",
            "best_score_so_far", "best_f17_so_far"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in history:
            p = list(r["params"]) + [""] * (5 - len(r["params"]))
            w.writerow([r["method"], r["eval"], *p[:5],
                        f"{r['good']:.2f}", f"{r['bad']:.2f}", f"{r['missed']:.2f}",
                        f"{r['trie_nodes']:.1f}", r["n_patterns"],
                        f"{r['f_17']:.6f}", f"{r['score']:.6f}",
                        f"{r['best_score_so_far']:.6f}", f"{r['best_f17_so_far']:.6f}"])


def make_markdown_table(rows: List[dict]) -> str:
    lines = ["| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |",
             "|---|---|---|---|---|"]
    for r in rows:
        params = "-" if r["params"] is None else " ".join(str(x) for x in r["params"])
        score = "-" if r["score"] is None else f"{r['score']:.5f}"
        lines.append(f"| {r['label']} | {r['f_17']:.5f} | {r['trie_nodes']:.0f} "
                     f"| {params} | {score} |")
    return "\n".join(lines)


def make_latex_table(dataset: str, rows: List[dict], nfold: int = 10,
                     budget: int = 100) -> str:
    best_f17 = max(r["f_17"] for r in rows)
    min_trie = min(r["trie_nodes"] for r in rows)
    out = [
        r"\begin{table}[tb]",
        r"  \centering",
        rf"  \caption{{Budget-matched HPO comparison on {dataset.replace('_', '/')}: "
        rf"hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each "
        rf"given the same budget of {budget} patgen evaluations. F$_{{1/7}}$ and trie "
        rf"nodes are {nfold}-fold cross-validation means of each method's winning profile.}}",
        rf"  \label{{tab:hpo-baselines-{dataset}}}",
        r"  \begin{tabular}{l c c}",
        r"    \toprule",
        r"    Method & F$_{1/7}$ & Trie nodes \\",
        r"    \midrule",
    ]
    for r in rows:
        f = f"{r['f_17']:.4f}"
        t = f"{r['trie_nodes']:.0f}"
        if abs(r["f_17"] - best_f17) < 1e-9:
            f = rf"\textbf{{{f}}}"
        if abs(r["trie_nodes"] - min_trie) < 0.5:
            t = rf"\textbf{{{t}}}"
        out.append(rf"    {r['label']} & {f} & {t} \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_objective(args, trie_normalizer: Optional[float]) -> ObjectiveFunction:
    if args.objective == "f17":
        return get_objective("f17", beta=args.beta)
    if args.objective == "f17_trie":
        return get_objective("f17_trie", beta=args.beta,
                             trie_weight=args.trie_weight,
                             trie_normalizer=trie_normalizer)
    # Fall back to the factory for any other registered objective.
    return get_objective(args.objective)


def parse_baseline_arg(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    out = []
    for it in items or []:
        if "=" in it:
            name, path = it.split("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(it))[0], it
        out.append((name, path))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Budget-matched HPO baseline comparison (GP vs Random vs TPE)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset ids, e.g. de/wortliste cssk/cshyphen")
    parser.add_argument("--methods", nargs="+", default=["gp", "random", "tpe"],
                        choices=["gp", "random", "tpe"])
    parser.add_argument("--objective", default="f17_trie")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Optimization rounds per method = number of model "
                             "re-plans/GP fits. Total objective evaluations (the "
                             "matched budget) = iterations * batch_size.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Candidates proposed per round and evaluated in parallel "
                             "(via ProcessPoolExecutor). Total budget = iterations * "
                             "batch_size, identical across methods.")
    parser.add_argument("--good-weight", type=int, default=1)
    parser.add_argument("--max-bad-weight", type=int, default=30)
    parser.add_argument("--max-threshold", type=int, default=5)
    parser.add_argument("--ucb-kappa", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1 / 7)
    parser.add_argument("--trie-weight", type=float, default=0.0005)
    add_trie_normalizer_args(parser)
    parser.add_argument("--nfold", type=int, default=10,
                        help="Folds for the final k-fold CV of each method's winner and "
                             "the baselines (search itself always uses single full-train)")
    parser.add_argument("--tpe-startup", type=int, default=10,
                        help="TPE random startup trials (clamped to iterations)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=str,
                        help="Profile file providing pat_start/pat_finish ranges for the search")
    parser.add_argument("--baseline-profile", action="append",
                        help="name=path of a hand-tuned profile to CV as a reference row "
                             "(repeatable)")
    parser.add_argument("--reuse-existing-gp", action="store_true",
                        help="Reuse a saved GP state pkl instead of re-running GP (off by "
                             "default; fresh runs are recommended for a fair comparison)")
    parser.add_argument("--output-dir", type=str, default="results/hpo_baselines")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.batch_size > 1:
        print(f"[parallel] batch_size={args.batch_size}: candidates per round "
              f"evaluated across up to {args.batch_size} worker processes.")

    os.makedirs(args.output_dir, exist_ok=True)
    pat_ranges = parse_profile(args.profile) if args.profile else DEFAULT_PAT_RANGES
    baselines = parse_baseline_arg(args.baseline_profile)

    all_summaries = {}
    fixed_trie_normalizers = []

    for dataset in args.datasets:
        print("\n" + "=" * 72)
        print(f"DATASET: {dataset}")
        print("=" * 72)
        wl, tr = find_dataset(dataset)
        print(f"Wordlist:  {wl}")
        print(f"Translate: {tr}")
        resolved_entries, _, _ = resolve_word_entries(wl)


        fixed_trie_normalizer = False
        trie_normalizer = None
        if args.objective == "f17_trie":
            trie_normalizer, fixed_trie_normalizer = resolve_trie_normalizer(
                args,
                wl,
                "scripts.compare_hpo_methods",
                dataset=dataset,
                wordlist_size=len(resolved_entries),
            )
            if fixed_trie_normalizer:
                fixed_trie_normalizers.append(trie_normalizer)

        objective = build_objective(args, trie_normalizer)
        bounds = [(1, args.max_bad_weight)] * 4 + [(1, args.max_threshold)]
        total_budget = args.iterations * args.batch_size
        print(f"Objective: {objective.name} | bounds={bounds} | "
              f"pat_ranges={pat_ranges} | nfold={args.nfold} | "
              f"budget={args.iterations} rounds x {args.batch_size} = {total_budget} evals")

        ctx = {
            "wl": wl, "tr": tr, "lang": dataset, "pat_ranges": pat_ranges,
            "nfold": args.nfold, "good_weight": args.good_weight,
            "objective": objective, "bounds": bounds,
            "iterations": args.iterations, "seed": args.seed,
            "ucb_kappa": args.ucb_kappa, "tpe_startup": args.tpe_startup,
            "batch_size": args.batch_size,
            "verbose": args.verbose, "cache": {},
            "scorer": PatgenScorer("patgen", wl, tr, verbose=args.verbose,
                                   tmp_suffix=f"_hpo_{uuid.uuid4().hex[:8]}"),
        }

        safe = dataset.replace("/", "_")

        # HPO methods: budget-matched search on the single full-train objective.
        method_results = {}
        for method in args.methods:
            print(f"\n--- method: {method} (search, {args.iterations} rounds x "
                  f"{args.batch_size} = {total_budget} evals) ---")
            t0 = time.time()
            mr = run_method(method, ctx)
            method_results[method] = mr
            write_history_csv(
                os.path.join(args.output_dir, f"{safe}_{method}_history.csv"),
                mr["history"])
            b = mr["best"]
            print(f"  best {method} (search): params={b['params']} "
                  f"train_f17={b['f_17']:.5f} train_trie={b['trie_nodes']:.0f} "
                  f"score={b['score']:.5f} "
                  f"({time.time() - t0:.1f}s, {len(ctx['cache'])} unique evals cached)")

        ctx["scorer"].clean()

        # Generalization: k-fold CV of the baselines + each method's winner. These
        # are independent jobs, run in parallel (each via the tested CV harness).
        label_map = {"gp": "GP (ours)", "random": "Random Search", "tpe": "TPE"}
        rows: List[Optional[dict]] = [None] * (len(baselines) + len(args.methods))
        print(f"\n--- {args.nfold}-fold CV of {len(rows)} profiles in parallel "
              f"(up to {max(1, args.batch_size)} workers) ---")
        t0 = time.time()
        max_cv_workers = min(len(rows), max(1, args.batch_size)) or 1
        with ProcessPoolExecutor(max_workers=max_cv_workers) as ex:
            futs = {}
            for oi, (name, path) in enumerate(baselines):
                fut = ex.submit(cross_validate_profile, path, wl, tr,
                                args.nfold, args.verbose)
                futs[fut] = ("baseline", oi, name, None)
            for mi, method in enumerate(args.methods):
                b = method_results[method]["best"]
                fut = ex.submit(run_parallel_cross_validation, wl, tr, dataset,
                                list(b["params"]), pat_ranges, args.nfold,
                                args.good_weight, args.verbose)
                futs[fut] = ("method", len(baselines) + mi, method, b)
            for fut in as_completed(futs):
                kind, order, key, b = futs[fut]
                if kind == "baseline":
                    res = fut.result()
                    rows[order] = {"label": f"Hand-tuned ({key})", "params": None,
                                   "f_17": res["f_17"], "trie_nodes": res["trie_nodes"],
                                   "search_f17": None, "score": None}
                    print(f"    [{key}] CV f17={res['f_17']:.5f} "
                          f"trie={res['trie_nodes']:.0f}")
                else:
                    _, res = fut.result()
                    rows[order] = {"label": label_map[key], "params": b["params"],
                                   "f_17": res["f_17"], "trie_nodes": res["trie_nodes"],
                                   "search_f17": b["f_17"], "score": b["score"]}
                    print(f"    [{key}] winner {b['params']} CV f17={res['f_17']:.5f} "
                          f"trie={res['trie_nodes']:.0f}")
        print(f"  CV phase done ({time.time() - t0:.1f}s)")

        md = make_markdown_table(rows)
        latex = make_latex_table(safe, rows, nfold=args.nfold, budget=total_budget)
        print("\n" + md + "\n")

        summary = {
            "dataset": dataset, "budget_evals": total_budget,
            "rounds": args.iterations, "batch_size": args.batch_size,
            "nfold": args.nfold,
            "objective": objective.name, "trie_normalizer": trie_normalizer,
            "bounds": bounds, "pat_ranges": pat_ranges, "good_weight": args.good_weight,
            "seed": args.seed, "rows": rows,
            "unique_evaluations": len(ctx["cache"]),
        }
        all_summaries[dataset] = summary
        with open(os.path.join(args.output_dir, f"{safe}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.output_dir, f"{safe}_summary.md"), "w") as f:
            f.write(f"# HPO comparison: {dataset}\n\n")
            f.write(f"Budget {total_budget} evals/method "
                    f"({args.iterations} rounds x {args.batch_size}), "
                    f"{args.nfold}-fold CV, objective {objective.name}.\n\n")
            f.write(md + "\n\n## LaTeX\n\n```latex\n" + latex + "\n```\n")

    with open(os.path.join(args.output_dir, "summary_all.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nAll results written to:", args.output_dir)
    for trie_normalizer in fixed_trie_normalizers:
        warn_fixed_trie_normalizer(
            "scripts.compare_hpo_methods",
            trie_normalizer,
            "END WARNING",
        )


if __name__ == "__main__":
    main()
