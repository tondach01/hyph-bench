#!/usr/bin/env python3
"""Analyze held-out optimization results.

This script uses deterministic 8/1/1 held-out runs to generate:

* a paired bootstrap table for optimized versus hand-tuned held-out scores,
* selector-ablation summaries over stored validation histories,
* a frontier figure of held-out F-score gain against trie-size ratio.
"""

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset
from .hyphenator.hyphenator import Hyphenator
from .hyperparameters.sample import Sample
from .hyperparameters.score import PatgenScorer
from .legacy_split import create_legacy_mod10_split
from .optimize_validation import train_patgen_multilevel


PROFILE_PATHS = {
    "cshyphen": Path("profiles/cshyphen.in"),
    "wortliste": Path("profiles/wortliste.in"),
}


def f17(good: int, bad: int, missed: int, beta: float = 1 / 7) -> float:
    if good == 0:
        return 0.0
    precision = good / (good + bad) if good + bad else 0.0
    recall = good / (good + missed) if good + missed else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def precision(good: int, bad: int) -> float:
    return good / (good + bad) if good + bad else 0.0


def recall(good: int, missed: int) -> float:
    return good / (good + missed) if good + missed else 0.0


def safe_name(dataset: str) -> str:
    return dataset.replace("/", "_")


def parse_profile(path: Path) -> List[Tuple[int, int, int, int, int]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [int(x) for x in line.split()]
            if len(parts) != 5:
                raise ValueError(f"Expected 5 values in {path}: {line}")
            rows.append(tuple(parts))
    return rows


def train_hand_profile(
    profile_path: Path,
    train_path: str,
    translate_path: str,
    tmp_suffix: str,
) -> Tuple[str, Dict[str, int], PatgenScorer]:
    scorer = PatgenScorer("patgen", train_path, translate_path, tmp_suffix=tmp_suffix)
    prev_id = 0
    total_patterns = 0
    final_stats = None
    try:
        for level, (pat_start, pat_finish, good_weight, bad_weight, threshold) in enumerate(
            parse_profile(profile_path), start=1
        ):
            sample = Sample(
                {
                    "level": level,
                    "prev": prev_id,
                    "pat_start": pat_start,
                    "pat_finish": pat_finish,
                    "good_weight": good_weight,
                    "bad_weight": bad_weight,
                    "threshold": threshold,
                }
            )
            scorer.score(sample)
            prev_id = sample.run_id
            total_patterns += sample.stats.get("level_patterns", 0)
            final_stats = sample.stats
        if final_stats is None:
            raise RuntimeError(f"No profile levels in {profile_path}")
        pattern_path = os.path.join(scorer.temp_dir, f"{prev_id}.pat")
        stats = {
            "train_good": final_stats["tp"],
            "train_bad": final_stats["fp"],
            "train_missed": final_stats["fn"],
            "n_patterns": total_patterns,
            "trie_nodes": final_stats["trie_nodes"],
        }
        return pattern_path, stats, scorer
    except Exception:
        scorer.clean()
        raise


def train_optimized_profile(
    params: Iterable[int],
    train_path: str,
    translate_path: str,
    tmp_suffix: str,
) -> Tuple[str, Dict[str, int], PatgenScorer]:
    scorer = PatgenScorer("patgen", train_path, translate_path, tmp_suffix=tmp_suffix)
    try:
        pattern_path, stats = train_patgen_multilevel(
            scorer,
            tuple(int(x) for x in params),
            DEFAULT_PAT_RANGES,
            good_weight=3,
        )
        return pattern_path, stats, scorer
    except Exception:
        scorer.clean()
        raise


def line_counts(correct: str, predicted: str) -> Tuple[int, int, int]:
    good = bad = missed = 0
    i_corr = i_hyph = 0
    while i_corr < len(correct) and i_hyph < len(predicted):
        if correct[i_corr] == "-" and predicted[i_hyph] == "-":
            good += 1
            i_corr += 1
            i_hyph += 1
        elif predicted[i_hyph] == "-":
            bad += 1
            i_hyph += 1
        elif correct[i_corr] == "-":
            missed += 1
            i_corr += 1
        else:
            i_corr += 1
            i_hyph += 1
    return good, bad, missed


def per_line_counts(test_path: str, pattern_path: str, translate_path: str) -> np.ndarray:
    hyphenator = Hyphenator(pattern_path, hyphenation_mark="-", translate_file=translate_path)
    rows = []
    with open(test_path, encoding="utf-8") as handle:
        for line in handle:
            correct = line.strip()
            rows.append(line_counts(correct, hyphenator.hyphenate(correct)))
    return np.asarray(rows, dtype=np.int64)


def aggregate(arr: np.ndarray) -> Dict[str, int]:
    good, bad, missed = arr.sum(axis=0)
    return {"good": int(good), "bad": int(bad), "missed": int(missed)}


def bootstrap_delta(
    opt_counts: np.ndarray,
    base_counts: np.ndarray,
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(opt_counts)
    deltas = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        og, ob, om = opt_counts[idx].sum(axis=0)
        bg, bb, bm = base_counts[idx].sum(axis=0)
        deltas[i] = f17(int(og), int(ob), int(om)) - f17(int(bg), int(bb), int(bm))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_history(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_row(history: List[dict], baseline_validation_f17: float, rule: str) -> dict:
    def as_float(row, key):
        return float(row[key])

    best_val = max(as_float(row, "validation_f17") for row in history)
    if rule == "max_objective":
        return max(history, key=lambda row: as_float(row, "objective_score"))
    if rule == "max_validation_f17":
        return max(history, key=lambda row: as_float(row, "validation_f17"))
    if rule == "min_trie_above_baseline":
        candidates = [
            row for row in history
            if as_float(row, "validation_f17") >= baseline_validation_f17
        ]
    elif rule.startswith("vc_"):
        tau = float(rule.split("_", 1)[1])
        candidates = [
            row for row in history
            if (
                as_float(row, "validation_f17") >= baseline_validation_f17
                and as_float(row, "validation_f17") >= best_val - tau
            )
        ]
    else:
        raise ValueError(f"Unknown selector rule: {rule}")
    if not candidates:
        candidates = history
    return min(candidates, key=lambda row: int(float(row["trie_nodes"])))


def params_from_history(row: dict) -> List[int]:
    return [int(float(row[f"param_{i}"])) for i in range(1, 6)]


def evaluate_profile_counts(pattern_path: str, test_path: str, translate_path: str) -> Dict[str, int]:
    counts = aggregate(per_line_counts(test_path, pattern_path, translate_path))
    counts["precision"] = precision(counts["good"], counts["bad"])
    counts["recall"] = recall(counts["good"], counts["missed"])
    counts["f17"] = f17(counts["good"], counts["bad"], counts["missed"])
    return counts


def dataset_splits(dataset: str, output_dir: Path) -> Tuple[str, str, dict]:
    wl, tr = find_dataset(dataset)
    split_dir = output_dir / safe_name(dataset) / "splits"
    splits = create_legacy_mod10_split(wl, str(split_dir))
    return wl, tr, splits


def make_frontier(rows: List[dict], output_pdf: Path, output_png: Path) -> None:
    x = [row["trie_ratio"] for row in rows]
    y = [row["delta_f17"] for row in rows]
    labels = [row["dataset"] for row in rows]

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axhline(0, color="#555555", lw=0.8)
    ax.axvline(1.0, color="#555555", lw=0.8)
    ax.axvspan(0, 1.0, color="#d7f0d0", alpha=0.35, label="smaller than baseline")
    ax.axvspan(1.0, 1.05, color="#e6f2ff", alpha=0.45, label="within 5% size")
    ax.scatter(x, y, s=28, color="#1b5e8a", edgecolor="white", linewidth=0.5, zorder=3)

    for xi, yi, label in zip(x, y, labels):
        dx = 0.006 if xi < 1.08 else -0.006
        ha = "left" if xi < 1.08 else "right"
        ax.annotate(label.replace("_", "\\_"), (xi, yi), xytext=(dx * 250, 4),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=6.5)

    ax.set_xlabel("Optimized trie nodes / best hand-tuned trie nodes")
    ax.set_ylabel(r"Held-out $\Delta F_{1/7}$")
    ax.set_xlim(min(x) - 0.03, max(x) + 0.05)
    ax.set_ylim(0, max(y) + 0.006)
    ax.grid(True, color="#dddddd", linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=240)
    plt.close(fig)


def latex_ci_table(rows: List[dict]) -> str:
    lines = [
        r"\begin{table*}[tb]",
        r"\centering\small",
        r"\caption{Paired bootstrap uncertainty for held-out optimized profiles against the best regenerated hand-tuned baseline. CIs are percentile 95\% intervals from resampling held-out word-list lines; $p_{\mathrm{sign}}$ is the dataset-level sign-test probability of observing 18/18 positive deltas under a symmetric null.}",
        r"\label{tab:bootstrap-ci}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Dataset & $\Delta F_{1/7}$ & 95\% CI & Opt. trie/Base trie & Baseline \\",
        r"\midrule",
    ]
    for row in rows:
        ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        ratio = f"{row['opt_trie']}/{row['base_trie']}"
        dataset = row["dataset"].replace("_", r"\_")
        lines.append(
            f"{dataset} & "
            f"${row['delta_f17']:+.4f}$ & {ci} & {ratio} & {row['best_baseline']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def latex_selector_table(rows: List[dict]) -> str:
    label = {
        "max_objective": "max objective",
        "max_validation_f17": "max validation $F_{1/7}$",
        "min_trie_above_baseline": "minimum trie above baseline",
        "vc_0.002": "VC-0.002",
        "vc_0.005": "VC-0.005",
        "vc_0.010": "VC-0.010",
    }
    lines = [
        r"\begin{table}[tb]",
        r"\centering\small",
        r"\caption{Validation-only selector ablation over the stored \GP candidate histories. Each rule is applied before held-out test evaluation.}",
        r"\label{tab:selector-ablation}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Selector & F wins & smaller & within 5\% & median trie ratio \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{label[row['rule']]} & {row['wins_f17']}/18 & "
            f"{row['smaller_tries']}/18 & {row['within_5pct']}/18 & "
            f"{row['median_trie_ratio']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def run_bootstrap(args, selected: List[dict], output_dir: Path) -> List[dict]:
    rows = []
    for i, item in enumerate(selected, start=1):
        dataset = item["dataset"]
        print(f"[bootstrap] {i}/{len(selected)} {dataset}")
        _, translate_path, splits = dataset_splits(dataset, output_dir)
        run_tag = uuid.uuid4().hex[:8]
        opt_pattern = base_pattern = None
        opt_scorer = base_scorer = None
        try:
            opt_pattern, opt_stats, opt_scorer = train_optimized_profile(
                item["params"],
                splits["train"],
                translate_path,
                f"_heldout_ci_opt_{run_tag}",
            )
            base_name = item["best_baseline_name"]
            base_pattern, base_stats, base_scorer = train_hand_profile(
                PROFILE_PATHS[base_name],
                splits["train"],
                translate_path,
                f"_heldout_ci_base_{run_tag}",
            )
            opt_counts = per_line_counts(splits["test"], opt_pattern, translate_path)
            base_counts = per_line_counts(splits["test"], base_pattern, translate_path)
            opt_agg = aggregate(opt_counts)
            base_agg = aggregate(base_counts)
            delta = f17(**opt_agg) - f17(**base_agg)
            low, high = bootstrap_delta(
                opt_counts,
                base_counts,
                reps=args.bootstrap_reps,
                seed=args.seed + i,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "delta_f17": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "opt_f17": f17(**opt_agg),
                    "base_f17": f17(**base_agg),
                    "opt_trie": opt_stats["trie_nodes"],
                    "base_trie": base_stats["trie_nodes"],
                    "best_baseline": base_name,
                    "test_lines": int(len(opt_counts)),
                }
            )
        finally:
            if opt_scorer:
                opt_scorer.clean()
            if base_scorer:
                base_scorer.clean()
    return rows


def run_selector_ablation(
    selected: List[dict],
    baseline_validation: Dict[str, dict],
    output_dir: Path,
) -> List[dict]:
    rules = [
        "max_objective",
        "max_validation_f17",
        "min_trie_above_baseline",
        "vc_0.002",
        "vc_0.005",
        "vc_0.010",
    ]
    accum = {rule: [] for rule in rules}

    for i, item in enumerate(selected, start=1):
        dataset = item["dataset"]
        print(f"[selector] {i}/{len(selected)} {dataset}")
        _, translate_path, splits = dataset_splits(dataset, output_dir)
        history_path = Path("results/gpoptval4") / dataset / "gpoptval4_history.csv"
        history = read_history(history_path)
        baseline_name = item["best_baseline_name"]
        baseline_val_f17 = baseline_validation[dataset][baseline_name]["f17"]
        base_pattern = base_scorer = None
        run_tag = uuid.uuid4().hex[:8]
        try:
            base_pattern, base_stats, base_scorer = train_hand_profile(
                PROFILE_PATHS[baseline_name],
                splits["train"],
                translate_path,
                f"_heldout_selector_base_{run_tag}",
            )
            base_eval = evaluate_profile_counts(base_pattern, splits["test"], translate_path)
            for rule in rules:
                row = choose_row(history, baseline_val_f17, rule)
                opt_pattern = opt_scorer = None
                try:
                    opt_pattern, opt_stats, opt_scorer = train_optimized_profile(
                        params_from_history(row),
                        splits["train"],
                        translate_path,
                        f"_heldout_selector_{rule}_{run_tag}",
                    )
                    opt_eval = evaluate_profile_counts(opt_pattern, splits["test"], translate_path)
                    accum[rule].append(
                        {
                            "dataset": dataset,
                            "params": params_from_history(row),
                            "opt_f17": opt_eval["f17"],
                            "base_f17": base_eval["f17"],
                            "delta_f17": opt_eval["f17"] - base_eval["f17"],
                            "opt_trie": opt_stats["trie_nodes"],
                            "base_trie": base_stats["trie_nodes"],
                            "trie_ratio": opt_stats["trie_nodes"] / base_stats["trie_nodes"],
                        }
                    )
                finally:
                    if opt_scorer:
                        opt_scorer.clean()
        finally:
            if base_scorer:
                base_scorer.clean()

    summaries = []
    for rule in rules:
        rows = accum[rule]
        summaries.append(
            {
                "rule": rule,
                "wins_f17": sum(1 for row in rows if row["delta_f17"] > 0),
                "smaller_tries": sum(1 for row in rows if row["trie_ratio"] < 1),
                "within_5pct": sum(1 for row in rows if row["trie_ratio"] <= 1.05),
                "within_10pct": sum(1 for row in rows if row["trie_ratio"] <= 1.10),
                "median_delta_f17": median(row["delta_f17"] for row in rows),
                "median_trie_ratio": median(row["trie_ratio"] for row in rows),
                "rows": rows,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="results/fixed_search_vc005_results.json")
    parser.add_argument("--baseline-validation", default="results/baseline_validation_metrics.json")
    parser.add_argument("--output-dir", default="results/heldout_analysis")
    parser.add_argument("--frontier-pdf", default="../brain/overleaf/latex/pics/heldout_frontier.pdf")
    parser.add_argument("--frontier-png", default="../brain/overleaf/latex/pics/heldout_frontier.png")
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-selector", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_json(Path(args.selected))
    baseline_validation = load_json(Path(args.baseline_validation))

    make_frontier(selected, Path(args.frontier_pdf), Path(args.frontier_png))

    sign_p_one_sided = math.pow(0.5, sum(1 for row in selected if row["delta_f17"] > 0))
    summary = {
        "datasets": len(selected),
        "positive_delta_count": sum(1 for row in selected if row["delta_f17"] > 0),
        "sign_test_one_sided_p": sign_p_one_sided,
        "median_delta_f17": median(row["delta_f17"] for row in selected),
        "median_trie_ratio": median(row["trie_ratio"] for row in selected),
    }

    if not args.skip_bootstrap:
        ci_rows = run_bootstrap(args, selected, output_dir)
        summary["bootstrap_positive_ci_count"] = sum(1 for row in ci_rows if row["ci_low"] > 0)
        with (output_dir / "bootstrap_ci.json").open("w", encoding="utf-8") as handle:
            json.dump(ci_rows, handle, indent=2)
        (output_dir / "bootstrap_ci_table.tex").write_text(latex_ci_table(ci_rows), encoding="utf-8")

    if not args.skip_selector:
        selector_rows = run_selector_ablation(selected, baseline_validation, output_dir)
        with (output_dir / "selector_ablation.json").open("w", encoding="utf-8") as handle:
            json.dump(selector_rows, handle, indent=2)
        (output_dir / "selector_ablation_table.tex").write_text(
            latex_selector_table(selector_rows),
            encoding="utf-8",
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
