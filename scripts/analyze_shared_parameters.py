#!/usr/bin/env python3
"""Select and evaluate profiles from the shared-parameter search.

The script applies the validation-constrained VC-0.005 selector to the fixed
and shared-parameter search spaces. It selects profiles from stored validation
histories, evaluates them once on the held-out test split, and computes paired
bootstrap confidence intervals against the same regenerated hand-tuned
baseline.
"""

import argparse
import csv
import json
import math
import os
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset
from .hyperparameters.sample import Sample
from .hyperparameters.score import PatgenScorer
from .legacy_split import create_legacy_mod10_split
from .optimize_validation import evaluate_parameter_set, f17_score
from .analyze_heldout_results import (
    aggregate,
    bootstrap_delta,
    make_frontier,
    per_line_counts,
    precision,
    recall,
)


PROFILE_PATHS = {
    "cshyphen": Path("profiles/cshyphen.in"),
    "wortliste": Path("profiles/wortliste.in"),
}

CAMERA_READY_DATASETS = [
    "cssk/cshyphen",
    "cs/cshyphen_cstenten",
    "cs/cshyphen_ujc",
    "cs/wiktionary",
    "de/wiktionary",
    "de/wortliste",
    "el/wiktionary",
    "es/wiktionary",
    "is/hyphenation-is",
    "it/wiktionary",
    "nl/wiktionary",
    "pl/wiktionary",
    "pt/wiktionary",
    "ru/wiktionary",
    "th/orchid",
    "tr/wiktionary",
    "uk/wiktionary",
]


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_history(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if len(rows) != 153:
        raise ValueError(f"Expected 153 evaluations in {path}, found {len(rows)}")
    expected_params = {f"param_{i}" for i in range(1, 7)}
    actual_params = {name for name in fieldnames if name.startswith("param_")}
    if actual_params != expected_params:
        raise ValueError(
            f"Expected a 6-D GPopt6 history in {path}, found {sorted(actual_params)}"
        )
    if max(float(row["validation_f17"]) for row in rows) <= 0:
        raise ValueError(f"History has no successful evaluation: {path}")
    return rows


def select_vc005(history: List[dict], baseline_validation_f17: float) -> dict:
    best_validation_f17 = max(float(row["validation_f17"]) for row in history)
    candidates = [
        row
        for row in history
        if float(row["validation_f17"]) >= baseline_validation_f17
        and float(row["validation_f17"]) >= best_validation_f17 - 0.005
    ]
    if not candidates:
        candidates = history
    return min(candidates, key=lambda row: int(float(row["trie_nodes"])))


def shared_params(row: dict) -> Tuple[int, ...]:
    return tuple(int(float(row[f"param_{i}"])) for i in range(1, 7))


def parse_profile(path: Path) -> List[Tuple[int, int, int, int, int]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = tuple(int(value) for value in line.split())
            if len(values) != 5:
                raise ValueError(f"Expected five values in {path}: {line}")
            rows.append(values)
    return rows


def train_hand_profile(
    patgen_path: str,
    profile_path: Path,
    train_path: str,
    translate_path: str,
) -> Tuple[str, Dict[str, int], PatgenScorer]:
    scorer = PatgenScorer(
        patgen_path,
        train_path,
        translate_path,
        tmp_suffix=f"_gpopt6_baseline_{uuid.uuid4().hex[:8]}",
    )
    previous_id = 0
    pattern_count = 0
    final_stats = None
    try:
        for level, (start, finish, good_weight, bad_weight, threshold) in enumerate(
            parse_profile(profile_path), start=1
        ):
            sample = Sample(
                {
                    "level": level,
                    "prev": previous_id,
                    "pat_start": start,
                    "pat_finish": finish,
                    "good_weight": good_weight,
                    "bad_weight": bad_weight,
                    "threshold": threshold,
                }
            )
            scorer.score(sample)
            previous_id = sample.run_id
            pattern_count += sample.stats.get("level_patterns", 0)
            final_stats = sample.stats
        if final_stats is None:
            raise RuntimeError(f"No profile levels in {profile_path}")
        return (
            os.path.join(scorer.temp_dir, f"{previous_id}.pat"),
            {
                "trie_nodes": final_stats["trie_nodes"],
                "n_patterns": pattern_count,
            },
            scorer,
        )
    except Exception:
        scorer.clean()
        raise


def binomial_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(wins, trials + 1)) / (2**trials)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_markdown(rows: List[dict], path: Path) -> None:
    lines = [
        "# GPopt6 VC-0.005 held-out results",
        "",
        "| Dataset | GPopt4 F1/7 | GPopt6 F1/7 | Difference | Baseline F1/7 | GPopt6 trie | Baseline trie |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | {row['gpopt4_f17']:.6f} | {row['gpopt6_f17']:.6f} | "
            f"{row['delta_vs_gpopt4']:+.6f} | {row['baseline_f17']:.6f} | "
            f"{row['gpopt6_trie']:,} | {row['baseline_trie']:,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: List[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Held-out GPopt6 results selected by the same validation-constrained VC-0.005 rule used for GPopt4.}",
        r"\label{tab:gpopt6-comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & GPopt4 $F_{1/7}$ & GPopt6 $F_{1/7}$ & $\Delta$ & Baseline $F_{1/7}$ & GPopt6 trie & Baseline trie \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"\texttt{{{latex_escape(row['dataset'])}}} & {row['gpopt4_f17']:.4f} & "
            rf"{row['gpopt6_f17']:.4f} & {row['delta_vs_gpopt4']:+.4f} & "
            rf"{row['baseline_f17']:.4f} & {row['gpopt6_trie']} & {row['baseline_trie']} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bootstrap_latex(rows: List[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Paired bootstrap confidence intervals for held-out GPopt6 improvement over the regenerated hand-tuned baseline.}",
        r"\label{tab:gpopt6-bootstrap}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & $\Delta F_{1/7}$ & 95\% CI low & 95\% CI high \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"\texttt{{{latex_escape(row['dataset'])}}} & "
            rf"{row['delta_vs_baseline']:+.4f} & "
            rf"{row['bootstrap_ci_low']:+.4f} & "
            rf"{row['bootstrap_ci_high']:+.4f} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-results",
        default="results/fixed_search_vc005_results.json",
    )
    parser.add_argument("--search-root", default="results/shared_parameter_search")
    parser.add_argument("--output-dir", default="results/shared_parameter_analysis")
    parser.add_argument("--patgen", default="patgen")
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    all_reference_results = read_json(Path(args.reference_results))
    reference_results_by_dataset = {
        row["dataset"]: row for row in all_reference_results
    }
    missing = [
        dataset
        for dataset in CAMERA_READY_DATASETS
        if dataset not in reference_results_by_dataset
    ]
    if missing:
        raise ValueError(f"Reference results do not contain: {', '.join(missing)}")
    reference_results = [
        reference_results_by_dataset[dataset] for dataset in CAMERA_READY_DATASETS
    ]
    search_root = Path(args.search_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, reference_row in enumerate(reference_results, start=1):
        started = time.monotonic()
        dataset = reference_row["dataset"]
        print(f"[{index}/{len(reference_results)}] {dataset}", flush=True)
        dataset_dir = search_root / dataset
        history = read_history(dataset_dir / "wider_history.csv")
        baseline_validation_f17 = float(reference_row["validation_best_baseline"]["f17"])
        selected = select_vc005(history, baseline_validation_f17)
        params = shared_params(selected)

        wordlist_path, translate_path = find_dataset(dataset)
        splits = create_legacy_mod10_split(wordlist_path, str(dataset_dir / "splits"))
        selected_pattern = dataset_dir / "wider_vc005_final.pat"
        _, test = evaluate_parameter_set(
            args.patgen,
            splits["train"],
            splits["test"],
            translate_path,
            params[:5],
            DEFAULT_PAT_RANGES,
            params[5],
            False,
            f"gpopt6_analysis_{index}",
            export_patterns_path=str(selected_pattern),
        )
        gpopt6_f17 = f17_score(test["good"], test["bad"], test["missed"])

        baseline_name = reference_row["best_baseline_name"]
        baseline_pattern = baseline_scorer = None
        try:
            baseline_pattern, baseline_stats, baseline_scorer = train_hand_profile(
                args.patgen,
                PROFILE_PATHS[baseline_name],
                splits["train"],
                translate_path,
            )
            gpopt6_counts = per_line_counts(
                splits["test"], str(selected_pattern), translate_path
            )
            baseline_counts = per_line_counts(
                splits["test"], baseline_pattern, translate_path
            )
            gpopt6_aggregate = aggregate(gpopt6_counts)
            baseline_aggregate = aggregate(baseline_counts)
            if gpopt6_aggregate != {
                "good": test["good"],
                "bad": test["bad"],
                "missed": test["missed"],
            }:
                raise RuntimeError(f"Per-line and aggregate GPopt6 counts differ for {dataset}")
            ci_low, ci_high = bootstrap_delta(
                gpopt6_counts,
                baseline_counts,
                args.bootstrap_reps,
                args.seed + index,
            )
        finally:
            if baseline_scorer is not None:
                baseline_scorer.clean()

        baseline_f17 = f17_score(**baseline_aggregate)
        gpopt4_f17 = float(reference_row["optimized"]["f17"])
        row = {
            "dataset": dataset,
            "selector": "vc_0.005",
            "params": list(params),
            "validation_f17": float(selected["validation_f17"]),
            "best_validation_f17": max(float(item["validation_f17"]) for item in history),
            "validation_trie": int(float(selected["trie_nodes"])),
            "gpopt6": {
                "good": test["good"],
                "bad": test["bad"],
                "missed": test["missed"],
                "precision": precision(test["good"], test["bad"]),
                "recall": recall(test["good"], test["missed"]),
                "f17": gpopt6_f17,
                "trie_nodes": test["trie_nodes"],
                "n_patterns": test["n_patterns"],
            },
            "baseline_name": baseline_name,
            "baseline": {
                **baseline_aggregate,
                "f17": baseline_f17,
                "trie_nodes": baseline_stats["trie_nodes"],
                "n_patterns": baseline_stats["n_patterns"],
            },
            "gpopt4_f17": gpopt4_f17,
            "gpopt4_trie": int(reference_row["optimized"]["trie_nodes"]),
            "gpopt6_f17": gpopt6_f17,
            "gpopt6_trie": test["trie_nodes"],
            "baseline_f17": baseline_f17,
            "baseline_trie": baseline_stats["trie_nodes"],
            "delta_vs_gpopt4": gpopt6_f17 - gpopt4_f17,
            "delta_vs_baseline": gpopt6_f17 - baseline_f17,
            "trie_ratio_vs_baseline": test["trie_nodes"] / baseline_stats["trie_nodes"],
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "elapsed_seconds": time.monotonic() - started,
        }
        rows.append(row)
        (output_dir / "vc005_results.partial.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  GPopt6={gpopt6_f17:.6f}, GPopt4={gpopt4_f17:.6f}, "
            f"baseline={baseline_f17:.6f}, trie={test['trie_nodes']}",
            flush=True,
        )

    wins_baseline = sum(row["delta_vs_baseline"] > 0 for row in rows)
    losses_baseline = sum(row["delta_vs_baseline"] < 0 for row in rows)
    wins_gpopt4 = sum(row["delta_vs_gpopt4"] > 0 for row in rows)
    losses_gpopt4 = sum(row["delta_vs_gpopt4"] < 0 for row in rows)
    summary = {
        "datasets": len(rows),
        "selector": "vc_0.005",
        "gpopt6_wins_vs_baseline": wins_baseline,
        "gpopt6_losses_vs_baseline": losses_baseline,
        "sign_test_vs_baseline_one_sided_p": binomial_sign_test(
            wins_baseline, losses_baseline
        ),
        "gpopt6_wins_vs_gpopt4": wins_gpopt4,
        "gpopt6_losses_vs_gpopt4": losses_gpopt4,
        "median_delta_vs_baseline": median(row["delta_vs_baseline"] for row in rows),
        "median_delta_vs_gpopt4": median(row["delta_vs_gpopt4"] for row in rows),
        "smaller_tries_than_baseline": sum(
            row["trie_ratio_vs_baseline"] < 1 for row in rows
        ),
        "median_trie_ratio_vs_baseline": median(
            row["trie_ratio_vs_baseline"] for row in rows
        ),
        "positive_bootstrap_ci_vs_baseline": sum(
            row["bootstrap_ci_low"] > 0 for row in rows
        ),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
    }
    (output_dir / "vc005_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(rows, output_dir / "vc005_results.md")
    write_latex(rows, output_dir / "vc005_table.tex")
    write_bootstrap_latex(rows, output_dir / "bootstrap_ci_table.tex")
    make_frontier(
        [
            {
                "dataset": row["dataset"],
                "trie_ratio": row["trie_ratio_vs_baseline"],
                "delta_f17": row["delta_vs_baseline"],
            }
            for row in rows
        ],
        output_dir / "frontier.pdf",
        output_dir / "frontier.png",
    )
    (output_dir / "vc005_results.partial.json").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
