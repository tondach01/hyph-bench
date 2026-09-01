#!/usr/bin/env python3
"""Validation-only selector ablation over the final per-level GP histories.

Alternative selectors inspect only each stored validation history.  After a row
has been selected, its profile is regenerated on the unchanged training split
and evaluated once on the held-out test split.  No optimizer search is rerun.
"""

import argparse
import csv
import json
import math
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Dict, List

from .analyze_gpopt260828 import (
    EXPECTED_DATASETS,
    HAND_PROFILES,
    OUTPUT_DIR,
    RESULTS_DIR,
    patgen_binary,
    train_hand_profile,
    train_optimized_profile,
)
from .analyze_heldout_results import aggregate, f17, per_line_counts
from .dataset_utls import find_dataset

RULES = (
    "max_objective",
    "max_validation_f17",
    "min_trie_above_baseline",
    "vc_0.002",
    "vc_0.005",
    "vc_0.010",
)


def read_history(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 153:
        raise AssertionError(f"expected 153 history rows in {path}, found {len(rows)}")
    return rows


def choose_row(history: List[dict], baseline_validation_f17: float, rule: str) -> tuple[dict, bool]:
    def number(row: dict, key: str) -> float:
        return float(row[key])

    best_validation = max(number(row, "validation_f17") for row in history)
    if rule == "max_objective":
        return max(history, key=lambda row: number(row, "objective_score")), False
    if rule == "max_validation_f17":
        return max(history, key=lambda row: number(row, "validation_f17")), False
    if rule == "min_trie_above_baseline":
        candidates = [
            row for row in history
            if number(row, "validation_f17") >= baseline_validation_f17
        ]
    elif rule.startswith("vc_"):
        tolerance = float(rule.split("_", 1)[1])
        candidates = [
            row for row in history
            if number(row, "validation_f17") >= baseline_validation_f17
            and number(row, "validation_f17") >= best_validation - tolerance
        ]
    else:
        raise ValueError(f"unknown selector rule: {rule}")

    used_fallback = not candidates
    if used_fallback:
        candidates = history
    return min(candidates, key=lambda row: int(float(row["trie_nodes"]))), used_fallback


def profile_from_row(row: dict) -> dict:
    return {
        "good_weights": [int(float(row[f"good_wt_{level}"])) for level in range(1, 5)],
        "bad_weights": [int(float(row[f"bad_wt_{level}"])) for level in range(1, 5)],
        "thresholds": [int(float(row[f"threshold_{level}"])) for level in range(1, 5)],
    }


def score(pattern_path: str, test_path: str, translate_path: str) -> float:
    counts = aggregate(per_line_counts(test_path, pattern_path, translate_path))
    return f17(counts["good"], counts["bad"], counts["missed"])


def analyze_dataset(profile_path: Path, baseline_row: dict) -> List[dict]:
    run_dir = profile_path.parent
    dataset = baseline_row["dataset"]
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    selected = json.loads(profile_path.read_text(encoding="utf-8"))
    if config["dataset"] != dataset:
        raise AssertionError(f"dataset mismatch for {run_dir}")

    history = read_history(run_dir / "final_history.csv")
    objective_row, _ = choose_row(history, -math.inf, "max_objective")
    if profile_from_row(objective_row) != {
        "good_weights": selected["good_weights"],
        "bad_weights": selected["bad_weights"],
        "thresholds": selected["thresholds"],
    }:
        raise AssertionError(f"max-objective row does not reproduce selected profile for {dataset}")

    _, translate_path = find_dataset(dataset)
    splits = {
        name: str(run_dir / "splits" / f"data.{name}.wlh")
        for name in ("train", "validation", "test")
    }
    pat_ranges = [tuple(int(value) for value in pair) for pair in config["pattern_ranges"]]
    patgen = patgen_binary(config)
    run_tag = uuid.uuid4().hex[:8]

    hand_validation = {}
    for name, hand_profile_path in HAND_PROFILES.items():
        hand_pattern = hand_scorer = None
        try:
            hand_pattern, _, hand_scorer = train_hand_profile(
                patgen,
                hand_profile_path,
                splits["train"],
                translate_path,
                f"_selector_hand_{name}_{run_tag}",
            )
            hand_validation[name] = score(hand_pattern, splits["validation"], translate_path)
        finally:
            if hand_scorer:
                hand_scorer.clean()
    baseline_name = max(hand_validation, key=hand_validation.get)
    baseline_validation_f17 = hand_validation[baseline_name]
    baseline_test = baseline_row["hand_baselines"][baseline_name]

    selections = {
        rule: choose_row(history, baseline_validation_f17, rule)
        for rule in RULES
    }
    evaluations = {}
    for history_row, _ in selections.values():
        observation = int(history_row["observation"])
        if observation in evaluations:
            continue
        profile = profile_from_row(history_row)
        pattern = scorer = None
        try:
            pattern, stats, scorer = train_optimized_profile(
                patgen,
                profile,
                pat_ranges,
                splits["train"],
                translate_path,
                f"_selector_observation_{observation}_{run_tag}",
            )
            evaluations[observation] = {
                "test_f17": score(pattern, splits["test"], translate_path),
                "trie_nodes": stats["trie_nodes"],
            }
        finally:
            if scorer:
                scorer.clean()

    rows = []
    for rule in RULES:
        history_row, used_fallback = selections[rule]
        observation = int(history_row["observation"])
        evaluation = evaluations[observation]
        rows.append({
            "dataset": dataset,
            "rule": rule,
            "observation": observation,
            "changed_from_max_objective": history_row["observation"] != objective_row["observation"],
            "fallback_to_unconstrained_minimum_trie": used_fallback,
            "profile": profile_from_row(history_row),
            "validation_f17": float(history_row["validation_f17"]),
            "validation_objective": float(history_row["objective_score"]),
            "baseline_name_selected_on_validation": baseline_name,
            "baseline_validation_f17": baseline_validation_f17,
            "test_f17": evaluation["test_f17"],
            "baseline_test_f17": baseline_test["f17"],
            "delta_test_f17": evaluation["test_f17"] - baseline_test["f17"],
            "trie_nodes": evaluation["trie_nodes"],
            "baseline_trie_nodes": baseline_test["trie_nodes"],
            "trie_ratio": evaluation["trie_nodes"] / baseline_test["trie_nodes"],
        })
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    best_test_by_dataset = {
        dataset: max(row["test_f17"] for row in rows if row["dataset"] == dataset)
        for dataset in {row["dataset"] for row in rows}
    }
    summaries = []
    for rule in RULES:
        selected = [row for row in rows if row["rule"] == rule]
        summaries.append({
            "rule": rule,
            "datasets": len(selected),
            "wins_f17_vs_baseline": sum(row["delta_test_f17"] > 0 for row in selected),
            "smaller_tries_vs_baseline": sum(row["trie_ratio"] < 1 for row in selected),
            "tries_within_5pct_of_baseline": sum(row["trie_ratio"] <= 1.05 for row in selected),
            "test_f17_within_0.005_of_best_selector": sum(
                row["test_f17"] >= best_test_by_dataset[row["dataset"]] - 0.005
                for row in selected
            ),
            "changed_from_max_objective": sum(row["changed_from_max_objective"] for row in selected),
            "fallbacks": sum(row["fallback_to_unconstrained_minimum_trie"] for row in selected),
            "median_delta_test_f17": median(row["delta_test_f17"] for row in selected),
            "median_trie_ratio": median(row["trie_ratio"] for row in selected),
        })
    return summaries


def latex_table(summaries: List[dict]) -> str:
    labels = {
        "max_objective": "max objective",
        "max_validation_f17": r"max validation $F_{1/7}$",
        "min_trie_above_baseline": "minimum trie above baseline",
        "vc_0.002": "VC-0.002",
        "vc_0.005": "VC-0.005",
        "vc_0.010": "VC-0.010",
    }
    lines = [
        r"\begin{table*}[tb]",
        r"\centering\small",
        r"\caption{Validation-only selector ablation over the 17 final per-level GP histories. Each rule selects a stored candidate before one held-out test evaluation.}",
        r"\label{tab:final-selector-ablation}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Selector & $F$ wins & smaller trie & trie $\leq 1.05\times$ & $F$ within .005 & changed & median trie ratio \\",
        r"\midrule",
    ]
    for row in summaries:
        n = row["datasets"]
        lines.append(
            f"{labels[row['rule']]} & {row['wins_f17_vs_baseline']}/{n} & "
            f"{row['smaller_tries_vs_baseline']}/{n} & {row['tries_within_5pct_of_baseline']}/{n} & "
            f"{row['test_f17_within_0.005_of_best_selector']}/{n} & "
            f"{row['changed_from_max_objective']}/{n} & {row['median_trie_ratio']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--bootstrap-rows", default=str(OUTPUT_DIR / "bootstrap_ci.json"))
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    baseline_rows = {
        row["dataset"]: row
        for row in json.loads(Path(args.bootstrap_rows).read_text(encoding="utf-8"))
    }
    profiles = sorted(results_dir.glob("*/*/selected_profile.json"))
    if len(profiles) != EXPECTED_DATASETS or len(baseline_rows) != EXPECTED_DATASETS:
        raise AssertionError("selector ablation requires all 17 final profiles and bootstrap baseline rows")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                analyze_dataset,
                profile_path,
                baseline_rows[profile_path.parent.relative_to(results_dir).as_posix()],
            ): profile_path
            for profile_path in profiles
        }
        for future in as_completed(futures):
            dataset_rows = future.result()
            rows.extend(dataset_rows)
            print(
                f"[selector {len(rows) // len(RULES)}/{len(profiles)}] "
                f"{dataset_rows[0]['dataset']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["dataset"], RULES.index(row["rule"])))

    summaries = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            "validation-only baseline and candidate selection over stored final histories; "
            "held-out test evaluated after selection"
        ),
        "results_dir": str(results_dir.resolve()),
        "datasets": EXPECTED_DATASETS,
        "history_rows_per_dataset": 153,
        "rules": list(RULES),
        "summaries": summaries,
        "rows": rows,
    }
    (output_dir / "selector_ablation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selector_ablation_table.tex").write_text(
        latex_table(summaries), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
