#!/usr/bin/env python3
"""
Aggregate threshold-ablation results (fixed1 vs shared vs perlayer x GP/TPE/Random).

Inputs:
  - fixed1 (paper) arm:  results/gpoptval4/<lang>/gpoptval4_history.csv
      best params by objective_score; held-out test metrics are recomputed in a
      single patgen run per dataset (same protocol code path as the ablation
      arms) and cached in <ablation-dir>/baselines_fixed1.json.
  - ablation arms: results/threshold_ablation/<lang>/<mode>_<method>/summary.json

Outputs (always derived, never overwriting inputs):
  - results/threshold_ablation/SUMMARY.md
  - results/threshold_ablation/ablation_summary.json
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset
from .legacy_split import create_legacy_mod10_split
from .optimize_validation import (
    evaluate_parameter_set,
    f17_score,
    precision,
    recall,
)

DATASETS = [
    "cssk/cshyphen",
    "es/wiktionary",
    "de/wiktionary",
    "cs/cshyphen_cstenten",
    "de/wortliste",
    "nl/wiktionary",
    "is/hyphenation-is",
    "ru/wiktionary",
    "pl/wiktionary",
    "cs/cshyphen_ujc",
    "it/wiktionary",
    "cs/wiktionary",
    "pt/wiktionary",
    "th/orchid",
    "el/wiktionary",
    "uk/wiktionary",
    "tr/wiktionary",
    "ms/wiktionary",
]

MODES = ["shared", "perlayer"]
METHODS = ["gp", "tpe", "random"]
BASELINE_BUDGET = 153


def read_best_baseline(history_path: str) -> Optional[Dict]:
    if not os.path.exists(history_path):
        return None
    with open(history_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    # Budget match: --resume runs of optimize_validation append a full new
    # budget to the pre-existing rows (e.g. el/wiktionary 115+153); only the
    # first BASELINE_BUDGET evaluations count for arm comparison.
    truncated = len(rows) > BASELINE_BUDGET
    rows = rows[:BASELINE_BUDGET]
    best = max(rows, key=lambda r: float(r["objective_score"]))
    return {
        "params": tuple(int(best[f"param_{i}"]) for i in range(1, 6)),
        "objective": float(best["objective_score"]),
        "validation_f17": float(best["validation_f17"]),
        "trie_nodes": int(best["trie_nodes"]),
        "n_evaluations": len(rows),
        "truncated": truncated,
    }


def split_paths_for(lang: str, splits_root: str) -> Dict[str, str]:
    wordlist_path, _ = find_dataset(lang)
    split_dir = os.path.join(splits_root, lang, "splits")
    return create_legacy_mod10_split(wordlist_path, split_dir)


def baseline_test_eval(lang: str, params: Tuple[int, ...], patgen: str,
                       splits_root: str) -> Dict:
    wordlist_path, translate_path = find_dataset(lang)
    paths = split_paths_for(lang, splits_root)
    _, res = evaluate_parameter_set(
        patgen, paths["train"], paths["test"], translate_path,
        tuple(params), DEFAULT_PAT_RANGES, 3, False, f"summ_{lang.replace('/', '_')}",
    )
    return {
        "test_f17": f17_score(res["good"], res["bad"], res["missed"]),
        "test_precision": precision(res["good"], res["bad"]),
        "test_recall": recall(res["good"], res["missed"]),
        "test_good": res["good"],
        "test_bad": res["bad"],
        "test_missed": res["missed"],
        "test_trie_nodes": res["trie_nodes"],
        "test_n_patterns": res["n_patterns"],
    }


def load_ablation_summaries(ablation_dir: str) -> Dict[Tuple[str, str, str], Dict]:
    out: Dict[Tuple[str, str, str], Dict] = {}
    for lang in DATASETS:
        for mode in MODES:
            for method in METHODS:
                path = os.path.join(ablation_dir, lang, f"{mode}_{method}", "summary.json")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as handle:
                        out[(lang, mode, method)] = json.load(handle)
    return out


def fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_delta(value: Optional[float], ref: Optional[float], digits: int = 4) -> str:
    if value is None or ref is None:
        return ""
    delta = value - ref
    return f" ({delta:+.{digits}f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", default="results/threshold_ablation")
    parser.add_argument("--baseline-dir", default="results/gpoptval4")
    parser.add_argument("--patgen", default=os.path.expanduser("~/patgen-10x"))
    parser.add_argument("--recompute-baselines", action="store_true",
                        help="Recompute held-out test metrics for the fixed1 arm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.ablation_dir, exist_ok=True)
    cache_path = os.path.join(args.ablation_dir, "baselines_fixed1.json")
    cache: Dict[str, Dict] = {}
    if os.path.exists(cache_path) and not args.force:
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)

    # fixed1 baseline arm (paper setting).
    baselines: Dict[str, Dict] = {}
    todo: List[Tuple[str, Dict]] = []
    for lang in DATASETS:
        history = os.path.join(args.baseline_dir, lang, "gpoptval4_history.csv")
        best = read_best_baseline(history)
        if best is None:
            print(f"[baseline] {lang}: no history found")
            continue
        cached = cache.get(lang)
        if cached and cached.get("params") == list(best["params"]) \
                and cached.get("n_evaluations") == best["n_evaluations"]:
            baselines[lang] = {**cached, "objective": best["objective"],
                               "validation_f17": best["validation_f17"]}
        elif args.recompute_baselines:
            todo.append((lang, best))
        else:
            baselines[lang] = {**best, "params": list(best["params"]), "test_f17": None}

    if todo:
        print(f"[baseline] recomputing held-out test for {len(todo)} datasets "
              f"with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(baseline_test_eval, lang, best["params"],
                                args.patgen, args.baseline_dir): (lang, best)
                for lang, best in todo
            }
            for future in as_completed(futures):
                lang, best = futures[future]
                test = future.result()
                entry = {**best, "params": list(best["params"]), **test}
                baselines[lang] = entry
                cache[lang] = entry
                print(f"[baseline] {lang}: test_f17={test['test_f17']:.6f}")
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)

    summaries = load_ablation_summaries(args.ablation_dir)

    # ------------------------------------------------------------------ #
    # Compose report
    # ------------------------------------------------------------------ #
    arms_gp = [("fixed1", None, None), ("shared", "gp", "shared_gp"), ("perlayer", "gp", "perlayer_gp")]
    arms_tpe = [("shared", "tpe", "shared_tpe"), ("perlayer", "tpe", "perlayer_tpe")]
    arms_random = [("shared", "random", "shared_random"), ("perlayer", "random", "perlayer_random")]

    def arm_values(lang: str, mode: str, method: Optional[str]):
        if method is None:
            base = baselines.get(lang)
            if not base:
                return None
            return {
                "test_f17": base.get("test_f17"),
                "val_f17": base.get("validation_f17"),
                "objective": base.get("objective"),
                "thresholds": "1,1,1,1",
                "params": tuple(base["params"]),
                "trie_nodes": base.get("test_trie_nodes", base.get("trie_nodes")),
            }
        summary = summaries.get((lang, mode, method))
        if not summary:
            return None
        thr = summary["thresholds"]
        thr_str = str(thr["shared"]) if thr["shared"] is not None else ",".join(
            str(t) for t in thr["per_level"])
        return {
            "test_f17": summary["test"]["f17"],
            "val_f17": summary["validation"]["f17"],
            "objective": summary["best_objective"],
            "thresholds": thr_str,
            "params": tuple(summary["best_params"]),
            "trie_nodes": summary["test"]["trie_nodes"],
        }

    lines: List[str] = []
    lines.append("# Threshold ablation (GPoptval4 protocol, budget 153 evals per arm)")
    lines.append("")
    lines.append("Test F_{1/7} on the held-out split; deltas vs the fixed1 paper arm. "
                 "`thr` = selected threshold(s), one per level. Missing arms shown as `-`.")
    lines.append("")

    def render_table(title: str, arm_specs) -> None:
        header = "| dataset | " + " | ".join(
            (spec[2] or spec[0]) if len(spec) > 2 else spec[0] for spec in arm_specs) + " |"
        sep = "|" + "---|" * (len(arm_specs) + 1)
        thr_row_label = "`thr` selected"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        deltas: Dict[str, List[float]] = {spec[0]: [] for spec in arm_specs[1:]}
        wins: Dict[str, int] = {spec[0]: 0 for spec in arm_specs[1:]}
        done = 0
        for lang in DATASETS:
            ref = arm_values(lang, "fixed1", None)
            cells = [f"`{lang}`"]
            for spec in arm_specs:
                mode, method = spec[0], spec[1]
                vals = arm_values(lang, mode, method)
                if vals is None or vals["test_f17"] is None:
                    cells.append("-")
                    continue
                cell = fmt(vals["test_f17"])
                if method is not None and ref and ref.get("test_f17") is not None:
                    cell += fmt_delta(vals["test_f17"], ref["test_f17"])
                    deltas[mode].append(vals["test_f17"] - ref["test_f17"])
                    if vals["test_f17"] > ref["test_f17"]:
                        wins[mode] += 1
                cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")
            thr_cells = [thr_row_label]
            for spec in arm_specs:
                vals = arm_values(lang, spec[0], spec[1])
                thr_cells.append(vals["thresholds"] if vals else "-")
            lines.append("| " + " | ".join(thr_cells) + " |")
            done += 1
        lines.append("")
        for spec in arm_specs[1:]:
            mode = spec[0]
            if deltas[mode]:
                mean_d = sum(deltas[mode]) / len(deltas[mode])
                lines.append(
                    f"- {mode}: mean delta vs fixed1 = {mean_d:+.4f} "
                    f"over {len(deltas[mode])} datasets; better on {wins[mode]}"
                )
        lines.append("")

    render_table("GP arms", arms_gp)
    if any(k[2] == "tpe" for k in summaries):
        render_table("TPE arms", [arms_gp[0]] + arms_tpe)
    if any(k[2] == "random" for k in summaries):
        render_table("Random arms", [arms_gp[0]] + arms_random)

    summary_md = "\n".join(lines)
    md_path = os.path.join(args.ablation_dir, "SUMMARY.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(summary_md + "\n")

    machine = {
        "baselines": baselines,
        "arms": {
            f"{lang}|{mode}|{method}": s for (lang, mode, method), s in summaries.items()
        },
    }
    json_path = os.path.join(args.ablation_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(machine, handle, indent=2)

    print(summary_md)
    print(f"\nWrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
