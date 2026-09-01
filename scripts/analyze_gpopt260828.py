#!/usr/bin/env python3
"""Final-run paired-bootstrap analysis for the 17 per-level GP profiles.

For every dataset with a stored ``selected_profile.json`` under the final run
directory this script:

* materializes the deterministic grouped train/validation/test split from the
  source word list when a published artifact set ships without split files
  (``--write-splits``),
* validates exact seeded hash membership, surface disjointness, counts, and
  SHA-256 hashes and, when a previous ``bootstrap_ci.json`` is present, checks
  its recorded split hashes,
* regenerates the selected optimized profile and the manuscript's hand-tuned
  baseline profiles on the stored train split with the recorded PATGEN binary,
* asserts that the regenerated optimized held-out Good/Bad/Missed aggregates
  and F_{1/7} exactly reproduce the recorded ``selected_profile.json`` values,
* evaluates optimized and baseline patterns per held-out test-list line and
  computes a paired percentile 95% bootstrap interval for the
  optimized-minus-baseline F_{1/7} delta with bounded-memory batching, and
* emits publication-neutral JSON/LaTeX artifacts under the analysis directory.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import numpy as np

from .analyze_heldout_results import (
    aggregate,
    f17,
    parse_profile,
    per_line_counts,
    precision,
    recall,
)
from .dataset_utls import find_dataset
from .hyperparameters.sample import Sample
from .hyperparameters.score import PatgenScorer
from .dataset_split import create_clean_split
from .optimize_validation import train_patgen_multilevel
from .per_level_search import WEIGHT_LABELS

RESULTS_DIR = Path("results/gpopt260828")
OUTPUT_DIR = Path("results/gpopt260828_analysis")
MANUSCRIPT_TABLES = Path("results/paper2_heldout_main_tables.json")
EXPECTED_DATASETS = 17

# The manuscript compares against hand-tuned profiles; the best-accuracy one is
# selected per dataset by regenerated held-out F_{1/7}.
HAND_PROFILES = {
    "cshyphen": Path("profiles/cshyphen.in"),
    "wortliste": Path("profiles/wortliste.in"),
}

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260829
# Upper bound on lines x batch elements materialized per bootstrap batch; keeps
# the (batch, n, 3) gather below ~24 MiB regardless of test-split size.
BOOTSTRAP_BATCH_ELEMENT_CAP = 1_000_000

SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]




def validate_splits(
    wordlist_path: str, split_dir: Path, config_counts: Dict[str, str]
) -> Dict[str, object]:
    """Assert stored splits exactly match the canonical grouped hash split."""
    split_paths = {name: split_dir / f"data.{name}.wlh" for name in SPLIT_NAMES}
    stored: Dict[str, List[str]] = {}
    evidence: Dict[str, Dict[str, object]] = {}
    for name, path in split_paths.items():
        assert path.is_file(), f"missing stored split file: {path}"
        stored[name] = read_lines(str(path))
        evidence[name] = {"count": len(stored[name]), "sha256": sha256_file(str(path))}

    with tempfile.TemporaryDirectory(prefix="pat-gen-opt-split-audit-") as temp_dir:
        expected = create_clean_split(wordlist_path, temp_dir)
        for name in SPLIT_NAMES:
            expected_path = expected[name]
            assert sha256_file(str(split_paths[name])) == sha256_file(expected_path), (
                f"{name} split does not match canonical grouped hash membership"
            )

    for name in SPLIT_NAMES:
        recorded = config_counts.get(name)
        assert recorded is not None and int(recorded) == len(stored[name]), (
            f"{name} count {len(stored[name])} does not match run_config {recorded}"
        )

    line_sets = {name: set(stored[name]) for name in SPLIT_NAMES}
    overlap = {
        f"{a}_vs_{b}": len(line_sets[a] & line_sets[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    assert not any(overlap.values()), f"split content overlap detected: {overlap}"

    return {
        "splits": evidence,
        "unique_word_types": int(expected["unique_count"]),
        "grouped_hash_membership_ok": True,
        "surface_disjoint_ok": True,
        "content_line_overlap": overlap,
    }


def patgen_binary(config: Dict[str, object]) -> str:
    recorded = str(config.get("patgen") or "")
    if recorded and os.path.exists(recorded):
        return recorded
    found = shutil.which(recorded) if recorded else None
    assert found or shutil.which("patgen"), "no usable PATGEN binary found"
    return found or "patgen"


def expected_weight_labels(good_weights: List[int], bad_weights: List[int]) -> Tuple[str, ...]:
    labels = []
    for good, bad in zip(good_weights, bad_weights):
        if good == 1:
            label = str(bad)
        elif bad == 1:
            label = f"1/{good}"
        else:
            raise AssertionError(f"weight pair ({good}, {bad}) outside the GP ratio space")
        assert label in WEIGHT_LABELS, f"weight label {label} outside the searched space"
        labels.append(label)
    return tuple(labels)


def train_hand_profile(
    patgen_path: str,
    profile_path: Path,
    train_path: str,
    translate_path: str,
    tmp_suffix: str,
) -> Tuple[str, Dict[str, int], PatgenScorer]:
    """train_hand_profile from analyze_heldout_results with a configurable binary."""
    scorer = PatgenScorer(patgen_path, train_path, translate_path, tmp_suffix=tmp_suffix)
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
    patgen_path: str,
    profile: Dict[str, object],
    pat_ranges: List[Tuple[int, int]],
    train_path: str,
    translate_path: str,
    tmp_suffix: str,
) -> Tuple[str, Dict[str, int], PatgenScorer]:
    scorer = PatgenScorer(patgen_path, train_path, translate_path, tmp_suffix=tmp_suffix)
    try:
        params = tuple(int(x) for x in profile["bad_weights"]) + tuple(
            int(x) for x in profile["thresholds"]
        )
        pattern_path, stats = train_patgen_multilevel(
            scorer,
            params,
            pat_ranges,
            good_weight=tuple(int(x) for x in profile["good_weights"]),
        )
        return pattern_path, stats, scorer
    except Exception:
        scorer.clean()
        raise


def f17_batch(good: np.ndarray, bad: np.ndarray, missed: np.ndarray, beta: float = 1 / 7) -> np.ndarray:
    """Vectorized twin of analyze_heldout_results.f17."""
    good = good.astype(np.float64)
    bad = bad.astype(np.float64)
    missed = missed.astype(np.float64)
    beta_sq = beta * beta
    p_den = good + bad
    r_den = good + missed
    prec = np.divide(good, p_den, out=np.zeros_like(good), where=p_den > 0)
    rec = np.divide(good, r_den, out=np.zeros_like(good), where=r_den > 0)
    num = (1 + beta_sq) * prec * rec
    den = (beta_sq * prec) + rec
    return np.divide(num, den, out=np.zeros_like(good), where=(den > 0) & (good > 0))


def bootstrap_delta_batched(
    opt_counts: np.ndarray,
    base_counts: np.ndarray,
    reps: int,
    seed: int,
    element_cap: int = BOOTSTRAP_BATCH_ELEMENT_CAP,
) -> Tuple[float, float]:
    """Paired percentile 95% CI with bounded-memory line resampling batches."""
    assert opt_counts.shape == base_counts.shape and opt_counts.ndim == 2
    rng = np.random.default_rng(seed)
    n = opt_counts.shape[0]
    batch = max(1, min(reps, element_cap // n))
    deltas = np.empty(reps, dtype=np.float64)
    done = 0
    while done < reps:
        size = min(batch, reps - done)
        idx = rng.integers(0, n, size=(size, n))
        opt_sums = opt_counts[idx].sum(axis=1)
        base_sums = base_counts[idx].sum(axis=1)
        deltas[done : done + size] = f17_batch(
            opt_sums[:, 0], opt_sums[:, 1], opt_sums[:, 2]
        ) - f17_batch(base_sums[:, 0], base_sums[:, 1], base_sums[:, 2])
        done += size
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def score_block(counts: Dict[str, int]) -> Dict[str, object]:
    return {
        "good": counts["good"],
        "bad": counts["bad"],
        "missed": counts["missed"],
        "precision": precision(counts["good"], counts["bad"]),
        "recall": recall(counts["good"], counts["missed"]),
        "f17": f17(counts["good"], counts["bad"], counts["missed"]),
    }


def analyze_dataset(
    profile_path_str: str,
    manuscript_baselines: Dict[str, str],
    reps: int,
    seed: int,
    row_index: int,
    results_dir: str,
    write_splits: bool,
    recorded_split_hashes: Dict[str, str],
) -> Dict[str, object]:
    profile_path = Path(profile_path_str)
    run_dir = profile_path.parent
    dataset = run_dir.relative_to(Path(results_dir)).as_posix()

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["dataset"] == dataset, f"dataset mismatch: {config['dataset']} vs {dataset}"
    pat_ranges = [tuple(int(v) for v in pair) for pair in config["pattern_ranges"]]
    n_levels = len(pat_ranges)
    assert len(profile["good_weights"]) == n_levels
    assert len(profile["bad_weights"]) == n_levels
    assert len(profile["thresholds"]) == n_levels
    assert expected_weight_labels(profile["good_weights"], profile["bad_weights"]) == tuple(
        profile["weight_ratios"]
    ), "stored weight labels disagree with decoded PATGEN weights"

    wordlist_path, translate_path = find_dataset(dataset)
    split_dir = run_dir / "splits"
    if write_splits:
        create_clean_split(wordlist_path, str(split_dir))
        print(f"[splits] regenerated canonical grouped split for {dataset}", flush=True)
    split_evidence = validate_splits(
        wordlist_path, split_dir, config.get("split_counts", {})
    )
    for name, evidence in split_evidence["splits"].items():
        recorded = recorded_split_hashes.get(name)
        assert recorded is None or recorded == evidence["sha256"], (
            f"{dataset} {name} split SHA-256 {evidence['sha256']} does not match the "
            f"recorded {recorded}; the source word list is not the one used by the run"
        )
    splits = {name: str(split_dir / f"data.{name}.wlh") for name in SPLIT_NAMES}
    patgen = patgen_binary(config)
    run_tag = uuid.uuid4().hex[:8]

    opt_pattern = opt_scorer = None
    hand = {}
    opt_counts = None
    opt_validation_agg = None
    expected_test = profile["held_out_test"]
    try:
        opt_pattern, opt_stats, opt_scorer = train_optimized_profile(
            patgen, profile, pat_ranges, splits["train"], translate_path, f"_ci_opt_{run_tag}"
        )
        opt_validation_agg = aggregate(
            per_line_counts(splits["validation"], opt_pattern, translate_path)
        )
        opt_counts = per_line_counts(splits["test"], opt_pattern, translate_path)
        opt_agg = aggregate(opt_counts)
        assert (opt_agg["good"], opt_agg["bad"], opt_agg["missed"]) == (
            expected_test["good"],
            expected_test["bad"],
            expected_test["missed"],
        ), f"optimized held-out aggregates do not reproduce {profile_path}"
        assert math.isclose(
            f17(opt_agg["good"], opt_agg["bad"], opt_agg["missed"]),
            profile["held_out_test_f17"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ), "optimized held-out F_{1/7} does not reproduce the selected profile"
        for key in ("train_good", "train_bad", "train_missed", "n_patterns", "trie_nodes"):
            assert opt_stats[key] == expected_test[key], (
                f"optimized {key}={opt_stats[key]} does not reproduce {expected_test[key]}"
            )

        for name, hand_path in HAND_PROFILES.items():
            hand_pattern = hand_scorer = None
            try:
                hand_pattern, hand_stats, hand_scorer = train_hand_profile(
                    patgen, hand_path, splits["train"], translate_path, f"_ci_{name}_{run_tag}"
                )
                hand_counts = per_line_counts(splits["test"], hand_pattern, translate_path)
                validation_aggregate = aggregate(
                    per_line_counts(splits["validation"], hand_pattern, translate_path)
                )
                hand[name] = {
                    "aggregate": aggregate(hand_counts),
                    "stats": hand_stats,
                    "validation_aggregate": validation_aggregate,
                    "counts": hand_counts,
                }
            finally:
                if hand_scorer:
                    hand_scorer.clean()
    finally:
        if opt_scorer:
            opt_scorer.clean()

    assert opt_counts is not None and opt_validation_agg is not None
    for name in HAND_PROFILES:
        assert name in hand, f"hand baseline {name} failed"

    hand_scores = {name: score_block(entry["aggregate"]) for name, entry in hand.items()}
    best_name = max(hand_scores, key=lambda name: hand_scores[name]["f17"])
    manuscript_name = manuscript_baselines.get(dataset)
    manuscript_consistent = manuscript_name is None or manuscript_name == best_name
    assert manuscript_consistent, (
        f"regenerated best baseline {best_name} != manuscript baseline {manuscript_name}"
    )

    best = hand[best_name]
    base_agg = best["aggregate"]
    opt_block = score_block(opt_agg)
    base_block = hand_scores[best_name]
    delta = opt_block["f17"] - base_block["f17"]
    row_seed = seed + row_index
    ci_low, ci_high = bootstrap_delta_batched(opt_counts, best["counts"], reps, row_seed)

    return {
        "dataset": dataset,
        "n_levels": n_levels,
        "pattern_ranges": [list(pair) for pair in pat_ranges],
        "weight_ratios": list(profile["weight_ratios"]),
        "thresholds": [int(x) for x in profile["thresholds"]],
        "validation_objective": profile["validation_objective"],
        "patgen": patgen,
        "split_evidence": split_evidence,
        "optimized": {
            **opt_block,
            "train_good": opt_stats["train_good"],
            "train_bad": opt_stats["train_bad"],
            "train_missed": opt_stats["train_missed"],
            "n_patterns": opt_stats["n_patterns"],
            "trie_nodes": opt_stats["trie_nodes"],
            "reproduces_selected_profile": True,
        },
        "validation_optimized": {
            **score_block(opt_validation_agg),
            "trie_nodes": opt_stats["trie_nodes"],
            "n_patterns": opt_stats["n_patterns"],
        },
        "hand_baselines": {
            name: {
                **hand_scores[name],
                "trie_nodes": hand[name]["stats"]["trie_nodes"],
                "n_patterns": hand[name]["stats"]["n_patterns"],
            }
            for name in sorted(hand_scores)
        },
        "validation_hand_baselines": {
            name: {
                **score_block(hand[name]["validation_aggregate"]),
                "trie_nodes": hand[name]["stats"]["trie_nodes"],
                "n_patterns": hand[name]["stats"]["n_patterns"],
            }
            for name in sorted(hand_scores)
        },
        "best_baseline": best_name,
        "manuscript_baseline": manuscript_name,
        "manuscript_consistent": manuscript_consistent,
        "delta_f17": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "opt_trie": opt_stats["trie_nodes"],
        "base_trie": best["stats"]["trie_nodes"],
        "trie_ratio": opt_stats["trie_nodes"] / best["stats"]["trie_nodes"],
        "test_lines": int(len(opt_counts)),
        "resamples": reps,
        "resample_seed": row_seed,
    }


def latex_ci_table(rows: List[dict], reps: int) -> str:
    lines = [
        r"\begin{table*}[tb]",
        r"\centering\small",
        (
            r"\caption{Paired bootstrap uncertainty for the held-out $F_{1/7}$ of the final "
            r"per-level optimized profiles against the best regenerated hand-tuned baseline. "
            r"Intervals are percentile 95\% intervals from "
            + f"{reps:,}".replace(",", r"{,}")
            + r" paired resamples of held-out word-list lines.}"
        ),
        r"\label{tab:final-bootstrap-ci}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Dataset & Opt.\ $F_{1/7}$ & Base $F_{1/7}$ & $\Delta F_{1/7}$ & 95\% CI & Opt.\ trie/Base trie \\",
        r"\midrule",
    ]
    for row in rows:
        ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        ratio = f"{row['opt_trie']}/{row['base_trie']}"
        dataset = row["dataset"].replace("_", r"\_")
        lines.append(
            f"{dataset} & "
            f"{row['optimized']['f17']:.4f} & "
            f"{row['hand_baselines'][row['best_baseline']]['f17']:.4f} & "
            f"${row['delta_f17']:+.4f}$ & {ci} & {ratio} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def load_manuscript_baselines(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["dataset"]: row["best_baseline_name"] for row in rows}


def load_recorded_split_hashes(path: Path) -> Dict[str, Dict[str, str]]:
    """Per-dataset hashes from a previous canonical grouped-split audit."""
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    recorded: Dict[str, Dict[str, str]] = {}
    for row in rows:
        split_evidence = row.get("split_evidence", {})
        if not split_evidence.get("grouped_hash_membership_ok"):
            continue
        splits = split_evidence.get("splits", {})
        hashes = {
            name: evidence["sha256"]
            for name, evidence in splits.items()
            if isinstance(evidence, dict) and "sha256" in evidence
        }
        if hashes:
            recorded[row["dataset"]] = hashes
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help=(
            "regenerate the deterministic grouped split from the source word list "
            "when a run directory ships without split files"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    profiles = sorted(str(path) for path in results_dir.glob("*/*/selected_profile.json"))
    assert len(profiles) == EXPECTED_DATASETS, (
        f"expected {EXPECTED_DATASETS} selected profiles, found {len(profiles)}"
    )

    manuscript_baselines = load_manuscript_baselines(MANUSCRIPT_TABLES)
    recorded_split_hashes = load_recorded_split_hashes(
        Path(args.output_dir) / "bootstrap_ci.json"
    )
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                analyze_dataset,
                profile_path,
                manuscript_baselines,
                args.reps,
                args.seed,
                index,
                str(results_dir),
                args.write_splits,
                recorded_split_hashes.get(
                    Path(profile_path).parent.relative_to(results_dir).as_posix(), {}
                ),
            ): profile_path
            for index, profile_path in enumerate(profiles, start=1)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[done {len(rows)}/{len(profiles)}] {row['dataset']}: "
                f"delta={row['delta_f17']:+.4f} CI=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]",
                flush=True,
            )
    rows.sort(key=lambda row: row["dataset"])
    assert len(rows) == EXPECTED_DATASETS

    crossing_zero = [row["dataset"] for row in rows if row["ci_low"] <= 0 <= row["ci_high"]]
    summary = {
        "datasets": len(rows),
        "results_dir": str(results_dir.resolve()),
        "resamples": args.reps,
        "seed": args.seed,
        "bootstrap": "paired percentile 95% intervals over held-out test word-list lines",
        "positive_delta_count": sum(1 for row in rows if row["delta_f17"] > 0),
        "bootstrap_positive_ci_count": sum(1 for row in rows if row["ci_low"] > 0),
        "crossing_zero_count": len(crossing_zero),
        "crossing_zero_datasets": crossing_zero,
        "negative_ci_count": sum(1 for row in rows if row["ci_high"] < 0),
        "median_delta_f17": median(row["delta_f17"] for row in rows),
        "median_trie_ratio": median(row["trie_ratio"] for row in rows),
        "manuscript_baseline_mismatches": [
            row["dataset"] for row in rows if not row["manuscript_consistent"]
        ],
        "all_split_assertions_passed": True,
        "all_aggregate_assertions_passed": True,
        "artifacts": {
            "rows": "bootstrap_ci.json",
            "table": "bootstrap_ci_table.tex",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bootstrap_ci.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    (output_dir / "bootstrap_ci_table.tex").write_text(
        latex_ci_table(rows, args.reps) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
