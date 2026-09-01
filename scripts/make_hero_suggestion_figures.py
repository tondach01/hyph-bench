#!/usr/bin/env python3
"""Candidate replacements for Figure 1 (the hero figure) of the per-level GP paper.

The published Figure 1 draws one arrow per dataset from the hand-tuned baseline to
the selected optimized profile in the (trie nodes, F_1/7) plane.  That encoding was
informative when the run produced mixed outcomes: arrow direction and colour told
the reader which datasets won on both axes and which traded one for the other.  In
the reported run all 17 datasets win on both axes, so direction and colour are
constant and the only surviving visual variable is arrow length -- a compound of a
log displacement in trie size and a linear displacement in F_1/7 that no reader can
decompose.  The arrowheads pile up against F = 1 and the labels fight for one band
of canvas.

This script builds candidate heroes from data already on disk:

  * held-out test points          results/gpopt260828_analysis/bootstrap_ci.json
  * validation baseline points    same canonical bootstrap_ci.json audit
  * the 153 recorded evaluations  results/gpopt260828/<lang>/<name>/final_history.csv
  * exported optimized patterns   results/gpopt260828/<lang>/<name>/final_patterns.pat
  * the recorded test split       results/gpopt260828/<lang>/<name>/splits/data.test.wlh
  * the classical profiles        profiles/cshyphen.in, profiles/wortliste.in

No patgen call, no re-optimization, no dataset regeneration.  Every number quoted in
the companion document is written to `stats.json` next to the figures.

Conventions shared by every candidate, so that they can be compared fairly:
  * compactness on x, "smaller is better", pointing left;
  * accuracy on y, "better is up" -- either F_1/7 on a log-residual axis or the
    error-reduction factor (baseline residual / optimized residual);
  * hollow navy marker = hand-tuned, solid green = optimized.

Usage:
    uv run python -m scripts.make_hero_suggestion_figures
    uv run python -m scripts.make_hero_suggestion_figures \
        --copy-to ../brain/overleaf/latex/pics/hero
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

# ---------------------------------------------------------------------------
# Palette. Mirrors make_frontier_visual, so a candidate placed beside the
# published figure does not shift colour.
# ---------------------------------------------------------------------------

INK = "#1f2933"
GREEN = "#236b43"  # optimized
BLUE = "#1b3a6b"  # hand-tuned
WARM = "#a33a35"  # attention / worse
MUTED = "#9aa6b2"
FAINT = "#dde2e7"

SHORT_NAME_OVERRIDES = {
    "cs/cshyphen_cstenten": "cs/ctt",
    "cs/cshyphen_ujc": "cs/ujc",
    "cssk/cshyphen": "cssk",
    "de/wortliste": "de/wl",
    "is/hyphenation-is": "is",
    "th/orchid": "th",
}

FLAGSHIP = "cssk/cshyphen"  # the dataset the cshyphen profile was hand-tuned for
ALTERNATE = "cs/cshyphen_ujc"  # the dataset with the most steeply curved frontier
QUALITATIVE = "cs/wiktionary"
TRANSLATE_FILES = {
    "cs/wiktionary": "data/cs/wiktionary/cs_wiktionary.wlh_dis.wlh.tra",
    "cssk/cshyphen": "data/cssk/cshyphen/cssk-all-weighted.wlhw_expanded.wlh.tra",
}

# patgen profiles as (good_wt, bad_wt, threshold) per level; see profiles/*.in.
CLASSICAL = {
    "cshyphen": [(1, 5, 1), (1, 5, 1), (1, 3, 1), (1, 3, 1)],
    "wortliste": [(2, 3, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)],
}

F_GRID = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)
N_RANDOM = 5  # the initial batch is drawn without a surrogate


def short_name(dataset: str) -> str:
    if dataset in SHORT_NAME_OVERRIDES:
        return SHORT_NAME_OVERRIDES[dataset]
    lang, name = dataset.split("/", 1)
    return lang if name == "wiktionary" else dataset


def style() -> None:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": MUTED,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    trie: int
    f17: float
    precision: float
    recall: float
    bad: int
    missed: int
    good: int

    @property
    def err(self) -> float:
        return max(1e-6, 1.0 - self.f17)


@dataclass(frozen=True)
class Case:
    dataset: str
    label: str
    test_base: Metrics
    test_opt: Metrics
    val_base: Metrics
    val_opt_trie: int
    val_opt_err: float
    history: np.ndarray  # (153, 2): trie nodes, validation residual error
    weight_ratios: Tuple[float, ...]
    thresholds: Tuple[int, ...]
    lines: int

    @property
    def trie_ratio(self) -> float:
        return self.test_opt.trie / self.test_base.trie

    @property
    def err_ratio(self) -> float:
        return self.test_opt.err / self.test_base.err

    @property
    def err_gain(self) -> float:
        """Error-reduction factor: >1 means fewer errors left than hand-tuned."""
        return self.test_base.err / self.test_opt.err

    @property
    def bad_ratio(self) -> float:
        return self.test_opt.bad / self.test_base.bad


def _metrics(entry: dict) -> Metrics:
    return Metrics(
        trie=int(entry["trie_nodes"]),
        f17=float(entry["f17"]),
        precision=float(entry["precision"]),
        recall=float(entry["recall"]),
        bad=int(entry["bad"]),
        missed=int(entry["missed"]),
        good=int(entry["good"]),
    )


def _weight(text: str) -> float:
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den)
    return float(text)


def load_cases(repo_root: Path) -> List[Case]:
    analysis = json.loads(
        (repo_root / "results/gpopt260828_analysis/bootstrap_ci.json").read_text("utf-8")
    )
    cases: List[Case] = []
    for entry in sorted(analysis, key=lambda e: e["dataset"]):
        dataset = entry["dataset"]
        baseline_name = entry["best_baseline"]
        history_rows: List[Tuple[float, float]] = []
        best: Tuple[Optional[Tuple[float, float]], float] = (None, -math.inf)
        path = repo_root / "results/gpopt260828" / dataset / "final_history.csv"
        with path.open(encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                trie = int(record["trie_nodes"])
                err = 1.0 - float(record["validation_f17"])
                history_rows.append((trie, err))
                score = float(record["objective_score"])
                if score > best[1]:
                    best = ((trie, err), score)
        assert best[0] is not None
        cases.append(
            Case(
                dataset=dataset,
                label=short_name(dataset),
                test_base=_metrics(entry["hand_baselines"][baseline_name]),
                test_opt=_metrics(entry["optimized"]),
                val_base=_metrics(entry["validation_hand_baselines"][baseline_name]),
                val_opt_trie=int(best[0][0]),
                val_opt_err=float(best[0][1]),
                history=np.asarray(history_rows, dtype=float),
                weight_ratios=tuple(_weight(w) for w in entry["weight_ratios"]),
                thresholds=tuple(int(t) for t in entry["thresholds"]),
                lines=int(entry["test_lines"]) * 10,
            )
        )
    return cases


def attainment(points: np.ndarray) -> np.ndarray:
    """Lower-left attainment frontier of (trie, error): nothing beats it on both."""
    order = points[np.lexsort((points[:, 1], points[:, 0]))]
    keep: List[Tuple[float, float]] = []
    best = math.inf
    for trie, err in order:
        if err < best - 1e-12:
            keep.append((float(trie), float(err)))
            best = err
    return np.asarray(keep, dtype=float)


def step_curve(front: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Staircase boundary of the achievable set implied by an attainment frontier."""
    xs: List[float] = []
    ys: List[float] = []
    for index, (trie, err) in enumerate(front):
        if index:
            xs.append(trie)
            ys.append(ys[-1])
        xs.append(trie)
        ys.append(err)
    return np.asarray(xs), np.asarray(ys)


def frontier_gain(case: Case) -> np.ndarray:
    """Frontier in gain coordinates: (trie/base_trie, base_err/err), better = up-left."""
    front = attainment(case.history)
    return np.column_stack([front[:, 0] / case.val_base.trie, case.val_base.err / front[:, 1]])


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def ratio_formatter(value: float, _pos: int = 0) -> str:
    """Tick labels on a multiplicative axis: 1/5, 1x, 3x."""
    if value <= 0:
        return ""
    if abs(value - 1.0) < 1e-9:
        return "1×"
    if value < 1.0:
        inverse = 1.0 / value
        rounded = round(inverse)
        if abs(inverse - rounded) < 0.03 * rounded:
            return f"1/{rounded:g}"
        return f"1/{inverse:.1f}"
    rounded = round(value)
    if abs(value - rounded) < 0.03 * max(1.0, rounded):
        return f"{rounded:g}×"
    return f"{value:.1f}×"


def f17_axis(ax, errors: Iterable[float]) -> Tuple[float, float]:
    """Log axis on the residual 1 - F_1/7, ticked with F_1/7 itself, better upwards."""
    values = list(errors)
    lo, hi = min(values), max(values)
    grid = [f for f in F_GRID if lo / 1.6 <= 1.0 - f <= hi * 1.6]
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1.0 - f for f in grid]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{1.0 - v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    limits = (hi * 1.45, lo / 1.55)
    ax.set_ylim(*limits)
    return limits


def log_ratio_axis(ax, axis: str, ticks: Sequence[float]) -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(FuncFormatter(ratio_formatter))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())


def trie_axis(ax, values: Sequence[float]) -> None:
    """Decade-plus-half ticks for a raw trie-size axis."""
    lo, hi = min(values), max(values)
    ticks = [t for t in (200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000)
             if lo / 1.9 <= t <= hi * 1.9]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def place_labels(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    color=GREEN,
    fontsize: float = 7.4,
    radius: float = 9.0,
) -> None:
    """Eight-direction greedy label placement in display space.

    Enough for a 17-point scatter: each label takes the cheapest ring position that
    does not overlap an already-placed label or any data marker. Deterministic, and
    it must be called after the figure's final layout, because the seats are chosen
    in display space and then frozen into data coordinates.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    trans = ax.transData
    pts = [trans.transform((x, y)) for x, y in zip(xs, ys)]
    marker_boxes = [(px - 3.5, py - 3.5, px + 3.5, py + 3.5) for px, py in pts]
    placed: List[Tuple[float, float, float, float]] = []
    directions = [
        (0.0, 1.0),
        (0.72, 0.72),
        (-0.72, 0.72),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.72, -0.72),
        (-0.72, -0.72),
        (0.0, -1.0),
    ]
    x0, x1 = sorted(ax.transData.transform([(v, ax.get_ylim()[0]) for v in ax.get_xlim()])[:, 0])
    y0, y1 = sorted(ax.transData.transform([(ax.get_xlim()[0], v) for v in ax.get_ylim()])[:, 1])
    for index in sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1])):
        px, py = pts[index]
        probe = ax.text(0, 0, labels[index], fontsize=fontsize, alpha=0.0)
        bbox = probe.get_window_extent(renderer=renderer)
        width, height = bbox.width, bbox.height
        probe.remove()
        best: Optional[Tuple[float, float, float, Tuple[float, float, float, float]]] = None
        for scale in (1.0, 1.5, 2.1, 3.0, 4.2):
            for dx, dy in directions:
                cx = px + dx * (radius * scale + width / 2.0)
                cy = py + dy * (radius * scale + height / 2.0)
                box = (cx - width / 2 - 1.0, cy - height / 2 - 1.0,
                       cx + width / 2 + 1.0, cy + height / 2 + 1.0)
                cost = 0.0
                for other in placed + marker_boxes:
                    ox = min(box[2], other[2]) - max(box[0], other[0])
                    oy = min(box[3], other[3]) - max(box[1], other[1])
                    if ox > 0 and oy > 0:
                        cost += min(ox, oy)
                cost += 6.0 * max(0.0, x0 - box[0]) + 6.0 * max(0.0, box[2] - x1)
                cost += 6.0 * max(0.0, y0 - box[1]) + 6.0 * max(0.0, box[3] - y1)
                cost += 0.3 * (scale - 1.0) * radius + 0.7 * (1.0 - dy)
                if best is None or cost < best[0]:
                    best = (cost, cx, cy, box)
            if best is not None and best[0] < 0.25:
                break
        assert best is not None
        _, cx, cy, box = best
        placed.append(box)
        ax.annotate(
            labels[index],
            xy=trans.inverted().transform((cx, cy)),
            xycoords="data",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=9,
        )


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=190)
    plt.close(fig)
    print(f"  {stem}")


# ---------------------------------------------------------------------------
# Diagnostic: what the accuracy ceiling does to the published axis
# ---------------------------------------------------------------------------


def fig_diag_axes(cases: List[Case], out_dir: Path, stats: dict) -> None:
    opt = np.array([c.test_opt.f17 for c in cases])
    base = np.array([c.test_base.f17 for c in cases])
    crowded = opt[opt >= 0.989]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35))

    ax = axes[0]
    lo, hi = 0.8555, 1.0055  # what _axis_limits() produces for this run
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    box = (crowded.min(), crowded.max())
    frac = (box[1] - box[0]) / (hi - lo)
    ax.add_patch(
        Rectangle((box[0], 0.10), box[1] - box[0], 0.34, facecolor=WARM, alpha=0.16, lw=0, zorder=0)
    )
    for value in base:
        ax.plot([value, value], [0.60, 0.86], color=BLUE, lw=1.0, alpha=0.8)
    for value in opt:
        ax.plot([value, value], [0.14, 0.40], color=GREEN, lw=1.2)
    ax.text(lo + 0.004, 0.88, "hand-tuned", fontsize=7.4, color=BLUE, va="bottom")
    ax.text(lo + 0.004, 0.42, "optimized", fontsize=7.4, color=GREEN, va="bottom")
    ax.annotate(
        f"{len(crowded)} of the 17 results\nthat matter share\n{100 * frac:.0f}% of the axis",
        xy=(box[0], 0.27),
        xytext=(0.936, 0.27),
        fontsize=7.2,
        color=WARM,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.13", color=WARM, lw=0.7),
    )
    ax.set_xlabel("$F_{1/7}$, linear axis (as published)")
    ax.set_title("(a) the axis Figure 1 uses", loc="left", fontsize=8.6)

    ax = axes[1]
    ax.set_xscale("log")
    errs = np.concatenate([1 - base, 1 - opt])
    ax.set_xlim(errs.max() * 1.5, errs.min() / 1.6)
    grid = [f for f in F_GRID if errs.min() / 1.6 <= 1 - f <= errs.max() * 1.6]
    ax.xaxis.set_major_locator(FixedLocator([1 - f for f in grid]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{1 - v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    tlo, thi = 1 - crowded.max(), 1 - crowded.min()
    ax.add_patch(Rectangle((tlo, 0.10), thi - tlo, 0.34, facecolor=GREEN, alpha=0.13, lw=0, zorder=0))
    for value in 1 - base:
        ax.plot([value, value], [0.60, 0.86], color=BLUE, lw=1.0, alpha=0.8)
    for value in 1 - opt:
        ax.plot([value, value], [0.14, 0.40], color=GREEN, lw=1.2)
    span = math.log10(errs.max() * 1.5 / (errs.min() / 1.6))
    frac2 = math.log10(thi / tlo) / span
    ax.annotate(
        f"the same {len(crowded)} results\nnow share {100 * frac2:.0f}%",
        xy=(thi, 0.27),
        xytext=(0.062, 0.27),
        fontsize=7.2,
        color=GREEN,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.13", color=GREEN, lw=0.7),
    )
    ax.set_xlabel("$F_{1/7}$ on a log-residual axis ($1-F_{1/7}$)")
    ax.set_title("(b) the same 34 numbers, decompressed", loc="left", fontsize=8.6)

    fig.tight_layout()
    save(fig, out_dir, "hero_diag_axes")
    stats["axis_decompression"] = {
        "n_crowded": int(len(crowded)),
        "linear_axis_share_pct": round(100 * float(frac), 1),
        "log_axis_share_pct": round(100 * float(frac2), 1),
        "factor": round(float(frac2 / frac), 1),
    }


# ---------------------------------------------------------------------------
# B: the gain plane
# ---------------------------------------------------------------------------


def _gain_axes(
    ax,
    cases: List[Case],
    *,
    x_label: str = "trie size, optimized / hand-tuned\n←\u2002smaller is better",
) -> Tuple[float, float]:
    tr = np.array([c.trie_ratio for c in cases])
    eg = np.array([c.err_gain for c in cases])
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_lo, x_hi = tr.min() / 1.30, 1.28
    y_lo, y_hi = 0.90, eg.max() * 1.30
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.add_patch(
        Rectangle((1.0, y_lo), x_hi - 1.0, y_hi - y_lo, facecolor=WARM, alpha=0.08, lw=0, zorder=0)
    )
    ax.add_patch(
        Rectangle((x_lo, y_lo), x_hi - x_lo, 1.0 - y_lo, facecolor=WARM, alpha=0.08, lw=0, zorder=0)
    )
    ax.axvline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)

    med_t = float(np.median(tr))
    med_e = float(np.median(eg))
    ax.plot([x_lo, med_t], [med_e, med_e], color=GREEN, lw=0.7, ls=(0, (1, 1.8)), zorder=2)
    ax.plot([med_t, med_t], [y_lo, med_e], color=GREEN, lw=0.7, ls=(0, (1, 1.8)), zorder=2)
    ax.scatter(tr, eg, s=18, color=GREEN, zorder=6, linewidths=0)
    ax.plot([1.0], [1.0], "o", ms=6.5, mfc="white", mec=BLUE, mew=1.4, zorder=7)

    log_ratio_axis(ax, "x", [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0])
    log_ratio_axis(ax, "y", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    ax.set_xlabel(x_label)
    ax.set_ylabel("errors left, hand-tuned / optimized\n→\u2002fewer is better")
    return med_t, med_e


def fig_gain(cases: List[Case], out_dir: Path, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    med_t, med_e = _gain_axes(ax, cases)
    ax.annotate(
        "every hand-tuned baseline\nis this one point",
        xy=(1.0, 1.0),
        xytext=(0.68, 1.03),
        fontsize=7.6,
        color=BLUE,
        ha="center",
        va="bottom",
    )
    ax.text(
        1 / 5.8,
        1.22,
        f"median: {1 / med_t:.1f}× smaller,\n{med_e:.1f}× fewer errors",
        fontsize=7.6,
        color=GREEN,
        ha="left",
        va="bottom",
    )
    ax.set_title("All 17 datasets", loc="left", fontsize=8.6)
    fig.tight_layout()
    place_labels(ax, [c.trie_ratio for c in cases], [c.err_gain for c in cases],
                 [c.label for c in cases])
    save(fig, out_dir, "hero_b_gain")
    stats["gain_plane"] = {
        "median_trie_ratio": round(float(np.median([c.trie_ratio for c in cases])), 4),
        "median_error_gain": round(float(np.median([c.err_gain for c in cases])), 3),
        "weakest_trie_ratio": round(max(c.trie_ratio for c in cases), 3),
        "weakest_error_gain": round(min(c.err_gain for c in cases), 3),
        "all_in_better_quadrant": int(
            sum(1 for c in cases if c.trie_ratio < 1 and c.err_gain > 1)
        ),
    }


# ---------------------------------------------------------------------------
# C: paired slopegraph
# ---------------------------------------------------------------------------


def fig_slope(cases: List[Case], out_dir: Path, stats: dict) -> None:
    ordered = sorted(cases, key=lambda c: c.trie_ratio)
    n = len(ordered)
    ys = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.9), sharey=True)

    for ax, kind in zip(axes, ("trie", "err")):
        ax.set_xscale("log")
        for y, case in zip(ys, ordered):
            if kind == "trie":
                a, b, factor = case.test_base.trie, case.test_opt.trie, 1 / case.trie_ratio
            else:
                a, b, factor = case.test_base.err, case.test_opt.err, case.err_gain
            ax.plot([a, b], [y, y], color=FAINT, lw=2.6, zorder=1, solid_capstyle="butt")
            ax.plot([a], [y], "o", ms=4.2, mfc="white", mec=BLUE, mew=1.0, zorder=3)
            ax.plot([b], [y], "o", ms=4.2, color=GREEN, zorder=3)
            ax.annotate(
                f"{factor:.1f}×",
                xy=(b, y),
                xytext=(-6, 0),
                textcoords="offset points",
                fontsize=6.8,
                color=GREEN,
                ha="right",
                va="center",
            )
        ax.set_ylim(-0.7, n - 0.3)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c.label for c in ordered], fontsize=7.2)
    axes[0].set_xlim(230, 90000)
    trie_axis(axes[0], [400, 60000])
    axes[0].set_xlabel("pattern trie size (nodes)\u2003←\u2002smaller")
    axes[0].set_title("(a) compactness", loc="left", fontsize=8.6)
    axes[1].set_xlim(5.5e-4, 0.36)
    axes[1].set_xlabel("errors left, $1-F_{1/7}$\u2003←\u2002fewer")
    axes[1].set_title("(b) accuracy", loc="left", fontsize=8.6)
    axes[1].xaxis.set_major_locator(FixedLocator([1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    axes[1].xaxis.set_minor_locator(NullLocator())
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.0, ms=4.6,
                   label="hand-tuned"),
            Line2D([], [], marker="o", ls="", color=GREEN, ms=4.6, label="optimized"),
        ],
        loc="lower center",
        ncols=2,
        fontsize=7.4,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out_dir, "hero_c_slope")


# ---------------------------------------------------------------------------
# D: the frontier family
# ---------------------------------------------------------------------------


def frontier_band(cases: List[Case], grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Per size ratio, the distribution over datasets of the best reachable gain."""
    rows = np.full((len(cases), len(grid)), np.nan)
    for i, case in enumerate(cases):
        front = frontier_gain(case)
        for j, g in enumerate(grid):
            usable = front[front[:, 0] <= g]
            if len(usable):
                rows[i, j] = usable[:, 1].max()
    out = {}
    counts = np.sum(~np.isnan(rows), axis=0)
    keep = counts >= len(cases) // 2
    out["grid"] = grid[keep]
    for name, q in (("lo", 25), ("med", 50), ("hi", 75)):
        out[name] = np.nanpercentile(rows[:, keep], q, axis=0)
    out["count"] = counts[keep]
    return out


def fig_frontiers(cases: List[Case], out_dir: Path, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    for case in cases:
        xs, ys = step_curve(frontier_gain(case)[:, ::-1][:, ::-1])
        front = frontier_gain(case)
        xs = np.repeat(front[:, 0], 2)[1:]
        ys = np.repeat(front[:, 1], 2)[:-1]
        keep = ys >= 0.9
        ax.plot(xs[keep], ys[keep], color=MUTED, lw=0.7, alpha=0.6, zorder=2)
    band = frontier_band(cases, np.geomspace(0.05, 1.0, 140))
    ax.fill_between(band["grid"], band["lo"], band["hi"], color=GREEN, alpha=0.16, lw=0, zorder=3)
    ax.plot(band["grid"], band["med"], color=GREEN, lw=2.0, zorder=5)
    ax.scatter(
        [c.val_opt_trie / c.val_base.trie for c in cases],
        [c.val_base.err / c.val_opt_err for c in cases],
        s=17,
        color=GREEN,
        zorder=6,
        linewidths=0,
    )
    ax.plot([1.0], [1.0], "o", ms=7.0, mfc="white", mec=BLUE, mew=1.4, zorder=7)
    ax.axvline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(0.035, 1.5)
    ax.set_ylim(0.82, 10.5)
    log_ratio_axis(ax, "x", [1 / 20, 1 / 10, 1 / 5, 1 / 3, 1 / 2, 1.0])
    log_ratio_axis(ax, "y", [1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    ax.set_xlabel("trie size, relative to hand-tuned\u2003←\u2002smaller")
    ax.set_ylabel("errors left, hand-tuned / this profile\n↑\u2002fewer")
    ax.annotate(
        "hand-tuned profile,\nall 17 datasets",
        xy=(1.0, 1.0),
        xytext=(0.60, 1.15),
        fontsize=7.6,
        color=BLUE,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=1, shrinkB=4),
    )
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=0.9, label="attainable frontier, one per dataset"),
            Line2D([], [], color=GREEN, lw=2.0, label="median frontier (band: quartiles)"),
            Line2D([], [], marker="o", ls="", color=GREEN, ms=4.6, label="profile the paper reports"),
            Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.3, ms=5.4,
                   label="hand-tuned baseline"),
        ],
        loc="upper left",
        fontsize=7.2,
    )
    fig.tight_layout()
    save(fig, out_dir, "hero_d_frontiers")
    reachable = band["med"]
    stats["frontier_band"] = {
        "median_gain_at_one_fifth_size": round(
            float(np.interp(0.2, band["grid"], reachable)), 2
        ),
        "median_gain_at_one_third_size": round(
            float(np.interp(1 / 3, band["grid"], reachable)), 2
        ),
        "median_gain_at_equal_size": round(float(reachable[-1]), 2),
    }


# ---------------------------------------------------------------------------
# E: mechanism + generality (the recommended hero)
# ---------------------------------------------------------------------------


def _mechanism_panel(ax, case: Case, *, guides: bool = True) -> dict:
    hist = case.history
    front = attainment(hist)
    bt, be = float(case.val_base.trie), case.val_base.err
    ax.set_xscale("log")
    ax.scatter(hist[:, 0], hist[:, 1], s=8, color=MUTED, alpha=0.5, linewidths=0, zorder=2)
    xs, ys = step_curve(front)
    ax.plot(xs, ys, color=GREEN, lw=1.6, zorder=4)

    same_acc = hist[hist[:, 1] <= be]
    same_size = hist[hist[:, 0] <= bt]
    smaller = bt / same_acc[:, 0].min()
    fewer = be / same_size[:, 1].min()

    y_lo, y_hi = f17_axis(ax, hist[:, 1].tolist() + [be])
    ax.set_ylim(1 - 0.88, y_hi)
    x_lo, x_hi = min(hist[:, 0].min(), bt) / 1.8, bt * 2.0
    ax.set_xlim(x_lo, x_hi)
    trie_axis(ax, [x_lo, x_hi])

    if guides:
        ax.plot([same_acc[:, 0].min(), bt], [be, be], color=WARM, lw=0.9, ls=(0, (3, 2)), zorder=3)
        ax.plot([same_acc[:, 0].min()], [be], "o", ms=3.6, color=WARM, zorder=5)
        ax.annotate(
            f"same accuracy,\n{smaller:.1f}× smaller",
            xy=(math.sqrt(same_acc[:, 0].min() * bt), be),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=7.2,
            color=WARM,
            ha="center",
            va="top",
        )
        best_same_size = same_size[np.argmin(same_size[:, 1])]
        ax.plot([bt, bt], [be, best_same_size[1]], color=WARM, lw=0.9, ls=(0, (3, 2)), zorder=3)
        ax.plot([bt], [best_same_size[1]], "o", ms=3.6, color=WARM, zorder=5)
        ax.annotate(
            f"same size,\n{fewer:.1f}× fewer errors",
            xy=(bt, math.sqrt(be * best_same_size[1])),
            xytext=(-6, 0),
            textcoords="offset points",
            fontsize=7.2,
            color=WARM,
            ha="right",
            va="center",
        )

    ax.plot([bt], [be], "o", ms=7.0, mfc="white", mec=BLUE, mew=1.5, zorder=6)
    ax.plot([case.val_opt_trie], [case.val_opt_err], "*", ms=12.0, color=GREEN, zorder=7)
    ax.annotate(
        "hand-tuned",
        xy=(bt, be),
        xytext=(7, -8),
        textcoords="offset points",
        fontsize=7.4,
        color=BLUE,
        ha="left",
        va="top",
    )
    ax.annotate(
        "reported",
        xy=(case.val_opt_trie, case.val_opt_err),
        xytext=(11, -3),
        textcoords="offset points",
        fontsize=7.4,
        color=GREEN,
        ha="left",
        va="top",
    )
    ax.set_xlabel("pattern trie size (nodes)\u2003←\u2002smaller")
    ax.set_ylabel("$F_{1/7}$ on validation\u2002→\u2002more accurate")
    del y_lo, y_hi
    return {"smaller": smaller, "fewer": fewer}


def _twopanel(cases: List[Case], dataset: str, out_dir: Path, stem: str) -> dict:
    case = next(c for c in cases if c.dataset == dataset)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.35))
    info = _mechanism_panel(axes[0], case)
    axes[0].set_title(
        f"(a) one dataset ({case.label}): the trade-off is a curve", loc="left", fontsize=8.6
    )
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=3.6, label="153 evaluated profiles"),
            Line2D([], [], color=GREEN, lw=1.6, label="attainable frontier"),
        ],
        loc="best",
        fontsize=7.2,
    )
    med_t, med_e = _gain_axes(
        axes[1], cases, x_label="trie size, optimized / hand-tuned\u2002←\u2002smaller"
    )
    axes[1].text(
        1 / 5.8,
        1.28,
        f"median: {1 / med_t:.1f}× smaller,\n{med_e:.1f}× fewer errors",
        fontsize=7.8,
        color=GREEN,
        ha="left",
        va="bottom",
    )
    axes[1].annotate(
        "hand-tuned",
        xy=(1.0, 1.0),
        xytext=(0.62, 1.02),
        fontsize=7.4,
        color=BLUE,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=1, shrinkB=4),
    )
    axes[1].set_title("(b) all 17 datasets: it always happens", loc="left", fontsize=8.6)
    fig.tight_layout()
    place_labels(axes[1], [c.trie_ratio for c in cases], [c.err_gain for c in cases],
                 [c.label for c in cases], fontsize=7.0, radius=8.0)
    save(fig, out_dir, stem)
    return info


def fig_twopanel(cases: List[Case], out_dir: Path, stats: dict) -> None:
    stats["twopanel"] = {
        FLAGSHIP: _twopanel(cases, FLAGSHIP, out_dir, "hero_e_twopanel"),
        ALTERNATE: _twopanel(cases, ALTERNATE, out_dir, "hero_e2_twopanel_alt"),
    }

def fig_mechanism(cases: List[Case], out_dir: Path, stats: dict) -> None:
    case = next(c for c in cases if c.dataset == FLAGSHIP)
    fig, ax = plt.subplots(figsize=(4.0, 3.35))
    stats["mechanism"] = _mechanism_panel(ax, case)
    ax.set_title(
        f"One dataset ({case.label}): the trade-off is a curve", loc="left", fontsize=8.6
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=3.6, label="153 evaluated profiles"),
            Line2D([], [], color=GREEN, lw=1.6, label="attainable frontier"),
        ],
        loc="best",
        fontsize=7.2,
    )
    fig.tight_layout()
    save(fig, out_dir, "hero_e_mechanism")


def fig_hero_stack(cases: List[Case], out_dir: Path, stats: dict) -> None:
    """Panels B and E as one two-row figure sharing frame geometry.

    Saving the panels separately with a tight bounding box lets each panel's
    y-label extent set its own left edge, so the frames do not align once both
    are scaled to \\columnwidth. One figure, one subplot column: same axes
    rectangle for both rows.
    """
    case = next(c for c in cases if c.dataset == FLAGSHIP)
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 6.55))

    med_t, med_e = _gain_axes(axes[0], cases)
    axes[0].annotate(
        "every hand-tuned baseline\nis this one point",
        xy=(1.0, 1.0),
        xytext=(0.585, 1.06),
        fontsize=7.4,
        color=BLUE,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=1, shrinkB=6),
    )
    axes[0].text(
        math.sqrt(axes[0].get_xlim()[0] * med_t),
        1.06,
        f"median: {1 / med_t:.1f}× smaller,\n{med_e:.1f}× fewer errors",
        fontsize=7.4,
        color=GREEN,
        ha="center",
        va="bottom",
    )
    axes[0].set_title("All 17 datasets", loc="left", fontsize=8.6)

    stats["hero_stack"] = _mechanism_panel(axes[1], case)
    axes[1].set_title(
        f"One dataset ({case.label}): the trade-off is a curve", loc="left", fontsize=8.6
    )
    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=3.6, label="153 evaluated profiles"),
            Line2D([], [], color=GREEN, lw=1.6, label="attainable frontier"),
        ],
        loc="upper right",
        fontsize=7.2,
    )

    fig.tight_layout(h_pad=2.2)
    fig.align_ylabels(axes)
    place_labels(axes[0], [c.trie_ratio for c in cases], [c.err_gain for c in cases],
                 [c.label for c in cases])
    save(fig, out_dir, "hero_stack")




# ---------------------------------------------------------------------------
# F: the two-way readout
# ---------------------------------------------------------------------------


def two_way_readout(cases: List[Case]) -> List[dict]:
    out = []
    for case in cases:
        hist = case.history
        bt, be = float(case.val_base.trie), case.val_base.err
        same_acc = hist[hist[:, 1] <= be]
        same_size = hist[hist[:, 0] <= bt]
        dominating = (hist[:, 0] <= bt) & (hist[:, 1] <= be)
        out.append(
            dict(
                dataset=case.dataset,
                label=case.label,
                smaller=float(bt / same_acc[:, 0].min()) if len(same_acc) else float("nan"),
                fewer=float(be / same_size[:, 1].min()) if len(same_size) else float("nan"),
                dominating=int(dominating.sum()),
                dominating_random=int(dominating[:N_RANDOM].sum()),
            )
        )
    return out


def fig_readout(cases: List[Case], out_dir: Path, stats: dict) -> None:
    read = sorted(two_way_readout(cases), key=lambda r: r["smaller"])
    n = len(read)
    ys = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), sharey=True)
    for ax, key, colour, title, unit in (
        (axes[0], "smaller", GREEN, "(a) at the hand-tuned accuracy", "× smaller trie"),
        (axes[1], "fewer", BLUE, "(b) at the hand-tuned trie size", "× fewer errors left"),
    ):
        vals = np.array([r[key] for r in read])
        ax.barh(ys, vals, height=0.6, color=colour, alpha=0.85, linewidth=0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(1.0, vals.max() * 2.4)
        med = float(np.median(vals))
        ax.axvline(med, color=WARM, lw=1.0, ls=(0, (4, 2)), zorder=4)
        ax.axvline(1.0, color=INK, lw=0.9, zorder=4)
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 40]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.grid(axis="y", visible=False)
        ax.text(
            med,
            n - 0.25,
            f"median {med:.1f}×  ",
            fontsize=7.4,
            color=WARM,
            va="center",
            ha="right",
        )
        for y, value in zip(ys, vals):
            ax.annotate(
                f"{value:.1f}×",
                xy=(value, y),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=6.8,
                color=INK,
                va="center",
            )
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["label"] for r in read], fontsize=7.2)
    axes[0].set_ylim(-0.7, n + 0.1)
    fig.tight_layout()
    save(fig, out_dir, "hero_f_readout")
    stats["two_way_readout"] = {
        "median_smaller_at_equal_accuracy": round(float(np.median([r["smaller"] for r in read])), 2),
        "median_fewer_errors_at_equal_size": round(float(np.median([r["fewer"] for r in read])), 2),
        "per_dataset": {r["label"]: [round(r["smaller"], 2), round(r["fewer"], 2)] for r in read},
    }


# ---------------------------------------------------------------------------
# G: small multiples
# ---------------------------------------------------------------------------


def fig_multiples(cases: List[Case], out_dir: Path, stats: dict) -> None:
    ordered = sorted(cases, key=lambda c: c.trie_ratio)
    fig, axes = plt.subplots(3, 6, figsize=(7.5, 3.6), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, case in zip(flat, ordered):
        norm_t = case.history[:, 0] / case.val_base.trie
        norm_g = case.val_base.err / case.history[:, 1]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(norm_t, norm_g, s=2.6, color=MUTED, alpha=0.55, linewidths=0)
        front = frontier_gain(case)
        ax.plot(np.repeat(front[:, 0], 2)[1:], np.repeat(front[:, 1], 2)[:-1], color=GREEN, lw=1.0)
        ax.plot([1.0], [1.0], "o", ms=3.6, mfc="white", mec=BLUE, mew=0.9)
        ax.plot(
            [case.val_opt_trie / case.val_base.trie],
            [case.val_base.err / case.val_opt_err],
            "*",
            ms=6.5,
            color=GREEN,
        )
        ax.set_title(case.label, fontsize=7.2, pad=2.0)
        ax.tick_params(labelsize=6.2, length=2)
    for ax in flat[len(ordered):]:
        ax.axis("off")
    flat[0].set_xlim(0.025, 2.0)
    flat[0].set_ylim(0.2, 22.0)
    log_ratio_axis(flat[0], "x", [1 / 20, 1 / 5, 1.0])
    log_ratio_axis(flat[0], "y", [1 / 3, 1.0, 3.0, 10.0])
    fig.supxlabel("trie size / hand-tuned\u2003←\u2002smaller", fontsize=8.2, y=0.012)
    fig.supylabel("errors left, hand-tuned / this profile\u2002↑\u2002fewer", fontsize=8.2, x=0.004)
    fig.tight_layout(rect=(0.028, 0.035, 1, 1))
    save(fig, out_dir, "hero_g_multiples")


# ---------------------------------------------------------------------------
# H: minimum-risk lollipop
# ---------------------------------------------------------------------------


def fig_lollipop(cases: List[Case], out_dir: Path, stats: dict) -> None:
    ordered = sorted(cases, key=lambda c: -c.trie_ratio)
    n = len(ordered)
    ys = np.arange(n)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.set_xscale("log")
    for y, case in zip(ys, ordered):
        ax.plot([case.trie_ratio, 1.0], [y, y], color=FAINT, lw=3.2, solid_capstyle="butt", zorder=1)
        ax.plot([case.trie_ratio], [y], "o", ms=5.0, color=GREEN, zorder=3)
        ax.plot([case.err_ratio], [y], "D", ms=4.0, color=BLUE, zorder=3)
    ax.axvline(1.0, color=INK, lw=0.9)
    for value, colour, label, dy in (
        (float(np.median([c.trie_ratio for c in cases])), GREEN, "median trie", 0.55),
        (float(np.median([c.err_ratio for c in cases])), BLUE, "median errors", 1.35),
    ):
        ax.axvline(value, color=colour, lw=0.9, ls=(0, (4, 2)), alpha=0.85)
        ax.text(value, n - 1 + dy, f" {label} {value:.2f}", fontsize=7.2, color=colour, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([c.label for c in ordered], fontsize=7.2)
    ax.set_ylim(-0.7, n + 0.9)
    ax.set_xlim(0.11, 1.42)
    log_ratio_axis(ax, "x", [1 / 8, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0])
    ax.set_xlabel("optimized / hand-tuned\u2003←\u2002better")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=GREEN, ms=5.0, label="trie size ratio"),
            Line2D([], [], marker="D", ls="", color=BLUE, ms=4.2, label="errors-left ratio"),
        ],
        loc="lower left",
        fontsize=7.2,
    )
    fig.tight_layout()
    save(fig, out_dir, "hero_h_lollipop")


# ---------------------------------------------------------------------------
# I: the precision/recall mechanism
# ---------------------------------------------------------------------------


def fig_pr(cases: List[Case], out_dir: Path, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for case in cases:
        ax.annotate(
            "",
            xy=(case.test_opt.recall, case.test_opt.precision),
            xytext=(case.test_base.recall, case.test_base.precision),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.4,head_width=0.16",
                color=GREEN,
                lw=0.95,
                shrinkA=1.5,
                shrinkB=1.0,
            ),
        )
        ax.plot(
            [case.test_base.recall],
            [case.test_base.precision],
            "o",
            ms=3.6,
            mfc="white",
            mec=BLUE,
            mew=0.9,
            zorder=4,
        )
    ax.set_xlabel("recall\u2003→\u2002finds more of the legal break points")
    ax.set_ylabel("precision\u2002↑\u2002inserts fewer wrong hyphens")
    ax.set_xlim(0.655, 1.035)
    ax.set_ylim(0.876, 1.006)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.0, ms=4.6,
                   label="hand-tuned"),
            Line2D([], [], marker=">", ls="", color=GREEN, ms=5.0, label="optimized (arrow head)"),
        ],
        loc="lower left",
        fontsize=7.2,
    )
    fig.tight_layout()
    place_labels(
        ax,
        [c.test_opt.recall for c in cases],
        [c.test_opt.precision for c in cases],
        [c.label for c in cases],
        radius=7.5,
        fontsize=7.0,
    )
    save(fig, out_dir, "hero_i_pr")
    stats["precision_recall"] = {
        "median_precision_gain": round(
            float(np.median([c.test_opt.precision - c.test_base.precision for c in cases])), 4
        ),
        "median_recall_loss": round(
            float(np.median([c.test_opt.recall - c.test_base.recall for c in cases])), 4
        ),
        "worst_recall": min((round(c.test_opt.recall, 4), c.label) for c in cases),
        "datasets_losing_recall": int(
            sum(1 for c in cases if c.test_opt.recall < c.test_base.recall)
        ),
        "median_wrong_hyphen_ratio": round(float(np.median([c.bad_ratio for c in cases])), 4),
    }


# ---------------------------------------------------------------------------
# J: the space, not the optimizer
# ---------------------------------------------------------------------------


def fig_space(cases: List[Case], out_dir: Path, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    all_t: List[float] = []
    all_g: List[float] = []
    rand_t: List[float] = []
    rand_g: List[float] = []
    for case in cases:
        t = case.history[:, 0] / case.val_base.trie
        g = case.val_base.err / case.history[:, 1]
        all_t.extend(t.tolist())
        all_g.extend(g.tolist())
        rand_t.extend(t[:N_RANDOM].tolist())
        rand_g.extend(g[:N_RANDOM].tolist())
    ax.add_patch(
        Rectangle((0.004, 1.0), 1.0 - 0.004, 40.0, facecolor=GREEN, alpha=0.07, lw=0, zorder=0)
    )
    ax.scatter(all_t, all_g, s=3.2, color=MUTED, alpha=0.32, linewidths=0, zorder=2)
    ax.scatter(rand_t, rand_g, s=12, color=WARM, alpha=0.9, linewidths=0, zorder=3)
    ax.plot([1.0], [1.0], "o", ms=8.0, mfc="white", mec=BLUE, mew=1.6, zorder=6)
    ax.axvline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhline(1.0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    rt = np.asarray(rand_t)
    rg = np.asarray(rand_g)
    frac = float(((rt <= 1.0) & (rg >= 1.0)).mean())
    ax.set_xlim(0.008, 3.2)
    ax.set_ylim(0.03, 34.0)
    log_ratio_axis(ax, "x", [1 / 100, 1 / 20, 1 / 5, 1.0, 3.0])
    log_ratio_axis(ax, "y", [1 / 20, 1 / 5, 1.0, 5.0, 20.0])
    ax.set_xlabel("trie size / hand-tuned\u2003←\u2002smaller")
    ax.set_ylabel("errors left, hand-tuned / this profile\u2002↑\u2002fewer")
    ax.text(
        0.0105,
        22.0,
        f"{100 * frac:.0f}% of the {len(rt)} purely random starting draws\n"
        "already beat the hand-tuned profile on both axes",
        fontsize=7.8,
        color=WARM,
        ha="left",
        va="top",
    )
    ax.annotate(
        "hand-tuned",
        xy=(1.0, 1.0),
        xytext=(-8, -10),
        textcoords="offset points",
        fontsize=7.6,
        color=BLUE,
        ha="right",
        va="top",
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=4.0,
                   label=f"all {len(all_t):,} evaluated profiles"),
            Line2D([], [], marker="o", ls="", color=WARM, ms=4.6,
                   label=f"initial random batch ({N_RANDOM} per dataset)"),
        ],
        loc="lower left",
        fontsize=7.2,
    )
    fig.tight_layout()
    save(fig, out_dir, "hero_j_space")
    read = two_way_readout(cases)
    stats["space"] = {
        "random_draws_dominating_pct": round(100 * frac, 1),
        "all_evals_dominating_pct": round(
            100 * sum(r["dominating"] for r in read) / (len(cases[0].history) * len(read)), 1
        ),
        "datasets_with_dominating_random_draw": sum(1 for r in read if r["dominating_random"] > 0),
    }


# ---------------------------------------------------------------------------
# K: the qualitative panel
# ---------------------------------------------------------------------------


def _breaks(word: str) -> set:
    out, index = set(), 0
    for char in word:
        if char == "-":
            out.add(index)
        else:
            index += 1
    return out


def hyphenation_examples(repo_root: Path, dataset: str, limit: int = 3000) -> dict:
    """Real optimized-profile hyphenations against gold, from the recorded test split."""
    import sys

    sys.path.insert(0, str(repo_root))
    from scripts.hyphenator.hyphenator import Hyphenator  # noqa: E402

    hyphenator = Hyphenator(
        str(repo_root / "results/gpopt260828" / dataset / "final_patterns.pat"),
        hyphenation_mark="-",
        translate_file=str(repo_root / TRANSLATE_FILES[dataset]),
    )
    gold_path = repo_root / "results/gpopt260828" / dataset / "splits/data.test.wlh"
    words = [w.strip() for w in gold_path.read_text("utf-8").splitlines() if w.strip()]
    false_pos, missed, exact = [], [], []
    for word in words[:limit]:
        plain = word.replace("-", "")
        if not (8 <= len(plain) <= 12):
            continue
        got = hyphenator.hyphenate(plain)
        gold, guess = _breaks(word), _breaks(got)
        if guess - gold:
            false_pos.append((word, got))
        elif gold - guess:
            missed.append((word, got))
        else:
            exact.append((word, got))
    return {"false_pos": false_pos, "missed": missed, "exact": exact}


def fig_words(cases: List[Case], repo_root: Path, out_dir: Path, stats: dict) -> None:
    case = next(c for c in cases if c.dataset == QUALITATIVE)
    examples = hyphenation_examples(repo_root, QUALITATIVE)
    picks = (
        [("exact", *p) for p in examples["exact"][:4]]
        + [("missed", *p) for p in examples["missed"][:3]]
        + [("false", *p) for p in examples["false_pos"][:2]]
    )
    total = sum(len(v) for v in examples.values())

    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    columns = (0.015, 0.30, 0.585)
    for x, head in zip(columns, ("gold hyphenation", "reported profile", "counted as")):
        ax.text(x, 0.985, head, fontsize=8.2, weight="bold", va="top",
                color=GREEN if head == "reported profile" else INK)
    y = 0.875
    for kind, gold, got in picks:
        colour = {"exact": GREEN, "missed": MUTED, "false": WARM}[kind]
        ax.text(columns[0], y, gold, fontsize=9.2, va="top", family="monospace")
        ax.text(columns[1], y, got, fontsize=9.2, va="top", family="monospace", color=colour)
        gold_set, guess = _breaks(gold), _breaks(got)
        note = {
            "exact": "every break point, and no wrong one",
            "missed": f"{len(gold_set - guess)} break point(s) not offered",
            "false": f"{len(guess - gold_set)} wrong hyphen(s) inserted",
        }[kind]
        ax.text(columns[2], y, note, fontsize=7.8, va="top", color=colour)
        y -= 0.093
    ax.text(
        columns[0],
        0.02,
        f"First {total} words of 8\u201312 letters in the {case.label} held-out split: exact on "
        f"{100 * len(examples['exact']) / total:.0f}%, at least one break point not offered on "
        f"{100 * len(examples['missed']) / total:.0f}%, a wrong hyphen on "
        f"{100 * len(examples['false_pos']) / total:.0f}%.",
        fontsize=7.4,
        va="bottom",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out_dir, "hero_k_words")
    stats["qualitative"] = {
        "dataset": QUALITATIVE,
        "n_words": total,
        "exact_pct": round(100 * len(examples["exact"]) / total, 1),
        "missed_pct": round(100 * len(examples["missed"]) / total, 1),
        "false_pct": round(100 * len(examples["false_pos"]) / total, 1),
    }


# ---------------------------------------------------------------------------
# L: where the classical profiles sit in the searched space
# ---------------------------------------------------------------------------


def fig_profiles(cases: List[Case], out_dir: Path, stats: dict) -> None:
    weights = np.array([c.weight_ratios for c in cases])
    thresholds = np.array([c.thresholds for c in cases], dtype=float)
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, values, title, ylabel in (
        (axes[0], weights, "(a) weight ratio $\\mathit{bad\\_wt}/\\mathit{good\\_wt}$",
         "ratio (log scale)"),
        (axes[1], thresholds,
         "(b) threshold $\\mathit{thresh}$: both classical profiles use 1 everywhere",
         "threshold"),
    ):
        for level in range(4):
            jitter = rng.uniform(-0.16, 0.16, len(cases))
            ax.scatter(
                level + 1 + jitter,
                values[:, level],
                s=17,
                color=GREEN,
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
            ax.plot(
                [level + 0.72, level + 1.28],
                [np.median(values[:, level])] * 2,
                color=GREEN,
                lw=1.6,
                zorder=4,
            )
        for name, marker, colour, size in (
            ("cshyphen", "o", BLUE, 6.6),
            ("wortliste", "s", WARM, 3.9),
        ):
            spec = CLASSICAL[name]
            ys = [
                (bad / good) if values is weights else thresh
                for good, bad, thresh in spec
            ]
            ax.plot(
                range(1, 5),
                ys,
                marker=marker,
                ls=(0, (4, 2)),
                ms=size,
                mfc="white",
                mec=colour,
                mew=1.3,
                color=colour,
                lw=1.0,
                zorder=6 if name == "cshyphen" else 7,
            )
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f"level {i}" for i in range(1, 5)])
        ax.set_xlim(0.55, 4.45)
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 30])
    axes[0].set_yticklabels(["1/5", "1/2", "1", "2", "5", "10", "20", "30"])
    axes[0].set_ylim(0.15, 45)
    axes[1].set_ylim(-1, 45)
    axes[1].set_yticks([1, 10, 20, 30, 42])
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=GREEN, ms=4.6,
                   label="selected per-level profile (17)"),
            Line2D([], [], color=GREEN, lw=1.8, label="median over datasets"),
            Line2D([], [], marker="o", ls=(0, (4, 2)), mfc="white", mec=BLUE, mew=1.2, ms=5.0,
                   color=BLUE, label="cshyphen (hand-tuned)"),
            Line2D([], [], marker="s", ls=(0, (4, 2)), mfc="white", mec=WARM, mew=1.2, ms=5.0,
                   color=WARM, label="wortliste (hand-tuned)"),
        ],
        loc="lower center",
        ncols=4,
        fontsize=7.2,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, out_dir, "hero_l_profiles")
    stats["profile_space"] = {
        "median_weight_ratios": [round(float(v), 3) for v in np.median(weights, axis=0)],
        "median_thresholds": [float(v) for v in np.median(thresholds, axis=0)],
        "classical_thresholds_all_one": all(
            t == 1 for spec in CLASSICAL.values() for _, _, t in spec
        ),
        "datasets_with_threshold_above_one": {
            f"level_{i + 1}": int((thresholds[:, i] > 1).sum()) for i in range(4)
        },
        "datasets_with_level1_ratio_above_cshyphen": int((weights[:, 0] > 5).sum()),
        "datasets_with_level4_ratio_below_one": int((weights[:, 3] < 1).sum()),
    }


# ---------------------------------------------------------------------------
# Numbers quoted in the companion document
# ---------------------------------------------------------------------------


def diagnosis_stats(cases: List[Case], stats: dict) -> None:
    bt = np.array([float(c.test_base.trie) for c in cases])
    ot = np.array([float(c.test_opt.trie) for c in cases])
    be = np.array([c.test_base.err for c in cases])
    oe = np.array([c.test_opt.err for c in cases])

    decomposition = {}
    for name, base, opt in (("trie", bt, ot), ("error", be, oe)):
        mid = (np.log10(base) + np.log10(opt)) / 2.0
        disp = np.log10(opt) - np.log10(base)
        between = float(mid.var())
        within = float(np.mean((disp / 2.0) ** 2))
        decomposition[name] = {
            "between_dataset_var": round(between, 4),
            "method_var": round(within, 4),
            "between_share_pct": round(100 * between / (between + within), 1),
            "axis_decades": round(
                float(np.log10(max(base.max(), opt.max()) / min(base.min(), opt.min()))), 2
            ),
            "median_displacement_decades": round(float(abs(np.median(disp))), 2),
        }
    stats["variance_decomposition"] = decomposition

    per_kd = np.concatenate(
        [
            np.array([c.test_base.trie / c.lines * 1000 for c in cases]),
            np.array([c.test_opt.trie / c.lines * 1000 for c in cases]),
        ]
    )
    raw = np.concatenate([bt, ot])
    stats["size_normalization"] = {
        "raw_trie_decades": round(float(np.log10(raw.max() / raw.min())), 2),
        "trie_per_1k_words_decades": round(float(np.log10(per_kd.max() / per_kd.min())), 2),
    }

    per_dataset = {}
    for case in cases:
        front = attainment(case.history)
        per_dataset[case.label] = {
            "n_points": int(len(front)),
            "trie_sweep": round(float(front[:, 0].max() / front[:, 0].min()), 2),
            "error_sweep": round(float(front[:, 1].max() / front[:, 1].min()), 2),
            "selected_is_largest_trie_on_frontier": bool(
                abs(front[:, 0].max() - case.val_opt_trie) < 0.5
            ),
        }
    stats["frontiers"] = {
        "median_points": int(statistics.median(v["n_points"] for v in per_dataset.values())),
        "median_trie_sweep": round(
            statistics.median(v["trie_sweep"] for v in per_dataset.values()), 2
        ),
        "selected_at_large_trie_end": sum(
            1 for v in per_dataset.values() if v["selected_is_largest_trie_on_frontier"]
        ),
        "per_dataset": per_dataset,
    }

    halved = []
    for case in cases:
        front = attainment(case.history)
        candidates = front[front[:, 0] <= case.val_opt_trie / 2.0]
        if len(candidates):
            halved.append(float(candidates[np.argmax(candidates[:, 0])][1] / case.val_opt_err))
    stats["frontier_cost_of_halving"] = {
        "datasets_with_half_size_point": len(halved),
        "median_error_inflation": round(statistics.median(halved), 2),
    }

    stats["headline"] = {
        "median_trie_ratio": round(float(np.median(ot / bt)), 4),
        "median_error_ratio": round(float(np.median(oe / be)), 4),
        "median_error_gain": round(float(np.median(be / oe)), 3),
        "median_wrong_hyphen_ratio": round(float(np.median([c.bad_ratio for c in cases])), 4),
        "datasets_better_on_both_axes": int(
            sum(1 for c in cases if c.trie_ratio < 1 and c.err_ratio < 1)
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="results/hero_suggestion_figures")
    parser.add_argument("--copy-to", default=None, help="also copy every PDF here")
    parser.add_argument("--only", action="append", help="render only these keys (repeatable)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    style()
    cases = load_cases(repo_root)
    print(f"{len(cases)} datasets, {len(cases[0].history)} evaluations each")
    stats: dict = {}
    diagnosis_stats(cases, stats)

    builders = {
        "diag": lambda: fig_diag_axes(cases, out_dir, stats),
        "gain": lambda: fig_gain(cases, out_dir, stats),
        "slope": lambda: fig_slope(cases, out_dir, stats),
        "frontiers": lambda: fig_frontiers(cases, out_dir, stats),
        "twopanel": lambda: fig_twopanel(cases, out_dir, stats),
        "mechanism": lambda: fig_mechanism(cases, out_dir, stats),
        "stack": lambda: fig_hero_stack(cases, out_dir, stats),
        "readout": lambda: fig_readout(cases, out_dir, stats),
        "multiples": lambda: fig_multiples(cases, out_dir, stats),
        "lollipop": lambda: fig_lollipop(cases, out_dir, stats),
        "pr": lambda: fig_pr(cases, out_dir, stats),
        "space": lambda: fig_space(cases, out_dir, stats),
        "words": lambda: fig_words(cases, repo_root, out_dir, stats),
        "profiles": lambda: fig_profiles(cases, out_dir, stats),
    }
    for key in args.only or list(builders):
        builders[key]()

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True), "utf-8")
    print(f"stats -> {out_dir / 'stats.json'}")

    if args.copy_to:
        target = Path(args.copy_to)
        if not target.is_absolute():
            target = repo_root / target
        target.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(out_dir.glob("hero_*.pdf"))
        for pdf in pdfs:
            shutil.copy2(pdf, target / pdf.name)
        print(f"copied {len(pdfs)} PDFs -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
