#!/usr/bin/env python3
"""Candidate appendix visualizations for the per-level GP search.

The published appendix illustrates the optimizer with a *restricted*
five-parameter experiment, because 1D slices of the real eight-parameter
search were judged unreadable.  This script builds candidate replacements
directly from the recorded search histories: no patgen runs, no new
optimization, only replotting of `final_history.csv`.

Every surrogate fitted here mirrors the optimizer in `scripts/gp_optimizer.py`
(Matern nu=2.5 with per-dimension length scales, additive WhiteKernel,
`normalize_y`, two restarts, raw integer coordinates) so that a posterior
drawn in a figure is the posterior the search actually used.

Usage:
    uv run python -m scripts.make_gp_suggestion_figures
    uv run python -m scripts.make_gp_suggestion_figures --copy-to ../brain/overleaf/latex/pics/suggestions
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.inspection import permutation_importance

# --- search space ---------------------------------------------------------
# Mirrors per_level_search.WEIGHT_LABELS and the --min/--max-threshold
# defaults recorded in every run_config.json of the reported run.
WEIGHT_LABELS = ("1/5", "1/4", "1/3", "1/2") + tuple(str(v) for v in range(1, 31))
N_LEVELS = 4
LOW = np.array([0] * N_LEVELS + [1] * N_LEVELS, dtype=float)
HIGH = np.array([len(WEIGHT_LABELS) - 1] * N_LEVELS + [42] * N_LEVELS, dtype=float)
COORD_COLUMNS = [f"weight_code_{i}" for i in range(1, N_LEVELS + 1)] + [
    f"threshold_{i}" for i in range(1, N_LEVELS + 1)
]
COORD_LABELS = [f"$w_{i}$" for i in range(1, N_LEVELS + 1)] + [
    f"$t_{i}$" for i in range(1, N_LEVELS + 1)
]
SHORT_LABELS = ["w1", "w2", "w3", "w4", "t1", "t2", "t3", "t4"]

WEIGHT_TICKS = [0, 1, 2, 3, 4, 8, 13, 18, 23, 28, 33]
WEIGHT_TICKLABELS = ["1/5", "1/4", "1/3", "1/2", "1", "5", "10", "15", "20", "25", "30"]
SPARSE_WEIGHT_TICKS = [0, 4, 13, 23, 33]
SPARSE_WEIGHT_TICKLABELS = ["1/5", "1", "10", "20", "30"]
LENGTH_SCALE_BOUNDS = (1e-3, 1e3)

BATCH_SIZE = 5
UNIFORM_STD = 1.0 / np.sqrt(12.0)  # std of U(0,1); the no-concentration reference

INK = "#1f2933"
ACCENT = "#1f5fa8"
WARM = "#d1495b"
MUTED = "#9aa6b2"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 8.5,
            "axes.titlesize": 9,
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
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
        }
    )


# --- data -----------------------------------------------------------------
class Run:
    """One dataset's recorded search history."""

    def __init__(self, name: str, directory: Path):
        self.name = name
        self.dir = directory
        self.df = pd.read_csv(directory / "final_history.csv")
        self.X = self.df[COORD_COLUMNS].to_numpy(dtype=float)
        self.y = self.df["objective_score"].to_numpy(dtype=float)
        self.f17 = self.df["validation_f17"].to_numpy(dtype=float)
        self.trie = self.df["trie_nodes"].to_numpy(dtype=float)
        self.selected = json.loads((directory / "selected_profile.json").read_text())
        self.config = json.loads((directory / "run_config.json").read_text())

    @property
    def winner(self) -> int:
        return int(self.y.argmax())

    @property
    def selected_point(self) -> np.ndarray:
        codes = [WEIGHT_LABELS.index(w) for w in self.selected["weight_ratios"]]
        return np.array(codes + list(self.selected["thresholds"]), dtype=float)

    def unit(self, X: np.ndarray | None = None) -> np.ndarray:
        return ((self.X if X is None else X) - LOW) / (HIGH - LOW)

    def tie_set(self, tol: float = 1e-3) -> np.ndarray:
        return np.where(self.y >= self.y.max() - tol)[0]


def load_runs(run_dir: Path) -> Dict[str, Run]:
    runs: Dict[str, Run] = {}
    for history in sorted(run_dir.glob("*/*/final_history.csv")):
        directory = history.parent
        name = f"{directory.parent.name}/{directory.name}"
        runs[name] = Run(name, directory)
    if not runs:
        raise SystemExit(f"no final_history.csv under {run_dir}")
    return runs


def fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 42) -> GaussianProcessRegressor:
    """Fit the surrogate the optimizer itself used (scripts/gp_optimizer.py)."""
    kernel = Matern(
        nu=2.5, length_scale=[1.0] * X.shape[1], length_scale_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2
    )
    gp.fit(X, y)
    return gp


def weight_axis(ax, axis: str = "x", dense: bool = False) -> None:
    """Label a weight-code axis with the ratios the codes stand for.

    The four fractional codes sit on top of each other in a narrow panel, so
    only the wide panels ask for the dense labelling.
    """
    ticks, labels = (WEIGHT_TICKS, WEIGHT_TICKLABELS) if dense else (
        SPARSE_WEIGHT_TICKS, SPARSE_WEIGHT_TICKLABELS
    )
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    labeler = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    setter(ticks)
    labeler(labels)


def save(fig: plt.Figure, out: Path, name: str, index: List[str]) -> None:
    path = out / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    index.append(name)
    print(f"  wrote {path.name}")


# --- statistics used by both the figures and the write-up ------------------
def anytime_fractions(run: Run) -> np.ndarray:
    """Fraction of the run's total gain achieved after each evaluation.

    The origin is the best of the initial random batch, so the curve measures
    what the *model-driven* part of the search added, not the luck of the
    first draw.
    """
    best = np.maximum.accumulate(run.y)
    start = best[BATCH_SIZE - 1]
    span = best[-1] - start
    if span <= 0:
        return np.ones_like(best)
    return np.clip((best - start) / span, 0.0, 1.0)


def first_reaching(run: Run, fraction: float) -> int:
    frac = anytime_fractions(run)
    hits = np.where(frac >= fraction)[0]
    return int(hits[0]) + 1 if len(hits) else len(frac)


def concentration(unit_points: np.ndarray) -> np.ndarray:
    """1 = every point shares a coordinate value, 0 = spread like uniform."""
    return 1.0 - unit_points.std(axis=0) / UNIFORM_STD


def concentration_pvalues(
    run: Run, top_k: int, permutations: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Concentration of the best `top_k` points, and its permutation p-value.

    The null resamples `top_k` of the *same* evaluated points, so the test asks
    whether the good region is tighter than the search's own sampling, not
    tighter than a uniform grid.
    """
    unit = run.unit()
    top = unit[np.argsort(run.y)[-top_k:]]
    observed = concentration(top)
    null = np.empty((permutations, unit.shape[1]))
    for i in range(permutations):
        null[i] = concentration(unit[rng.choice(len(unit), top_k, replace=False)])
    return observed, (null >= observed).mean(axis=0)


def importance_shares(run: Run, seed: int = 0) -> np.ndarray:
    """Share of objective variation attributable to each coordinate."""
    forest = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    forest.fit(run.X, run.y)
    result = permutation_importance(
        forest, run.X, run.y, n_repeats=5, random_state=seed, n_jobs=-1
    )
    values = np.clip(result.importances_mean, 0.0, None)
    total = values.sum()
    return values / total if total > 0 else values


def snapshot(run: Run, n: int) -> GaussianProcessRegressor:
    return fit_gp(run.X[:n], run.y[:n])


def incumbent(run: Run, n: int) -> np.ndarray:
    return run.X[: n][run.y[:n].argmax()].copy()


# --- candidate figures ----------------------------------------------------
def fig_explore_plane(run: Run, out: Path, index: List[str]) -> None:
    """Where the search put its evaluations, and where the good ones landed."""
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    order = np.arange(1, len(run.y) + 1)
    top = np.argsort(run.y)[-15:]
    sel = run.selected_point
    scatter = None

    for ax, (i, j) in zip(axes[:2], [(0, 2), (4, 6)]):
        scatter = ax.scatter(
            run.X[:, i], run.X[:, j], c=order, cmap="viridis", s=22,
            edgecolor="white", linewidth=0.3, zorder=3,
        )
        ax.scatter(
            run.X[top, i], run.X[top, j], s=95, facecolor="none",
            edgecolor=WARM, linewidth=1.1, zorder=4, label="best 15",
        )
        ax.scatter(sel[i], sel[j], marker="*", s=190, color=WARM,
                   edgecolor="white", linewidth=0.6, zorder=5, label="selected")
        ax.set_xlabel(COORD_LABELS[i])
        ax.set_ylabel(COORD_LABELS[j])
        if i < N_LEVELS:
            weight_axis(ax, "x")
        if j < N_LEVELS:
            weight_axis(ax, "y")
    axes[0].set_title("level-1 vs level-3 weight ratio")
    axes[1].set_title("level-1 vs level-3 threshold")
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=WARM, mew=1.1,
               ms=7, label="best 15"),
        Line2D([], [], marker="*", ls="none", color=WARM, ms=11, label="selected"),
    ]
    axes[1].legend(handles=handles, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, -0.22))
    fig.colorbar(scatter, ax=axes[1], label="evaluation", pad=0.02)

    ax = axes[2]
    grid = np.linspace(0, 1, 200)
    unit = run.unit()
    for coord, colour, label in [(0, ACCENT, "$w_1$"), (5, MUTED, "$t_2$")]:
        ax.plot(grid, [(unit[:, coord] <= g).mean() for g in grid],
                color=colour, lw=1.4, ls="--", label=f"{label}: all 153")
        ax.plot(grid, [(unit[top, coord] <= g).mean() for g in grid],
                color=colour, lw=2.0, label=f"{label}: best 15")
    ax.plot(grid, grid, color=INK, lw=0.8, ls=":", label="uniform")
    ax.set_xlabel("coordinate, normalised to its range")
    ax.set_ylabel("empirical CDF")
    ax.set_title("the good region is tight in $w_1$, not in $t_2$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "suggestion_explore_plane", index)


def fig_anytime(runs: Dict[str, Run], stats: dict, out: Path, index: List[str]) -> None:
    """How quickly each search reached its own final answer."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    curves = np.vstack([anytime_fractions(r) for r in runs.values()])
    evals = np.arange(1, curves.shape[1] + 1)
    for row in curves:
        ax.plot(evals, row, color=MUTED, lw=0.8, alpha=0.8)
    ax.plot(evals, np.median(curves, axis=0), color=ACCENT, lw=2.4,
            label="median over 17 datasets")
    winners = np.array([r.winner + 1 for r in runs.values()])
    ax.scatter(winners, np.ones_like(winners) * 1.035, marker="v", s=26,
               color=WARM, clip_on=False,
               label="evaluation that produced the selected profile")
    for value, height, text in [
        (stats["median_eval_99pct"], 0.46,
         f"99% of the gain by eval {stats['median_eval_99pct']}"),
        (stats["median_winner_eval"], 0.32,
         f"selected profile at eval {stats['median_winner_eval']}"),
    ]:
        ax.axvline(value, color=INK, ls="--", lw=0.9)
        ax.annotate(text, xy=(value, height), xytext=(-6, 0),
                    textcoords="offset points", fontsize=7.5, va="center",
                    ha="right", bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="none", alpha=0.85))
    ax.axvspan(150, 153, color=WARM, alpha=0.10)
    ax.set_xlabel("patgen evaluation (30 iterations $\\times$ batch 5, then 3 exploitation)")
    ax.set_ylabel("fraction of the run's total gain\nover its initial random batch")
    ax.set_xlim(1, 153)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "suggestion_anytime", index)


def fig_concentration(stats: dict, out: Path, index: List[str]) -> None:
    """Which coordinates the good region actually pins down, across datasets."""
    excess = np.array(stats["excess_concentration"])
    pvals = np.array(stats["concentration_pvalues"])
    names = stats["datasets"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    limit = float(np.abs(excess).max())
    image = ax.imshow(excess, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(excess.shape[0]):
        for j in range(excess.shape[1]):
            if pvals[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9,
                        color=INK, fontweight="bold")
    ax.set_xticks(range(len(COORD_LABELS)))
    ax.set_xticklabels(COORD_LABELS)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("how much tighter the best 15 profiles are than the search's own sampling\n"
                 "(* = $p<0.05$, permutation test on the same evaluations)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="excess concentration", pad=0.02)
    fig.tight_layout()
    save(fig, out, "suggestion_concentration", index)


def fig_two_modes(run: Run, out: Path, index: List[str]) -> None:
    """Two different profiles that tie on the objective at very different sizes."""
    tie = run.tie_set()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    ax.scatter(run.X[:, 0], run.y, s=20, color=MUTED, edgecolor="white",
               linewidth=0.3, label="evaluation")
    ax.scatter(run.X[tie, 0], run.y[tie], s=70, color=WARM, edgecolor="white",
               linewidth=0.5, zorder=4, label="within 0.001 of the best")
    ax.set_xlabel("level-1 weight ratio $w_1$")
    ax.set_ylabel("validation objective")
    weight_axis(ax, "x")
    ax.set_title("two separated groups reach the same objective")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.scatter(run.trie, run.f17, s=20, color=MUTED, edgecolor="white", linewidth=0.3)
    ax.scatter(run.trie[tie], run.f17[tie], s=70, color=WARM, edgecolor="white",
               linewidth=0.5, zorder=4)
    ax.set_ylim(run.f17.min() - 0.010, run.f17.max() + 0.006)
    for idx in (tie[run.trie[tie].argmin()], tie[run.trie[tie].argmax()]):
        ax.annotate(f"{int(run.trie[idx])} nodes\n$w_1$={WEIGHT_LABELS[int(run.X[idx, 0])]}",
                    xy=(run.trie[idx], run.f17[idx]), xytext=(0, -34),
                    textcoords="offset points", fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, alpha=0.7))
    ax.set_xscale("log")
    ax.set_xlabel("trie nodes (log scale)")
    ax.set_ylabel("validation $F_{1/7}$")
    ax.set_title("the tie spans a %.1f$\\times$ size difference"
                 % (run.trie[tie].max() / run.trie[tie].min()))
    fig.tight_layout()
    save(fig, out, "suggestion_two_modes", index)


def fig_tie_sets(runs: Dict[str, Run], out: Path, index: List[str]) -> None:
    """Size headroom hiding inside each dataset's set of tied profiles."""
    rows = []
    for name, run in runs.items():
        tie = run.tie_set()
        sel = run.trie[run.winner]
        rows.append((name, run.trie[tie].min() / sel, run.trie[tie].max() / sel, len(tie)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (name, low, high, count) in enumerate(rows):
        ax.plot([low, high], [i, i], color=MUTED, lw=2.0, solid_capstyle="round")
        ax.scatter([low], [i], s=26, color=ACCENT, zorder=3)
        ax.scatter([1.0], [i], marker="*", s=90, color=WARM, zorder=4)
        ax.annotate(f"n={count}", xy=(high, i), xytext=(5, 0),
                    textcoords="offset points", fontsize=6.5, va="center")
    ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
    ax.set_xticklabels(["0.5$\\times$", "0.7$\\times$", "1$\\times$",
                        "1.4$\\times$", "2$\\times$"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("trie size relative to the selected profile (log scale)")
    ax.set_title("profiles tied within 0.001 of the selected objective\n"
                 "star = selected, dot = smallest tied profile")
    fig.tight_layout()
    save(fig, out, "suggestion_tie_sets", index)


def fig_slice_evolution(run: Run, out: Path, index: List[str], coord: int = 0,
                        snapshots: Sequence[int] = (25, 50, 100, 150)) -> None:
    """The posterior along the dominant coordinate, tightening with evidence."""
    fig, axes = plt.subplots(1, len(snapshots), figsize=(9.6, 2.7), sharey=True)
    grid = np.linspace(LOW[coord], HIGH[coord], 160)
    for ax, n in zip(axes, snapshots):
        gp = snapshot(run, n)
        centre = incumbent(run, n)
        query = np.tile(centre, (len(grid), 1))
        query[:, coord] = grid
        mean, sd = gp.predict(query, return_std=True)
        ax.fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=ACCENT, alpha=0.20, lw=0)
        ax.plot(grid, mean, color=ACCENT, lw=1.8)
        ax.scatter(run.X[:n, coord], run.y[:n], s=12, color=WARM, alpha=0.55,
                   edgecolor="none", zorder=3)
        ax.axvline(centre[coord], color=INK, ls="--", lw=0.9)
        ax.set_title(f"after {n} evaluations")
        ax.set_xlabel(COORD_LABELS[coord])
        weight_axis(ax, "x")
    axes[0].set_ylabel("validation objective")
    fig.suptitle("posterior along $w_1$, other coordinates held at the incumbent best", y=1.04)
    fig.tight_layout()
    save(fig, out, "suggestion_slice_evolution", index)


def fig_surface(run: Run, out: Path, index: List[str],
                pair: Tuple[int, int] = (0, 2), n: int = 150) -> None:
    """Mean and uncertainty over the two coordinates that carry the objective."""
    i, j = pair
    gp = snapshot(run, n)
    centre = incumbent(run, n)
    xs = np.linspace(LOW[i], HIGH[i], 70)
    ys = np.linspace(LOW[j], HIGH[j], 70)
    xx, yy = np.meshgrid(xs, ys)
    query = np.tile(centre, (xx.size, 1))
    query[:, i] = xx.ravel()
    query[:, j] = yy.ravel()
    mean, sd = gp.predict(query, return_std=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for ax, field, title, cmap in [
        (axes[0], mean.reshape(xx.shape), "posterior mean", "viridis"),
        (axes[1], sd.reshape(xx.shape), "posterior uncertainty", "magma"),
    ]:
        image = ax.pcolormesh(xx, yy, field, cmap=cmap, shading="auto")
        ax.contour(xx, yy, field, levels=6, colors="white", linewidths=0.4, alpha=0.6)
        ax.scatter(run.X[:n, i], run.X[:n, j], s=10, color="white", alpha=0.7,
                   edgecolor=INK, linewidth=0.2, zorder=3)
        ax.scatter(centre[i], centre[j], marker="*", s=170, color=WARM,
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_xlabel(COORD_LABELS[i])
        ax.set_ylabel(COORD_LABELS[j])
        ax.set_title(title)
        ax.grid(False)
        weight_axis(ax, "x")
        weight_axis(ax, "y")
        fig.colorbar(image, ax=ax, pad=0.02)
    fig.suptitle("after %d evaluations, other coordinates at the incumbent best" % n, y=1.02)
    fig.tight_layout()
    save(fig, out, "suggestion_surface", index)


def fig_surrogate_blindness(run: Run, out: Path, index: List[str], coord: int = 4,
                            n: int = 150, top_k: int = 15,
                            permutations: int = 2000, seed: int = 42) -> None:
    """What the surrogate learned about a threshold, against what the data shows.

    The left panel is the honest version of a one-dimensional threshold plot:
    the fitted posterior is flat, because the marginal likelihood pushed this
    coordinate's length scale to its upper bound.  The right panel is the same
    coordinate measured directly from the evaluations, where the best profiles
    clearly avoid part of the range.
    """
    gp = snapshot(run, n)
    centre = incumbent(run, n)
    grid = np.linspace(LOW[coord], HIGH[coord], 160)
    query = np.tile(centre, (len(grid), 1))
    query[:, coord] = grid
    mean, sd = gp.predict(query, return_std=True)
    scales = np.asarray(gp.kernel_.k1.length_scale)

    observed, pvals = concentration_pvalues(
        run, top_k, permutations, np.random.default_rng(seed)
    )
    top = np.argsort(run.y)[-top_k:]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    ax = axes[0]
    ax.fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=ACCENT, alpha=0.20, lw=0)
    ax.plot(grid, mean, color=ACCENT, lw=1.8)
    ax.scatter(run.X[:n, coord], run.y[:n], s=12, color=WARM, alpha=0.5,
               edgecolor="none", zorder=3)
    ax.set_xlabel(COORD_LABELS[coord])
    ax.set_ylabel("validation objective")
    ax.set_title(f"fitted posterior: flat\n(length scale {scales[coord]:.0f}, "
                 f"bound {LENGTH_SCALE_BOUNDS[1]:.0f})")

    ax = axes[1]
    bins = np.linspace(LOW[coord], HIGH[coord], 9)
    ax.hist(run.X[:, coord], bins=bins, color=MUTED, alpha=0.85,
            density=True, label="all %d evaluations" % len(run.y))
    ax.hist(run.X[top, coord], bins=bins, histtype="step", color=WARM, lw=1.8,
            density=True, label=f"best {top_k}")
    ax.set_xlabel(COORD_LABELS[coord])
    ax.set_ylabel("density")
    ax.set_title("measured directly: the best profiles\n"
                 f"avoid low {COORD_LABELS[coord]} ($p={pvals[coord]:.3f}$)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "suggestion_surrogate_blindness", index)


def fig_length_scales(runs: Dict[str, Run], out: Path, index: List[str],
                      shown: Sequence[str] = ("cssk/cshyphen", "ru/wiktionary"),
                      checkpoints: Sequence[int] = tuple(range(10, 151, 5))) -> dict:
    """What the surrogate believes each coordinate is worth, as evidence arrives.

    A length scale at the upper bound means the coordinate has been switched
    off; one at the lower bound means the fit has collapsed and the posterior
    carries no information away from the observations.
    """
    traces: Dict[str, Dict[str, List[float]]] = {}
    fig, axes = plt.subplots(1, len(shown), figsize=(8.6, 3.3), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), shown):
        run = runs[name]
        per_coord = {label: [] for label in SHORT_LABELS}
        for n in checkpoints:
            scales = np.asarray(snapshot(run, n).kernel_.k1.length_scale)
            for label, value in zip(SHORT_LABELS, scales):
                per_coord[label].append(float(value))
        traces[name] = per_coord
        for label, values in per_coord.items():
            is_weight = label.startswith("w")
            ax.plot(checkpoints, values, lw=2.0 if is_weight else 1.3,
                    ls="-" if is_weight else "--",
                    color=ACCENT if is_weight else WARM,
                    alpha=0.9 if is_weight else 0.75)
            ax.annotate(label, xy=(checkpoints[-1], values[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, va="center",
                        color=ACCENT if is_weight else WARM)
        ax.axhline(LENGTH_SCALE_BOUNDS[1], color=INK, lw=0.9, ls=":")
        ax.axhline(LENGTH_SCALE_BOUNDS[0], color=INK, lw=0.9, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 6e3)
        ax.set_xlim(checkpoints[0], checkpoints[-1] + 12)
        ax.set_xlabel("evaluations used to fit the surrogate")
        ax.set_title(name)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("fitted length scale per coordinate")
    first.annotate("upper bound: coordinate switched off",
                   xy=(checkpoints[0], 1.3e3), fontsize=7, va="bottom")
    first.annotate("lower bound: fit collapsed",
                   xy=(checkpoints[0], 2.6e-4), fontsize=7, va="bottom")
    fig.suptitle("solid: weight ratios, dashed: thresholds", y=1.02)
    fig.tight_layout()
    save(fig, out, "suggestion_length_scales", index)
    return traces


def fig_uncertainty(runs: Dict[str, Run], out: Path, index: List[str],
                    checkpoints: Sequence[int] = (10, 20, 30, 50, 75, 100, 125, 150),
                    n_probe: int = 1200, seed: int = 7) -> Dict[str, List[float]]:
    """When the surrogate stopped being surprised."""
    rng = np.random.default_rng(seed)
    probe = LOW + rng.random((n_probe, len(LOW))) * (HIGH - LOW)
    traces: Dict[str, List[float]] = {}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for name, run in runs.items():
        values = []
        for n in checkpoints:
            gp = snapshot(run, n)
            values.append(float(gp.predict(probe, return_std=True)[1].mean()))
        relative = np.array(values) / values[0]
        traces[name] = relative.tolist()
        ax.plot(checkpoints, relative, color=MUTED, lw=0.9, alpha=0.85)
    median = np.median(np.vstack(list(traces.values())), axis=0)
    ax.plot(checkpoints, median, color=ACCENT, lw=2.4, label="median over 17 datasets")
    ax.set_xlabel("evaluations used to fit the surrogate")
    ax.set_ylabel("mean posterior uncertainty over\nthe whole search box (relative)")
    ax.set_title("the surrogate's uncertainty collapses early, then plateaus")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "suggestion_uncertainty", index)
    return traces


def fig_baselines(hpo_dir: Path, out: Path, index: List[str],
                  datasets: Sequence[str] = ("cssk_cshyphen", "de_wortliste",
                                             "nl_wiktionary", "th_orchid")) -> None:
    """Model-driven search against random search and TPE, same budget.

    Absolute objectives differ in the fourth decimal, so the panels show
    distance to the best value any of the three methods reached: a log axis
    where lower is better and the curves are comparable across datasets.
    """
    methods = [("gp", ACCENT, "GP"), ("tpe", "#5c8001", "TPE"), ("random", MUTED, "random")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.6, 2.8), sharex=True)
    for ax, dataset in zip(axes, datasets):
        frames = {}
        for method, _, _ in methods:
            path = hpo_dir / f"{dataset}_{method}_history.csv"
            if path.is_file():
                frames[method] = pd.read_csv(path)
        if not frames:
            continue
        target = max(float(f["best_score_so_far"].max()) for f in frames.values())
        floor = min(float(f["best_score_so_far"].min()) for f in frames.values())
        span = target - floor
        for method, colour, label in methods:
            frame = frames.get(method)
            if frame is None:
                continue
            regret = (target - frame["best_score_so_far"]) / span
            ax.plot(frame["eval"], np.maximum(regret, 1e-4), color=colour, lw=1.6,
                    label=label)
        ax.set_yscale("log")
        ax.set_title(dataset.replace("_", "/"), fontsize=8.5)
        ax.set_xlabel("evaluation")
    axes[0].set_ylabel("distance to the best value\nany method reached (relative)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("restricted five-parameter comparison at equal budget; lower is better",
                 y=1.04)
    fig.tight_layout()
    save(fig, out, "suggestion_baselines", index)


def fig_search_cloud(run: Run, out: Path, index: List[str]) -> None:
    """The whole search drawn in the accuracy/size plane of the main figure."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    order = np.arange(1, len(run.y) + 1)
    scatter = ax.scatter(run.trie, run.f17, c=order, cmap="viridis", s=26,
                         edgecolor="white", linewidth=0.3, zorder=3)
    front = []
    for idx in np.argsort(run.trie):
        if not front or run.f17[idx] > run.f17[front[-1]]:
            front.append(idx)
    ax.step(run.trie[front], run.f17[front], where="post", color=WARM, lw=1.3,
            alpha=0.9, label="attainable frontier")
    ax.scatter(run.trie[run.winner], run.f17[run.winner], marker="*", s=210,
               color=WARM, edgecolor="white", linewidth=0.6, zorder=5, label="selected")
    ax.set_xscale("log")
    ax.set_ylim(np.percentile(run.f17, 5) - 0.005, run.f17.max() + 0.004)
    ax.set_xlabel("trie nodes (log scale)")
    ax.set_ylabel("validation $F_{1/7}$")
    ax.set_title("every evaluation of one search, in the plane of Figure 1")
    ax.legend(loc="lower right")
    fig.colorbar(scatter, ax=ax, label="evaluation", pad=0.02)
    fig.tight_layout()
    save(fig, out, "suggestion_search_cloud", index)


# --- driver ---------------------------------------------------------------
def collect_stats(runs: Dict[str, Run], top_k: int, permutations: int,
                  seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = list(runs)
    excess, pvalues, shares, rows = [], [], [], []
    for name in names:
        run = runs[name]
        observed, pvals = concentration_pvalues(run, top_k, permutations, rng)
        excess.append((observed - concentration(run.unit())).tolist())
        pvalues.append(pvals.tolist())
        shares.append(importance_shares(run).tolist())
        tie = run.tie_set()
        selected_trie = float(run.trie[run.winner])
        rows.append(
            {
                "dataset": name,
                "winner_eval": run.winner + 1,
                "eval_99pct": first_reaching(run, 0.99),
                "eval_999pct": first_reaching(run, 0.999),
                "tied_profiles": int(len(tie)),
                "selected_trie": selected_trie,
                "smallest_tied_trie": float(run.trie[tie].min()),
                "largest_tied_trie": float(run.trie[tie].max()),
                "tie_span_ratio": float(run.trie[tie].max() / run.trie[tie].min()),
                "shrink_available": float(selected_trie / run.trie[tie].min()),
                "f17_cost_of_shrink": float(
                    run.f17[run.winner] - run.f17[tie][run.trie[tie].argmin()]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    shares_array = np.array(shares)
    final_scales = np.array(
        [np.asarray(snapshot(runs[name], 150).kernel_.k1.length_scale) for name in names]
    )
    ignored = final_scales >= 0.9 * LENGTH_SCALE_BOUNDS[1]
    return {
        "datasets": names,
        "n_evaluations": int(len(next(iter(runs.values())).y)),
        "top_k": top_k,
        "permutations": permutations,
        "per_dataset": rows,
        "excess_concentration": excess,
        "concentration_pvalues": pvalues,
        "importance_shares": shares,
        "median_winner_eval": int(frame.winner_eval.median()),
        "median_eval_99pct": int(frame.eval_99pct.median()),
        "median_eval_999pct": int(frame.eval_999pct.median()),
        "winner_after_eval_100": int((frame.winner_eval > 100).sum()),
        "mean_importance": dict(zip(SHORT_LABELS, shares_array.mean(axis=0).round(4).tolist())),
        "median_threshold_importance": float(
            np.median(shares_array[:, N_LEVELS:].sum(axis=1)).round(4)
        ),
        "significant_concentration_counts": dict(
            zip(SHORT_LABELS, (np.array(pvalues) < 0.05).sum(axis=0).tolist())
        ),
        "datasets_with_shrink_1_3x": int((frame.shrink_available >= 1.3).sum()),
        "max_shrink_available": float(frame.shrink_available.max()),
        "max_shrink_dataset": frame.loc[frame.shrink_available.idxmax(), "dataset"],
        "max_f17_cost_of_shrink": float(frame.f17_cost_of_shrink.max()),
        "final_length_scales": final_scales.round(2).tolist(),
        "ignored_coordinate_counts": dict(
            zip(SHORT_LABELS, ignored.sum(axis=0).tolist())
        ),
        "datasets_ignoring_a_threshold": int(ignored[:, N_LEVELS:].any(axis=1).sum()),
        "median_w1_length_scale": float(np.median(final_scales[:, 0]).round(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/gpopt260828"))
    parser.add_argument("--hpo-dir", type=Path,
                        default=Path("results/hpo_representative_150"))
    parser.add_argument("--out", type=Path, default=Path("results/gp_suggestion_figures"))
    parser.add_argument("--copy-to", type=Path, default=None,
                        help="also copy every rendered figure into this directory")
    parser.add_argument("--focus", default="cssk/cshyphen",
                        help="dataset used for the single-run figures")
    parser.add_argument("--multimodal", default="el/wiktionary",
                        help="dataset used for the tied-profiles figure")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    style()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_dir)
    print(f"loaded {len(runs)} histories from {args.run_dir}")

    stats = collect_stats(runs, args.top_k, args.permutations, args.seed)
    index: List[str] = []

    fig_anytime(runs, stats, args.out, index)
    fig_explore_plane(runs[args.focus], args.out, index)
    fig_concentration(stats, args.out, index)
    fig_two_modes(runs[args.multimodal], args.out, index)
    fig_tie_sets(runs, args.out, index)
    fig_slice_evolution(runs[args.focus], args.out, index)
    fig_surface(runs[args.focus], args.out, index)
    fig_surrogate_blindness(runs[args.focus], args.out, index,
                            permutations=args.permutations, top_k=args.top_k,
                            seed=args.seed)
    fig_search_cloud(runs[args.focus], args.out, index)
    stats["length_scale_traces"] = fig_length_scales(runs, args.out, index)
    stats["uncertainty_traces"] = fig_uncertainty(runs, args.out, index)
    if args.hpo_dir.is_dir():
        fig_baselines(args.hpo_dir, args.out, index)

    stats["figures"] = index
    stats["focus_dataset"] = args.focus
    stats["multimodal_dataset"] = args.multimodal
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"wrote {args.out / 'stats.json'}")

    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for name in index:
            shutil.copy2(args.out / f"{name}.pdf", args.copy_to / f"{name}.pdf")
        print(f"copied {len(index)} figures to {args.copy_to}")


if __name__ == "__main__":
    main()
