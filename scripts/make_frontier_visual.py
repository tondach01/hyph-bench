#!/usr/bin/env python3
"""Accuracy-compactness frontier figures: published baseline -> optimized profile.

One arrow per dataset. Tail = currently published best hand-tuned baseline,
head = profile selected by a search run. x = pattern trie size (log scale),
y = held-out F_{1/7}.

Label placement is *solved*, never hand-tuned. `solve_labels` guarantees that every
label, at a visually uniform standoff from its own arrowhead:

  * does not overlap any arrow shaft or arrowhead (capsule geometry, not endpoints),
  * does not overlap any other label, baseline marker, legend, or note,
  * stays inside the axes,
  * and is strictly closer to its own arrow than to any other arrow.

That last constraint is what stops a label from appearing to annotate a neighbouring
arrow when two arrowheads land near each other. `audit_layout` re-checks all of the
above after rendering; `--check` turns any residual violation into a nonzero exit.

Usage:
    uv run python -m scripts.make_frontier_visual                 # every known run
    uv run python -m scripts.make_frontier_visual --run gpopt8
    uv run python -m scripts.make_frontier_visual --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# --------------------------------------------------------------------------------------
# Display names
# --------------------------------------------------------------------------------------

# Overrides reproducing the manuscript's short names exactly. Unknown datasets fall back
# to `derive_short_names`, so the figure also works for result sets not listed here.
SHORT_NAME_OVERRIDES: Dict[str, str] = {
    "cs/cshyphen_cstenten": "cs/ctt",
    "cs/cshyphen_ujc": "cs/ujc",
}


def derive_short_names(datasets: Sequence[str]) -> Dict[str, str]:
    """Shortest unambiguous label per dataset.

    `<lang>/wiktionary` and languages contributing a single dataset collapse to the bare
    language code; anything else keeps a disambiguating source suffix.
    """
    per_lang: Dict[str, List[str]] = {}
    for dataset in datasets:
        lang = dataset.split("/", 1)[0]
        per_lang.setdefault(lang, []).append(dataset)

    names: Dict[str, str] = {}
    for dataset in datasets:
        lang, _, source = dataset.partition("/")
        if dataset in SHORT_NAME_OVERRIDES:
            names[dataset] = SHORT_NAME_OVERRIDES[dataset]
        elif len(per_lang[lang]) == 1 or source == "wiktionary":
            names[dataset] = lang
        else:
            names[dataset] = f"{lang}/{source}"

    # Collapsing must stay injective; fall back to the full dataset id on any clash.
    seen: Dict[str, str] = {}
    for dataset, name in names.items():
        if name in seen:
            names[dataset] = dataset
            names[seen[name]] = seen[name]
        else:
            seen[name] = dataset
    return names


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One dataset: published baseline point and optimized point."""

    dataset: str
    name: str
    base_trie: int
    base_f17: float
    opt_trie: int
    opt_f17: float

    @property
    def smaller_trie(self) -> bool:
        return self.opt_trie < self.base_trie

    @property
    def more_accurate(self) -> bool:
        return self.opt_f17 > self.base_f17


@dataclass(frozen=True)
class RunSpec:
    """A frontier figure to produce."""

    key: str
    display: str
    # Self-contained bootstrap analysis (nested or flat schema), if the run has one.
    analysis: Optional[str] = None
    # Otherwise: per-dataset selected_profile.json under this directory ...
    profiles: Optional[str] = None
    # ... paired with baselines from this flat bootstrap analysis.
    baselines: Optional[str] = None
    note: Optional[str] = None


KNOWN_RUNS: Tuple[RunSpec, ...] = (
    RunSpec(
        key="gpopt4",
        display="GPopt4",
        analysis="results/paper2_revision_analysis_currentci/bootstrap_ci.json",
        note="published camera-ready selection",
    ),
    RunSpec(
        # The reported run. Its figure is the one that can reach the manuscript, so
        # the in-plot label must be the manuscript's method name, never the dated
        # internal identifier: the submission bundler rejects a figure that renders
        # `gpopt\d{6,8}`, and the identifier is repository-internal by policy.
        key="gpopt260828",
        display="Per-level GP search",
        analysis="results/gpopt260828_analysis/bootstrap_ci.json",
    ),
    RunSpec(
        key="gpopt8",
        display="GPopt8",
        # Directory is spelled `gpopt8` on disk; the run itself is GPopt8.
        profiles="results/gpopt8",
        baselines="results/paper2_revision_analysis_currentci/bootstrap_ci.json",
        note="best-validation-objective export; VC-0.005 post-selection not applied",
    ),
)


def _baseline_point(entry: dict) -> Tuple[int, float]:
    """Baseline trie/F_1/7 from either bootstrap schema."""
    if "base_f17" in entry:
        return int(entry["base_trie"]), float(entry["base_f17"])
    hand = entry["hand_baselines"][entry["best_baseline"]]
    return int(hand["trie_nodes"]), float(hand["f17"])


def _optimized_point(entry: dict) -> Tuple[int, float]:
    """Optimized trie/F_1/7 from either bootstrap schema."""
    if "opt_f17" in entry:
        return int(entry["opt_trie"]), float(entry["opt_f17"])
    optimized = entry["optimized"]
    return int(optimized["trie_nodes"]), float(optimized["f17"])


def load_rows(repo_root: Path, spec: RunSpec) -> List[Row]:
    """Resolve a run to baseline/optimized point pairs."""
    if spec.analysis:
        path = repo_root / spec.analysis
        if not path.exists():
            raise FileNotFoundError(f"{spec.key}: missing analysis {path}")
        entries = json.loads(path.read_text(encoding="utf-8"))
        points = {}
        for entry in entries:
            base = _baseline_point(entry)
            opt = _optimized_point(entry)
            points[entry["dataset"]] = (base, opt)
    else:
        baseline_path = repo_root / spec.baselines
        profile_root = repo_root / spec.profiles
        if not baseline_path.exists():
            raise FileNotFoundError(f"{spec.key}: missing baselines {baseline_path}")
        if not profile_root.is_dir():
            raise FileNotFoundError(f"{spec.key}: missing profiles {profile_root}")
        baselines = {e["dataset"]: e for e in json.loads(baseline_path.read_text(encoding="utf-8"))}
        points = {}
        for profile_path in sorted(profile_root.glob("*/*/selected_profile.json")):
            dataset = f"{profile_path.parent.parent.name}/{profile_path.parent.name}"
            entry = baselines.get(dataset)
            if entry is None:
                continue
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            points[dataset] = (
                _baseline_point(entry),
                (int(profile["held_out_test"]["trie_nodes"]), float(profile["held_out_test_f17"])),
            )

    if not points:
        raise ValueError(f"{spec.key}: no datasets resolved")

    names = derive_short_names(sorted(points))
    return [
        Row(dataset, names[dataset], base[0], base[1], opt[0], opt[1])
        for dataset, (base, opt) in sorted(points.items())
    ]


# --------------------------------------------------------------------------------------
# Pixel geometry. Boxes are (x0, y0, x1, y1); capsules are (a, b, half_width).
# --------------------------------------------------------------------------------------

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]
Capsule = Tuple[Point, Point, float]


def box_from_center(cx: float, cy: float, w: float, h: float, pad: float = 0.0) -> Box:
    return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)


def box_overlap_depth(a: Box, b: Box) -> float:
    """Minimum translation distance separating two boxes (0 when disjoint)."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return min(dx, dy) if dx > 0.0 and dy > 0.0 else 0.0


def point_box_dist(p: Point, b: Box) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def point_seg_dist(p: Point, a: Point, b: Point) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    len2 = vx * vx + vy * vy
    if len2 <= 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


def seg_box_penetration(a: Point, b: Point, box: Box) -> float:
    """Length of segment a->b lying inside `box` (Liang-Barsky slab clipping)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, a[0] - box[0]),
        (dx, box[2] - a[0]),
        (-dy, a[1] - box[1]),
        (dy, box[3] - a[1]),
    ):
        if p == 0.0:
            if q < 0.0:
                return 0.0
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return 0.0
            t0 = max(t0, r)
        else:
            if r < t0:
                return 0.0
            t1 = min(t1, r)
    return max(0.0, t1 - t0) * math.hypot(dx, dy)


def seg_box_dist(a: Point, b: Point, box: Box) -> float:
    """Exact distance between a segment and an axis-aligned box."""
    if seg_box_penetration(a, b, box) > 0.0:
        return 0.0
    best = min(point_box_dist(a, box), point_box_dist(b, box))
    x0, y0, x1, y1 = box
    for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        best = min(best, point_seg_dist(corner, a, b))
    return best


def capsule_box_dist(cap: Capsule, box: Box) -> float:
    a, b, half_width = cap
    return max(0.0, seg_box_dist(a, b, box) - half_width)


def capsule_box_overlap(cap: Capsule, box: Box) -> float:
    """Penetration length of a capsule into a box (0 when clear)."""
    a, b, half_width = cap
    inflated = (
        box[0] - half_width,
        box[1] - half_width,
        box[2] + half_width,
        box[3] + half_width,
    )
    return seg_box_penetration(a, b, inflated)


def box_overflow(box: Box, bounds: Box) -> float:
    """How far `box` pokes outside `bounds`."""
    return (
        max(0.0, bounds[0] - box[0])
        + max(0.0, box[2] - bounds[2])
        + max(0.0, bounds[1] - box[1])
        + max(0.0, box[3] - bounds[3])
    )


# --------------------------------------------------------------------------------------
# Label placement solver
# --------------------------------------------------------------------------------------


@dataclass
class ArrowGeom:
    """Rendered arrow in pixel space.

    `anchor` is the data point the label is attached to; `tip_point` is where the
    arrowhead is actually drawn. They differ: FancyArrowPatch pulls the tip back by
    `shrinkB` plus the head geometry, so spacing labels off the data point would leave
    a visibly uneven gap. All geometry uses the measured tip.
    """

    anchor: Point
    tip_point: Point
    shaft: Capsule
    tip: Capsule

    def clearance(self, box: Box) -> float:
        return min(capsule_box_dist(self.shaft, box), capsule_box_dist(self.tip, box))

    def overlap(self, box: Box) -> float:
        return capsule_box_overlap(self.shaft, box) + capsule_box_overlap(self.tip, box)


def measure_arrow(
    patch: FancyArrowPatch, tail: Point, head: Point, px_per_point: float
) -> ArrowGeom:
    """Build an ArrowGeom from the patch's real display-space path.

    `shrinkA`/`shrinkB` and the arrowhead geometry mean the drawn arrow is materially
    shorter than tail->head, so the endpoints are measured rather than assumed. Falls
    back to the nominal endpoints if matplotlib stops exposing the display path.
    """
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    norm = math.hypot(dx, dy) or 1.0
    unit = (dx / norm, dy / norm)

    points: List[Point] = []
    try:
        result = patch._get_path_in_displaycoord()
        paths = result[0] if isinstance(result, tuple) else result
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        for path in paths:
            codes = path.codes
            for k, vertex in enumerate(path.vertices):
                if codes is not None and codes[k] == MplPath.CLOSEPOLY:
                    continue
                points.append((float(vertex[0]), float(vertex[1])))
    except Exception:
        points = []
    if not points:
        points = [tail, head]

    along = [(p[0] - tail[0]) * unit[0] + (p[1] - tail[1]) * unit[1] for p in points]
    t_start, t_tip = min(along), max(along)
    start = (tail[0] + unit[0] * t_start, tail[1] + unit[1] * t_start)
    tip_point = (tail[0] + unit[0] * t_tip, tail[1] + unit[1] * t_tip)

    head_len = min(0.9 * ARROW_MUTATION * px_per_point, max(t_tip - t_start, 1.0))
    half_width = 0.5 * ARROW_LINEWIDTH * px_per_point + 0.5
    head_half = half_width
    for point, t in zip(points, along):
        if t >= t_tip - head_len:
            perp = abs((point[0] - tail[0]) * -unit[1] + (point[1] - tail[1]) * unit[0])
            head_half = max(head_half, perp + 0.5)

    return ArrowGeom(
        anchor=head,
        tip_point=tip_point,
        shaft=(start, tip_point, half_width),
        tip=(
            (tip_point[0] - unit[0] * head_len, tip_point[1] - unit[1] * head_len),
            tip_point,
            head_half,
        ),
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_seg_dist(a: Point, b: Point, c: Point, d: Point) -> float:
    """Distance between two segments; 0 when they cross."""
    if (_orientation(c, d, a) > 0.0) != (_orientation(c, d, b) > 0.0) and (
        _orientation(a, b, c) > 0.0
    ) != (_orientation(a, b, d) > 0.0):
        return 0.0
    return min(
        point_seg_dist(a, c, d),
        point_seg_dist(b, c, d),
        point_seg_dist(c, a, b),
        point_seg_dist(d, a, b),
    )


@dataclass
class LabelGeom:
    """Text extent for one label, in pixels."""

    name: str
    width: float
    height: float


@dataclass
class Obstacles:
    """Everything a label must avoid besides the arrows and the other labels."""

    axes: Box
    markers: List[Tuple[Point, float]] = field(default_factory=list)
    reserved: List[Box] = field(default_factory=list)


# Penalty weights. Collision terms are lengths in pixels and share one hard weight so
# that no aesthetic preference can ever outbid a real overlap.
W_HARD = 100.0
W_OWNERSHIP = 100.0
W_MARGIN = 5.0
W_DIRECTION = 1.5
W_DISTANCE = 3.0

N_DIRECTIONS = 32
ESCAPE_STEPS = (0.0, 3.0, 7.0, 12.0, 20.0, 30.0, 44.0, 62.0, 84.0)
LABEL_PAD = 1.5
# A label must never come closer than this to an arrow that is not its own, even when
# the two never intersect: near-tangency reads as "glued to the wrong arrow".
MIN_FOREIGN_CLEARANCE = 4.0
# Ownership is decided between *arrowheads*, which is what the eye pairs a label with;
# a foreign shaft passing at a safe distance does not claim the label. Being nearest is
# a hard requirement; the extra breathing room on top of it is only a preference, so
# that a label in a tight pocket settles for a thin margin instead of fleeing its arrow.
OWNERSHIP_MARGIN = 4.0
# Beyond this gap proximity alone no longer pairs a label with its arrowhead, so the
# label earns a leader line. Leaders are a last resort: W_LEADER prices them above any
# reachable clean seat, and the leader's own crossings are charged at W_HARD.
MAX_OWN_GAP = 14.0
W_LEADER = 60.0
W_LEADER_CROSS = 25.0


def _standoff_center(arrow: ArrowGeom, w: float, h: float, u: Point, standoff: float) -> Point:
    """Centre a label along direction `u` so its gap to the own arrowhead equals `standoff`.

    Distance to the arrowhead capsule increases monotonically with travel along `u`, so a
    bisection converges. Solving for the gap (rather than a fixed centre offset) is what
    makes the visual spacing uniform in every direction.
    """
    origin = arrow.tip_point
    lo, hi = 0.0, 600.0
    for _ in range(28):
        mid = (lo + hi) / 2.0
        box = box_from_center(origin[0] + u[0] * mid, origin[1] + u[1] * mid, w, h, LABEL_PAD)
        if capsule_box_dist(arrow.tip, box) < standoff:
            lo = mid
        else:
            hi = mid
    return (origin[0] + u[0] * hi, origin[1] + u[1] * hi)


def _candidates(
    arrow: ArrowGeom, label: LabelGeom, standoff: float
) -> List[Tuple[Point, float, float]]:
    """(centre, angular deviation from preferred direction, extra distance) triples."""
    start = arrow.shaft[0]
    dx, dy = arrow.tip_point[0] - start[0], arrow.tip_point[1] - start[1]
    norm = math.hypot(dx, dy)
    # Preferred reading direction: beyond the arrowhead, continuing the arrow's travel.
    preferred = math.atan2(dy, dx) if norm > 1e-9 else math.pi / 2.0

    out: List[Tuple[Point, float, float]] = []
    for k in range(N_DIRECTIONS):
        theta = 2.0 * math.pi * k / N_DIRECTIONS
        u = (math.cos(theta), math.sin(theta))
        base = _standoff_center(arrow, label.width, label.height, u, standoff)
        deviation = abs(math.atan2(math.sin(theta - preferred), math.cos(theta - preferred)))
        for extra in ESCAPE_STEPS:
            out.append(((base[0] + u[0] * extra, base[1] + u[1] * extra), deviation, extra))
    return out


def leader_segment(
    box: Box, arrow: ArrowGeom
) -> Optional[Tuple[Point, Point]]:
    """Connector from the label box edge to the arrowhead, or None if none is needed."""
    if capsule_box_dist(arrow.tip, box) <= MAX_OWN_GAP:
        return None

    centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    head = arrow.tip_point
    dx, dy = head[0] - centre[0], head[1] - centre[1]
    span = math.hypot(dx, dy)
    if span <= 1e-9:
        return None
    u = (dx / span, dy / span)

    # Leave the label box, stop just shy of the arrowhead.
    half_w, half_h = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
    t_exit = min(
        half_w / abs(u[0]) if u[0] else math.inf,
        half_h / abs(u[1]) if u[1] else math.inf,
    )
    t_stop = span - arrow.tip[2]
    if t_stop <= t_exit + 1.0:
        return None
    return (
        (centre[0] + u[0] * t_exit, centre[1] + u[1] * t_exit),
        (centre[0] + u[0] * t_stop, centre[1] + u[1] * t_stop),
    )


def _leader_obstruction(
    leader: Tuple[Point, Point],
    index: int,
    arrows: Sequence[ArrowGeom],
    boxes: Sequence[Optional[Box]],
) -> Tuple[float, float]:
    """Leader obstruction as (arrow crossings, text penetration).

    A label deep inside a bundle of arrows cannot be reached without crossing one, so
    arrow crossings are merely minimised. Running a leader through another label's text
    is never acceptable, so that cost is reported separately and audited.
    """
    start, end = leader
    arrow_cost = 0.0
    text_cost = 0.0
    for other_index, arrow in enumerate(arrows):
        if other_index == index:
            continue
        shaft_a, shaft_b, half_width = arrow.shaft
        clearance = seg_seg_dist(start, end, shaft_a, shaft_b) - half_width
        if clearance < 1.0:
            arrow_cost += 1.0 - clearance
    for other_index, box in enumerate(boxes):
        if box is None or other_index == index:
            continue
        text_cost += seg_box_penetration(start, end, box)
    return arrow_cost, text_cost


def _score(
    index: int,
    centre: Point,
    deviation: float,
    extra: float,
    label: LabelGeom,
    arrows: Sequence[ArrowGeom],
    obstacles: Obstacles,
    boxes: Sequence[Optional[Box]],
    standoff: float,
) -> float:
    box = box_from_center(centre[0], centre[1], label.width, label.height, LABEL_PAD)
    own = arrows[index]

    penalty = W_HARD * box_overflow(box, obstacles.axes)

    for arrow in arrows:
        overlap = arrow.overlap(box)
        if overlap:
            penalty += W_HARD * overlap

    for centre_px, radius in obstacles.markers:
        intrusion = radius + LABEL_PAD - point_box_dist(centre_px, box)
        if intrusion > 0.0:
            penalty += W_HARD * intrusion

    for reserved in obstacles.reserved:
        penalty += W_HARD * box_overlap_depth(box, reserved)

    for other_index, other in enumerate(boxes):
        if other is None or other_index == index:
            continue
        penalty += W_HARD * box_overlap_depth(box, other)

    # Never sit flush against somebody else's arrow.
    for other_index, arrow in enumerate(arrows):
        if other_index == index:
            continue
        gap = arrow.clearance(box)
        if gap < MIN_FOREIGN_CLEARANCE:
            penalty += W_HARD * (MIN_FOREIGN_CLEARANCE - gap)

    # Ownership: this label's own arrowhead must be the nearest arrowhead.
    own_gap = capsule_box_dist(own.tip, box)
    for other_index, arrow in enumerate(arrows):
        if other_index == index:
            continue
        gap = capsule_box_dist(arrow.tip, box)
        if gap < own_gap:
            penalty += W_OWNERSHIP * (own_gap - gap)
        if gap < own_gap + OWNERSHIP_MARGIN:
            penalty += W_MARGIN * (own_gap + OWNERSHIP_MARGIN - gap)

    leader = leader_segment(box, own)
    if leader is not None:
        arrow_cost, text_cost = _leader_obstruction(leader, index, arrows, boxes)
        penalty += W_LEADER + W_LEADER_CROSS * arrow_cost + W_HARD * text_cost

    return penalty + W_DIRECTION * deviation + W_DISTANCE * extra


def solve_labels(
    arrows: Sequence[ArrowGeom],
    labels: Sequence[LabelGeom],
    obstacles: Obstacles,
    standoff: float,
    max_sweeps: int = 30,
) -> List[Point]:
    """Assign a label centre per arrow. Deterministic: greedy seed, then sweeps to a fixpoint."""
    candidates = [_candidates(a, l, standoff) for a, l in zip(arrows, labels)]
    boxes: List[Optional[Box]] = [None] * len(arrows)
    chosen: List[Point] = [a.tip_point for a in arrows]

    # Most constrained first: heads with the most nearby arrowheads get first pick.
    def crowding(i: int) -> Tuple[float, str]:
        tip = arrows[i].tip_point
        near = sum(
            1
            for j, other in enumerate(arrows)
            if j != i and math.dist(tip, other.tip_point) < 4.0 * standoff + labels[i].width
        )
        return (-near, labels[i].name)

    order = sorted(range(len(arrows)), key=crowding)

    for _ in range(max_sweeps):
        changed = False
        for i in order:
            best_centre, best_score = None, math.inf
            for centre, deviation, extra in candidates[i]:
                score = _score(
                    i, centre, deviation, extra, labels[i], arrows, obstacles, boxes, standoff
                )
                if score < best_score:
                    best_centre, best_score = centre, score
            assert best_centre is not None
            if boxes[i] is None or best_centre != chosen[i]:
                changed = True
            chosen[i] = best_centre
            boxes[i] = box_from_center(
                best_centre[0], best_centre[1], labels[i].width, labels[i].height, LABEL_PAD
            )
        if not changed:
            break

    return chosen


def audit_layout(
    arrows: Sequence[ArrowGeom],
    labels: Sequence[LabelGeom],
    boxes: Sequence[Box],
    obstacles: Obstacles,
) -> List[str]:
    """Re-check the layout against the *rendered* label extents.

    Taking real extents rather than the solver's predicted boxes means the audit
    validates what matplotlib actually drew, so any drift between the solver's text
    model and the renderer surfaces as a violation instead of hiding behind it.
    """
    problems: List[str] = []

    for i, box in enumerate(boxes):
        name = labels[i].name

        overflow = box_overflow(box, obstacles.axes)
        if overflow > 0.5:
            problems.append(f"{name}: {overflow:.1f}px outside axes")

        for j, arrow in enumerate(arrows):
            overlap = arrow.overlap(box)
            if overlap > 0.5:
                problems.append(f"{name}: overlaps {labels[j].name} arrow by {overlap:.1f}px")

        for centre_px, radius in obstacles.markers:
            intrusion = radius - point_box_dist(centre_px, box)
            if intrusion > 0.5:
                problems.append(f"{name}: covers a baseline marker by {intrusion:.1f}px")

        for reserved in obstacles.reserved:
            depth = box_overlap_depth(box, reserved)
            if depth > 0.5:
                problems.append(f"{name}: overlaps figure furniture by {depth:.1f}px")

        for j in range(i + 1, len(boxes)):
            depth = box_overlap_depth(box, boxes[j])
            if depth > 0.5:
                problems.append(f"{name}: overlaps label {labels[j].name} by {depth:.1f}px")

        own_gap = capsule_box_dist(arrows[i].tip, box)
        leader = leader_segment(box, arrows[i])
        if own_gap > MAX_OWN_GAP + 0.5 and leader is None:
            problems.append(
                f"{name}: {own_gap:.1f}px from its own arrowhead with no leader "
                f"(max {MAX_OWN_GAP:.1f}px)"
            )
        if leader is not None:
            _, text_cost = _leader_obstruction(leader, i, arrows, boxes)
            if text_cost > 0.5:
                problems.append(
                    f"{name}: leader line runs through other label text by {text_cost:.1f}px"
                )

        for j, arrow in enumerate(arrows):
            if j == i:
                continue
            clearance = arrow.clearance(box)
            if clearance + 0.5 < MIN_FOREIGN_CLEARANCE:
                problems.append(
                    f"{name}: only {clearance:.1f}px from the {labels[j].name} arrow "
                    f"(min {MIN_FOREIGN_CLEARANCE:.1f}px)"
                )
            tip_gap = capsule_box_dist(arrow.tip, box)
            if tip_gap + 0.5 < own_gap:
                problems.append(
                    f"{name}: nearer the {labels[j].name} arrowhead ({tip_gap:.1f}px) "
                    f"than its own ({own_gap:.1f}px)"
                )
    return problems


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

COLOR_WIN = "#236b43"  # more accurate and smaller trie
COLOR_BIGGER = "#1b3a6b"  # more accurate but bigger trie
COLOR_LOSS = "#a33a35"  # less accurate

ARROW_LINEWIDTH = 1.35
ARROW_MUTATION = 9.5
MARKER_SIZE = 4.2
STANDOFF_POINTS = 5.0
FIGURE_SIZE = (7.6, 4.7)
# Floor for the residual 1 - F_1/7 so a perfect score stays plottable on a log axis.
_MIN_ERROR = 1e-5


def setup_typography() -> None:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["svg.fonttype"] = "none"
    # ACL/IEEE reject Type 3 fonts.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _row_color(row: Row) -> str:
    if not row.more_accurate:
        return COLOR_LOSS
    return COLOR_WIN if row.smaller_trie else COLOR_BIGGER


# Nice F_1/7 gridlines for the log-residual-error axis, filtered to the data range.
F_GRID = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998)


def _axis_limits(rows: Sequence[Row], y_scale: str) -> Box:
    """Data-driven limits with headroom for labels, the legend, and the note.

    On the "error" scale the y value plotted is the residual 1 - F_1/7 on a log axis.
    As the optimizer improves, endpoints bunch against F = 1 and a linear axis squeezes
    them into a sliver; the log residual expands exactly that region, so a better result
    spreads the cloud out instead of collapsing it.
    """
    tries = [v for row in rows for v in (row.base_trie, row.opt_trie)]
    scores = [v for row in rows for v in (row.base_f17, row.opt_f17)]

    if y_scale == "error":
        errors = [math.log10(max(_MIN_ERROR, 1.0 - s)) for s in scores]
        span = (max(errors) - min(errors)) or 0.1
        # Inverted: larger error (worse) at the bottom. The bottom pad only has to clear
        # the legend and the note; on a log axis 0.38 of the span would be a wasteful
        # half-decade of white space.
        return (
            min(tries) / 1.9,
            10.0 ** (max(errors) + 0.28 * span),
            max(tries) * 1.9,
            10.0 ** (min(errors) - 0.10 * span),
        )

    span = max(scores) - min(scores)
    if span <= 0.0:
        span = max(1e-3, abs(max(scores)) * 0.01)
    return (
        min(tries) / 1.9,
        min(scores) - 0.38 * span,
        max(tries) * 1.9,
        max(scores) + 0.10 * span,
    )


def render_frontier(
    rows: Sequence[Row],
    display: str,
    output_png: Path,
    output_pdf: Optional[Path] = None,
    subtitle: Optional[str] = None,
    y_scale: str = "error",
) -> Tuple[List[str], int]:
    """Render one figure. Returns (layout violations, number of leader lines drawn)."""
    setup_typography()

    to_y = (lambda f: max(_MIN_ERROR, 1.0 - f)) if y_scale == "error" else (lambda f: f)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xscale("log")
    x_lo, y_lo, x_hi, y_hi = _axis_limits(rows, y_scale)
    ax.set_xlim(x_lo, x_hi)
    if y_scale == "error":
        ax.set_yscale("log")
        # Ticks stay in F_1/7 so the axis reads the same as the linear version.
        shown = [f for f in F_GRID if y_hi <= 1.0 - f <= y_lo]
        ax.yaxis.set_major_locator(FixedLocator([1.0 - f for f in shown]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda e, _: f"{1.0 - e:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(y_lo, y_hi)

    arrow_patches: List[FancyArrowPatch] = []
    for row in rows:
        color = _row_color(row)
        patch = FancyArrowPatch(
            (row.base_trie, to_y(row.base_f17)),
            (row.opt_trie, to_y(row.opt_f17)),
            arrowstyle="-|>",
            mutation_scale=ARROW_MUTATION,
            lw=ARROW_LINEWIDTH,
            color=color,
            alpha=0.88,
            zorder=3,
            shrinkA=3.0,
            shrinkB=3.0,
        )
        ax.add_patch(patch)
        arrow_patches.append(patch)
        ax.plot(
            [row.base_trie],
            [to_y(row.base_f17)],
            "o",
            mfc="white",
            mec=color,
            mew=1.1,
            ms=MARKER_SIZE,
            zorder=3,
            alpha=0.88,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(colors="#444", labelsize=8.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_xlabel("pattern trie size (nodes, log scale)  —  left = more compact", fontsize=10.2)
    ax.set_ylabel(
        r"$F_{1/7}$  (held-out, log residual)  —  up = more accurate"
        if y_scale == "error"
        else r"$F_{1/7}$  (held-out)  —  up = more accurate",
        fontsize=11,
    )

    total = len(rows)
    n_win = sum(1 for r in rows if r.more_accurate and r.smaller_trie)
    n_bigger = sum(1 for r in rows if r.more_accurate and not r.smaller_trie)
    n_loss = total - n_win - n_bigger

    handles = []
    for color, count, text in (
        (COLOR_WIN, n_win, "more accurate $+$ smaller trie"),
        (COLOR_BIGGER, n_bigger, "more accurate $+$ bigger trie"),
        (COLOR_LOSS, n_loss, "less accurate"),
    ):
        if count:
            handles.append(
                Line2D(
                    [0], [0], color=color, lw=2.0, marker=">", ms=6,
                    label=f"{text} ({count}/{total})",
                )
            )
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=False,
        fontsize=8.2,
        handlelength=2.2,
        borderpad=0.2,
    )

    # Run metadata sits above the axes as a sub-caption: inside the data area it would
    # compete with the legend and the summary note for the same empty corner.
    if subtitle:
        ax.set_title(subtitle, fontsize=8.0, color="#666", loc="left", pad=6.0)

    note = ax.text(
        0.045,
        0.165,
        f"{display} moves {n_win}/{total} onto\n"
        r"better-$F_{1/7}$, smaller-trie points",
        transform=ax.transAxes,
        fontsize=8.6,
        va="top",
        ha="left",
        color="#333",
        linespacing=1.35,
    )

    # tight_layout resizes the axes, which changes the data->pixel mapping. It must run
    # before any pixel measurement, or the solved layout would not be the saved layout.
    fig.tight_layout()

    # ---- solve label placement in pixel space -------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_point = fig.dpi / 72.0

    arrows: List[ArrowGeom] = [
        measure_arrow(
            patch,
            tuple(ax.transData.transform((row.base_trie, to_y(row.base_f17)))),
            tuple(ax.transData.transform((row.opt_trie, to_y(row.opt_f17)))),
            px_per_point,
        )
        for row, patch in zip(rows, arrow_patches)
    ]

    annotations = [
        ax.annotate(
            row.name,
            (row.opt_trie, to_y(row.opt_f17)),
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.6,
            color=_row_color(row),
            zorder=5,
        )
        for row in rows
    ]
    fig.canvas.draw()
    labels = [
        LabelGeom(row.name, ann.get_window_extent(renderer).width, ann.get_window_extent(renderer).height)
        for row, ann in zip(rows, annotations)
    ]

    obstacles = Obstacles(
        axes=tuple(ax.get_window_extent(renderer).extents),
        markers=[
            (tuple(ax.transData.transform((row.base_trie, to_y(row.base_f17)))),
             0.5 * MARKER_SIZE * px_per_point + 0.5 * ARROW_LINEWIDTH * px_per_point)
            for row in rows
        ],
        reserved=[
            tuple(legend.get_window_extent(renderer).extents),
            tuple(note.get_window_extent(renderer).extents),
        ],
    )

    standoff = STANDOFF_POINTS * px_per_point
    centres = solve_labels(arrows, labels, obstacles, standoff)
    for ann, arrow, centre in zip(annotations, arrows, centres):
        ann.xyann = (
            (centre[0] - arrow.anchor[0]) / px_per_point,
            (centre[1] - arrow.anchor[1]) / px_per_point,
        )

    # Everything from here on uses the *rendered* text extents rather than the solver's
    # predicted boxes, so the audit certifies the figure as drawn.
    fig.canvas.draw()
    placed = [tuple(ann.get_window_extent(renderer).extents) for ann in annotations]

    # A label that could not be seated near its arrowhead gets a leader line, so the
    # pairing stays unambiguous instead of relying on proximity that is not there.
    to_data = ax.transData.inverted().transform
    leaders = 0
    for row, arrow, box in zip(rows, arrows, placed):
        leader = leader_segment(box, arrow)
        if leader is None:
            continue
        leaders += 1
        (sx, sy), (ex, ey) = leader
        (dsx, dsy), (dex, dey) = to_data((sx, sy)), to_data((ex, ey))
        ax.plot(
            [dsx, dex],
            [dsy, dey],
            lw=0.6,
            color=_row_color(row),
            alpha=0.55,
            zorder=2,
            solid_capstyle="butt",
        )

    problems = audit_layout(arrows, labels, placed, obstacles)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return problems, leaders


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument(
        "--output-dir",
        default="results/frontier_figures",
        help="directory for generated figures",
    )
    parser.add_argument(
        "--run",
        action="append",
        choices=[r.key for r in KNOWN_RUNS],
        help="render only this run (repeatable); default renders every available run",
    )
    parser.add_argument(
        "--y-scale",
        choices=("error", "linear"),
        default="error",
        help="error: log residual 1-F_1/7 (default, separates strong results); "
             "linear: raw F_1/7 as in the published figure",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any figure has a label collision",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    selected = [r for r in KNOWN_RUNS if not args.run or r.key in args.run]
    failures = 0
    rendered = 0

    for spec in selected:
        try:
            rows = load_rows(repo_root, spec)
        except (FileNotFoundError, ValueError) as exc:
            if args.run:
                print(f"error: {exc}", file=sys.stderr)
                failures += 1
            else:
                print(f"skipping {spec.key}: {exc}")
            continue

        suffix = "" if args.y_scale == "error" else f"_{args.y_scale}"
        stem = f"published_baseline_to_{spec.key}_frontier{suffix}"
        problems, leaders = render_frontier(
            rows,
            spec.display,
            out_dir / f"{stem}.png",
            out_dir / f"{stem}.pdf",
            subtitle=spec.note,
            y_scale=args.y_scale,
        )
        rendered += 1
        status = "clean" if not problems else f"{len(problems)} COLLISIONS"
        if leaders:
            status += f", {leaders} leader{'s' if leaders > 1 else ''}"
        print(f"{spec.key:12s} {len(rows):2d} datasets  ->  {out_dir / stem}.png  [{status}]")
        for problem in problems:
            print(f"    ! {problem}")
        failures += len(problems)

    if not rendered:
        print("error: no runs rendered", file=sys.stderr)
        return 1
    if failures:
        print(
            "\nResidual collisions mean the layout is over-subscribed: too many arrowheads\n"
            "for the canvas. Widen FIGURE_SIZE or drop series; the solver cannot invent space.",
            file=sys.stderr,
        )
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
