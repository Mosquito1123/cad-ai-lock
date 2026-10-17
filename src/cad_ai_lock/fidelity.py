"""Floorplan fidelity checks — the missing correction link.

Commercial rule: a run is NOT successful if the locked RoomPlan or a generated
image cannot be shown to agree with the source CAD floorplan. Soft reference
image generation (prompt + "reference paths") is NOT ControlNet and must not
be treated as structure-locked output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.control import render_canny_from_plan, render_canny_from_image
from cad_ai_lock.geometry import RoomPlan


@dataclass
class FidelityResult:
    score: float
    passed: bool
    threshold: float
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "issues": self.issues,
            "metrics": self.metrics,
        }


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Return (distance, t in 0..1)."""
    vx, vy = x2 - x1, y2 - y1
    length2 = vx * vx + vy * vy
    if length2 < 1e-6:
        return float((px - x1) ** 2 + (py - y1) ** 2) ** 0.5, 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / length2
    t_clamped = max(0.0, min(1.0, t))
    qx = x1 + t_clamped * vx
    qy = y1 + t_clamped * vy
    return float(((px - qx) ** 2 + (py - qy) ** 2) ** 0.5), t_clamped


def _near_opening(x: float, y: float, plan: RoomPlan, tol: float = 10.0) -> bool:
    for op in plan.openings:
        dist, _t = _point_segment_distance(
            x, y, op.start.x, op.start.y, op.end.x, op.end.y
        )
        if dist <= tol:
            return True
    return False


def measure_wall_deviation_px(
    plan: RoomPlan,
    source_bgr: np.ndarray,
    *,
    step: float = 4.0,
) -> dict[str, Any]:
    """Distance from wall centerlines to the source ink, in pixels.

    Samples that fall on a door/window gap are skipped — the wall is drawn
    through the opening in the plan, but the ink is intentionally missing.
    """
    h, w = plan.image_height, plan.image_width
    src = source_bgr
    if src.shape[0] != h or src.shape[1] != w:
        src = cv2.resize(source_bgr, (w, h))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    ink = (gray < 140).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # distanceTransform expects non-zero = free space
    free = np.where(ink > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    samples: list[float] = []
    for wall in plan.walls:
        length = wall.length_px
        n = max(2, int(length / max(step, 1.0)))
        for t in np.linspace(0.0, 1.0, n):
            x = wall.start.x + (wall.end.x - wall.start.x) * float(t)
            y = wall.start.y + (wall.end.y - wall.start.y) * float(t)
            if _near_opening(x, y, plan):
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                samples.append(float(dist[yi, xi]))
    if not samples:
        return {"mean_px": None, "median_px": None, "p95_px": None, "samples": 0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_px": round(float(arr.mean()), 3),
        "median_px": round(float(np.median(arr)), 3),
        "p95_px": round(float(np.percentile(arr, 95)), 3),
        "samples": int(arr.size),
    }


def _edge_coverage(ctrl: np.ndarray, gen: np.ndarray, dilate: int = 5) -> tuple[float, float]:
    if ctrl.ndim == 3:
        ctrl = cv2.cvtColor(ctrl, cv2.COLOR_BGR2GRAY)
    if gen.ndim == 3:
        gen = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
    ctrl_e = (ctrl > 40).astype(np.uint8) * 255
    gen_e = cv2.Canny(cv2.GaussianBlur(gen, (3, 3), 0), 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    gen_d = cv2.dilate(gen_e, k, 1)
    ctrl_px = int(np.count_nonzero(ctrl_e))
    if ctrl_px == 0:
        return 0.0, 0.0
    covered = int(np.count_nonzero((ctrl_e > 0) & (gen_d > 0)))
    coverage = covered / ctrl_px
    gen_px = int(np.count_nonzero(gen_e))
    precision = (
        int(np.count_nonzero((gen_e > 0) & (cv2.dilate(ctrl_e, k, 1) > 0))) / gen_px
        if gen_px
        else 0.0
    )
    return float(coverage), float(precision)


def assess_plan_vs_source(
    plan: RoomPlan,
    source_bgr: np.ndarray,
    *,
    threshold: float = 0.28,
    expect_min_rooms: int = 5,
) -> FidelityResult:
    """Does RoomPlan actually explain the source floorplan edges?"""
    issues: list[str] = []
    h, w = plan.image_height, plan.image_width
    src = cv2.resize(source_bgr, (w, h))
    src_e = render_canny_from_image(src)
    plan_e = render_canny_from_plan(plan)
    coverage, precision = _edge_coverage(src_e, plan_e, dilate=6)
    # Also: plan edges should land on source ink
    cov2, _ = _edge_coverage(plan_e, src_e, dilate=6)
    deviation = measure_wall_deviation_px(plan, source_bgr)
    score = 0.55 * coverage + 0.45 * cov2

    n_rooms = len(plan.rooms or [])
    names = [str(r.get("name") or "") for r in (plan.rooms or [])]
    types = [str(r.get("type") or "") for r in (plan.rooms or [])]
    has_living = any("客厅" in n or t == "living" for n, t in zip(names, types))
    has_kitchen = any("厨" in n or t == "kitchen" for n, t in zip(names, types))
    n_bed = sum(1 for n, t in zip(names, types) if "卧" in n or t == "bedroom")

    if n_rooms < expect_min_rooms:
        issues.append(f"rooms_too_few:{n_rooms}<{expect_min_rooms}")
        score -= 0.2
    if not has_living:
        issues.append("missing_living_room")
        score -= 0.15
    if not has_kitchen:
        issues.append("missing_kitchen")
        score -= 0.1
    if n_bed < 2:
        issues.append(f"bedrooms_too_few:{n_bed}")
        score -= 0.1
    if plan.scale_px_per_m is None:
        issues.append("missing_scale")
        score -= 0.05
    if coverage < 0.15:
        issues.append(f"plan_misses_source_edges:coverage={coverage:.2f}")
    if cov2 < 0.15:
        issues.append(f"plan_edges_not_on_source:cov={cov2:.2f}")
    median_px = deviation.get("median_px")
    mean_px = deviation.get("mean_px")
    # Centerlines within a few pixels of the ink count as aligned even when
    # the sheet is a small demo (fewer named rooms than a full apartment).
    geometry_ok = (
        median_px is not None
        and mean_px is not None
        and float(median_px) <= 3.5
        and float(mean_px) <= 6.0
        and cov2 >= 0.45
    )
    if median_px is not None and float(median_px) > 3.5:
        issues.append(f"wall_centerline_offset:median_px={float(median_px):.2f}")
        score -= min(0.35, float(median_px) / 40.0)

    score = float(max(0.0, min(1.0, score)))
    semantic_ok = (
        score >= threshold
        and "missing_living_room" not in issues
        and n_rooms >= expect_min_rooms
    )
    passed = bool(geometry_ok or semantic_ok)
    return FidelityResult(
        score=score,
        passed=passed,
        threshold=threshold,
        issues=issues,
        metrics={
            "source_edge_coverage_by_plan": round(coverage, 4),
            "plan_edge_on_source": round(cov2, 4),
            "wall_deviation_px": deviation,
            "geometry_aligned": geometry_ok,
            "rooms": n_rooms,
            "bedrooms": n_bed,
            "has_living": has_living,
            "has_kitchen": has_kitchen,
            "scale_px_per_m": plan.scale_px_per_m,
        },
    )


def assess_image_vs_plan(
    image_bgr: np.ndarray,
    plan: RoomPlan,
    *,
    threshold: float = 0.30,
    label: str = "image",
) -> FidelityResult:
    """Does a generated image keep plan wall structure?"""
    h, w = plan.image_height, plan.image_width
    img = cv2.resize(image_bgr, (w, h))
    plan_e = render_canny_from_plan(plan)
    coverage, precision = _edge_coverage(plan_e, img, dilate=5)
    score = 0.7 * coverage + 0.3 * precision
    issues: list[str] = []
    if coverage < threshold:
        issues.append(f"{label}_structure_drift:coverage={coverage:.2f}")
    passed = coverage >= threshold
    return FidelityResult(
        score=float(score),
        passed=passed,
        threshold=threshold,
        issues=issues,
        metrics={
            "control_coverage": round(coverage, 4),
            "precision": round(precision, 4),
            "label": label,
        },
    )


def assess_image_vs_source(
    image_bgr: np.ndarray,
    source_bgr: np.ndarray,
    *,
    threshold: float = 0.22,
    label: str = "image",
) -> FidelityResult:
    """Top-down reno must still look like the same apartment envelope."""
    h, w = source_bgr.shape[:2]
    img = cv2.resize(image_bgr, (w, h))
    src_e = render_canny_from_image(source_bgr)
    coverage, precision = _edge_coverage(src_e, img, dilate=7)
    score = 0.65 * coverage + 0.35 * precision
    issues: list[str] = []
    if coverage < threshold:
        issues.append(f"{label}_deviates_from_floorplan:coverage={coverage:.2f}")
    return FidelityResult(
        score=float(score),
        passed=coverage >= threshold,
        threshold=threshold,
        issues=issues,
        metrics={
            "source_coverage": round(coverage, 4),
            "precision": round(precision, 4),
            "label": label,
        },
    )
