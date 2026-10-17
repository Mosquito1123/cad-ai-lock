"""Extract RoomPlan from PDF / JPG CAD floorplans.

Pipeline:
  1. Load PDF page(s) via pypdfium2 → RGB, or load JPG/PNG via OpenCV.
  2. Edge / line detection (Canny + Hough) → candidate wall segments.
  3. Detect shorter gaps / perpendicular stubs as openings (doors/windows).
  4. Optional: OpenAI-compatible vision API to refine labels & merge segments.
  5. Emit RoomPlan JSON + write control maps via control.py.
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.config import get_settings
from cad_ai_lock.geometry import Opening, OpeningType, Point2D, RoomPlan, Wall


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def load_image(path: str | Path, pdf_page: int = 0, pdf_scale: float = 2.0) -> np.ndarray:
    """Load JPG/PNG/PDF → BGR uint8 image."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        if pdf_page < 0 or pdf_page >= len(doc):
            raise IndexError(f"PDF page {pdf_page} out of range (0..{len(doc) - 1})")
        page = doc[pdf_page]
        bitmap = page.render(scale=pdf_scale)
        pil = bitmap.to_pil()
        rgb = np.array(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _merge_collinear(
    segments: list[tuple[float, float, float, float]],
    angle_tol_deg: float = 8.0,
    dist_tol: float = 15.0,
    gap_tol: float = 25.0,
) -> list[tuple[float, float, float, float]]:
    """Greedy merge of near-collinear Hough segments into longer walls."""
    if not segments:
        return []

    def angle(s: tuple[float, float, float, float]) -> float:
        return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180

    def mid(s: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)

    def length(s: tuple[float, float, float, float]) -> float:
        return math.hypot(s[2] - s[0], s[3] - s[1])

    segs = sorted(segments, key=length, reverse=True)
    used = [False] * len(segs)
    merged: list[tuple[float, float, float, float]] = []

    for i, s in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        a = angle(s)
        points = [(s[0], s[1]), (s[2], s[3])]
        changed = True
        while changed:
            changed = False
            for j, t in enumerate(segs):
                if used[j]:
                    continue
                if abs(((angle(t) - a + 90) % 180) - 90) > angle_tol_deg:
                    continue
                # distance between midpoints projected roughly
                mx, my = mid(t)
                # point-line distance to s axis
                x1, y1, x2, y2 = points[0][0], points[0][1], points[-1][0], points[-1][1]
                denom = math.hypot(x2 - x1, y2 - y1) or 1.0
                dist = abs((y2 - y1) * mx - (x2 - x1) * my + x2 * y1 - y2 * x1) / denom
                if dist > dist_tol:
                    continue
                # gap between endpoint clouds
                all_pts = points + [(t[0], t[1]), (t[2], t[3])]
                min_gap = min(
                    math.hypot(p[0] - q[0], p[1] - q[1])
                    for p in points
                    for q in [(t[0], t[1]), (t[2], t[3])]
                )
                if min_gap > gap_tol and length(t) < 40:
                    continue
                if min_gap > gap_tol * 2:
                    continue
                used[j] = True
                points.extend([(t[0], t[1]), (t[2], t[3])])
                changed = True

        # PCA-ish: extreme points along dominant direction
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # use angle a
        rad = math.radians(a)
        ux, uy = math.cos(rad), math.sin(rad)
        projs = [p[0] * ux + p[1] * uy for p in points]
        i_min, i_max = int(np.argmin(projs)), int(np.argmax(projs))
        merged.append((points[i_min][0], points[i_min][1], points[i_max][0], points[i_max][1]))

    return merged


def _structure_mask(gray: np.ndarray) -> np.ndarray:
    """Dark structural ink. Thin text is dropped; only a few border pixels are cropped.

    A previous version zeroed a wide strip on the right (``w // 14``), which
    could delete a real exterior wall that sat near the sheet edge.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = (blur < 130).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = gray.shape[:2]
    m = 4
    ink[:m, :] = 0
    ink[-m:, :] = 0
    ink[:, :m] = 0
    ink[:, -m:] = 0
    return ink


def _band_groups(flags: np.ndarray, gap: int = 2) -> list[tuple[int, int]]:
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v > prev + gap:
            groups.append((start, prev))
            start = v
        prev = v
    groups.append((start, prev))
    return groups


def _full_runs(counts: np.ndarray, dropout: int = 3) -> list[tuple[int, int]]:
    """Runs where the cross-section is a full wall, ignoring tapered caps.

    ``dropout`` merges 1–2 px holes (JPEG) without bridging a door gap.
    """
    if counts.size == 0 or int(counts.max()) == 0:
        return []
    positive = counts[counts > 0]
    thresh = max(2, int(round(0.85 * float(np.median(positive)))))
    on = counts >= thresh
    raw: list[tuple[int, int]] = []
    i = 0
    n = int(on.size)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i + 1
        while j < n and on[j]:
            j += 1
        raw.append((i, j - 1))
        i = j
    merged: list[list[int]] = []
    for a, b in raw:
        if merged and a - merged[-1][1] - 1 <= dropout:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _axis_wall_segments(
    mask: np.ndarray,
    orientation: str,
    *,
    min_len: int = 28,
    max_bridge: int = 140,
) -> tuple[list[tuple[float, float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Medial-axis walls along one direction, plus gaps that were bridged.

    Thick strokes are collapsed to their centerline (not the ink edge), so a
    6 px wall lands on the same pixel as the plan that drew it.
    """
    h, w = mask.shape[:2]
    segments: list[tuple[float, float, float, float, float]] = []
    gaps: list[tuple[float, float, float, float]] = []
    k = max(15, min(h, w) // 40)
    if orientation == "h":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        directed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        for y0, y1 in _band_groups(directed.any(axis=1)):
            band = directed[y0 : y1 + 1]
            counts = (band > 0).sum(axis=0).astype(np.int32)
            thick = float(y1 - y0 + 1)
            y = float(round((y0 + y1) / 2.0))
            runs = [(a, b) for a, b in _full_runs(counts) if b - a + 1 >= min_len]
            merged: list[list[int]] = []
            bridged: list[tuple[int, int]] = []
            for a, b in runs:
                if not merged:
                    merged.append([a, b])
                    continue
                gap = a - merged[-1][1] - 1
                if 0 < gap <= max_bridge:
                    bridged.append((merged[-1][1] + 1, a - 1))
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            for a, b in merged:
                segments.append((float(a), y, float(b), y, thick))
            for g0, g1 in bridged:
                if g1 >= g0:
                    gaps.append((float(g0), y, float(g1), y))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        directed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        for x0, x1 in _band_groups(directed.any(axis=0)):
            band = directed[:, x0 : x1 + 1]
            counts = (band > 0).sum(axis=1).astype(np.int32)
            thick = float(x1 - x0 + 1)
            x = float(round((x0 + x1) / 2.0))
            runs = [(a, b) for a, b in _full_runs(counts) if b - a + 1 >= min_len]
            merged = []
            bridged = []
            for a, b in runs:
                if not merged:
                    merged.append([a, b])
                    continue
                gap = a - merged[-1][1] - 1
                if 0 < gap <= max_bridge:
                    bridged.append((merged[-1][1] + 1, a - 1))
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            for a, b in merged:
                segments.append((x, float(a), x, float(b), thick))
            for g0, g1 in bridged:
                if g1 >= g0:
                    gaps.append((x, float(g0), x, float(g1)))
    return segments, gaps


def _snap_wall_junctions(walls: list[Wall], tol: float = 8.0) -> list[Wall]:
    """Pull wall ends onto crossing centerlines so corners actually meet."""
    vert_x: list[float] = []
    hor_y: list[float] = []
    for w in walls:
        if abs(w.end.x - w.start.x) <= abs(w.end.y - w.start.y):
            vert_x.append((w.start.x + w.end.x) / 2.0)
        else:
            hor_y.append((w.start.y + w.end.y) / 2.0)

    def nearest(v: float, axes: list[float]) -> float:
        if not axes:
            return v
        a = min(axes, key=lambda g: abs(g - v))
        return a if abs(a - v) <= tol else v

    out: list[Wall] = []
    for w in walls:
        horizontal = abs(w.end.y - w.start.y) <= abs(w.end.x - w.start.x)
        if horizontal:
            y = (w.start.y + w.end.y) / 2.0
            x1, x2 = nearest(w.start.x, vert_x), nearest(w.end.x, vert_x)
            if abs(x2 - x1) < 24:
                continue
            start, end = Point2D(x=min(x1, x2), y=y), Point2D(x=max(x1, x2), y=y)
        else:
            x = (w.start.x + w.end.x) / 2.0
            y1, y2 = nearest(w.start.y, hor_y), nearest(w.end.y, hor_y)
            if abs(y2 - y1) < 24:
                continue
            start, end = Point2D(x=x, y=min(y1, y2)), Point2D(x=x, y=max(y1, y2))
        out.append(
            Wall(
                id=w.id,
                start=start,
                end=end,
                thickness_px=w.thickness_px,
                exterior=w.exterior,
            )
        )
    return out


def _point_seg_dist(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    vx, vy = x2 - x1, y2 - y1
    L2 = vx * vx + vy * vy
    if L2 < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L2))
    return math.hypot(px - (x1 + t * vx), py - (y1 + t * vy))


def _colored_openings_on_walls(
    image_bgr: np.ndarray, walls: list[Wall]
) -> list[Opening]:
    """Door/window strokes drawn on a wall (sample: orange door, blue window).

    Endpoints are inset by the stroke's cap radius so they match the line
    the CAD author specified, not the antialiased fringe.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 90
    hue = hsv[:, :, 0]
    h, w = sat.shape[:2]
    found: list[Opening] = []
    for wall in walls:
        x1, y1 = wall.start.x, wall.start.y
        x2, y2 = wall.end.x, wall.end.y
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 16:
            continue
        n = max(2, int(round(length)))
        ts = np.linspace(0.0, 1.0, n)
        flags = np.zeros(n, dtype=bool)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        px, py = -uy, ux
        radius = max(2, int(round(wall.thickness_px)) // 2 + 1)
        for i, t in enumerate(ts):
            cx = x1 + (x2 - x1) * float(t)
            cy = y1 + (y2 - y1) * float(t)
            hit = False
            for s in range(-radius, radius + 1):
                xi = int(round(cx + px * s))
                yi = int(round(cy + py * s))
                if 0 <= xi < w and 0 <= yi < h and sat[yi, xi]:
                    hit = True
                    break
            flags[i] = hit
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i + 1
            while j < n and flags[j]:
                j += 1
            if j - i >= 16:
                t0 = float(ts[i])
                t1 = float(ts[j - 1])
                ax = x1 + (x2 - x1) * t0
                ay = y1 + (y2 - y1) * t0
                bx = x1 + (x2 - x1) * t1
                by = y1 + (y2 - y1) * t1
                mid_t = (t0 + t1) / 2.0
                mx = x1 + (x2 - x1) * mid_t
                my = y1 + (y2 - y1) * mid_t
                # perpendicular thickness → cap inset
                half = 0
                mxi, myi = int(round(mx)), int(round(my))
                while half < 6:
                    xi = int(round(mx + px * (half + 1)))
                    yi = int(round(my + py * (half + 1)))
                    if not (0 <= xi < w and 0 <= yi < h and sat[yi, xi]):
                        break
                    half += 1
                run = math.hypot(bx - ax, by - ay)
                inset = float(min(half, 4))
                if run > 2 * inset + 8:
                    ax += ux * inset
                    ay += uy * inset
                    bx -= ux * inset
                    by -= uy * inset
                hues = []
                for t in np.linspace(t0, t1, 9):
                    sx = int(round(x1 + (x2 - x1) * float(t)))
                    sy = int(round(y1 + (y2 - y1) * float(t)))
                    if 0 <= sx < w and 0 <= sy < h and sat[sy, sx]:
                        hues.append(int(hue[sy, sx]))
                med_hue = float(np.median(hues)) if hues else 0.0
                # OpenCV hue: orange door ~15, cyan window ~95.
                kind = (
                    OpeningType.DOOR
                    if med_hue < 35 or med_hue > 150
                    else OpeningType.WINDOW
                )
                found.append(
                    Opening(
                        id=_new_id("open"),
                        type=kind,
                        start=Point2D(x=float(ax), y=float(ay)),
                        end=Point2D(x=float(bx), y=float(by)),
                        wall_id=wall.id,
                    )
                )
            i = j
    return found


def _gap_openings(
    gaps: list[tuple[float, float, float, float]], walls: list[Wall]
) -> list[Opening]:
    openings: list[Opening] = []
    for x1, y1, x2, y2 in gaps:
        if math.hypot(x2 - x1, y2 - y1) < 16:
            continue
        wall_id = None
        best = 1e9
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for w in walls:
            d = _point_seg_dist(mx, my, w.start.x, w.start.y, w.end.x, w.end.y)
            if d < best:
                best = d
                wall_id = w.id
        openings.append(
            Opening(
                id=_new_id("open"),
                type=OpeningType.DOOR,
                start=Point2D(x=x1, y=y1),
                end=Point2D(x=x2, y=y2),
                wall_id=wall_id if best <= 12 else None,
            )
        )
    return openings


def _dedupe_openings(openings: list[Opening], tol: float = 22.0) -> list[Opening]:
    """Prefer the colored stroke when it overlaps a structural gap."""
    # Caller passes colored strokes first; a later structural gap at the same
    # place is dropped so the door/window keeps the stroke endpoints.
    kept: list[Opening] = []
    for op in openings:
        mx = (op.start.x + op.end.x) / 2.0
        my = (op.start.y + op.end.y) / 2.0
        dup = False
        for k in kept:
            kx = (k.start.x + k.end.x) / 2.0
            ky = (k.start.y + k.end.y) / 2.0
            if math.hypot(mx - kx, my - ky) <= tol:
                dup = True
                break
        if not dup:
            kept.append(op)
    return kept


def _walls_from_segments(
    segments: list[tuple[float, float, float, float, float]],
) -> list[Wall]:
    walls: list[Wall] = []
    for x1, y1, x2, y2, thick in segments:
        if math.hypot(x2 - x1, y2 - y1) < 24:
            continue
        walls.append(
            Wall(
                id=_new_id("wall"),
                start=Point2D(x=x1, y=y1),
                end=Point2D(x=x2, y=y2),
                thickness_px=max(4.0, thick),
            )
        )
    return _snap_wall_junctions(walls)


def _detect_centerline_walls(
    image_bgr: np.ndarray,
) -> tuple[list[Wall], list[Opening], np.ndarray] | None:
    """Centerline walls + openings. None if the sheet is not axis-aligned ink."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Structural ink only — colored door/window strokes are openings, not walls.
    core = ((gray < 130) & (hsv[:, :, 1] < 50)).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = core.shape[:2]
    m = 3
    core[:m, :] = 0
    core[-m:, :] = 0
    core[:, :m] = 0
    core[:, -m:] = 0
    max_bridge = max(48, int(0.22 * min(h, w)))
    h_segs, h_gaps = _axis_wall_segments(core, "h", max_bridge=max_bridge)
    v_segs, v_gaps = _axis_wall_segments(core, "v", max_bridge=max_bridge)
    walls = _walls_from_segments(h_segs + v_segs)
    if len(walls) < 4:
        return None
    colored = _colored_openings_on_walls(image_bgr, walls)
    gaps = _gap_openings(h_gaps + v_gaps, walls)
    openings = _dedupe_openings(colored + gaps)
    return walls, openings, core


def _snap_ortho_seg(
    x1: float, y1: float, x2: float, y2: float, tol_deg: float = 14.0
) -> tuple[float, float, float, float] | None:
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
    d0 = min(abs(ang - 0), abs(ang - 180))
    d90 = abs(ang - 90)
    if d0 <= tol_deg and d0 <= d90:
        y = (y1 + y2) / 2
        return (x1, y, x2, y)
    if d90 <= tol_deg:
        x = (x1 + x2) / 2
        return (x, y1, x, y2)
    return None


def detect_walls_and_openings(
    image_bgr: np.ndarray,
    min_line_length: int = 40,
    max_line_gap: int = 12,
    canny1: int = 50,
    canny2: int = 150,
) -> tuple[list[Wall], list[Opening], np.ndarray]:
    """Wall centerlines + door/window strokes.

    Orthogonal ink is traced on its medial axis so walls land on the same
    pixels as the drawing, and openings are the colored strokes (or real
    gaps) on those walls — not short leftover segments or title text.
    Falls back to Hough/LSD when the sheet is not an axis-aligned ink plan.
    """
    center = _detect_centerline_walls(image_bgr)
    if center is not None:
        return center

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    structure = _structure_mask(gray)
    edges = structure.copy()

    raw: list[tuple[float, float, float, float]] = []

    # Hough on structure mask
    lines = cv2.HoughLinesP(
        structure,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, min(w, h) // 45),
        minLineLength=max(28, min_line_length - 10),
        maxLineGap=max(8, max_line_gap),
    )
    if lines is not None:
        for line in lines:
            vals = line.reshape(-1)[:4]
            x1, y1, x2, y2 = map(float, vals)
            if math.hypot(x2 - x1, y2 - y1) >= min_line_length * 0.7:
                snapped = _snap_ortho_seg(x1, y1, x2, y2)
                if snapped:
                    raw.append(snapped)

    # LSD supplement (often recovers long exterior walls)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lsd_lines = lsd.detect(blur)[0]
        if lsd_lines is not None:
            for L in lsd_lines:
                x1, y1, x2, y2 = map(float, np.asarray(L).reshape(-1)[:4])
                ln = math.hypot(x2 - x1, y2 - y1)
                if ln < 40:
                    continue
                mx, my = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                if not (0 <= mx < w and 0 <= my < h):
                    continue
                if structure[
                    max(0, my - 2) : my + 3, max(0, mx - 2) : mx + 3
                ].max() == 0:
                    continue
                snapped = _snap_ortho_seg(x1, y1, x2, y2)
                if snapped:
                    raw.append(snapped)
    except Exception:
        pass

    # legacy Canny path if still sparse
    if len(raw) < 4:
        canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), canny1, canny2)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        edges = canny
        lines = cv2.HoughLinesP(
            canny,
            rho=1,
            theta=np.pi / 180,
            threshold=max(40, min(w, h) // 40),
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is not None:
            for line in lines:
                vals = line.reshape(-1)[:4]
                x1, y1, x2, y2 = map(float, vals)
                if math.hypot(x2 - x1, y2 - y1) >= min_line_length * 0.8:
                    raw.append((x1, y1, x2, y2))

    merged = _merge_collinear(raw)
    lengths = [math.hypot(s[2] - s[0], s[3] - s[1]) for s in merged]
    if lengths:
        thr = max(min_line_length * 1.2, float(np.percentile(lengths, 35)))
    else:
        thr = float(min_line_length)

    walls: list[Wall] = []
    openings: list[Opening] = []
    for s, L in zip(merged, lengths or [0] * len(merged)):
        x1, y1, x2, y2 = s
        if L >= thr:
            walls.append(
                Wall(
                    id=_new_id("wall"),
                    start=Point2D(x=x1, y=y1),
                    end=Point2D(x=x2, y=y2),
                    thickness_px=8.0,
                )
            )
        elif L >= 18:
            openings.append(
                Opening(
                    id=_new_id("open"),
                    type=OpeningType.DOOR if L < thr * 0.65 else OpeningType.WINDOW,
                    start=Point2D(x=x1, y=y1),
                    end=Point2D(x=x2, y=y2),
                )
            )

    if not walls:
        margin = 20
        corners = [
            (margin, margin, w - margin, margin),
            (w - margin, margin, w - margin, h - margin),
            (w - margin, h - margin, margin, h - margin),
            (margin, h - margin, margin, margin),
        ]
        for x1, y1, x2, y2 in corners:
            walls.append(
                Wall(
                    id=_new_id("wall"),
                    start=Point2D(x=float(x1), y=float(y1)),
                    end=Point2D(x=float(x2), y=float(y2)),
                    thickness_px=10.0,
                    exterior=True,
                )
            )

    return walls, openings, edges


def refine_with_vision(
    image_bgr: np.ndarray,
    plan: RoomPlan,
) -> RoomPlan:
    """Optional LLM vision refine via OpenAI-compatible chat completions."""
    settings = get_settings()
    if not settings.openai_api_key:
        return plan

    import httpx

    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return plan
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    schema_hint = {
        "walls": [{"id": "wall_1", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0}}],
        "openings": [
            {
                "id": "door_1",
                "type": "door",
                "start": {"x": 40, "y": 0},
                "end": {"x": 60, "y": 0},
            }
        ],
        "notes": "string",
    }
    prompt = (
        "You are a floorplan CAD parser. Given this floorplan image and a draft JSON "
        "of walls/openings in pixel coordinates, refine wall endpoints and label "
        "doors vs windows. Return ONLY valid JSON matching this shape:\n"
        f"{json.dumps(schema_hint)}\n"
        f"Image size: {plan.image_width}x{plan.image_height}. Draft:\n"
        f"{json.dumps(plan.to_json_dict(), ensure_ascii=False)[:6000]}"
    )

    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": settings.openai_vision_model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        # strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        walls = [Wall.model_validate(w) for w in data.get("walls", [])]
        openings = [Opening.model_validate(o) for o in data.get("openings", [])]
        if walls:
            plan.walls = walls
        if openings:
            plan.openings = openings
        if data.get("notes"):
            plan.notes = str(data["notes"])
        plan.meta["vision_refined"] = True
    except Exception as exc:  # noqa: BLE001 — optional path must not break CLI
        plan.meta["vision_error"] = str(exc)

    return plan


def extract_room_plan(
    path: str | Path,
    *,
    pdf_page: int = 0,
    use_vision: bool = False,
    scale_px_per_m: float | None = None,
    auto_correct: bool = True,
    use_ocr: bool = True,
) -> tuple[RoomPlan, np.ndarray, np.ndarray]:
    """Main entry: path → (RoomPlan, BGR image, edge/structure map).

    When ``auto_correct`` is True (default), runs assess→repair until topology
    health passes or repair budget is exhausted. Never requires hand-edited
    plan.json for the happy path.
    """
    path = Path(path)
    image = load_image(path, pdf_page=pdf_page)
    h, w = image.shape[:2]
    walls, openings, edges = detect_walls_and_openings(image)

    plan = RoomPlan(
        # Basename only — avoid leaking absolute local paths into shared artifacts
        source_path=path.name,
        image_width=w,
        image_height=h,
        scale_px_per_m=scale_px_per_m,
        walls=walls,
        openings=openings,
        plan_revision="extract-0",
        meta={
            "extractor": "wall_centerline",
            "pdf_page": pdf_page,
            "auto_correct": auto_correct,
        },
    )

    ocr_items: list[dict[str, Any]] = []
    if use_ocr:
        try:
            from cad_ai_lock.ocr_utils import run_ocr

            ocr_items = run_ocr(image)
            plan.meta["ocr_items"] = ocr_items
            plan.meta["ocr_count"] = len(ocr_items)
        except Exception as exc:  # noqa: BLE001
            plan.meta["ocr_error"] = str(exc)

    if use_vision:
        plan = refine_with_vision(image, plan)

    if auto_correct:
        from cad_ai_lock.assess import assess_plan
        from cad_ai_lock.repair import auto_repair, mark_exterior

        plan = mark_exterior(plan)
        health = assess_plan(plan)
        plan, health, applied = auto_repair(
            plan,
            health,
            image_bgr=image,
            ocr_items=ocr_items,
        )
        plan.meta["extract_health"] = health.to_dict()
        plan.meta["extract_repairs"] = applied
    elif ocr_items and scale_px_per_m is None:
        from cad_ai_lock.repair import (
            estimate_scale_from_ocr,
            infer_rooms,
            label_rooms_from_ocr,
            mark_exterior,
        )

        plan = mark_exterior(plan)
        plan = infer_rooms(plan, image)
        plan = label_rooms_from_ocr(plan, ocr_items)
        plan = estimate_scale_from_ocr(plan, ocr_items)

    return plan, image, edges


def make_synthetic_floorplan(
    out_path: str | Path,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Draw a simple rectangle room with door + window for demos / smoke tests."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((height, width, 3), dtype=np.uint8) * 245
    # outer walls
    margin = 60
    pt1 = (margin, margin)
    pt2 = (width - margin, height - margin)
    cv2.rectangle(img, pt1, pt2, (30, 30, 30), 6)
    # interior partition
    mid_x = width // 2
    cv2.line(img, (mid_x, margin), (mid_x, height - margin - 80), (30, 30, 30), 5)
    # door gap on bottom wall (erase then mark)
    door_cx = width // 2 - 80
    cv2.line(
        img,
        (door_cx - 35, height - margin),
        (door_cx + 35, height - margin),
        (245, 245, 245),
        8,
    )
    cv2.line(
        img,
        (door_cx - 35, height - margin),
        (door_cx + 35, height - margin),
        (0, 120, 255),
        3,
    )
    # window on top wall
    win_cx = width // 2 + 100
    cv2.line(
        img,
        (win_cx - 40, margin),
        (win_cx + 40, margin),
        (245, 245, 245),
        8,
    )
    cv2.line(
        img,
        (win_cx - 40, margin),
        (win_cx + 40, margin),
        (255, 180, 0),
        3,
    )
    # dimension-ish labels
    cv2.putText(
        img,
        "CAD FLOORPLAN DEMO",
        (margin, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)
    return out_path
