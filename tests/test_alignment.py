"""Sample floor plan must stay on the source walls, rooms, and openings."""

from __future__ import annotations

import math

import numpy as np

from cad_ai_lock.extract import extract_room_plan, load_image
from cad_ai_lock.extrude import _ppm as extrude_ppm
from cad_ai_lock.extrude import extrude_room_plan
from cad_ai_lock.fidelity import assess_plan_vs_source, measure_wall_deviation_px
from cad_ai_lock.generate import run_t2i
from cad_ai_lock.geometry import RoomPlan, pixels_per_meter
from cad_ai_lock.layout import _px_per_m, build_layout
from cad_ai_lock.perspective import _basis_from_camera, _ppm as perspective_ppm
from cad_ai_lock.perspective import place_living_camera, project_point
from cad_ai_lock.render3d import project_topdown_points
from cad_ai_lock.repair import infer_openings
from cad_ai_lock.validate import validate_alignment, validate_file


def _pair_distance(a_start, a_end, b_start, b_end) -> float:
    direct = math.hypot(a_start[0] - b_start[0], a_start[1] - b_start[1]) + math.hypot(
        a_end[0] - b_end[0], a_end[1] - b_end[1]
    )
    swapped = math.hypot(a_start[0] - b_end[0], a_start[1] - b_end[1]) + math.hypot(
        a_end[0] - b_start[0], a_end[1] - b_start[1]
    )
    return min(direct, swapped) / 2.0


def _bbox(poly):
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def test_sample_extract_matches_plan_geometry():
    image = load_image("examples/sample_floorplan.jpg")
    gt = RoomPlan.from_file("examples/sample_plan.json")
    plan, _img, _edges = extract_room_plan(
        "examples/sample_floorplan.jpg", use_ocr=False, auto_correct=True
    )

    assert len(plan.walls) == len(gt.walls)
    for wall in gt.walls:
        best = min(
            _pair_distance(
                (wall.start.x, wall.start.y),
                (wall.end.x, wall.end.y),
                (w.start.x, w.start.y),
                (w.end.x, w.end.y),
            )
            for w in plan.walls
        )
        assert best <= 2.0, (wall.id, best)

    for opening in gt.openings:
        best = min(
            _pair_distance(
                (opening.start.x, opening.start.y),
                (opening.end.x, opening.end.y),
                (o.start.x, o.start.y),
                (o.end.x, o.end.y),
            )
            for o in plan.openings
            if o.type == opening.type
        )
        assert best <= 3.0, (opening.id, best)

    # Title band must not become a wall or a door.
    for wall in plan.walls:
        mid_y = (wall.start.y + wall.end.y) / 2
        assert mid_y > 50

    gt_boxes = [_bbox(r["polygon"]) for r in gt.rooms]
    got_boxes = [_bbox(r["polygon"]) for r in plan.rooms]
    assert len(got_boxes) >= 2
    for box in gt_boxes:
        assert max(_iou(box, other) for other in got_boxes) >= 0.9

    deviation = measure_wall_deviation_px(plan, image)
    assert deviation["median_px"] <= 1.5
    assert deviation["mean_px"] <= 2.0
    fidelity = assess_plan_vs_source(plan, image, expect_min_rooms=5)
    assert fidelity.metrics["geometry_aligned"] is True
    assert fidelity.passed is True


def test_shifted_walls_report_large_deviation():
    image = load_image("examples/sample_floorplan.jpg")
    plan = RoomPlan.from_file("examples/sample_plan.json").model_copy(deep=True)
    for wall in plan.walls:
        wall.start.x += 28
        wall.end.x += 28
        wall.start.y += 28
        wall.end.y += 28
    deviation = measure_wall_deviation_px(plan, image)
    assert deviation["median_px"] >= 15


def test_topdown_keeps_north_up_and_east_right():
    plan = RoomPlan.from_file("examples/sample_plan.json")
    scene = extrude_room_plan(plan)
    screen = project_topdown_points(
        scene,
        np.array([[60.0, 60.0], [60.0, 420.0], [580.0, 60.0]], dtype=np.float64),
        size=480,
    )
    north, south, east = screen
    assert north[1] < south[1]
    assert east[0] > north[0]


def test_perspective_is_not_mirrored_and_stays_in_room():
    plan, _img, _edges = extract_room_plan(
        "examples/sample_floorplan.jpg", use_ocr=False, auto_correct=True
    )
    cam = place_living_camera(plan)
    assert cam.vanishing_point is not None
    # Living room ends at x=320. The bedroom window (x≈420) must not steal the aim.
    assert cam.vanishing_point.x < 340
    assert cam.vanishing_point.y > 400

    eye = (190.0, 180.0)
    fwd, right, _up = _basis_from_camera(eye, (190.0, 420.0))
    assert fwd[1] > 0.9
    assert right[0] < 0  # facing south, screen-right is west
    projected = project_point(
        250.0,
        300.0,
        1.2,
        cam_xy=eye,
        fwd=fwd,
        right=right,
        cam_h_m=1.5,
        ppm=40.0,
        focal=612.0,
        canvas_w=720,
        canvas_h=1280,
        horizon_y=537.0,
    )
    assert projected is not None
    assert projected[0] < 360  # east of the camera appears on the left


def test_scale_fallback_is_shared():
    plan = RoomPlan.from_file("examples/sample_plan.json").model_copy(deep=True)
    plan.scale_px_per_m = None
    ppm = pixels_per_meter(plan)
    assert ppm == _px_per_m(plan) == perspective_ppm(plan) == extrude_ppm(plan)
    plan.scale_px_per_m = 40
    assert pixels_per_meter(plan) == 40


def test_existing_openings_are_not_replaced():
    plan, image, _edges = extract_room_plan(
        "examples/sample_floorplan.jpg", use_ocr=False, auto_correct=True
    )
    before = [
        (o.type.value, round(o.start.x, 1), round(o.start.y, 1), round(o.end.x, 1), round(o.end.y, 1))
        for o in plan.openings
    ]
    infer_openings(plan, image)
    after = [
        (o.type.value, round(o.start.x, 1), round(o.start.y, 1), round(o.end.x, 1), round(o.end.y, 1))
        for o in plan.openings
    ]
    assert before == after


def test_mock_render_stays_on_control_lines(tmp_path):
    plan, _img, _edges = extract_room_plan(
        "examples/sample_floorplan.jpg", use_ocr=False, auto_correct=True
    )
    layout = build_layout(plan)
    out = tmp_path / "t2i.png"
    run_t2i(plan, "现代简约", out, seed=1, layout=layout)
    result = validate_file(out, plan, layout=layout)
    assert result.passed
    assert result.mean_deviation_px is not None
    assert result.mean_deviation_px < 4.0

    # A generated image shifted off the plan fails the deviation check.
    import cv2

    img = cv2.imread(str(out))
    shifted = np.roll(img, 40, axis=1)
    bad = validate_alignment(shifted, plan)
    assert bad.mean_deviation_px is not None
    assert bad.mean_deviation_px > result.mean_deviation_px
