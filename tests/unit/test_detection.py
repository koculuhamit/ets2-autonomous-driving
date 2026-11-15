"""Turning a detector's raw outputs into what is ahead of the truck."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from core.perception.detection import (
    DEFAULT_CAB_VIEW,
    SPEC_FORMAT,
    CabView,
    Detection,
    Detector,
    Kind,
    ModelSpec,
    Rect,
    decode,
    merge,
    preprocess,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COCO_HEAD = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
SPEC = ModelSpec(input_size=(8, 24), classes=(*COCO_HEAD, None))
KINDS = SPEC.kinds()
CAR, TRAIN, TRUCK = COCO_HEAD.index("car"), COCO_HEAD.index("train"), COCO_HEAD.index("truck")


def logits_for(*winners: tuple[int, float]) -> np.ndarray:
    """One row of logits per query, `score` for its class and near zero for the rest."""
    rows = np.full((len(winners), len(SPEC.classes)), -9.0)
    for row, (index, score) in enumerate(winners):
        rows[row, index] = np.log(score / (1 - score))
    return rows


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #


def test_coco_names_become_kinds_and_the_rest_are_ignored() -> None:
    kinds = list(Kind)
    assert kinds[KINDS[CAR]] is Kind.CAR
    assert kinds[KINDS[COCO_HEAD.index("motorbike")]] is Kind.MOTORCYCLE
    assert KINDS[TRAIN] == -1
    assert KINDS[-1] == -1, "an unused class index"


def test_a_spec_from_another_program_is_refused() -> None:
    with pytest.raises(ValueError, match="not a detector spec"):
        ModelSpec.from_json({"format": "something-else", "input_size": [1, 1], "classes": []})


def test_a_spec_reads_back_its_normalisation() -> None:
    spec = ModelSpec.from_json(
        {
            "format": SPEC_FORMAT,
            "input_size": [448, 1344],
            "classes": COCO_HEAD,
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        }
    )
    assert spec.input_size == (448, 1344)
    assert spec.mean == (0.5, 0.5, 0.5) and spec.std == (0.25, 0.25, 0.25)


def test_only_a_model_trained_on_signal_heads_is_taken_to_find_them() -> None:
    """The vehicle-tuned nano keeps COCO's traffic light class and never finds
    one: lamps read only inside its heads would be none at all."""
    from core.perception.detection import Kind

    classes = [*COCO_HEAD, "boat", "traffic light"]
    plain = {"format": SPEC_FORMAT, "input_size": [448, 1344], "classes": classes}
    assert not ModelSpec.from_json(plain).signal_heads
    trained = ModelSpec.from_json({**plain, "signal_heads": True})
    assert trained.signal_heads
    order = list(Kind)
    assert order.index(Kind.TRAFFIC_LIGHT) in trained.kinds()


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_the_input_is_rgb_scaled_to_one_at_the_model_size() -> None:
    blue = np.zeros((30, 90, 3), dtype=np.uint8)
    blue[..., 0] = 255  # BGR
    tensor = preprocess(blue, SPEC)
    assert tensor.shape == (1, 3, 8, 24)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert np.allclose(tensor[0, 2], 1.0) and np.allclose(tensor[0, :2], 0.0)


def test_normalisation_is_applied_per_channel_when_the_model_wants_it() -> None:
    spec = ModelSpec((4, 4), COCO_HEAD, mean=(0.5, 0.0, 0.0), std=(0.5, 1.0, 2.0))
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    tensor = preprocess(white, spec)
    assert np.allclose(tensor[0, 0], 1.0)
    assert np.allclose(tensor[0, 1], 1.0)
    assert np.allclose(tensor[0, 2], 0.5)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


CROP = (100, 50, 500, 250)  # 400 x 200 pixels of the frame


def test_a_box_lands_in_frame_pixels_through_the_crop() -> None:
    [car] = decode(np.array([[0.5, 0.5, 0.25, 0.5]]), logits_for((CAR, 0.9)), KINDS, CROP)
    assert car.kind is Kind.CAR
    assert car.score == pytest.approx(0.9)
    assert car.box == pytest.approx((250.0, 100.0, 350.0, 200.0))
    assert car.ground_point == pytest.approx((300.0, 200.0))


def test_boxes_past_the_crop_edge_are_cut_at_it() -> None:
    [car] = decode(np.array([[0.95, 0.5, 0.2, 0.5]]), logits_for((CAR, 0.9)), KINDS, CROP)
    assert car.box[2] == pytest.approx(500.0)


def test_unsure_detections_and_other_things_are_dropped() -> None:
    boxes = np.array([[0.2, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]])
    found = decode(boxes, logits_for((CAR, 0.3), (TRAIN, 0.95), (TRUCK, 0.6)), KINDS, CROP)
    assert [d.kind for d in found] == [Kind.TRUCK]


def test_one_vehicle_seen_as_car_and_truck_is_kept_once_as_the_surer() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.505, 0.5, 0.2, 0.2]])
    found = decode(boxes, logits_for((CAR, 0.6), (TRUCK, 0.8)), KINDS, CROP)
    assert [(d.kind, round(d.score, 2)) for d in found] == [(Kind.TRUCK, 0.8)]


def test_two_vehicles_side_by_side_are_both_kept() -> None:
    """Overlapping, as a car half hidden behind another is, but not one object."""
    boxes = np.array([[0.45, 0.5, 0.2, 0.2], [0.55, 0.5, 0.2, 0.2]])
    found = decode(boxes, logits_for((CAR, 0.7), (CAR, 0.9)), KINDS, CROP)
    assert [round(d.score, 2) for d in found] == [0.9, 0.7]


def test_logits_for_another_model_are_refused() -> None:
    with pytest.raises(ValueError, match="class logits"):
        decode(np.zeros((1, 4)), np.zeros((1, 3)), KINDS, CROP)


# --------------------------------------------------------------------------- #
# The cab
# --------------------------------------------------------------------------- #


def test_a_region_outside_the_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="not a region"):
        Rect(0.5, 0.0, 1.2, 1.0)


def test_the_road_ahead_is_seen_and_the_mirrors_and_hud_are_not() -> None:
    view, width, height = DEFAULT_CAB_VIEW, 1920, 1080
    ahead = (1000, 450)
    assert view.windscreen.contains(*ahead, width, height)
    assert not view.is_covered(*ahead, width, height)
    for mirror in [(130, 240), (1780, 240), (120, 820), (1490, 630)]:
        assert view.is_covered(*mirror, width, height), mirror
    assert view.is_covered(1700, 900, width, height), "the navigation HUD"


def _fake_run(seen: list, boxes: np.ndarray, logits: np.ndarray):
    def run(tensor):
        seen.append(tensor)
        return boxes, logits

    return run


def test_the_model_is_shown_only_the_windscreen() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:60, 50:200] = 255
    seen: list = []
    view = CabView(windscreen=Rect(0.25, 0.2, 1.0, 0.6))
    detector = Detector(_fake_run(seen, np.zeros((0, 4)), np.zeros((0, 9))), SPEC, view)
    assert detector.detect(frame) == []
    assert np.allclose(seen[0], 1.0), "everything the model saw was the white windscreen"


def test_a_vehicle_in_a_mirror_is_not_ahead() -> None:
    view = CabView(windscreen=Rect(0.0, 0.0, 1.0, 1.0), covered=(Rect(0.75, 0.0, 1.0, 0.5),))
    boxes = np.array([[0.3, 0.5, 0.1, 0.1], [0.9, 0.2, 0.1, 0.1]])
    run = _fake_run([], boxes, logits_for((CAR, 0.9), (CAR, 0.8)))
    found = Detector(run, SPEC, view).detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [round(d.score, 2) for d in found] == [0.9]
    assert found[0].centre == pytest.approx((60.0, 50.0))


def test_a_big_vehicle_close_up_is_kept_at_a_lower_certainty() -> None:
    """A coach crossing ahead at night scored 0.25 to 0.58: gone under the
    threshold, the road was clear, and the truck drove into its side."""
    view = CabView(windscreen=Rect(0.0, 0.0, 1.0, 1.0))
    boxes = np.array([[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.4, 0.4]])
    run = _fake_run([], boxes, logits_for((TRUCK, 0.35), (CAR, 0.35), (TRUCK, 0.2)))
    found = Detector(run, SPEC, view).detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [round(d.score, 2) for d in found] == [0.35], (
        "the big one; not a small one, nor a faint one"
    )


# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #


def test_a_model_that_is_not_the_one_described_is_refused(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"another model")
    spec = {
        "format": SPEC_FORMAT,
        "input_size": [8, 24],
        "classes": COCO_HEAD,
        "sha256": hashlib.sha256(b"the model").hexdigest(),
    }
    (tmp_path / "model.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="is not the model"):
        Detector.open(tmp_path)


MODEL_DIR = REPO_ROOT / "data" / "models" / "dfine-nano"


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or not (MODEL_DIR / "model.onnx").exists(),
    reason="needs onnxruntime and a model from tools/export_detector.py",
)
def test_the_exported_model_runs_and_sees_nothing_in_an_empty_frame() -> None:
    detector = Detector.open(MODEL_DIR, providers=("CPUExecutionProvider",))
    assert detector.spec.licence == "apache-2.0"
    grey = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    assert detector.detect(grey) == []


def test_looks_at_one_frame_merge_into_one_box_per_object() -> None:
    wide = [
        Detection(Kind.CAR, 0.6, (100, 100, 140, 120)),
        Detection(Kind.TRUCK, 0.9, (0, 0, 50, 50)),
    ]
    zoomed = [
        Detection(Kind.CAR, 0.8, (101, 100, 141, 121)),
        Detection(Kind.CAR, 0.7, (300, 90, 306, 94)),
    ]
    merged = merge(wide, zoomed)
    assert [(d.kind, d.score) for d in merged] == [
        (Kind.TRUCK, 0.9),
        (Kind.CAR, 0.8),
        (Kind.CAR, 0.7),
    ], "the car both found is kept once, from the surer look; the far one only the zoom saw"


def test_a_car_alongside_is_not_lost_to_the_small_door_mirror() -> None:
    """What happened: a car close on the right, its centre on the door mirror,
    was dropped as if it were in the mirror."""
    view, width, height = DEFAULT_CAB_VIEW, 1920, 1080
    alongside = (1300.0, 520.0, 1700.0, 700.0)  # centre (1500, 610), on the right door mirror
    assert view.is_covered(1500, 610, width, height)
    assert not view.hides(alongside, width, height)
    in_the_inset = (1700.0, 200.0, 1760.0, 240.0)
    assert view.hides(in_the_inset, width, height)


def test_a_box_on_part_of_an_object_is_merged_into_the_whole() -> None:
    van = Detection(Kind.CAR, 0.8, (100, 100, 300, 260))
    roof = Detection(Kind.TRUCK, 0.9, (105, 100, 295, 170))
    beside = Detection(Kind.CAR, 0.7, (310, 150, 360, 190))
    assert merge([roof], [van, beside]) == [van, beside]


def test_a_vague_box_over_the_road_does_not_swallow_the_cars_in_it() -> None:
    """What happened: a 0.35 box over the whole windscreen removed a 0.87 car."""
    vague = Detection(Kind.CAR, 0.35, (330, 260, 1920, 690))
    car = Detection(Kind.CAR, 0.87, (805, 468, 890, 528))
    assert merge([vague, car]) == [car, vague]


def test_a_box_over_the_whole_windscreen_is_not_a_car() -> None:
    """What happened: a dark windscreen, and a band of sky, boxed as a car --
    which would be a car at the bumper."""
    boxes = np.array([[0.5, 0.5, 0.98, 0.9], [0.5, 0.5, 0.3, 0.3]])
    found = decode(boxes, logits_for((CAR, 0.8), (CAR, 0.7)), KINDS, CROP)
    assert [round(d.score, 2) for d in found] == [0.7]


# --------------------------------------------------------------------------- #
# Mirrors
# --------------------------------------------------------------------------- #


def test_a_crop_keeps_its_shape_at_the_left_of_a_grey_canvas() -> None:
    from core.perception.detection import kept_canvas

    crop = np.full((325, 262, 3), 255, dtype=np.uint8)
    canvas, used = kept_canvas(crop, (8, 24))
    assert canvas.shape == (8, 24, 3)
    assert used == round(262 * 8 / 325)
    assert (canvas[:, :used] == 255).all() and (canvas[:, used:] == 114).all()


def test_far_in_the_mirror_counts_vehicles_not_the_trailer_nor_big_boxes_nor_the_grey() -> None:
    """Stretched to the model's wide input, a car far back in the left mirror
    was found 0.7 s before it hit the truck; kept, 1.8 s."""
    from core.perception.detection import MIRROR_FAR_MAX

    used = round(262 * 8 / 325) / 24  # of the canvas, the share the mirror takes
    boxes = np.array(
        [
            [0.4 * used, 0.5, 0.05, 0.1],  # a car, in the lane on the left
            [0.8 * used, 0.5, 0.05, 0.2],  # the truck's own trailer (OWN_RIG)
            [0.3 * used, 0.5, 0.05, MIRROR_FAR_MAX + 0.1],  # a glow by night
            [0.5, 0.5, 0.05, 0.3],  # on the grey beside the mirror
        ]
    )
    run = _fake_run([], boxes, logits_for(*[(CAR, 0.9)] * 4))
    detector = Detector(run, SPEC, DEFAULT_CAB_VIEW)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert detector.far_in_mirror(frame, -1) == pytest.approx(0.1)
