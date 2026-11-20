"""Perception in its own process: real process, fake window, fake model."""

from __future__ import annotations

import time

import pytest

from core.perception.camera import GroundCamera
from core.perception.detection import Detection, Kind
from core.perception.service import PerceptionService
from core.perception.tracking import lead_vehicle
from pal.base import WindowGeometry, WindowNotInFrontError
from pal.fake import FakeScreenCapture

CAMERA = GroundCamera(focal_px=820.0, horizon_row=438.0, centre_col=960.0, height_m=2.9)


class CarAhead:
    """A 'model' that sees a car 40 m ahead, whatever the image."""

    def detect(self, image):
        scale = CAMERA.focal_px / 40.0
        bottom = CAMERA.horizon_row + CAMERA.focal_px * CAMERA.height_m / 40.0
        return [Detection(Kind.CAR, 0.9, (960 - scale, bottom - 1.5 * scale, 960 + scale, bottom))]


class Covered(FakeScreenCapture):
    def grab(self):
        raise WindowNotInFrontError("a browser is in front")


class Broken:
    def __init__(self) -> None:
        raise OSError("no GPU here")


def small_window() -> FakeScreenCapture:
    return FakeScreenCapture(WindowGeometry(0, 0, 64, 48))


def full_window() -> FakeScreenCapture:
    return FakeScreenCapture(WindowGeometry(0, 0, 1920, 1080))


def covered_window() -> Covered:
    return Covered(WindowGeometry(0, 0, 64, 48))


def wait_for(service: PerceptionService, condition, timeout: float = 5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        report = service.latest()
        if report is not None and condition(report):
            return report
        time.sleep(0.02)
    raise AssertionError("no such report in time")


def test_a_car_ahead_arrives_as_a_confirmed_lead_with_its_distance() -> None:
    with PerceptionService(small_window, CarAhead, CAMERA, rate_hz=50.0) as service:
        report = wait_for(service, lambda r: any(t.confirmed for t in r.tracks))
        _, (distance, closing, lateral, _) = lead_vehicle(report.tracks)
        assert distance == pytest.approx(40.0, rel=0.01)
        assert abs(closing) < 0.5 and abs(lateral) < 0.1
        assert time.perf_counter() - report.grabbed_at < 1.0, "fresh, on the caller's clock"


def test_a_covered_window_is_reported_not_hidden() -> None:
    with PerceptionService(covered_window, CarAhead, CAMERA, rate_hz=50.0) as service:
        report = wait_for(service, lambda r: True)
        assert report.covered and report.tracks == ()
        assert report.covered_why == "a browser is in front", "and why, as the capture said"


def test_a_child_that_cannot_start_says_why() -> None:
    service = PerceptionService(small_window, Broken, CAMERA)
    with pytest.raises(RuntimeError, match="no GPU here"):
        service.start()
    assert not service.alive


class OnTheGpu(CarAhead):
    provider = "CUDAExecutionProvider"


def test_the_child_says_where_its_model_runs() -> None:
    """What happened: CUDA died with a driver fault, ONNX Runtime fell back to
    the CPU in silence, and a drive went ahead on three times slower sight."""
    with PerceptionService(small_window, OnTheGpu, CAMERA) as service:
        assert service.provider == "CUDAExecutionProvider"
    with PerceptionService(small_window, CarAhead, CAMERA) as service:
        assert service.provider == ""


def test_a_snapshot_asked_for_while_nothing_reads_the_reports_is_still_written(tmp_path) -> None:
    """After a hit the drive stops reading the reports; the pipe filled, the
    process blocked on it, was killed, and the black box was never written."""
    service = PerceptionService(small_window, CarAhead, CAMERA, rate_hz=200.0, black_box=tmp_path)
    service.start()
    try:
        wait_for(service, lambda r: True)
        time.sleep(3.0)  # nothing read: the pipe fills
        service.snapshot("hit-0001.0")
        # Written by the child in its own time: stopped at once, a loaded
        # runner stopped it first now and then. A child blocked on the pipe
        # never writes, and this still fails.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not any((tmp_path / "hit-0001.0").glob("*.jpg")):
            time.sleep(0.1)
    finally:
        service.stop()
    assert any((tmp_path / "hit-0001.0").glob("*.jpg"))


def test_the_newest_report_is_fresh_after_the_caller_was_busy() -> None:
    """What happened on Windows: follow.py set up for 1.25 s after perception
    started without reading it, the child stood blocked on the full pipe all
    that time, and the loop's first report was 1.25 s old -- the drive stopped
    before it began. A full-size view's report carries the minimap's masks,
    23 kB: on Windows the pipe holds less than one."""
    with PerceptionService(full_window, CarAhead, CAMERA, rate_hz=200.0) as service:
        wait_for(service, lambda r: True)
        time.sleep(1.5)  # busy elsewhere: nothing read
        report = service.latest()
        assert report is not None
        assert time.perf_counter() - report.grabbed_at < 0.3


def test_the_child_starts_as_windows_starts_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no fork: the child is spawned, a fresh interpreter, and
    everything handed to it must survive being pickled."""
    import multiprocessing

    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    try:
        with PerceptionService(small_window, CarAhead, CAMERA, rate_hz=50.0) as service:
            report = wait_for(service, lambda r: any(t.confirmed for t in r.tracks), timeout=20.0)
            assert report.tracks
    finally:
        multiprocessing.set_start_method(previous, force=True)


def test_views_are_kept_only_while_asked_for() -> None:
    import itertools

    import cv2
    import numpy as np

    from core.perception.service import KeptFrame

    with PerceptionService(small_window, CarAhead, CAMERA, rate_hz=50.0) as service:
        wait_for(service, lambda r: True)
        time.sleep(0.2)
        assert service.kept() == [], "none unless asked"
        service.keep_frames(0.05)
        deadline = time.perf_counter() + 5.0
        kept: list[KeptFrame] = []
        while len(kept) < 3 and time.perf_counter() < deadline:
            kept += service.kept()
            time.sleep(0.02)
        service.keep_frames(0.0)
        time.sleep(0.2)
        service.kept()
        time.sleep(0.3)
        assert service.kept() == [], "stopped when told"
    assert len(kept) >= 3
    first = kept[0]
    assert (first.width, first.height) == (64, 48)
    image = cv2.imdecode(np.frombuffer(first.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (48, 64, 3)
    assert all(b.timestamp - a.timestamp >= 0.04 for a, b in itertools.pairwise(kept))


def test_held_it_looks_at_nothing_until_resumed() -> None:
    """Waiting for the beacon, the model stays loaded and the GPU is the game's."""
    with PerceptionService(small_window, CarAhead, CAMERA, rate_hz=50.0) as service:
        wait_for(service, lambda r: True)
        service.hold()
        time.sleep(0.2)
        held = service.latest()
        time.sleep(0.3)
        assert service.latest() is held, "a new report while held"
        service.resume()
        fresh = wait_for(service, lambda r: r is not held)
        assert time.perf_counter() - fresh.grabbed_at < 1.0


def test_a_slow_turn_of_perception_is_told_part_by_part() -> None:
    """On a motorway no view came for over a second, and nothing said where
    the time had gone."""
    from core.perception.service import _slow_turn

    said = _slow_turn(10.0, [("grab", 10.85), ("detect", 10.9), ("the rest", 10.92)])
    assert said == "0.92 s: grab 0.85 s, detect 0.05 s, the rest 0.02 s"


def test_the_mirror_is_looked_at_one_way_then_the_other() -> None:
    """Watching the mirror costs one look in two frames, as it did."""
    from types import SimpleNamespace

    import numpy as np

    from core.perception.service import _far_in_mirror, _in_mirror

    detector = SimpleNamespace(
        nearest_in_mirror=lambda _image, _side: 0.25, far_in_mirror=lambda _image, _side: 0.05
    )
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    side = SimpleNamespace(value=-1)
    looks = []
    for sequence in range(8):
        frame = SimpleNamespace(sequence=sequence, image=image)
        looks.append((_in_mirror(detector, frame, side), _far_in_mirror(detector, frame, side)))
    assert sum(near is not None for near, _ in looks) == 2
    assert sum(far is not None for _, far in looks) == 2
    assert all(near is None or far is None for near, far in looks), "one look a frame"
    assert (-1, 0.05) in [far for _, far in looks]
    none = SimpleNamespace(value=0)
    frame = SimpleNamespace(sequence=0, image=image)
    assert (
        _in_mirror(detector, frame, none) is None and _far_in_mirror(detector, frame, none) is None
    )
