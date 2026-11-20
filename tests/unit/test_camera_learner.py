"""Learning a new truck's camera from its driver's driving."""

from __future__ import annotations

import time
from pathlib import Path

from tools.camera_learner import CameraLearner


class _Recorder:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.running = True
        (folder / "images").mkdir(parents=True)
        (folder / "images" / "000000.jpg").write_bytes(b"jpeg")
        (folder / "manifest.json").write_text("{}")

    def poll(self):
        return None if self.running else 0

    def send_signal(self, _signal) -> None:
        self.running = False

    def wait(self, timeout=None) -> int:
        return 0


def _learner(tmp_path: Path, answers: list):
    launched, saved = [], []

    def launch(command, **_kwargs):
        launched.append(_Recorder(Path(command[2])))
        return launched[-1]

    def calibrate(_cache, chunks):
        return answers.pop(0) if answers else (None, "more")

    learner = CameraLearner(
        "daf.vehicle.daf.xf_electric",
        Path("map"),
        root=tmp_path,
        chunk_s=0.0,  # every chunk ran its time
        launch=launch,
        calibrate=calibrate,
        save=lambda truck, document, chunks: saved.append((truck, document, chunks)),
    )
    return learner, launched, saved


def _settle(learner: CameraLearner) -> None:
    while learner._working is not None and learner._working.is_alive():
        time.sleep(0.01)


def test_a_chunk_is_recorded_then_the_camera_worked_out_from_it(tmp_path: Path) -> None:
    learner, launched, saved = _learner(tmp_path, [(None, "3 of 40 views of a straight road")])
    learner.step()
    assert len(launched) == 1
    assert launched[0].folder.parent == tmp_path / "daf.vehicle.daf.xf_electric"
    learner.step()
    assert len(launched) == 1, "one chunk at a time"
    launched[0].running = False
    learner.step()
    _settle(learner)
    assert not learner.learned and "straight road" in learner.why
    learner.step()
    assert len(launched) == 2, "not enough: another chunk"
    assert saved == []


def test_once_enough_the_camera_is_kept_and_nothing_more_recorded(tmp_path: Path) -> None:
    camera = {"focal_px": 1086.0, "horizon_row": 390.2, "height_m": 2.45}
    learner, launched, saved = _learner(tmp_path, [(camera, "")])
    learner.step()
    launched[0].running = False
    learner.step()
    _settle(learner)
    assert learner.learned and learner.why == ""
    assert saved == [("daf.vehicle.daf.xf_electric", camera, 1)]
    learner.step()
    assert len(launched) == 1


def test_stopping_ends_the_chunk_being_recorded(tmp_path: Path) -> None:
    learner, launched, _ = _learner(tmp_path, [])
    learner.step()
    learner.stop()
    assert not launched[0].running


def test_a_chunk_that_ends_at_once_is_not_started_again_at_once(tmp_path: Path) -> None:
    """The game not running, tools/record.py gives up at once: tried again
    after `RETRY_S`, not on every turn of the mission's loop."""
    from tools.camera_learner import RETRY_S

    now = [0.0]
    launched = []

    def launch(command, **_kwargs):
        launched.append(_Recorder(Path(command[2])))
        return launched[-1]

    learner = CameraLearner(
        "scania.vehicle.scania.r_2016",
        Path("map"),
        root=tmp_path,
        chunk_s=300.0,
        launch=launch,
        calibrate=lambda *_a: (None, "more"),
        save=lambda *_a: None,
        clock=lambda: now[0],
    )
    learner.step()
    launched[0].running = False
    now[0] = 1.0
    learner.step()
    _settle(learner)
    now[0] = 2.0
    learner.step()
    assert len(launched) == 1
    now[0] = 1.0 + RETRY_S + 0.1
    learner.step()
    assert len(launched) == 2
