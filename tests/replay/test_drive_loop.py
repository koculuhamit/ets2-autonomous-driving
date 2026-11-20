"""The drive's whole loop, run on a recorded drive instead of the game.

tools/follow.py's loop is some 1800 lines that no unit test runs: a name
reused for the trailer's geometry stopped six drives in their first seconds
while every unit test passed (15 November). Here the loop runs for real --
reader, commander, planner, controllers, lights -- against a recording
served through the replay backend, paced by the wall clock, the truck's
pedals and wheel read back as the game would show the stack's commands.
The truck goes where the recording went, whatever is commanded: this finds
a loop that breaks, not a drive that goes wrong.

Needs a recorded drive with its plan and the map it was planned on, which
are not in the repository: skipped without them. `ETS2AD_REPLAY_DRIVE` names
a drive's folder; otherwise the newest under data/missions is taken.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSIONS = REPO_ROOT / "data" / "missions"
LOOP_S = 6.0


def _drive() -> Path | None:
    named = os.environ.get("ETS2AD_REPLAY_DRIVE")
    candidates = [Path(named)] if named else sorted(MISSIONS.glob("*/drive-*"), reverse=True)
    for drive in candidates:
        plan = drive.parent / f"{drive.name}.npz"
        manifest = drive / "manifest.json"
        if not (plan.is_file() and manifest.is_file() and (drive / "telemetry.bin").is_file()):
            continue
        if json.loads(manifest.read_text()).get("duration", 0.0) < 60.0:
            continue
        with np.load(plan) as saved:
            if "metadata" not in saved.files:
                continue
            cache = Path(json.loads(str(saved["metadata"])).get("cache", ""))
        if (cache / "manifest.json").is_file():
            return drive
    return None


DRIVE = _drive()


def _start_s(drive: Path) -> float:
    """Where in the drive to begin: a second before the first vehicle it
    followed after 20 s, so that following is gone through; else 20 s."""
    acc = drive / "acc.npy"
    if acc.is_file():
        log = np.load(acc)
        led = log[(log[:, 0] >= 20.0) & np.isfinite(log[:, 3])]
        if len(led):
            return max(float(led[0, 0]) - 1.0, 0.0)
    return 20.0


def _led_in(drive: Path, start: float, seconds: float) -> bool:
    """Whether the recorded drive followed a vehicle in that stretch."""
    acc = drive / "acc.npy"
    if not acc.is_file():
        return False
    log = np.load(acc)
    stretch = log[(log[:, 0] >= start) & (log[:, 0] <= start + seconds)]
    return bool(np.isfinite(stretch[:, 3]).any())


pytestmark = pytest.mark.skipif(
    DRIVE is None, reason="no recorded drive with its plan and map on this machine"
)


@pytest.fixture
def replayed(monkeypatch, tmp_path):
    """The PAL backend serving `DRIVE`, from 20 s in, as the game would."""
    import pal
    from core.bridge import command as command_module
    from core.bridge import reader as reader_module
    from core.bridge.command import COMMAND_BLOCK_DTYPE, COMMAND_MEMORY_NAME
    from core.bridge.layout import BLOCK_DTYPE
    from core.recording.reader import open_recording
    from core.recording.replay import ReplayBackend

    recording = open_recording(DRIVE)
    backend = ReplayBackend(recording, root=tmp_path / "pal")
    replay = backend.replay_reader
    begin = float(recording.telemetry["t"][0]) + _start_s(DRIVE)
    # The wall clock, shared with the perception child (forked): its views
    # and the telemetry come from the same moment of the drive.
    started = time.time()
    backend.drive_time = lambda: begin + time.time() - started
    backend.recording = recording
    served = replay.read

    def as_the_game_shows_it() -> bytes:
        # Paced by the wall clock, not stepped per read: the reader reads a
        # block twice to check it is whole.
        replay.seek_time(backend.drive_time())
        block = np.frombuffer(bytearray(served()), dtype=BLOCK_DTYPE)[0]
        writer = backend._writers.get(COMMAND_MEMORY_NAME)
        steer = throttle = brake = 0.0
        if writer is not None:
            command = np.frombuffer(bytes(writer.buffer), dtype=COMMAND_BLOCK_DTYPE)[0]
            steer, throttle, brake = (float(v) for v in command["axes"][:3])
        # The pedals and wheel the stack sent, not the recorded drive's:
        # else they read as the driver's foot, and the truck is handed over.
        truck = block["truck"]
        for name, value in (("steering", steer), ("throttle", throttle), ("brake", brake)):
            truck[f"input_{name}"] = value
            truck[f"effective_{name}"] = value
        return block.tobytes()

    replay.read = as_the_game_shows_it
    for module in (pal, reader_module, command_module):
        monkeypatch.setattr(module, "get_backend", lambda *_a, **_k: backend)
    return backend


class RecordedView:
    """The game window as the recording kept it (`--keep-frames`): the view
    kept last before the drive's moment now."""

    def __init__(self, backend) -> None:
        from core.recording.format import IMAGES_DIR

        self._backend = backend
        self._images = backend.recording.path / IMAGES_DIR
        self._times = backend.recording.frames["t"]
        self._shown: tuple[int, np.ndarray] | None = None
        self._sequence = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def grab(self):
        import cv2

        from core.recording.format import image_name
        from pal.base import Frame

        index = max(int(np.searchsorted(self._times, self._backend.drive_time())) - 1, 0)
        if self._shown is None or self._shown[0] != index:
            self._shown = (index, cv2.imread(str(self._images / image_name(index))))
        self._sequence += 1
        time.sleep(1.0 / 30.0)
        return Frame(image=self._shown[1], timestamp=time.perf_counter(), sequence=self._sequence)


def _drive_loop(monkeypatch, tmp_path, *extra: str):
    """tools/follow.py's main on `DRIVE`'s plan for `LOOP_S`: its exit code,
    its outcome file, and how long it took."""
    import tools.follow as follow

    # The recorded truck does not stop when told to; waiting 25 s for it
    # tells nothing about the loop.
    monkeypatch.setattr(follow, "_brake_to_rest", lambda *_a, **_k: None)
    outcome = tmp_path / "outcome.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "follow.py",
            str(DRIVE.parent / f"{DRIVE.name}.npz"),
            # As tools/mission.py drives: the recorded truck goes as fast as
            # it went, and a lower setting is a safety stop for overspeed.
            "--speed",
            "90",
            "--distance",
            "10000000",
            "--duration",
            str(LOOP_S),
            "--yes",
            "--output",
            str(tmp_path / "drive"),
            "--outcome",
            str(outcome),
            *extra,
        ],
    )
    began = time.perf_counter()
    code = follow.main()
    return code, outcome, time.perf_counter() - began


def test_the_loop_drives_a_recorded_route_until_its_time_is_up(
    replayed, monkeypatch, tmp_path
) -> None:
    code, outcome, took = _drive_loop(monkeypatch, tmp_path)
    reason = json.loads(outcome.read_text())
    # The recorded truck goes its own way: stopped at a light while the loop
    # pulls away, the guard stops it for something in the way (the Iveco's
    # drive to Uppsala, 16 November). A safety stop is the loop working.
    assert reason["reason"] in ("duration", "unsafe"), reason
    assert code == (0 if reason["reason"] == "duration" else 1)
    assert "Traceback" not in reason["detail"]
    assert took < LOOP_S + 30.0
    messages = (tmp_path / "drive" / "messages.txt").read_text()
    assert "Traceback" not in messages
    manifest = json.loads((tmp_path / "drive" / "manifest.json").read_text())
    assert manifest["telemetry"]["count"] > 100, "the loop went round"


@pytest.mark.skipif(
    DRIVE is None or not (DRIVE / "images").is_dir() or not any((DRIVE / "images").iterdir()),
    reason="the drive kept no views of the road (--keep-frames)",
)
def test_the_loop_drives_with_perception_on_the_views_the_drive_kept(
    replayed, monkeypatch, tmp_path
) -> None:
    """With --acc: the perception child on the recorded views, the cruise,
    the lights read and stopped for, the lane changes gated."""
    import tools.follow as follow

    monkeypatch.setattr(follow, "game_window", lambda: RecordedView(replayed))
    _, outcome, _ = _drive_loop(monkeypatch, tmp_path, "--acc", "--allow-cpu")
    reason = json.loads(outcome.read_text())
    assert reason["reason"] in ("duration", "unsafe"), reason
    if reason["reason"] == "unsafe":
        # The recorded truck goes its own way: a safety stop is the loop
        # working. A traceback is not.
        assert "Traceback" not in reason["detail"]
    messages = (tmp_path / "drive" / "messages.txt").read_text()
    assert "Traceback" not in messages, messages[-2000:]
    acc = np.load(tmp_path / "drive" / "acc.npy")
    if _led_in(DRIVE, _start_s(DRIVE), LOOP_S):
        # The drive followed a vehicle here; seen through this truck's own
        # camera, so must the loop. On the Actros's, a DAF XD saw the lorry
        # ahead too wide to be one and ran into it (15 November).
        assert np.isfinite(acc[:, 3]).any(), "the vehicle the drive followed was not followed"
