"""Tests for recording and replay.

The cases that matter are the ugly ones: a recording whose writer was killed, a
recording from an older plugin, and a replay that has to behave identically to
the live path. Those are the situations in which a recording is most valuable
and most likely to be quietly wrong.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE, BLOCK_SIZE, LAYOUT_VERSION, MAGIC
from core.bridge.reader import TelemetryReader, TelemetrySnapshot
from core.recording import (
    Recorder,
    RecordingFormatError,
    ReplayBackend,
    open_recording,
    replay_frames,
)
from core.recording.format import MANIFEST_FILE, TELEMETRY_FILE, TELEMETRY_RECORD_DTYPE
from pal import PalNotConnectedError


def make_snapshot(index: int, speed: float = 0.0, t: float = 0.0) -> TelemetrySnapshot:
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    header = block["header"]
    header["magic"] = MAGIC
    header["layout_version"] = LAYOUT_VERSION
    header["block_size"] = BLOCK_SIZE
    header["write_count"] = index + 1
    header["sequence"] = (index + 1) * 2
    block["truck"]["speed"] = speed
    return TelemetrySnapshot(block=block[0], received_at=t, attempts=1)


def write_recording(path: Path, frames: int = 10, notes: str = "") -> Path:
    with Recorder(path, notes=notes) as recorder:
        for index in range(frames):
            recorder.write_telemetry(make_snapshot(index, speed=index * 1.5, t=index * 0.016))
    return path


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_round_trip_preserves_every_frame(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=10))
    assert len(recording.telemetry) == 10
    assert list(recording.telemetry["block"]["truck"]["speed"]) == pytest.approx(
        [i * 1.5 for i in range(10)]
    )


def test_timeline_starts_at_the_first_record(tmp_path: Path) -> None:
    """Setup time before the first frame must not appear as dead air."""
    with Recorder(tmp_path / "run") as recorder:
        recorder.write_telemetry(make_snapshot(0, t=1234.5))
        recorder.write_telemetry(make_snapshot(1, t=1234.6))
    recording = open_recording(tmp_path / "run")
    assert recording.telemetry["t"][0] == pytest.approx(0.0)
    assert recording.telemetry["t"][1] == pytest.approx(0.1)


def test_manifest_records_notes_and_counts(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=5, notes="A2 east"))
    assert recording.manifest["notes"] == "A2 east"
    assert recording.manifest["telemetry"]["count"] == 5
    assert recording.truncated is False


def test_refuses_to_overwrite_an_existing_recording(tmp_path: Path) -> None:
    """Appending to somebody else's run, or silently replacing one, are both
    worse than failing here."""
    write_recording(tmp_path / "run")
    with pytest.raises(FileExistsError):
        Recorder(tmp_path / "run").open()


def test_frame_at_finds_the_frame_in_effect(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=10))
    assert float(recording.frame_at(0.0)["block"]["truck"]["speed"]) == pytest.approx(0.0)
    # 0.05 s falls between frames 3 (0.048) and 4 (0.064).
    assert float(recording.frame_at(0.05)["block"]["truck"]["speed"]) == pytest.approx(4.5)
    assert float(recording.frame_at(999.0)["block"]["truck"]["speed"]) == pytest.approx(13.5)


# --------------------------------------------------------------------------- #
# Damage and version mismatches
# --------------------------------------------------------------------------- #


def test_a_killed_writer_still_leaves_a_readable_recording(tmp_path: Path) -> None:
    """The run somebody most wants to look at is the one that crashed."""
    path = write_recording(tmp_path / "run", frames=10)

    # Simulate a process killed part-way through writing record 11.
    stream = path / TELEMETRY_FILE
    with stream.open("ab") as handle:
        handle.write(b"\x00" * (TELEMETRY_RECORD_DTYPE.itemsize // 3))

    recording = open_recording(path)
    assert len(recording.telemetry) == 10
    assert float(recording.telemetry["block"]["truck"]["speed"][-1]) == pytest.approx(13.5)


class SlowDisk:
    """A file whose writes wait, as a laptop's hard disk did behind the game's."""

    def __init__(self, stream, wait_s: float) -> None:
        self._stream = stream
        self._wait = wait_s

    def write(self, data: bytes) -> int:
        time.sleep(self._wait)
        return self._stream.write(data)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def test_a_slow_disk_does_not_hold_up_the_writer_s_caller(tmp_path: Path) -> None:
    """On Windows a write waiting on the disk held the control loop up 0.9 s,
    and the plugin stopped the truck: the files are written on a thread."""
    recorder = Recorder(tmp_path / "run")
    recorder.open()
    recorder._telemetry = SlowDisk(recorder._telemetry, 0.2)
    began = time.perf_counter()
    for index in range(5):
        recorder.write_telemetry(make_snapshot(index, speed=index * 1.5, t=index * 0.016))
    handed_over = time.perf_counter() - began
    recorder.close()
    assert handed_over < 0.1
    assert len(open_recording(tmp_path / "run").telemetry) == 5, "and all of it is written"


def test_a_write_that_failed_is_told_by_the_next(tmp_path: Path) -> None:
    class FullDisk:
        def write(self, data: bytes) -> int:
            raise OSError(28, "No space left on device")

        def __getattr__(self, name: str):
            return lambda *args: None

    recorder = Recorder(tmp_path / "run")
    recorder.open()
    stream, recorder._telemetry = recorder._telemetry, FullDisk()
    recorder.write_telemetry(make_snapshot(0))
    deadline = time.perf_counter() + 2.0
    with pytest.raises(OSError, match="No space"):
        while time.perf_counter() < deadline:
            recorder.write_telemetry(make_snapshot(1))
            time.sleep(0.01)
    recorder._telemetry = stream
    recorder.close()


def test_counts_come_from_the_file_not_the_manifest(tmp_path: Path) -> None:
    """A writer killed before finalising never updated its counts."""
    path = write_recording(tmp_path / "run", frames=8)
    manifest_path = path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry"]["count"] = 0  # as an interrupted writer would leave it
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    recording = open_recording(path)
    assert len(recording.telemetry) == 8


def test_truncated_flag_reports_a_manifest_disagreement(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "run", frames=8)
    manifest_path = path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry"]["count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert open_recording(path).truncated is True


def test_rejects_a_recording_from_a_different_block_layout(tmp_path: Path) -> None:
    """Reinterpreting old blocks under the current layout would produce
    numbers rather than errors."""
    path = write_recording(tmp_path / "run", frames=4)
    manifest_path = path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry"]["layout_version"] = LAYOUT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RecordingFormatError, match="layout v"):
        open_recording(path)


def test_rejects_a_future_recording_format(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "run", frames=4)
    manifest_path = path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RecordingFormatError, match="format v99"):
        open_recording(path)


def test_rejects_a_directory_that_is_not_a_recording(tmp_path: Path) -> None:
    (tmp_path / "not-a-run").mkdir()
    with pytest.raises(RecordingFormatError, match="not a recording"):
        open_recording(tmp_path / "not-a-run")


def test_rejects_a_corrupt_manifest(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "run", frames=2)
    (path / MANIFEST_FILE).write_text("{ broken", encoding="utf-8")
    with pytest.raises(RecordingFormatError):
        open_recording(path)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def test_replay_runs_through_the_live_reader(tmp_path: Path) -> None:
    """Replay must exercise the production path, not a parallel one: the same
    seqlock check, the same layout validation, the same parsing."""
    path = write_recording(tmp_path / "run", frames=6)
    speeds = [float(s.block["truck"]["speed"]) for s in replay_frames(path)]
    assert speeds == pytest.approx([i * 1.5 for i in range(6)])


def test_replay_is_deterministic(tmp_path: Path) -> None:
    """Repeatable runs are the point; a replay that varies is not a fixture."""
    path = write_recording(tmp_path / "run", frames=6)
    first = [s.write_count for s in replay_frames(path)]
    second = [s.write_count for s in replay_frames(path)]
    assert first == second


def test_replay_can_start_part_way_through(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "run", frames=10)
    speeds = [float(s.block["truck"]["speed"]) for s in replay_frames(path, start=0.05)]
    assert speeds[0] == pytest.approx(4.5)


def test_replay_backend_serves_the_recording_as_shared_memory(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=4))
    backend = ReplayBackend(recording)
    with TelemetryReader(backend, stale_after=None) as reader:
        assert float(reader.read().block["truck"]["speed"]) == pytest.approx(0.0)
        backend.replay_reader.advance()
        assert float(reader.read().block["truck"]["speed"]) == pytest.approx(1.5)


def test_replay_reports_the_end_rather_than_repeating(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=2))
    backend = ReplayBackend(recording)
    with TelemetryReader(backend, stale_after=None) as reader:
        reader.read()
        assert backend.replay_reader.advance() is True
        reader.read()
        assert backend.replay_reader.advance() is False
        with pytest.raises(PalNotConnectedError, match="end of"):
            reader.read()


def test_replay_can_loop(tmp_path: Path) -> None:
    """Long-running tests want a recording they can keep feeding."""
    recording = open_recording(write_recording(tmp_path / "run", frames=3))
    backend = ReplayBackend(recording, loop=True)
    source = backend.replay_reader
    source.open()
    for _ in range(7):
        assert source.advance() is True
    assert source.index < 3


def test_replaying_an_empty_recording_fails_clearly(tmp_path: Path) -> None:
    with Recorder(tmp_path / "run"):
        pass
    recording = open_recording(tmp_path / "run")
    backend = ReplayBackend(recording)
    with pytest.raises(PalNotConnectedError, match="no telemetry"):
        backend.replay_reader.open()


def test_commander_records_what_it_published(tmp_path: Path) -> None:
    """Telemetry shows what the truck did; the command stream shows what the
    stack asked for. The gap between them is usually where the bug is."""
    from core.bridge.command import Commander, ControlCommand
    from pal.fake import FakeBackend

    with (
        Recorder(tmp_path / "run") as recorder,
        Commander(FakeBackend(), recorder=recorder) as commander,
    ):
        commander.send(ControlCommand(steer=0.25, throttle=0.5))
        commander.send(ControlCommand(steer=-0.25, brake=0.75))

    recording = open_recording(tmp_path / "run")
    # open() and close() each publish a neutral frame as well.
    axes = recording.commands["block"]["axes"]
    assert len(axes) == 4
    assert axes[1] == pytest.approx([0.25, 0.5, 0.0, 0.0])
    assert axes[2] == pytest.approx([-0.25, 0.0, 0.75, 0.0])


def test_command_stream_shares_the_telemetry_timeline(tmp_path: Path) -> None:
    """Commands and telemetry are only comparable if their clocks are."""
    from core.bridge.command import Commander, ControlCommand
    from pal.fake import FakeBackend

    with Recorder(tmp_path / "run") as recorder:
        recorder.write_telemetry(make_snapshot(0, t=recorder.duration))
        with Commander(FakeBackend(), recorder=recorder) as commander:
            commander.send(ControlCommand(throttle=0.3))

    recording = open_recording(tmp_path / "run")
    assert recording.commands["t"][0] >= recording.telemetry["t"][0]


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _picture(seed: int) -> np.ndarray:
    """Smooth enough that JPEG keeps it close; different for every seed."""
    y, x = np.mgrid[0:48, 0:64]
    image = np.stack([x * 2 + seed * 40, y * 3, (x + y) + seed * 70], axis=-1)
    return (image % 256).astype(np.uint8)


def test_images_come_back_on_the_telemetry_timeline(tmp_path: Path) -> None:
    from pal.base import Frame

    with Recorder(tmp_path / "run") as recorder:
        recorder.write_telemetry(make_snapshot(0, t=10.0))
        recorder.write_frame(Frame(_picture(1), timestamp=10.02, sequence=7))
        recorder.write_telemetry(make_snapshot(1, t=10.016))
        recorder.write_telemetry(make_snapshot(2, t=10.033))
        recorder.write_frame(Frame(_picture(2), timestamp=10.05, sequence=9))

    recording = open_recording(tmp_path / "run")
    assert list(recording.frames["t"]) == pytest.approx([0.02, 0.05])
    assert list(recording.frames["sequence"]) == [7, 9], "the gap shows a grab not recorded"
    assert (recording.frames["width"][0], recording.frames["height"][0]) == (64, 48)
    second = recording.image(1)
    assert second.shape == (48, 64, 3)
    assert np.abs(second.astype(int) - _picture(2)).mean() < 3, "JPEG, but the same picture"
    assert np.abs(second.astype(int) - _picture(1)).mean() > 3, "and not the other one"
    # The telemetry in effect when the image was taken.
    in_effect = recording.frame_at(float(recording.frames["t"][1]))
    assert in_effect["block"]["header"]["write_count"] == 3
    assert json.loads((tmp_path / "run" / MANIFEST_FILE).read_text())["frames"]["count"] == 2


def test_a_recording_without_images_has_no_frames(tmp_path: Path) -> None:
    recording = open_recording(write_recording(tmp_path / "run", frames=3))
    assert len(recording.frames) == 0
    with pytest.raises(IndexError):
        recording.image(0)


def test_a_recording_from_before_images_existed_still_opens(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "run", frames=3)
    (path / "frames.bin").unlink()
    (path / "images").rmdir()
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    del manifest["frames"]
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    recording = open_recording(path)
    assert len(recording.telemetry) == 3 and len(recording.frames) == 0


def test_a_missing_image_is_an_error_not_a_black_frame(tmp_path: Path) -> None:
    from pal.base import Frame

    with Recorder(tmp_path / "run") as recorder:
        recorder.write_frame(Frame(_picture(0), timestamp=1.0, sequence=1))
    (tmp_path / "run" / "images" / "000000.jpg").unlink()
    with pytest.raises(RecordingFormatError, match="missing"):
        open_recording(tmp_path / "run").image(0)


def test_a_view_encoded_elsewhere_is_recorded_on_the_drive_s_timeline(tmp_path: Path) -> None:
    """The perception child encodes the views kept for training; the
    recorder writes them as its own frames."""
    import cv2

    image = np.zeros((48, 64, 3), np.uint8)
    image[10:20, 10:30] = (0, 0, 255)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    with Recorder(tmp_path / "run") as recorder:
        recorder.write_telemetry(make_snapshot(0, t=100.0))
        recorder.write_encoded_frame(jpeg.tobytes(), 100.5, 7, 64, 48)
    recording = open_recording(tmp_path / "run")
    assert len(recording.frames) == 1
    assert recording.frames["t"][0] == pytest.approx(0.5)
    assert recording.frames["sequence"][0] == 7
    read = recording.image(0)
    assert read.shape == (48, 64, 3)
    assert read[15, 20, 2] > 200, "the red patch where it was"
