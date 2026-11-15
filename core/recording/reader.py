"""Reading recordings back.

Two ways in, for two different jobs.

`Recording` gives the whole session as memory-mapped arrays, for analysis:
plotting a run, fitting a model, checking a metric. Nothing is loaded until it
is touched, so opening a two-hour recording costs the same as opening a
two-second one.

`ReplaySharedMemoryReader` presents the same recording through the PAL's
shared-memory interface, so `TelemetryReader` and everything above it run
against a file exactly as they run against the game. That is the whole point:
offline tests exercise the real reader, the real seqlock check and the real
parsing, rather than a second implementation that agrees with it today.

Counts come from file size rather than the manifest. A recording whose writer
was killed never got its final manifest, and that is precisely the run somebody
will want to look at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from core.bridge.command import COMMAND_LAYOUT_VERSION
from core.bridge.layout import LAYOUT_VERSION
from core.recording.format import (
    COMMAND_RECORD_DTYPE,
    COMMANDS_FILE,
    FORMAT_NAME,
    FORMAT_VERSION,
    FRAME_RECORD_DTYPE,
    FRAMES_FILE,
    IMAGES_DIR,
    MANIFEST_FILE,
    TELEMETRY_FILE,
    TELEMETRY_RECORD_DTYPE,
    RecordingFormatError,
    image_name,
)
from pal.base import PalNotConnectedError, SharedMemoryReader

__all__ = ["Recording", "ReplaySharedMemoryReader", "open_recording"]


def _load_stream(path: Path, dtype: np.dtype) -> np.ndarray:
    """Memory-map a record stream, ignoring any partial trailing record."""
    if not path.is_file():
        return np.zeros(0, dtype=dtype)
    size = path.stat().st_size
    count = size // dtype.itemsize
    if count == 0:
        return np.zeros(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(count,))


@dataclass(frozen=True, slots=True)
class Recording:
    """A recorded session, opened lazily.

    `telemetry` and `commands` are structured arrays whose `block` field has
    exactly the dtype the live code uses, so analysis reads
    `recording.telemetry["block"]["truck"]["speed"]` and is done.
    """

    path: Path
    manifest: dict
    telemetry: np.ndarray
    commands: np.ndarray
    frames: np.ndarray
    """One record per captured image; empty for a recording without any."""

    @property
    def duration(self) -> float:
        """Seconds from the first to the last telemetry record."""
        if len(self.telemetry) < 2:
            return 0.0
        return float(self.telemetry["t"][-1] - self.telemetry["t"][0])

    @property
    def truncated(self) -> bool:
        """Whether the writer was interrupted.

        True when the manifest's counts disagree with what is on disk, which
        means the writer never ran its final update. The records themselves are
        unaffected -- this is metadata, not damage.
        """
        declared = self.manifest.get("telemetry", {}).get("count", 0)
        return bool(declared) and declared != len(self.telemetry)

    def frame_at(self, t: float) -> np.void:
        """The last telemetry frame at or before `t`.

        Binary search over the timestamps, which are monotonic by construction.
        """
        if len(self.telemetry) == 0:
            raise IndexError("recording contains no telemetry")
        index = int(np.searchsorted(self.telemetry["t"], t, side="right")) - 1
        return self.telemetry[max(index, 0)]

    def image(self, index: int) -> np.ndarray:
        """Captured image `index` as a (height, width, 3) BGR array."""
        import cv2

        if not 0 <= index < len(self.frames):
            raise IndexError(f"image {index} of {len(self.frames)}")
        path = self.path / IMAGES_DIR / image_name(index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RecordingFormatError(f"{path} is missing or not an image")
        return image

    def summary(self) -> str:
        notes = self.manifest.get("notes") or ""
        return (
            f"{self.path.name}: {len(self.telemetry)} telemetry, "
            f"{len(self.commands)} command frames, "
            + (f"{len(self.frames)} images, " if len(self.frames) else "")
            + f"{self.duration:.1f}s"
            + (f" -- {notes}" if notes else "")
            + (" [truncated]" if self.truncated else "")
        )


def open_recording(path: Path | str) -> Recording:
    """Open a recording directory.

    Raises:
        RecordingFormatError: not a recording, or one this build cannot read.
    """
    path = Path(path)
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RecordingFormatError(f"{path} has no {MANIFEST_FILE}; not a recording")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordingFormatError(f"could not read {manifest_path}: {exc}") from exc

    if manifest.get("format") != FORMAT_NAME:
        raise RecordingFormatError(f"{path} is not an {FORMAT_NAME} directory")

    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise RecordingFormatError(
            f"{path} uses recording format v{version}, this build reads v{FORMAT_VERSION}"
        )

    _check_layout(path, manifest, "telemetry", LAYOUT_VERSION)
    _check_layout(path, manifest, "commands", COMMAND_LAYOUT_VERSION)

    return Recording(
        path=path,
        manifest=manifest,
        telemetry=_load_stream(path / TELEMETRY_FILE, TELEMETRY_RECORD_DTYPE),
        commands=_load_stream(path / COMMANDS_FILE, COMMAND_RECORD_DTYPE),
        frames=_load_stream(path / FRAMES_FILE, FRAME_RECORD_DTYPE),
    )


def _check_layout(path: Path, manifest: dict, stream: str, expected: int) -> None:
    """Refuse a recording whose blocks have a different layout.

    Reinterpreting them under the current layout would produce numbers rather
    than errors, which is the failure this whole project keeps designing
    against.
    """
    recorded = manifest.get(stream, {}).get("layout_version")
    if recorded is not None and recorded != expected:
        raise RecordingFormatError(
            f"{path}: {stream} blocks are layout v{recorded}, this build understands "
            f"v{expected}. The recording is intact; a build matching that layout can "
            f"read it."
        )


class ReplaySharedMemoryReader(SharedMemoryReader):
    """Serves a recording through the PAL's shared-memory interface.

    Hand one to `TelemetryReader` and the whole stack runs offline against real
    recorded data, through the same code that talks to the game.

    Each `read()` returns the bytes of the current frame. `TelemetryReader`
    reads twice per attempt to check the seqlock, so the cursor advances on a
    caller's explicit `advance()` rather than on every read -- a recording has
    no writer to race, and pretending otherwise would make every replayed
    snapshot a torn one.
    """

    def __init__(self, recording: Recording, *, loop: bool = False) -> None:
        self._recording = recording
        self._loop = loop
        self._index = 0
        self._open = False

    @property
    def recording(self) -> Recording:
        return self._recording

    @property
    def index(self) -> int:
        """Index of the frame `read()` currently returns."""
        return self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._recording.telemetry)

    def open(self) -> None:
        if len(self._recording.telemetry) == 0:
            raise PalNotConnectedError(f"{self._recording.path} contains no telemetry frames")
        self._index = 0
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> int:
        return int(TELEMETRY_RECORD_DTYPE["block"].itemsize)

    def read(self) -> bytes:
        if not self._open:
            raise PalNotConnectedError("replay reader is not open")
        if self.exhausted:
            raise PalNotConnectedError(
                f"replay reached the end of {self._recording.path} "
                f"({len(self._recording.telemetry)} frames)"
            )
        return self._recording.telemetry[self._index]["block"].tobytes()

    def advance(self, frames: int = 1) -> bool:
        """Move to a later frame. Returns False once the recording runs out."""
        self._index += frames
        if self._index >= len(self._recording.telemetry):
            if not self._loop:
                return False
            self._index %= len(self._recording.telemetry)
        return True

    def seek_time(self, t: float) -> None:
        """Position on the last frame at or before `t`."""
        timestamps = self._recording.telemetry["t"]
        self._index = max(int(np.searchsorted(timestamps, t, side="right")) - 1, 0)

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
