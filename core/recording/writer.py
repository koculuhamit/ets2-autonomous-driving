"""Writing recordings.

A recorder that loses the session when something goes wrong is worse than no
recorder, because the loss is discovered after the interesting run rather than
before it. So the writer favours durability over throughput:

* Records are appended to plain files with no buffering layer of our own. A
  process killed at any moment leaves every completed record intact and at
  most one partial record at the end, which the reader handles.
* The manifest is written at `open()` with zero counts and rewritten at
  `close()` with the real ones, via a temporary file and an atomic rename. A
  crash therefore leaves either the old manifest or the new one, never a
  half-written one.
* `close()` is idempotent and does not raise.

The files are written on a thread of the recorder's own, never on the
caller's: the control loop hands a record over and goes on. On Windows the
recordings and the game shared a laptop's hard disk, and a write waiting
behind the game's held the loop up 0.3-0.9 s at a time, its thread idle,
until the plugin stopped the truck. What a killed process loses is the
records still queued, a few milliseconds of them.

Counts in the manifest are a convenience; the reader derives the truth from
file size, so a recording whose manifest never got its final update is still
completely readable.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Self

import numpy as np

from core.recording.format import (
    COMMAND_RECORD_DTYPE,
    COMMANDS_FILE,
    FRAME_RECORD_DTYPE,
    FRAMES_FILE,
    IMAGES_DIR,
    MANIFEST_FILE,
    TELEMETRY_FILE,
    TELEMETRY_RECORD_DTYPE,
    image_name,
    manifest_template,
)

if TYPE_CHECKING:
    from core.bridge.reader import TelemetrySnapshot
    from pal.base import Frame

#: JPEG quality for recorded images. At 90 a 1080p frame of the game is about
#: 430 kB and takes 7 ms to encode; the artefacts are well below what a
#: detector notices.
JPEG_QUALITY = 90

__all__ = ["Recorder"]


class Recorder:
    """Appends telemetry and command frames to a recording directory.

    Typical use:

        with Recorder(Path("data/runs/highway"), notes="A2 eastbound") as rec:
            while driving:
                snapshot = reader.read()
                rec.write_telemetry(snapshot)

    The timeline starts at the first record written, so a recording never opens
    with dead air from however long setup took.
    """

    def __init__(self, path: Path, *, notes: str = "", metadata: dict | None = None) -> None:
        self._path = Path(path)
        self._notes = notes
        self._metadata = dict(metadata or {})
        self._telemetry: BinaryIO | None = None
        self._commands: BinaryIO | None = None
        self._frames: BinaryIO | None = None
        self._telemetry_count = 0
        self._command_count = 0
        self._frame_count = 0
        self._origin: float | None = None
        # Images are encoded off the telemetry thread, so the start of the
        # timeline can be claimed from either.
        self._origin_lock = threading.Lock()
        self._manifest: dict[str, object] = {}
        self._writes: queue.SimpleQueue[Callable[[], object] | None] = queue.SimpleQueue()
        self._writer: threading.Thread | None = None
        self._failure: OSError | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def telemetry_count(self) -> int:
        return self._telemetry_count

    @property
    def command_count(self) -> int:
        return self._command_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        """Seconds since the first record. Zero before anything is written."""
        return 0.0 if self._origin is None else time.perf_counter() - self._origin

    def open(self) -> None:
        """Create the directory and open the streams.

        Raises:
            FileExistsError: the directory already holds a recording. Refusing
                is deliberate: appending to somebody else's run, or silently
                overwriting one, are both worse than failing here.
        """
        self.close()
        if (self._path / MANIFEST_FILE).exists():
            raise FileExistsError(f"{self._path} already contains a recording")
        self._path.mkdir(parents=True, exist_ok=True)

        self._telemetry = (self._path / TELEMETRY_FILE).open("wb")
        self._commands = (self._path / COMMANDS_FILE).open("wb")
        self._frames = (self._path / FRAMES_FILE).open("wb")
        (self._path / IMAGES_DIR).mkdir(exist_ok=True)
        self._telemetry_count = 0
        self._command_count = 0
        self._frame_count = 0
        self._origin = None
        self._failure = None
        self._writer = threading.Thread(target=self._write_on, name="recorder", daemon=True)
        self._writer.start()

        self._manifest = manifest_template(datetime.now(UTC).isoformat())
        self._manifest["notes"] = self._notes
        self._manifest["metadata"] = self._metadata
        self._write_manifest()

    def close(self) -> None:
        """Flush, finalise the manifest, and release the files. Never raises."""
        if self._writer is not None:
            self._writes.put(None)  # what is queued is written first
            self._writer.join()
            self._writer = None
        for stream in (self._telemetry, self._commands, self._frames):
            if stream is not None:
                # A failed flush leaves the completed records on disk
                # regardless; there is nothing useful to do about it here.
                with contextlib.suppress(OSError):
                    stream.flush()
                    os.fsync(stream.fileno())
                stream.close()
        opened = self._telemetry is not None
        self._telemetry = None
        self._commands = None
        self._frames = None
        if opened:
            # The reader derives counts from file size, so a manifest that
            # could not be rewritten costs metadata, not the recording.
            with contextlib.suppress(OSError):
                self._finalise_manifest()

    @property
    def is_open(self) -> bool:
        return self._telemetry is not None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        """Append one telemetry frame, stamped relative to the recording start."""
        if self._telemetry is None:
            raise RuntimeError("recorder is not open")
        record = np.zeros(1, dtype=TELEMETRY_RECORD_DTYPE)
        record[0]["t"] = self._relative(snapshot.received_at)
        record[0]["block"] = snapshot.block
        self._write(self._telemetry.write, record.tobytes())
        self._telemetry_count += 1

    def write_command(self, block: np.void, timestamp: float) -> None:
        """Append one command frame.

        Commands are recorded as published rather than as the game applied
        them: what a replay needs to reproduce is the decision, and what the
        game did with it is already in the telemetry stream.
        """
        if self._commands is None:
            raise RuntimeError("recorder is not open")
        record = np.zeros(1, dtype=COMMAND_RECORD_DTYPE)
        record[0]["t"] = self._relative(timestamp)
        record[0]["block"] = block
        self._write(self._commands.write, record.tobytes())
        self._command_count += 1

    def write_frame(self, frame: Frame, quality: int = JPEG_QUALITY) -> None:
        """Append one image of the game window, on the same timeline.

        Encoding takes milliseconds, so a recorder that must not miss
        telemetry calls this from another thread; it is safe to, as long as
        only one thread writes images.
        """
        import cv2

        if self._frames is None:
            raise RuntimeError("recorder is not open")
        ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("the image could not be encoded")
        height, width = frame.shape
        self.write_encoded_frame(encoded.tobytes(), frame.timestamp, frame.sequence, width, height)

    def write_encoded_frame(
        self, jpeg: bytes, timestamp: float, sequence: int, width: int, height: int
    ) -> None:
        """Append one image already JPEG-encoded -- by the perception child,
        which has the frames (core.perception.service.KeptFrame)."""
        if self._frames is None:
            raise RuntimeError("recorder is not open")
        self._write((self._path / IMAGES_DIR / image_name(self._frame_count)).write_bytes, jpeg)
        record = np.zeros(1, dtype=FRAME_RECORD_DTYPE)
        record[0]["t"] = self._relative(timestamp)
        record[0]["sequence"] = sequence
        record[0]["width"], record[0]["height"] = width, height
        self._write(self._frames.write, record.tobytes())
        self._frame_count += 1

    def _write(self, write: Callable[[bytes], object], data: bytes) -> None:
        """Have `data` written on the recorder's thread.

        Raises:
            OSError: an earlier write failed; the recording is incomplete.
        """
        if self._failure is not None:
            raise self._failure
        self._writes.put(lambda: write(data))

    def _write_on(self) -> None:
        while (write := self._writes.get()) is not None:
            if self._failure is not None:
                continue  # one failure is told; what follows it is dropped
            try:
                write()
            except OSError as exc:
                self._failure = exc

    def _relative(self, timestamp: float) -> float:
        with self._origin_lock:
            if self._origin is None:
                self._origin = timestamp
        return timestamp - self._origin

    def _write_manifest(self) -> None:
        """Write the manifest atomically, so a crash never leaves half of one."""
        target = self._path / MANIFEST_FILE
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    def _finalise_manifest(self) -> None:
        telemetry = self._manifest["telemetry"]
        commands = self._manifest["commands"]
        frames = self._manifest["frames"]
        assert isinstance(telemetry, dict) and isinstance(commands, dict)
        assert isinstance(frames, dict)
        telemetry["count"] = self._telemetry_count
        commands["count"] = self._command_count
        frames["count"] = self._frame_count
        self._manifest["duration"] = round(self.duration, 3)
        self._manifest["ended_at"] = datetime.now(UTC).isoformat()
        self._write_manifest()
