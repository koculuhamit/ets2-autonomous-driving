"""Running the live stack against a recording.

`ReplayBackend` is a PAL backend whose shared memory is a recorded file. Give
one to `TelemetryReader` and every layer above it runs unchanged: the same
seqlock check, the same layout validation, the same parsing. Offline tests
therefore exercise the production path rather than a second implementation of
it that happens to agree today.

    for snapshot in replay_frames(open_recording("data/runs/a2-east")):
        controller.step(snapshot)

`replay_frames()` walks the recording as fast as the consumer takes it. Nothing waits
on wall time: a test should not take as long as the drive did, and a controller
that is fed frames in order does not care how quickly they arrive. Use
`realtime=True` when the point is to watch it happen.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from core.recording.reader import Recording, ReplaySharedMemoryReader, open_recording
from pal.base import SharedMemoryReader
from pal.fake import FakeBackend

if TYPE_CHECKING:
    from core.bridge.reader import TelemetrySnapshot

__all__ = ["ReplayBackend", "replay_frames"]


class ReplayBackend(FakeBackend):
    """A backend that serves a recording instead of a running game.

    Everything except shared memory is inherited from the fake backend, so a
    replay still has a game-paths layout, a hotkey listener and the rest --
    inert, but present, so code under test does not have to know it is being
    replayed.
    """

    name = "replay"

    def __init__(
        self, recording: Recording, *, loop: bool = False, root: Path | None = None
    ) -> None:
        super().__init__(root=root)
        self._replay = ReplaySharedMemoryReader(recording, loop=loop)

    @property
    def replay_reader(self) -> ReplaySharedMemoryReader:
        """The underlying reader, for stepping through the recording."""
        return self._replay

    def shared_memory(self, name: str, size: int) -> SharedMemoryReader:
        return self._replay


def replay_frames(
    recording: Recording | Path | str,
    *,
    start: float = 0.0,
    realtime: bool = False,
    stale_after: float | None = None,
) -> Iterator[TelemetrySnapshot]:
    """Yield every telemetry frame of a recording, through the live reader.

    Args:
        recording: an open recording, or a path to one.
        start: skip to this many seconds into the run.
        realtime: pace the frames to the recorded timestamps. Off by default --
            a test should not take as long as the drive did.
        stale_after: passed to `TelemetryReader`. `None` disables the staleness
            check, which is right here: a recording's publish counter advances
            only when the consumer asks for the next frame, so wall-clock
            freshness is meaningless.
    """
    import time

    from core.bridge.reader import TelemetryReader

    if not isinstance(recording, Recording):
        recording = open_recording(recording)

    backend = ReplayBackend(recording)
    source = backend.replay_reader

    with TelemetryReader(backend, stale_after=stale_after) as reader:
        if start > 0.0:
            source.seek_time(start)

        origin = time.perf_counter()
        first_t = float(recording.telemetry["t"][source.index])

        while True:
            frame_t = float(recording.telemetry["t"][source.index])
            if realtime:
                due = origin + (frame_t - first_t)
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield reader.read()
            if not source.advance():
                return
