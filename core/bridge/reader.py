"""Reads the plugin's telemetry block, consistently.

The plugin writes from the game's main thread whenever it likes; we read from
another process whenever *we* like. Nothing synchronises the two, so a naive
copy can catch a frame half-written -- a position from one physics step and a
heading from the next. At 90 km/h that is a metre of phantom lateral error,
arriving as a plausible number with no error attached.

The block carries a seqlock for exactly this, and `TelemetryReader` implements
the reader half: take a copy, check that the sequence counter did not move
while the copy was in progress, retry if it did. Torn frames become retries,
and retries that do not settle become an exception rather than bad data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from core.bridge.layout import BLOCK_DTYPE, BLOCK_SIZE, LAYOUT_VERSION, MAGIC, GameId, HeaderFlag
from pal import PalNotConnectedError, get_backend

if TYPE_CHECKING:
    from pal.base import Backend, SharedMemoryReader

__all__ = [
    "LayoutMismatchError",
    "StaleTelemetryError",
    "TelemetryError",
    "TelemetryReader",
    "TelemetrySnapshot",
    "TornReadError",
]

#: How many times to retry a torn read before giving up. The writer holds the
#: lock for one memcpy of a 4 KB block, so even two attempts almost always
#: suffice; this is headroom for a descheduled reader, not a busy-wait.
DEFAULT_MAX_ATTEMPTS: Final = 8

#: How long a write left half done may hold a read up before the game is taken
#: for stalled. The eight attempts above take microseconds; the game's main
#: thread paused in the middle of a write -- a hitch, an autosave -- for longer
#: than that on a motorway out of Uppsala, and the drive was given up at
#: 73 km/h. After the quick attempts, reading goes on every `STALL_POLL_S`
#: until this much time has passed.
DEFAULT_STALL_WAIT: Final = 0.5
STALL_POLL_S: Final = 0.001

#: How long the publish counter may stand still before the data is considered
#: stale. The plugin publishes at the game's frame rate -- 60 Hz on the
#: reference machine -- so a second is roughly sixty missed frames: far beyond
#: any plausible hitch, and short enough that a controller notices before it
#: has driven anywhere on bad data.
DEFAULT_STALE_AFTER: Final = 1.0

#: Sleep between polls while waiting for a frame. Well under one frame at
#: 60 Hz, so the wait adds no meaningful latency of its own.
_POLL_INTERVAL: Final = 0.002


class TelemetryError(Exception):
    """Base class for telemetry reading failures."""


class LayoutMismatchError(TelemetryError):
    """The block was written by a plugin this reader does not understand.

    Almost always a stale `.so` left in the game's `plugins/` directory after
    the layout changed. Reinstall with `scripts/install_plugin.py`.
    """


class TornReadError(TelemetryError):
    """The block kept changing underneath us.

    Not expected in normal operation: it means the writer held the lock across
    repeated attempts, which suggests the game's main thread is stalled.
    """


class StaleTelemetryError(TelemetryError):
    """Nobody has published for a while. The data is a corpse.

    This is not hypothetical. A POSIX shared memory object outlives the process
    that created it unless `shm_unlink` runs, and ETS2 does not reliably call
    `scs_telemetry_shutdown` on exit -- observed on 1.55.0.0/Linux, where the
    segment survived the game with its last frame intact. A reader attaching
    afterwards sees a perfectly well-formed block: right magic, right layout,
    plausible numbers, a truck standing still with the brake applied.

    Silently consuming that is the worst failure this bridge can have, because
    nothing about the data says it is wrong. So freshness is checked, not
    assumed, and a controller is told to stop rather than handed a snapshot
    from a game that is no longer running.

    Note that a paused game also stops publishing. The distinction does not
    matter to a caller -- in both cases, do not drive -- but `is_paused` on the
    last snapshot says which it was.
    """


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """One consistent view of the telemetry block.

    `block` is a NumPy structured scalar over a private copy of the bytes, so
    it stays valid and unchanging for as long as it is held.
    """

    block: np.void
    received_at: float
    """`time.perf_counter()` when the copy was taken. Shares a time axis with
    captured frames and control commands, which is what makes end-to-end
    latency measurable rather than estimated."""

    attempts: int
    """How many copies it took to get a consistent one. Persistently above 1
    means the reader is competing with the writer more than expected."""

    @property
    def sequence(self) -> int:
        return int(self.block["header"]["sequence"])

    @property
    def write_count(self) -> int:
        """Publishes since the plugin initialised. A stalled counter across
        several reads means the game is paused, minimised or hung."""
        return int(self.block["header"]["write_count"])

    @property
    def write_time(self) -> float:
        """Plugin-side steady-clock seconds at publish. Differences between
        consecutive snapshots give the game's real publish interval; the
        absolute value is only comparable within one plugin session."""
        return float(self.block["header"]["write_time"])

    @property
    def is_paused(self) -> bool:
        """Whether the game is paused or sitting in a menu.

        Commands must be cut here: the game ignores them while paused and
        applies whatever is held the instant it resumes.
        """
        return bool(int(self.block["header"]["flags"]) & HeaderFlag.PAUSED)

    @property
    def game_id(self) -> int:
        return int(self.block["header"]["game_id"])


class TelemetryReader:
    """Consistent reads of the shared telemetry block.

    Typical use:

        with TelemetryReader() as reader:
            snapshot = reader.read()
            speed = float(snapshot.block["truck"]["speed"])

    The backend comes from the PAL, so this class works unchanged against the
    real game on either platform and against a recorded block in tests.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        stale_after: float | None = DEFAULT_STALE_AFTER,
        stall_wait: float = DEFAULT_STALL_WAIT,
    ) -> None:
        """
        Args:
            backend: PAL backend; defaults to the running platform's.
            max_attempts: seqlock retries, back to back, before waiting.
            stall_wait: seconds, from the first attempt, a write left half
                done may take to finish before the read gives up.
            stale_after: seconds the publish counter may stand still before
                `read()` raises `StaleTelemetryError`. `None` disables the
                check, which is only appropriate for tools that deliberately
                inspect a dead block -- never for anything that drives.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if stale_after is not None and stale_after <= 0:
            raise ValueError("stale_after must be positive, or None to disable")
        self._backend = backend if backend is not None else get_backend()
        if stall_wait < 0:
            raise ValueError("stall_wait must not be negative")
        self._max_attempts = max_attempts
        self._stall_wait = stall_wait
        self._stale_after = stale_after
        self._source: SharedMemoryReader | None = None
        self._last_write_count: int | None = None
        self._last_change_at: float | None = None

    def open(self) -> None:
        """Map the block and verify it was written by a plugin we understand.

        Raises:
            PalNotConnectedError: the game is not running, or has no plugin.
            LayoutMismatchError: the plugin is a different build.
        """
        from core.bridge.layout import SHARED_MEMORY_NAME

        source = self._backend.shared_memory(SHARED_MEMORY_NAME, BLOCK_SIZE)
        source.open()
        self._source = source
        self._last_write_count = None
        self._last_change_at = None
        try:
            self._validate(self._decode(source.read()))
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def is_open(self) -> bool:
        return self._source is not None

    @property
    def last_write_count(self) -> int | None:
        """The publish counter at the last read; None before one."""
        return self._last_write_count

    def __enter__(self) -> TelemetryReader:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self) -> TelemetrySnapshot:
        """Return a consistent snapshot.

        The sequence counter sits near the start of the block, so a full copy
        observes it first and the rest of the payload after. Taking a second
        copy and comparing the two counters is therefore enough: the counter
        only ever increases, and every write brackets itself by making it odd
        and then even again. If both copies saw the same even value, no write
        began or ended between the start of the first copy and the start of the
        second -- which covers the whole of the first copy.

        Raises:
            PalNotConnectedError: the block went away (game closed).
            TornReadError: the block kept changing across every attempt.
        """
        if self._source is None:
            raise PalNotConnectedError("telemetry reader is not open")

        began = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            if attempt > self._max_attempts:
                if time.perf_counter() - began >= self._stall_wait:
                    break
                time.sleep(STALL_POLL_S)
            candidate = self._decode(self._source.read())
            sequence = int(candidate["header"]["sequence"])
            if sequence % 2 != 0:
                continue  # a write was in progress when the copy started

            confirmation = self._decode(self._source.read())
            if int(confirmation["header"]["sequence"]) != sequence:
                continue  # a write completed while we were copying

            received_at = time.perf_counter()
            snapshot = TelemetrySnapshot(block=candidate, received_at=received_at, attempts=attempt)
            self._check_freshness(snapshot, received_at)
            return snapshot

        raise TornReadError(
            f"could not read a consistent block in {attempt - 1} attempts over "
            f"{time.perf_counter() - began:.2f} s; the game's main thread may be stalled"
        )

    def wait_for_frame(self, timeout: float | None = None) -> TelemetrySnapshot:
        """Block until the plugin publishes a frame, and return it.

        A single `read()` cannot tell fresh data from stale: one sample of a
        counter says nothing about whether it is moving. That matters most at
        startup, which is exactly when a reader is most likely to attach to a
        block left behind by a game that has already exited.

        This waits for the counter to actually advance, so what comes back is
        known to be current. Anything that is about to act on telemetry should
        start here rather than with `read()`.

        Args:
            timeout: seconds to wait. Defaults to `stale_after`, or one second
                if the staleness check is disabled.

        Raises:
            StaleTelemetryError: nothing was published within the timeout.
            PalNotConnectedError: the block went away.
        """
        if self._source is None:
            raise PalNotConnectedError("telemetry reader is not open")

        limit = timeout if timeout is not None else (self._stale_after or DEFAULT_STALE_AFTER)
        deadline = time.perf_counter() + limit

        # Read with the staleness check suppressed: standing still is the
        # normal state here, and reporting it is this method's job.
        stale_after, self._stale_after = self._stale_after, None
        try:
            baseline = self.read()
            while time.perf_counter() < deadline:
                snapshot = self.read()
                if snapshot.write_count != baseline.write_count:
                    return snapshot
                time.sleep(_POLL_INTERVAL)
        finally:
            self._stale_after = stale_after
            # Whatever happens, the next read() judges freshness from now, not
            # from before the wait.
            self._last_write_count = None
            self._last_change_at = None

        reason = "the game is paused" if baseline.is_paused else "the game is not running"
        raise StaleTelemetryError(
            f"no frame published within {limit:.1f}s; {reason}. The shared memory "
            f"block outlives the game, so its contents are last-known, not current."
        )

    def _check_freshness(self, snapshot: TelemetrySnapshot, now: float) -> None:
        """Raise if the publish counter has stood still for too long.

        The counter, not a timestamp: `write_time` comes from the plugin's own
        clock and a dead plugin's last value would keep looking recent for as
        long as we cared to compare it against anything.
        """
        write_count = snapshot.write_count
        if write_count != self._last_write_count:
            self._last_write_count = write_count
            self._last_change_at = now
            return

        if self._stale_after is None or self._last_change_at is None:
            return

        standing_still = now - self._last_change_at
        if standing_still > self._stale_after:
            reason = "the game is paused" if snapshot.is_paused else "the game is not running"
            raise StaleTelemetryError(
                f"no new frame for {standing_still:.1f}s (publish counter stuck at "
                f"{write_count}); {reason}. The shared memory block outlives the "
                f"game, so its contents are last-known, not current."
            )

    @staticmethod
    def _decode(blob: bytes) -> np.void:
        if len(blob) < BLOCK_SIZE:
            raise LayoutMismatchError(f"shared memory is {len(blob)} bytes, expected {BLOCK_SIZE}")
        # frombuffer over `bytes` is read-only and does not copy; the bytes
        # themselves are already the private copy the PAL handed us.
        return np.frombuffer(blob, dtype=BLOCK_DTYPE, count=1)[0]

    @staticmethod
    def _validate(block: np.void) -> None:
        header = block["header"]
        magic = int(header["magic"])
        if magic != MAGIC:
            raise LayoutMismatchError(
                f"bad magic 0x{magic:08X}, expected 0x{MAGIC:08X}. The block exists "
                "but was not written by this project's plugin, or the plugin has "
                "not published a frame yet."
            )

        version = int(header["layout_version"])
        if version != LAYOUT_VERSION:
            raise LayoutMismatchError(
                f"plugin publishes layout v{version}, this reader understands "
                f"v{LAYOUT_VERSION}. Close the game and run the launcher again, "
                "which installs the plugin built for it; from a checkout, rebuild "
                "and reinstall: python scripts/install_plugin.py"
            )

        block_size = int(header["block_size"])
        if block_size != BLOCK_SIZE:
            raise LayoutMismatchError(
                f"plugin publishes a {block_size}-byte block, this reader expects "
                f"{BLOCK_SIZE}. Layout version matched, so one side was changed "
                "without bumping it."
            )

        game = int(header["game_id"])
        if game == GameId.ATS:
            raise LayoutMismatchError(
                "this block comes from American Truck Simulator; the stack targets "
                "Euro Truck Simulator 2"
            )
