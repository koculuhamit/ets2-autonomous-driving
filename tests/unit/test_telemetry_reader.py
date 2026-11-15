"""Tests for the seqlock reader.

The interesting cases are the ones that only happen when the writer and reader
race, which is precisely when they are hardest to reproduce against the real
game. The fake PAL backend lets us produce them deterministically: a block
frozen mid-write, a write that completes between two copies, a plugin whose
layout does not match.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE, BLOCK_SIZE, LAYOUT_VERSION, MAGIC, GameId, HeaderFlag
from core.bridge.reader import (
    LayoutMismatchError,
    StaleTelemetryError,
    TelemetryReader,
    TornReadError,
)
from pal import PalNotConnectedError
from pal.fake import FakeBackend, FakeSharedMemoryReader


def make_block(
    *,
    sequence: int = 2,
    speed: float = 0.0,
    magic: int = MAGIC,
    layout_version: int = LAYOUT_VERSION,
    block_size: int = BLOCK_SIZE,
    game_id: int = GameId.ETS2,
    flags: int = HeaderFlag.TELEMETRY_ACTIVE,
    write_count: int = 1,
) -> bytes:
    """Build a block the way the plugin would."""
    block = np.zeros(1, dtype=BLOCK_DTYPE)
    header = block["header"]
    header["magic"] = magic
    header["layout_version"] = layout_version
    header["block_size"] = block_size
    header["sequence"] = sequence
    header["flags"] = flags
    header["game_id"] = game_id
    header["write_count"] = write_count
    block["truck"]["speed"] = speed
    return block.tobytes()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def install(backend: FakeBackend, blobs: list[bytes], *, bootstrap: bool = True) -> None:
    """Script the exact byte sequence the reader will see.

    `TelemetryReader.open()` validates the block, which costs one read. Tests
    care about what `read()` does, so a valid bootstrap block is prepended by
    default and the script describes only the reads that follow. Validation
    tests pass `bootstrap=False` to put their malformed block first.

    Each `read()` attempt consumes one blob for the candidate copy, plus a
    second for the confirmation -- unless the candidate's counter was odd, in
    which case it is discarded immediately and no confirmation is taken.
    """
    script = ([make_block()] if bootstrap else []) + blobs
    backend.set_shared_memory("ets2ad_telemetry", FakeSharedMemoryReader(BLOCK_SIZE, script))


def test_reads_a_consistent_block(backend: FakeBackend) -> None:
    block = make_block(speed=22.5)
    install(backend, [block, block])
    with TelemetryReader(backend) as reader:
        snapshot = reader.read()
    assert float(snapshot.block["truck"]["speed"]) == pytest.approx(22.5)
    assert snapshot.attempts == 1
    assert snapshot.sequence == 2


def test_retries_while_a_write_is_in_progress(backend: FakeBackend) -> None:
    """An odd sequence means the writer is mid-write; the copy is not usable."""
    install(
        backend,
        [
            make_block(sequence=3, speed=99.0),  # odd: discarded, no confirmation read
            make_block(sequence=4, speed=11.0),  # settled: candidate
            make_block(sequence=4, speed=11.0),  # settled: confirmation
        ],
    )
    with TelemetryReader(backend) as reader:
        snapshot = reader.read()
    assert float(snapshot.block["truck"]["speed"]) == pytest.approx(11.0)
    assert snapshot.attempts == 2


def test_retries_when_a_write_completes_mid_copy(backend: FakeBackend) -> None:
    """Both copies were even, but the counter moved: the first copy may have
    picked up fields from either side of the write."""
    install(
        backend,
        [
            make_block(sequence=2, speed=10.0),  # first copy
            make_block(sequence=4, speed=20.0),  # counter moved -> discard
            make_block(sequence=4, speed=20.0),  # retry, stable
            make_block(sequence=4, speed=20.0),
        ],
    )
    with TelemetryReader(backend) as reader:
        snapshot = reader.read()
    assert float(snapshot.block["truck"]["speed"]) == pytest.approx(20.0)
    assert snapshot.attempts == 2


def test_gives_up_after_max_attempts(backend: FakeBackend) -> None:
    """A writer that never settles is a stalled game, not bad data to pass on."""
    install(backend, [make_block(sequence=3)])  # cycles: always odd
    with (
        TelemetryReader(backend, max_attempts=3, stall_wait=0.0) as reader,
        pytest.raises(TornReadError),
    ):
        reader.read()


def test_a_write_held_up_for_a_moment_is_waited_for(backend: FakeBackend) -> None:
    """Out of Uppsala the game paused in the middle of a write for longer
    than eight quick attempts, and the drive was given up at 73 km/h."""
    install(backend, [make_block(sequence=3)] * 40 + [make_block(sequence=4, speed=20.0)] * 2)
    with TelemetryReader(backend) as reader:
        snapshot = reader.read()
    assert float(snapshot.block["truck"]["speed"]) == pytest.approx(20.0)
    assert snapshot.attempts == 41


def test_a_write_never_finished_is_given_up_on_after_the_stall_wait(backend: FakeBackend) -> None:
    import time

    install(backend, [make_block(sequence=3)])
    began = time.perf_counter()
    with TelemetryReader(backend, stall_wait=0.05) as reader, pytest.raises(TornReadError):
        reader.read()
    assert time.perf_counter() - began >= 0.05


def test_rejects_a_block_with_the_wrong_magic(backend: FakeBackend) -> None:
    install(backend, [make_block(magic=0xDEADBEEF)], bootstrap=False)
    with pytest.raises(LayoutMismatchError, match="bad magic"):
        TelemetryReader(backend).open()


def test_rejects_a_stale_plugin_by_layout_version(backend: FakeBackend) -> None:
    """The single most likely real failure: an old .so left in plugins/."""
    install(backend, [make_block(layout_version=LAYOUT_VERSION + 1)], bootstrap=False)
    with pytest.raises(LayoutMismatchError, match="install_plugin"):
        TelemetryReader(backend).open()


def test_rejects_a_block_size_mismatch(backend: FakeBackend) -> None:
    install(backend, [make_block(block_size=BLOCK_SIZE - 8)], bootstrap=False)
    with pytest.raises(LayoutMismatchError, match="without bumping"):
        TelemetryReader(backend).open()


def test_rejects_american_truck_simulator(backend: FakeBackend) -> None:
    install(backend, [make_block(game_id=GameId.ATS)], bootstrap=False)
    with pytest.raises(LayoutMismatchError, match="American Truck Simulator"):
        TelemetryReader(backend).open()


def test_open_failure_leaves_the_reader_closed(backend: FakeBackend) -> None:
    """A half-open reader would fail later, somewhere less obvious."""
    install(backend, [make_block(magic=0)], bootstrap=False)
    reader = TelemetryReader(backend)
    with pytest.raises(LayoutMismatchError):
        reader.open()
    assert reader.is_open is False


def test_reading_before_open_raises(backend: FakeBackend) -> None:
    install(backend, [make_block()])
    with pytest.raises(PalNotConnectedError):
        TelemetryReader(backend).read()


def test_paused_flag_is_exposed(backend: FakeBackend) -> None:
    """Commands must stop while paused: the game ignores them, then applies
    whatever is held the moment it resumes."""
    paused = make_block(flags=HeaderFlag.TELEMETRY_ACTIVE | HeaderFlag.PAUSED)
    install(backend, [paused, paused])
    with TelemetryReader(backend) as reader:
        assert reader.read().is_paused is True


def test_snapshot_is_a_private_copy(backend: FakeBackend) -> None:
    """Snapshots are held across frames; one that mutates is a data race the
    caller cannot see."""
    slow, fast = make_block(speed=5.0), make_block(speed=77.0)
    install(backend, [slow, slow, fast, fast])
    with TelemetryReader(backend) as reader:
        first = reader.read()
        reader.read()
    assert float(first.block["truck"]["speed"]) == pytest.approx(5.0)


def test_max_attempts_must_be_positive(backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TelemetryReader(backend, max_attempts=0)


class FrozenClock:
    """Advances only when a test says so, so staleness is tested in zero time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FrozenClock:
    frozen = FrozenClock()
    monkeypatch.setattr("core.bridge.reader.time.perf_counter", frozen)
    return frozen


def test_a_standing_still_counter_becomes_stale(backend: FakeBackend, clock: FrozenClock) -> None:
    """The real case: the game exited, the segment outlived it, and the block
    still looks perfectly well-formed."""
    frozen = make_block(write_count=500)
    install(backend, [frozen] * 12)

    with TelemetryReader(backend, stale_after=1.0) as reader:
        reader.read()  # first read establishes the baseline
        clock.advance(0.5)
        reader.read()  # still within tolerance
        clock.advance(0.6)
        with pytest.raises(StaleTelemetryError, match="not running"):
            reader.read()


def test_advancing_counter_never_goes_stale(backend: FakeBackend, clock: FrozenClock) -> None:
    install(backend, [make_block(write_count=n) for n in range(1, 13)])
    with TelemetryReader(backend, stale_after=0.1) as reader:
        for _ in range(5):
            clock.advance(1.0)
            reader.read()


def test_stale_message_distinguishes_a_paused_game(
    backend: FakeBackend, clock: FrozenClock
) -> None:
    """Both mean 'do not drive', but which one it was matters when debugging."""
    paused = make_block(write_count=7, flags=HeaderFlag.TELEMETRY_ACTIVE | HeaderFlag.PAUSED)
    install(backend, [paused] * 8)
    with TelemetryReader(backend, stale_after=0.5) as reader:
        reader.read()
        clock.advance(2.0)
        with pytest.raises(StaleTelemetryError, match="paused"):
            reader.read()


def test_staleness_check_can_be_disabled(backend: FakeBackend, clock: FrozenClock) -> None:
    """Only for tools deliberately inspecting a dead block."""
    install(backend, [make_block(write_count=1)] * 8)
    with TelemetryReader(backend, stale_after=None) as reader:
        reader.read()
        clock.advance(3600.0)
        assert reader.read().write_count == 1


def test_reopening_resets_the_freshness_baseline(backend: FakeBackend, clock: FrozenClock) -> None:
    """A reconnect starts a new session; carrying the old timer over would
    declare a healthy game stale the moment it came back."""
    install(backend, [make_block(write_count=3)] * 16)
    reader = TelemetryReader(backend, stale_after=1.0)
    reader.open()
    reader.read()
    clock.advance(5.0)
    reader.close()

    reader.open()
    reader.read()  # no exception: the baseline was reset
    reader.close()


def test_stale_after_must_be_positive(backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TelemetryReader(backend, stale_after=0)


def test_wait_for_frame_returns_once_the_counter_advances(backend: FakeBackend) -> None:
    install(
        backend,
        [
            make_block(write_count=9),  # baseline candidate
            make_block(write_count=9),  # baseline confirmation
            make_block(write_count=9),  # still nothing new
            make_block(write_count=9),
            make_block(write_count=10, speed=8.0),  # published
            make_block(write_count=10, speed=8.0),
        ],
    )
    with TelemetryReader(backend) as reader:
        snapshot = reader.wait_for_frame(timeout=5.0)
    assert snapshot.write_count == 10
    assert float(snapshot.block["truck"]["speed"]) == pytest.approx(8.0)


def test_wait_for_frame_times_out_on_a_dead_block(backend: FakeBackend) -> None:
    """The startup case this exists for: attaching to a segment the game left
    behind. A single read would have accepted it."""
    install(backend, [make_block(write_count=42)])
    with (
        TelemetryReader(backend) as reader,
        pytest.raises(StaleTelemetryError, match="not running"),
    ):
        reader.wait_for_frame(timeout=0.05)


def test_wait_for_frame_reports_a_paused_game_as_such(backend: FakeBackend) -> None:
    paused = make_block(write_count=3, flags=HeaderFlag.TELEMETRY_ACTIVE | HeaderFlag.PAUSED)
    install(backend, [paused])
    with TelemetryReader(backend) as reader, pytest.raises(StaleTelemetryError, match="paused"):
        reader.wait_for_frame(timeout=0.05)


def test_wait_for_frame_does_not_leave_the_reader_looking_stale(
    backend: FakeBackend, clock: FrozenClock
) -> None:
    """Time spent waiting must not count against the next read's freshness."""
    install(backend, [make_block(write_count=n // 2 + 1) for n in range(20)])
    with TelemetryReader(backend, stale_after=1.0) as reader:
        reader.wait_for_frame(timeout=5.0)
        clock.advance(0.9)
        reader.read()  # would raise if the wait had started the stale timer


def test_wait_for_frame_before_open_raises(backend: FakeBackend) -> None:
    install(backend, [make_block()])
    with pytest.raises(PalNotConnectedError):
        TelemetryReader(backend).wait_for_frame()
