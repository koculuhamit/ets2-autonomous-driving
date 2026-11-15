"""Tests for the command publisher.

The safety behaviour is what matters here. Everything else in the stack can be
wrong and produce a bad trajectory; this is the layer that decides whether a
bad trajectory reaches the pedals at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.bridge.command import (
    BUTTON_BITS,
    COMMAND_BLOCK_DTYPE,
    COMMAND_BLOCK_SIZE,
    COMMAND_MAGIC,
    COMMAND_MEMORY_NAME,
    Commander,
    CommandFlag,
    ControlCommand,
)
from core.common.controls import Button
from pal import PalNotConnectedError
from pal.fake import FakeBackend


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def published(backend: FakeBackend) -> np.void:
    """Decode the block as the plugin would see it."""
    writer = backend.shared_memory_writer(COMMAND_MEMORY_NAME, COMMAND_BLOCK_SIZE)
    return np.frombuffer(bytes(writer.buffer), dtype=COMMAND_BLOCK_DTYPE, count=1)[0]


# --------------------------------------------------------------------------- #
# ControlCommand
# --------------------------------------------------------------------------- #


def test_neutral_is_all_zero() -> None:
    command = ControlCommand.neutral()
    assert command.axis_values() == (0.0, 0.0, 0.0, 0.0)
    assert command.button_mask() == 0


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (ControlCommand(steer=2.0), (1.0, 0.0, 0.0, 0.0)),
        (ControlCommand(steer=-9.0), (-1.0, 0.0, 0.0, 0.0)),
        (ControlCommand(throttle=1.5), (0.0, 1.0, 0.0, 0.0)),
        (ControlCommand(brake=-0.5), (0.0, 0.0, 0.0, 0.0)),
        (ControlCommand(steer=float("nan")), (0.0, 0.0, 0.0, 0.0)),
        (ControlCommand(throttle=float("inf")), (0.0, 1.0, 0.0, 0.0)),
    ],
)
def test_axis_values_are_clamped(command: ControlCommand, expected: tuple[float, ...]) -> None:
    """Out of range is a bug upstream, but not one the vehicle should act out."""
    assert command.axis_values() == pytest.approx(expected)


def test_button_mask_uses_the_documented_bits() -> None:
    command = ControlCommand(buttons=frozenset({Button.HANDBRAKE, Button.GEAR_UP}))
    assert command.button_mask() == (1 << 0) | (1 << 2)


def test_unsupported_buttons_are_dropped_not_shifted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bit positions are wire format; silently renumbering them would send
    the wrong command rather than none."""
    carried = {b: bit for b, bit in BUTTON_BITS.items() if b is not Button.TRAILER_ATTACH}
    monkeypatch.setattr("core.bridge.command.BUTTON_BITS", carried)
    command = ControlCommand(buttons=frozenset({Button.TRAILER_ATTACH, Button.HORN}))
    assert command.button_mask() == 1 << 6


def test_every_button_reaches_the_plugin() -> None:
    assert set(BUTTON_BITS) == set(Button)


def test_attach_is_the_last_bit() -> None:
    command = ControlCommand(buttons=frozenset({Button.TRAILER_ATTACH}))
    assert command.button_mask() == 1 << 13


# --------------------------------------------------------------------------- #
# Commander
# --------------------------------------------------------------------------- #


def test_open_publishes_a_neutral_disengaged_frame(backend: FakeBackend) -> None:
    """Until the magic is in place the plugin sees 'no commander'. Writing it
    on open keeps the first real command from doubling as the handshake."""
    with Commander(backend):
        block = published(backend)
    assert int(block["magic"]) == COMMAND_MAGIC
    assert int(block["block_size"]) == COMMAND_BLOCK_SIZE


def test_send_publishes_axes_and_flags(backend: FakeBackend) -> None:
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=-0.25, throttle=0.6))
        block = published(backend)
    assert block["axes"] == pytest.approx([-0.25, 0.6, 0.0, 0.0])
    assert int(block["flags"]) & CommandFlag.ENGAGED


def test_disengage_clears_the_flag_and_the_axes(backend: FakeBackend) -> None:
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=0.9, throttle=1.0))
        commander.disengage()
        block = published(backend)
    assert int(block["flags"]) & CommandFlag.ENGAGED == 0
    assert block["axes"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_disengaged_send_keeps_the_flag_clear(backend: FakeBackend) -> None:
    """The flag is the authority, not the values: the plugin emits neutral
    regardless, so a caller that forgets to zero its outputs is still safe."""
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=1.0, throttle=1.0), engaged=False)
        block = published(backend)
    assert int(block["flags"]) & CommandFlag.ENGAGED == 0


def test_command_count_advances_every_send(backend: FakeBackend) -> None:
    """The plugin's watchdog keys on this counter; a stuck one releases the
    controls, so it must move on every publish."""
    with Commander(backend) as commander:
        counts = []
        for _ in range(5):
            commander.send(ControlCommand(throttle=0.1))
            counts.append(int(published(backend)["command_count"]))
    assert counts == sorted(set(counts)) and len(counts) == 5


def test_sequence_is_even_after_a_publish(backend: FakeBackend) -> None:
    """An odd counter means 'mid-write'. Leaving it odd would make the plugin
    discard every frame and fall through to its watchdog."""
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=0.3))
        assert int(published(backend)["sequence"]) % 2 == 0


def test_sequence_advances_by_two_per_publish(backend: FakeBackend) -> None:
    with Commander(backend) as commander:
        commander.send(ControlCommand())
        first = int(published(backend)["sequence"])
        commander.send(ControlCommand())
        assert int(published(backend)["sequence"]) == first + 2


def test_close_releases_the_controls(backend: FakeBackend) -> None:
    """Exiting normally -- including by exception -- must not leave anything
    held down."""
    commander = Commander(backend)
    commander.open()
    commander.send(ControlCommand(steer=1.0, throttle=1.0))
    commander.close()

    block = published(backend)
    assert int(block["flags"]) & CommandFlag.ENGAGED == 0
    assert block["axes"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_close_survives_the_game_disappearing(backend: FakeBackend) -> None:
    """Releasing controls on a block that no longer exists is not an error."""
    commander = Commander(backend)
    commander.open()
    backend.shared_memory_writer(COMMAND_MEMORY_NAME, COMMAND_BLOCK_SIZE).close()
    commander.close()
    assert commander.is_open is False


def test_send_before_open_raises(backend: FakeBackend) -> None:
    with pytest.raises(PalNotConnectedError):
        Commander(backend).send(ControlCommand())


def test_engage_defaults_to_neutral(backend: FakeBackend) -> None:
    """Engaging should not resume whatever was last commanded."""
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=0.8, throttle=0.9))
        commander.disengage()
        commander.engage()
        block = published(backend)
    assert int(block["flags"]) & CommandFlag.ENGAGED
    assert block["axes"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_nan_never_reaches_the_block(backend: FakeBackend) -> None:
    with Commander(backend) as commander:
        commander.send(ControlCommand(steer=float("nan"), brake=float("nan")))
        block = published(backend)
    assert not np.isnan(np.asarray(block["axes"])).any()
