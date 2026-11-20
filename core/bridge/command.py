"""Writes control setpoints into the game.

The other half of the bridge. `core/bridge/reader.py` brings the world in;
this sends steering, throttle and brake back out, through the input device the
plugin registers with the game (see ADR-0002).

The mirror of the telemetry layout, with the roles reversed: this process is
the writer and the plugin is the reader, and the same seqlock keeps the plugin
from acting on a half-written frame.

Three things are structural rather than left to callers getting it right:

* **Values are clamped, and NaN becomes neutral.** A controller producing NaN
  is a bug to fix upstream, but not one the vehicle should act out.
* **Disengaged means zero.** `engaged=False` makes the plugin emit neutral
  whatever the axes hold, so stopping is one flag rather than a promise that
  every caller remembered to zero its outputs first.
* **Silence releases the controls.** The plugin watches the command counter
  and goes neutral if it stops advancing. `Commander` therefore has to keep
  publishing while engaged -- and a crashed stack, which publishes nothing at
  all, is exactly the case that behaviour exists for.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

from core.common.controls import Axis, Button, clamp_axis
from pal import PalNotConnectedError, get_backend

if TYPE_CHECKING:
    from core.recording import Recorder
    from pal.base import Backend, SharedMemoryWriter

__all__ = [
    "COMMAND_BLOCK_DTYPE",
    "COMMAND_BLOCK_SIZE",
    "COMMAND_MAGIC",
    "COMMAND_MEMORY_NAME",
    "CommandFlag",
    "Commander",
    "ControlCommand",
]

#: "ET2C" little-endian.
COMMAND_MAGIC: Final = 0x43325445
COMMAND_LAYOUT_VERSION: Final = 1
COMMAND_MEMORY_NAME: Final = "ets2ad_command"
SAFETY_STOP_S: Final = 1.0
"""Quiet this long, the plugin stops the truck itself and leaves it to the
driver (`kSafetyStopSeconds`, plugin/src/safety_stop.h); until then it holds
the steering where it was."""

#: Axis order in the block. Must match `CommandAxis` in command_layout.h.
AXIS_ORDER: Final = (Axis.STEER, Axis.THROTTLE, Axis.BRAKE, Axis.CLUTCH)

#: Bit position per button. Must match `CommandButton` in command_layout.h.
BUTTON_BITS: Final = {
    Button.HANDBRAKE: 0,
    Button.ENGINE: 1,
    Button.GEAR_UP: 2,
    Button.GEAR_DOWN: 3,
    Button.INDICATOR_LEFT: 4,
    Button.INDICATOR_RIGHT: 5,
    Button.HORN: 6,
    Button.LIGHTS: 7,
    Button.MOTOR_BRAKE: 8,
    Button.ENGINE_BRAKE_UP: 9,
    Button.ENGINE_BRAKE_DOWN: 10,
    Button.HAZARD: 11,
    Button.ACTIVATE: 12,
    Button.TRAILER_ATTACH: 13,
    Button.RETARDER_UP: 14,
    Button.RETARDER_DOWN: 15,
}


class CommandFlag:
    """Bits in the block's `flags` field."""

    ENGAGED: Final = 1 << 0


COMMAND_BLOCK_DTYPE: Final = np.dtype(
    {
        "names": [
            "magic",
            "layout_version",
            "block_size",
            "sequence",
            "command_count",
            "command_time",
            "flags",
            "buttons",
            "axes",
        ],
        "formats": ["<u4", "<u4", "<u4", "<u4", "<u8", "<f8", "<u4", "<u4", ("<f4", 4)],
        "offsets": [0, 4, 8, 12, 16, 24, 32, 36, 40],
        "itemsize": 96,
    }
)

COMMAND_BLOCK_SIZE: Final = COMMAND_BLOCK_DTYPE.itemsize

#: Offset of the seqlock counter, written separately from the payload.
_SEQUENCE_OFFSET: Final = COMMAND_BLOCK_DTYPE.fields["sequence"][1]
_SEQUENCE_SIZE: Final = 4


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """One set of control setpoints.

    Steering is bipolar in [-1, 1], negative left. The pedals are in [0, 1].
    Out-of-range values are clamped on the way out rather than rejected: a
    controller that overshoots its own limits should be fixed, but not by
    leaving the vehicle uncommanded in the meantime.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    buttons: frozenset[Button] = field(default_factory=frozenset)

    @classmethod
    def neutral(cls) -> ControlCommand:
        """Everything released. What the panic path sends."""
        return cls()

    def axis_values(self) -> tuple[float, ...]:
        """Clamped axis values in block order."""
        raw = (self.steer, self.throttle, self.brake, self.clutch)
        return tuple(clamp_axis(axis, value) for axis, value in zip(AXIS_ORDER, raw, strict=True))

    def button_mask(self) -> int:
        """Buttons as the bitmask the plugin expects.

        Buttons the plugin does not carry are dropped rather than silently
        shifting the others: the bit positions are part of the wire format.
        """
        mask = 0
        for button in self.buttons:
            bit = BUTTON_BITS.get(button)
            if bit is not None:
                mask |= 1 << bit
        return mask


class Commander:
    """Publishes control commands to the plugin.

    Typical use:

        with Commander() as commander:
            commander.send(ControlCommand(steer=0.1, throttle=0.3))
            ...
            commander.disengage()

    `close()` sends a neutral, disengaged frame before unmapping, so exiting
    normally -- including by exception -- leaves the truck with nothing held
    down. A process that dies without running `close()` is covered by the
    plugin's watchdog instead.
    """

    def __init__(self, backend: Backend | None = None, recorder: Recorder | None = None) -> None:
        """
        Args:
            backend: PAL backend; defaults to the running platform's.
            recorder: when given, every published command is also written to it.
                A recording of an autonomous run needs the decisions as well as
                their consequences -- telemetry alone shows what the truck did,
                not what the stack asked for, and the gap between those two is
                usually where the bug is.
        """
        self._backend = backend if backend is not None else get_backend()
        self._recorder = recorder
        self._writer: SharedMemoryWriter | None = None
        self._block = np.zeros(1, dtype=COMMAND_BLOCK_DTYPE)
        self._sequence = 0
        self._count = 0
        self._started_at = time.perf_counter()

    def open(self) -> None:
        """Map the plugin's command block.

        Raises:
            PalNotConnectedError: the game is not running with the plugin.
        """
        writer = self._backend.shared_memory_writer(COMMAND_MEMORY_NAME, COMMAND_BLOCK_SIZE)
        writer.open()
        self._writer = writer
        self._sequence = 0
        self._count = 0
        self._started_at = time.perf_counter()
        # Publish a neutral frame immediately: until the magic is in place the
        # plugin treats the block as "no commander", and leaving it that way
        # would mean the first real command doubles as the handshake.
        self.send(ControlCommand.neutral(), engaged=False)

    def close(self) -> None:
        """Release the controls, then unmap."""
        if self._writer is not None:
            # The game going away here is expected, not exceptional: there
            # is nothing left to release.
            with contextlib.suppress(PalNotConnectedError):
                self.send(ControlCommand.neutral(), engaged=False)
            self._writer.close()
            self._writer = None

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def __enter__(self) -> Commander:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def send(self, command: ControlCommand, *, engaged: bool = True) -> None:
        """Publish one command frame.

        Must be called repeatedly while engaged: the plugin releases the
        controls if the counter stops advancing.

        Args:
            command: the setpoints.
            engaged: when False the plugin emits neutral whatever `command`
                holds, so disengaging does not depend on the caller zeroing
                its own outputs.

        Raises:
            PalNotConnectedError: the block went away.
        """
        if self._writer is None:
            raise PalNotConnectedError("commander is not open")

        self._count += 1
        block = self._block[0]
        block["magic"] = COMMAND_MAGIC
        block["layout_version"] = COMMAND_LAYOUT_VERSION
        block["block_size"] = COMMAND_BLOCK_SIZE
        block["command_count"] = self._count
        block["command_time"] = time.perf_counter() - self._started_at
        block["flags"] = CommandFlag.ENGAGED if engaged else 0
        block["buttons"] = command.button_mask()
        block["axes"] = command.axis_values()

        self._publish()
        if self._recorder is not None:
            self._recorder.write_command(block, time.perf_counter())

    def engage(self, command: ControlCommand | None = None) -> None:
        """Start driving, from neutral unless told otherwise."""
        self.send(command if command is not None else ControlCommand.neutral(), engaged=True)

    def disengage(self) -> None:
        """Hand control back. Idempotent, and safe to call from an error path."""
        self.send(ControlCommand.neutral(), engaged=False)

    def _publish(self) -> None:
        """Write the block under the seqlock.

        The counter is written on its own, before and after the payload, so a
        plugin reading mid-update sees an odd value and retries rather than
        acting on a frame with one axis from each of two commands.
        """
        assert self._writer is not None
        payload = self._block.tobytes()

        self._sequence = (self._sequence + 1) & 0xFFFFFFFF  # odd: update starting
        self._writer.write(self._sequence_bytes(), _SEQUENCE_OFFSET)

        self._writer.write(payload[:_SEQUENCE_OFFSET], 0)
        after = _SEQUENCE_OFFSET + _SEQUENCE_SIZE
        self._writer.write(payload[after:], after)

        self._sequence = (self._sequence + 1) & 0xFFFFFFFF  # even: consistent
        self._writer.write(self._sequence_bytes(), _SEQUENCE_OFFSET)

    def _sequence_bytes(self) -> bytes:
        return int(self._sequence).to_bytes(_SEQUENCE_SIZE, "little")
