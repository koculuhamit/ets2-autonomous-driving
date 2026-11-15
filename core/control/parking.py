"""The parking brake: set it when stopped, release it before moving off.

Found in the game on 2026-09-24: with the truck at rest, a service brake
command reaches the game (`input_brake` 0.30) but is not applied
(`effective_brake` 0.00), and on a 1 degree slope the truck rolled back at
1.35 km/h within a quarter of a second of releasing the parking brake. A
brake applied while still moving did hold, through the stop. Either way, a
stopped truck left to the service brake alone is not safely held, so a stop
that lasts is held on the parking brake.

The game's parking brake is a toggle: each press of the button flips it.
Pressing blind could as easily release it as set it, so every press is
decided from the telemetry's `parking_brake` state, and after a press the
controller waits for the state to change before it will press again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["ParkingBrake", "StopLatch"]

PRESS_S: Final = 0.2
"""How long the button is held for one press."""

SETTLE_S: Final = 1.0
"""How long to wait for the state to change after a press before trying again."""

SET_AFTER_S: Final = 1.0
"""Stopped and asked to stay stopped this long, set the parking brake."""

STOPPED_MPS: Final = 0.2

RESUME_TARGET_MPS: Final = 1.5
"""Once stopped for something, the target must reach this to move off again.
Stopped 12.3 m behind a trailer, its measured distance wandered to 13.1 m, the
target to 0.31 m/s, just over the stop threshold: the hold was dropped, the
parking brake never set, and the truck rolled back on a slope."""

MOVE_OFF_S: Final = 0.7
"""Moving off, the throttle is applied this long before the parking brake is
released, so the truck is pulling before it is let go on a slope."""


@dataclass
class ParkingBrake:
    """Decides, each step, whether to press the parking-brake button -- or
    any toggle the telemetry shows the state of (the engine, at a pump)."""

    settle_s: float = SETTLE_S
    """How long the state is given to follow a press."""

    _pressed_at: float | None = None
    _stopped_since: float | None = None

    def press(self, *, want_set: bool, is_set: bool, now: float) -> bool:
        """Whether the button should be held down in this step.

        `want_set` is what the caller wants, `is_set` what the telemetry
        says. A press lasts `PRESS_S`; after it, nothing more until the
        state has had `settle_s` to follow.
        """
        if self._pressed_at is not None:
            since = now - self._pressed_at
            if since < PRESS_S:
                return True
            if since < PRESS_S + self.settle_s and want_set != is_set:
                return False
            self._pressed_at = None
        if want_set != is_set:
            self._pressed_at = now
            return True
        return False

    def should_hold(self, *, speed: float, stopping: bool, now: float) -> bool:
        """Whether a stop has lasted long enough to set the parking brake."""
        if not stopping or abs(speed) > STOPPED_MPS:
            self._stopped_since = None
            return False
        if self._stopped_since is None:
            self._stopped_since = now
        return now - self._stopped_since >= SET_AFTER_S


@dataclass
class StopLatch:
    """A stop, once made, lasts until there is a real reason to go."""

    holding: bool = False

    def stopping(
        self, *, target: float, speed: float, stop_target: float, hold_below: float
    ) -> bool:
        """Whether the truck should be stopped (or stay stopped) now."""
        if target <= stop_target and abs(speed) < hold_below:
            self.holding = True
        elif self.holding and target >= RESUME_TARGET_MPS:
            self.holding = False
        return self.holding or target <= stop_target
