"""Vehicle safety limits.

The invariants that decide whether the stack is allowed to keep driving. They
live here rather than in the tool that happens to use them, because they are
statements about the vehicle, not about an experiment -- and because logic that
stops a moving truck deserves tests, which is how the bug that prompted this
module would have been caught.

That bug: the distance budget fired *during* the emergency stop it had just
triggered, so the abort aborted itself and the truck was left rolling with no
commands at all. Hence `suspended()`, and hence the tests below it.

Each limit answers a different question, and none of them covers another:

* **Speed** -- going faster than allowed. Fires while moving.
* **Distance** -- used more road than the operator said was clear. Fires while
  moving, and is the only one that knows the road is finite.
* **Stuck** -- throttle applied, nothing happening. Fires at *zero* speed,
  which every other limit treats as perfectly safe.
* **Direction** -- moving the opposite way to the one commanded.
* **Impact** -- speed lost faster than any brake can shed it (`Impacts`):
  something was hit, at once.
* **Damage** -- the truck, its trailer or its cargo damaged since the drive
  began (`DamageWatch`): something was hit, whatever the stack thought. A
  scrape that does no damage shows only as an impact.
* **Standing** -- at rest, by its own choice, for longer than anything on
  the road makes a truck wait (`Standstill`).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

__all__ = [
    "Collided",
    "DamageWatch",
    "DistanceBudgetExceeded",
    "Impacts",
    "InputsNotApplied",
    "SafetyMonitor",
    "SpeedLimitExceeded",
    "Standstill",
    "StoodTooLong",
    "Stuck",
    "UnsafeCondition",
    "WrongDirection",
]

#: Throttle below this does not count as trying to move.
_THROTTLE_THRESHOLD: Final = 0.1

#: Speed below this counts as stationary.
DEFAULT_STUCK_SPEED_MS: Final = 0.3

#: Throttle applied with no movement for longer than this means something is in
#: the way. Long enough for a clutch to take up and an automatic to choose a
#: gear, short enough that nothing gets leaned on.
DEFAULT_STUCK_SECONDS: Final = 2.5

#: Reversing faster than this while forward was commanded is not rollback.
DEFAULT_REVERSE_TOLERANCE_MS: Final = 0.5

#: A pedal commanded and none applied for longer than this means the game is
#: not taking the stack's inputs. The automatic gearbox cuts the throttle for
#: every shift, for up to 0.78 s across all the recorded runs; twice that.
DEFAULT_UNAPPLIED_SECONDS: Final = 1.5

#: Applied pedal below this counts as none.
_PEDAL_APPLIED: Final = 0.01

#: Of a longer gap between the game's frames only this counts towards
#: `DEFAULT_UNAPPLIED_SECONDS`: the game stood still, it did not ignore the
#: inputs. In a roundabout in Torino a shift cut the throttle and the game
#: froze for 0.83 s in it, and the drive was stopped for the 1.5 s.
_FRAME_GAP_COUNTED_S: Final = 0.1


class UnsafeCondition(RuntimeError):
    """Base class: the vehicle must stop doing what it is doing."""


class SpeedLimitExceeded(UnsafeCondition):
    """Faster than the caller allowed."""


class DistanceBudgetExceeded(UnsafeCondition):
    """More road used than the operator said was clear.

    The only limit that knows the road runs out. Nothing in the stack can see
    ahead, so this number comes from a person and has to be asked for rather
    than remembered -- a stale budget is what put ten points of damage on a
    chassis.
    """


class Stuck(UnsafeCondition):
    """Throttle applied and the truck is not moving.

    Deliberately separate from the others, because zero speed is *inside* every
    other limit. An obstacle, a kerb, a closed gate and the end of the clear
    stretch all look identical from here, and none of them are worth leaning on.
    """


class InputsNotApplied(UnsafeCondition):
    """The stack commands a pedal and the game applies none.

    The game only polls the stack's input device while its own window has the
    focus. Telemetry keeps flowing either way, so from the stack's side nothing
    looks wrong: it commands, the truck does not respond. Standing still that
    looked like an obstacle; moving, nothing noticed at all, and the truck
    would coast on uncommanded.

    Nothing the stack sends will stop the truck in this state, braking
    included. What this is for is saying so, fast and plainly.
    """


class Collided(UnsafeCondition):
    """The truck, its trailer or its cargo has been damaged: something was hit.

    Whatever was hit is still there, and whatever the stack failed to see is
    not seen now either. Out of Milan the truck clipped an island's kerb 100 m
    from the depot and then pushed against it at full throttle for 2.5 s
    before anything stopped it.
    """


DAMAGE_HIT: Final = 0.001
"""A rise in damage, of 1 for wrecked, that is a hit: 0.0049 for clipping a
kerb in Milan; no drive without a hit changed it at all."""


HIT_DROP_MPS: Final = 2.5
HIT_WITHIN_S: Final = 0.15
"""Speed lost this fast -- 16.7 m/s^2 -- is a hit: the brakes give 9.7 at the
most. In the Italian-Swiss customs the truck scraped a booth's island at
31 km/h and lost 3.1 m/s in 0.1 s; it did no damage, so nothing stopped
the truck, and 12 s later, knocked off its line, it hit a barrier. Every
such drop in 110 recorded drives was a hit."""


class Impacts:
    """Raises `Collided` when the speed has fallen by `HIT_DROP_MPS` within
    `HIT_WITHIN_S`."""

    def __init__(self) -> None:
        self._speeds: deque[tuple[float, float]] = deque()

    def check(self, now: float, speed: float) -> None:
        self._speeds.append((now, speed))
        while now - self._speeds[0][0] > HIT_WITHIN_S:
            self._speeds.popleft()
        lost = max(v for _, v in self._speeds) - speed
        if lost >= HIT_DROP_MPS:
            raise Collided(f"hit something: {lost * 3.6:.0f} km/h lost at once")


class DamageWatch:
    """Raises `Collided` once the damage has risen by `DAMAGE_HIT` since the
    first telemetry block it saw."""

    def __init__(self) -> None:
        self._first: float | None = None

    @staticmethod
    def damage(block) -> float:
        """The truck's cabin and chassis, the trailer's chassis and its cargo, summed."""
        truck, trailer = block["truck"], block["trailers"][0]
        return float(
            truck["wear_cabin"]
            + truck["wear_chassis"]
            + trailer["wear_chassis"]
            + trailer["cargo_damage"]
        )

    def check(self, block) -> None:
        now = self.damage(block)
        if self._first is None:
            self._first = now
        elif now - self._first >= DAMAGE_HIT:
            raise Collided(f"hit something: {100.0 * (now - self._first):.1f} % damage")


class StoodTooLong(UnsafeCondition):
    """At rest for longer than a light, a junction or a queue holds a truck.

    Whatever it waits for -- a lead it remembers that has gone, a vehicle it
    gives way to that is parked -- is not going to change; the driver is
    asked to look.
    """


STANDSTILL_S: Final = 300.0
"""No light's red is longer than 46 s; five minutes is a jam or a mistake."""


class Standstill:
    """Raises `StoodTooLong` after `STANDSTILL_S` at rest, counted from the
    seconds it is told of: time paused in the game is not told."""

    def __init__(self, limit_s: float = STANDSTILL_S) -> None:
        self._limit_s = limit_s
        self._stood = 0.0

    def update(self, speed: float, dt: float) -> None:
        if abs(speed) >= DEFAULT_STUCK_SPEED_MS:
            self._stood = 0.0
            return
        self._stood += dt
        if self._stood >= self._limit_s:
            raise StoodTooLong(f"at rest for {self._stood / 60.0:.0f} min")


class WrongDirection(UnsafeCondition):
    """Moving the opposite way to the one commanded.

    Usually means the game selected reverse: `abackward` at a standstill is a
    request to reverse, not a brake, so anything holding the brake to keep the
    truck still selects it.
    """


class SafetyMonitor:
    """Checks the vehicle state against the limits, once per control step.

    Stateful only in what it must remember: where the run started, and how long
    the truck has been refusing to move.
    """

    def __init__(
        self,
        *,
        speed_limit_ms: float,
        distance_budget_m: float | None = None,
        stuck_seconds: float = DEFAULT_STUCK_SECONDS,
        stuck_speed_ms: float = DEFAULT_STUCK_SPEED_MS,
        reverse_tolerance_ms: float = DEFAULT_REVERSE_TOLERANCE_MS,
        unapplied_seconds: float = DEFAULT_UNAPPLIED_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if speed_limit_ms <= 0.0:
            raise ValueError("speed_limit_ms must be positive")
        if distance_budget_m is not None and distance_budget_m <= 0.0:
            raise ValueError("distance_budget_m must be positive, or None")
        self._speed_limit = speed_limit_ms
        self._distance_budget = distance_budget_m
        self._stuck_seconds = stuck_seconds
        self._stuck_speed = stuck_speed_ms
        self._reverse_tolerance = reverse_tolerance_ms
        self._unapplied_seconds = unapplied_seconds
        self._clock = clock
        self._unapplied_for: float | None = None
        self._applied_checked_at: float | None = None
        self._applied_frame: int | None = None

        self._previous: tuple[float, float] | None = None
        self._moving_since: float | None = None
        self._suspended = False
        self.distance_travelled = 0.0

    @property
    def suspended(self) -> bool:
        return self._suspended

    @contextmanager
    def suspend(self) -> Iterator[None]:
        """Stop enforcing while bringing the vehicle to rest.

        Distance and stuck both trip on states that a stop necessarily passes
        through -- the road is already overspent, and the truck is about to be
        stationary with the brake on. Enforcing them here makes the stop
        unrunnable, which leaves the vehicle moving and uncommanded: strictly
        worse than the condition being guarded against.

        Distance keeps accumulating so the figure reported afterwards is still
        the truth.
        """
        previous, self._suspended = self._suspended, True
        try:
            yield
        finally:
            self._suspended = previous

    def check(
        self,
        *,
        speed: float,
        throttle: float,
        position: tuple[float, float],
        expect_forward: bool = False,
        brake: float = 0.0,
        applied: float | None = None,
        frame: int | None = None,
    ) -> None:
        """Raise if any limit is breached.

        Args:
            speed: forward speed, m/s. Negative when reversing.
            throttle: commanded throttle, 0 to 1.
            position: world `(x, z)`, for the distance budget.
            expect_forward: whether forward motion was commanded.
            brake: commanded brake, 0 to 1.
            applied: the larger of the throttle and brake the game reports
                applying, or None to skip the check that inputs arrive.
            frame: the game's frame counter, if known: what `applied` is
                judged by is the time the game ran, not the time it stood.

        Raises:
            UnsafeCondition: one of its subclasses, naming what went wrong.
        """
        self._accumulate(position)
        if self._suspended:
            return

        if abs(speed) > self._speed_limit:
            raise SpeedLimitExceeded(
                f"{speed * 3.6:.1f} km/h exceeds the {self._speed_limit * 3.6:.1f} km/h limit"
            )

        if expect_forward and speed < -self._reverse_tolerance:
            raise WrongDirection(f"commanded forward but moving at {speed * 3.6:.1f} km/h")

        if self._distance_budget is not None and self.distance_travelled > self._distance_budget:
            raise DistanceBudgetExceeded(
                f"used {self.distance_travelled:.0f} m of a {self._distance_budget:.0f} m budget"
            )

        if applied is not None:
            self._check_applied(max(throttle, brake), applied, frame)
        self._check_stuck(speed=speed, throttle=throttle)

    def _check_applied(self, commanded: float, applied: float, frame: int | None) -> None:
        now = self._clock()
        if frame is not None:
            if frame == self._applied_frame:
                return  # nothing new from the game
            self._applied_frame = frame
        ran = 0.0
        if self._applied_checked_at is not None:
            ran = min(now - self._applied_checked_at, _FRAME_GAP_COUNTED_S)
        self._applied_checked_at = now
        if commanded < _THROTTLE_THRESHOLD or applied >= _PEDAL_APPLIED:
            self._unapplied_for = None
            return
        if self._unapplied_for is None:
            self._unapplied_for = 0.0
            return
        self._unapplied_for += ran
        if self._unapplied_for > self._unapplied_seconds:
            raise InputsNotApplied(
                f"{commanded:.0%} commanded, nothing applied for "
                f"{self._unapplied_for:.1f}s -- the game is not taking the stack's "
                "inputs; is its window focused? Take the controls"
            )

    def _accumulate(self, position: tuple[float, float]) -> None:
        """Add the road covered since the last sample.

        The budget is metres of road somebody looked at, so what it has to be
        spent against is road: the sum of the steps, not the distance from
        where the run began. Displacement undercounts every corner, and on a
        loop it shrinks back towards zero while the truck keeps driving, which
        would leave the one limit that knows the road runs out never firing.
        """
        if self._previous is None:
            self._previous = position
            return
        dx = position[0] - self._previous[0]
        dz = position[1] - self._previous[1]
        self._previous = position
        self.distance_travelled += (dx * dx + dz * dz) ** 0.5

    def _check_stuck(self, *, speed: float, throttle: float) -> None:
        now = self._clock()
        if throttle < _THROTTLE_THRESHOLD or abs(speed) > self._stuck_speed:
            self._moving_since = now
            return
        if self._moving_since is None:
            self._moving_since = now
            return
        stalled = now - self._moving_since
        if stalled > self._stuck_seconds:
            raise Stuck(
                f"{throttle:.0%} throttle for {stalled:.1f}s at {speed * 3.6:.1f} km/h "
                "-- something is in the way"
            )
