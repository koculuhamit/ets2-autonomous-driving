"""How fast the vehicle may go, given where the road bends.

A target speed is a wish; the corners decide. Cornering at a curvature `k`
costs `v^2 * k` of lateral acceleration, so every radius implies a ceiling --
and a truck with a trailer runs out of grip and stability well before a car
would.

The route this was written for makes the point. Ten per cent of it is tighter
than a 25 m radius, and holding 40 km/h through that would need 11.9 m/s^2 of
lateral acceleration. The vehicle would leave the road, or lie down.

Two passes build the profile:

**Curvature.** For each point, `v = sqrt(a_lat_max / |k|)`, which is the fastest
the corner allows.

**Deceleration.** Walking backwards from the end, no point may be faster than
what can be shed before reaching the next one. Without this the vehicle arrives
at a corner already too fast and the limit is a statement rather than a plan --
the same "predictive rather than proportional" argument the brake's dead zone
forced on the longitudinal controller, for the same reason.

The result is a speed the vehicle can actually hold, which the speed controller
then treats as its target.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from core.localization.path import ReferencePath

__all__ = ["SpeedProfile"]

#: Comfortable-to-firm lateral acceleration for a combination vehicle. The
#: taught drive this was built against reached 5.6 m/s^2, which is 0.57 g and
#: more than a loaded trailer should be asked for.
DEFAULT_MAX_LATERAL: Final = 2.5

#: Planning deceleration. Chosen against what the vehicle can actually deliver:
#: the service brake's smallest useful command gives 1.81 m/s^2 and nothing
#: between that and 0.22 is commandable, so planning for much less would mean
#: planning for a deceleration that does not exist.
DEFAULT_MAX_DECELERATION: Final = 1.5

#: How gently the truck plans to shed speed by lifting off, m/s^2: coasting
#: gives 0.22, so from where the profile at this rate would have it slow
#: down, it stops gaining speed, and the brake does the rest later and for
#: less. Planned at the coast's own 0.22 it would lift off half a kilometre
#: short of a motorway exit.
COAST_DECELERATION: Final = 0.8

#: Nor does it gain speed the road wants shed again within this many seconds:
#: between two bends out of Milan the profile rose by 5 km/h for 30 m, and
#: the truck went from the brake to full throttle and back. Over a city
#: drive, 5 quick changes from throttle to brake for 51, for 3 % more time.
LIFT_OFF_WINDOW_S: Final = 4.0

#: Through a turn tighter than this radius, metres, no faster than this, m/s:
#: the floor below, 11 km/h. The wheel turns at its own rate: into a 9-10 m
#: junction turn north of Zurich at 16 km/h, as the lateral limit allows,
#: the truck was 27 degrees behind the line before its wheel had come round,
#: and 2-3 m wide, both ways. At 11 km/h it has half again as much road to
#: come round in.
TIGHT_RADIUS_M: Final = 15.0
TIGHT_SPEED_MPS: Final = 3.0

#: Floor, m/s. A profile that reaches zero in a corner is stuck there.
DEFAULT_MIN_SPEED: Final = 3.0


class SpeedProfile:
    """Speed limits along a path, from its curvature.

        profile = SpeedProfile(path)
        target = min(wanted_speed, profile.speed_at(progress))

    Precomputed once per path: the backward pass is global, so it cannot be
    evaluated pointwise, and a path does not change while it is being followed.
    """

    def __init__(
        self,
        path: ReferencePath,
        *,
        max_lateral_acceleration: float = DEFAULT_MAX_LATERAL,
        max_deceleration: float = DEFAULT_MAX_DECELERATION,
        min_speed: float = DEFAULT_MIN_SPEED,
        max_speed: float | None = None,
    ) -> None:
        """
        Args:
            path: the route.
            max_lateral_acceleration: cornering ceiling, m/s^2.
            max_deceleration: how hard the backward pass may plan to brake,
                m/s^2.
            min_speed: floor, m/s.
            max_speed: optional ceiling, m/s.

        Raises:
            ValueError: any limit non-positive, or a floor above the ceiling.
        """
        if max_lateral_acceleration <= 0.0:
            raise ValueError("max_lateral_acceleration must be positive")
        if max_deceleration <= 0.0:
            raise ValueError("max_deceleration must be positive")
        if min_speed <= 0.0:
            raise ValueError("min_speed must be positive")
        if max_speed is not None and max_speed < min_speed:
            raise ValueError("max_speed must not be below min_speed")

        self._path = path
        self._s = np.array([path.sample_at(s).s for s in _arc_positions(path)])
        curvature = np.abs(np.array([path.sample_at(s).curvature for s in self._s]))

        ceiling = max_speed if max_speed is not None else np.inf
        with np.errstate(divide="ignore"):
            cornering = np.sqrt(max_lateral_acceleration / np.maximum(curvature, 1e-9))
        cornering = np.where(
            curvature > 1.0 / TIGHT_RADIUS_M, np.minimum(cornering, TIGHT_SPEED_MPS), cornering
        )
        self._limit = np.clip(cornering, min_speed, ceiling)

        self._apply_deceleration(max_deceleration, min_speed)
        self._coast = _slowed(self._s, self._limit.copy(), COAST_DECELERATION, min_speed)

    @property
    def speeds(self) -> np.ndarray:
        """The profile, m/s, at the arc lengths in `positions`."""
        return self._limit

    @property
    def positions(self) -> np.ndarray:
        """Arc lengths the profile is defined at, metres."""
        return self._s

    def speed_at(self, s: float) -> float:
        """The speed limit at arc length `s`, interpolated and clamped."""
        return float(np.interp(s, self._s, self._limit))

    def coasting_at(self, s: float, speed: float = 0.0) -> float:
        """The speed not to accelerate past at arc length `s`, going at
        `speed` (m/s): from it the bends ahead are come to by lifting off
        first, and none within `LIFT_OFF_WINDOW_S` asks for less
        (`SpeedController.step(coast_above=...)`)."""
        first = int(np.searchsorted(self._s, s))
        last = int(np.searchsorted(self._s, s + speed * LIFT_OFF_WINDOW_S, side="right"))
        ahead = self._limit[first:last]
        soon = float(ahead.min()) if len(ahead) else np.inf
        return min(float(np.interp(s, self._s, self._coast)), soon)

    def slowest(self) -> float:
        """The tightest point on the route, m/s."""
        return float(self._limit.min())

    def _apply_deceleration(self, max_deceleration: float, min_speed: float) -> None:
        """Walk backwards so every point can slow to the one after it.

        `v_i <= sqrt(v_{i+1}^2 + 2*a*ds)`. Without this the profile says
        "30 km/h in that corner" while the vehicle is still doing 60 a metre
        before it -- true, useless, and exactly the mistake of treating a limit
        as a plan.
        """
        self._limit = _slowed(self._s, self._limit, max_deceleration, min_speed)


def _slowed(s: np.ndarray, limit: np.ndarray, deceleration: float, floor: float) -> np.ndarray:
    """`limit`, in place, lowered so each point can slow to the next at `deceleration`."""
    for index in range(len(limit) - 2, -1, -1):
        span = s[index + 1] - s[index]
        if span <= 0.0:
            continue
        reachable = np.sqrt(limit[index + 1] ** 2 + 2.0 * deceleration * span)
        limit[index] = max(min(limit[index], reachable), floor)
    return limit


def _arc_positions(path: ReferencePath) -> np.ndarray:
    """One profile point per path point."""
    return np.linspace(0.0, path.length, len(path))
