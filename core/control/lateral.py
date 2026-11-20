"""Lateral control: following a path.

Pure Pursuit, which aims the vehicle at a point some distance ahead on the
path and works out the steering angle that would drive a bicycle model through
it. It is the right first controller here: one tuning knob with a physical
meaning, no gains to balance, and a failure mode -- cutting corners -- that is
obvious from the trace rather than subtle.

Two things between it and the game.

**The game wants a steering *rate*, not an angle.** `SteeringPositionController`
converts one to the other, and this module stops at "what angle do I want".

**The angle is not the road wheel angle.** `effective_steering` is a
normalised control position: 0.05 gives about 2 degrees of mean wheel angle,
0.95 about 39. That map was measured (`docs/results/sysid.md`) and is inverted
here, so the geometry is done in radians of actual wheel angle and the game
asked for whatever produces it.

Sign conventions come from `core.localization.path`, where they were measured
rather than assumed: positive is left throughout -- lateral error, curvature,
wheel angle and steering command alike.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Final

from core.localization.path import ReferencePath

__all__ = [
    "ACTROS_GEOMETRY",
    "LateralCommand",
    "PurePursuit",
    "TrailerGeometry",
    "VehicleGeometry",
]


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    """What the bicycle model needs, measured rather than looked up."""

    wheelbase: float
    """Front axle to driven axle, metres. Derived from the wheel positions the
    game reports, so it follows the truck rather than being a constant."""

    steering_map: tuple[tuple[float, float], ...]
    """`effective_steering` to mean road wheel angle, radians.

    Progressive: the ratio rises from 0.112 near centre to 0.132 at full lock,
    so a linear approximation is wrong by about 20 % at the extremes. The
    wheels are also not parallel -- Ackermann puts up to 12.65 degrees between
    inner and outer -- so this is their mean, which is exactly what a bicycle
    model's single wheel represents.
    """

    axle_behind: float = 0.0
    """The driven axle's distance behind the point the truck's position is
    given for, metres. The game's is the truck's origin, by the front axle:
    2.87 m ahead of the driven axles on the 8x4 driven in Bern and Linz.
    Pure pursuit steers the driven axle round its arc; given the origin for
    it, the truck turned about its front axle and the driven axle -- and the
    trailer hitched over it -- came round inside."""

    def __post_init__(self) -> None:
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if len(self.steering_map) < 2:
            raise ValueError("steering_map needs at least two points")
        commands = [c for c, _ in self.steering_map]
        angles = [a for _, a in self.steering_map]
        if commands != sorted(commands) or angles != sorted(angles):
            raise ValueError("steering_map must be monotone in both columns")
        if commands[0] != 0.0 or angles[0] != 0.0:
            raise ValueError("steering_map must start at (0, 0)")

    @property
    def max_wheel_angle(self) -> float:
        """Largest wheel angle the map covers, radians.

        A lower bound on the vehicle's real lock: the measurement ran out of
        hold time before the steering reached its stop.
        """
        return self.steering_map[-1][1]

    def command_for_angle(self, wheel_angle: float) -> float:
        """The `effective_steering` that produces this wheel angle, radians in.

        Clamped to the measured range. Beyond it the honest answer is the most
        we know how to ask for.
        """
        sign = -1.0 if wheel_angle < 0.0 else 1.0
        return sign * _invert(self.steering_map, abs(wheel_angle))

    def angle_for_command(self, command: float) -> float:
        """The wheel angle a command produces. The forward map, for prediction."""
        sign = -1.0 if command < 0.0 else 1.0
        return sign * _interpolate(self.steering_map, abs(command))


def _degrees(value: float) -> float:
    return math.radians(value)


#: Mercedes Actros 2014, from the stationary steering sweep and the reported
#: wheel positions. See `docs/results/sysid.md`.
#:
#: The command axis is the *applied* `effective_steering`, not the value the
#: sweep asked for. Those differ at the top -- the position loop reached 0.9488
#: when asked for 1.00, because the steering rate falls off near the stops and
#: the hold was too short. Indexing by the request instead makes the map appear
#: to *lose* sensitivity at full lock, when in fact it gains it: a quiet 3 %
#: error in the steering angle at the extremes, which a test caught.
#:
#: It therefore stops at 0.9488, and commands above that clamp. The vehicle has
#: a little more lock than this describes; how much was never measured.
ACTROS_GEOMETRY: Final = VehicleGeometry(
    wheelbase=3.761,
    steering_map=(
        (0.0000, 0.0),
        (0.0491, _degrees(1.97)),
        (0.0992, _degrees(3.97)),
        (0.1991, _degrees(7.98)),
        (0.2992, _degrees(12.01)),
        (0.4991, _degrees(20.14)),
        (0.6975, _degrees(28.33)),
        (0.8788, _degrees(35.90)),
        (0.9488, _degrees(38.82)),
    ),
)


@dataclass(frozen=True, slots=True)
class TrailerGeometry:
    """What a trailer does to the line the combination actually drives.

    The controller steers the tractor, and the tractor's error is not the
    vehicle's error: a trailer cuts inside every corner, by an amount that
    grows with the square of its length over the radius. Measured on this
    combination, the tractor held 0.70 m while the trailer was 2.12 m off the
    path.

    Built from the telemetry rather than configured -- the hook position and
    the wheel positions are both reported, and they change with the trailer.
    """

    length: float
    """Hook to mean axle, metres. The lever that produces the cutting."""

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("trailer length must be positive")

    @classmethod
    def from_block(cls, block: object, trailer: int = 0) -> TrailerGeometry | None:
        """Read the geometry out of a telemetry block, or None if unhitched."""
        data = block["trailers"][trailer]  # type: ignore[index]
        if not int(data["connected"]):
            return None
        count = int(data["wheels"]["count"])
        if count == 0:
            return None
        positions = data["wheels"]["position"][:count]
        axle = sum(float(p["z"]) for p in positions) / count
        return cls(length=abs(axle - float(data["hook_position"][2])))

    def off_tracking(self, curvature: float) -> float:
        """Steady-state lateral offset of the trailer, metres.

        Signed in the same convention as `lateral_error`: positive is left, so
        a right-hand corner (negative curvature) gives a negative offset,
        because the trailer cuts to the inside.

        Steady state, and corners are rarely long enough to reach it. Measured
        against a real run, the observed offset was 0.69 of this with a
        correlation of 0.56 -- the trend is right, the pointwise value is not.
        That is why nothing steers on this yet: a compensation built on it
        would be applying a number it cannot justify.
        """
        magnitude = abs(curvature)
        if magnitude < 1e-9:
            return 0.0
        radius = 1.0 / magnitude
        if radius <= self.length:
            # Tighter than the trailer is long: the geometry has no solution
            # and the real vehicle would be jackknifing.
            return math.copysign(radius, curvature)
        return math.copysign(radius - math.sqrt(radius**2 - self.length**2), curvature)


@dataclass(frozen=True, slots=True)
class LateralCommand:
    """What the lateral controller wants, and what it saw."""

    steering: float
    """`effective_steering` to hold. Positive steers left."""

    wheel_angle: float
    """The road wheel angle that command should produce, radians."""

    lateral_error: float
    """Signed distance from the path, metres. Positive when left of it."""

    heading_error: float
    """Vehicle heading minus path heading, radians, wrapped."""

    lookahead: float
    """Distance to the aim point actually used, metres."""

    progress: float
    """Arc length of the nearest point on the path, metres. Feed back as
    `near` on the next call so a self-crossing route resolves correctly."""

    at_end: bool
    """Whether the aim point ran past the end of the path."""


class PurePursuit:
    """Steers towards a point ahead on the path.

    One call per control step:

        command = controller.step(
            x=..., z=..., heading=..., speed=..., path=path, progress=previous
        )

    Stateless: the caller carries `progress` between steps. That keeps the
    controller safe to re-run over a recording -- replaying a drive and
    re-tuning against it is the whole reason the recording exists -- and makes
    a mistuned run reproducible rather than path-dependent.
    """

    def __init__(
        self,
        geometry: VehicleGeometry = ACTROS_GEOMETRY,
        *,
        lookahead_time: float = 1.8,
        min_lookahead: float = 8.0,
        max_lookahead: float = 60.0,
        curvature_gain: float = 0.6,
        curvature_samples: int = 8,
    ) -> None:
        """
        Args:
            geometry: wheelbase and steering map.
            lookahead_time: seconds of travel to aim ahead. The single knob:
                larger is smoother and cuts corners harder, smaller tracks
                tighter and oscillates sooner.
            min_lookahead: floor, metres. Below roughly a vehicle length the
                aim point sits inside the truck and the geometry degenerates --
                and a tractor-trailer is much longer than its wheelbase.
            max_lookahead: ceiling, metres.
            curvature_gain: shortens the lookahead where the path ahead bends,
                capping it at `gain / max_curvature` over the window. A fixed
                time ahead is too far into a tight corner: aiming 20 m ahead on
                a 10 m radius puts the target more than half way round, and the
                controller drives the chord. Scanning ahead rather than reading
                the curvature underfoot matters, because the corner has to be
                seen before it is entered. Zero disables it.
            curvature_samples: how many points to scan over the window.

        These are starting values, tuned only in simulation. A noise-free model
        rewards ever-shorter lookaheads without limit -- the same signature as
        the longitudinal gains -- so the floor is held at something with a
        physical meaning rather than at whatever the simulation liked best.
        """
        if lookahead_time <= 0.0:
            raise ValueError("lookahead_time must be positive")
        if not 0.0 < min_lookahead <= max_lookahead:
            raise ValueError("lookahead bounds must be positive and ordered")
        if curvature_gain < 0.0:
            raise ValueError("curvature_gain must not be negative")
        if curvature_samples < 2:
            raise ValueError("curvature_samples must be at least 2")
        self._geometry = geometry
        self._lookahead_time = lookahead_time
        self._min_lookahead = min_lookahead
        self._max_lookahead = max_lookahead
        self._curvature_gain = curvature_gain
        self._curvature_samples = curvature_samples

    @property
    def geometry(self) -> VehicleGeometry:
        return self._geometry

    def lookahead_for(self, speed: float) -> float:
        """Aim distance at this speed on a straight path, metres.

        Proportional to speed so the controller looks a fixed *time* ahead:
        the same geometry then produces the same response whatever the speed,
        which is what stops a tuning that works at 30 km/h oscillating at 80.
        """
        return min(
            max(abs(speed) * self._lookahead_time, self._min_lookahead),
            self._max_lookahead,
        )

    def _lookahead_on(self, path: ReferencePath, start: float, speed: float) -> float:
        """Aim distance allowing for how the path ahead bends.

        Scans the window for its sharpest curvature and caps the lookahead at
        `gain / curvature`, never below the floor. The scan is the part that
        works: reading the curvature underfoot would shorten the lookahead only
        once the corner had been entered, which is far too late to have aimed
        anywhere useful.
        """
        lookahead = self.lookahead_for(speed)
        if self._curvature_gain <= 0.0:
            return lookahead

        step = lookahead / (self._curvature_samples - 1)
        sharpest = max(
            abs(path.sample_at(start + index * step).curvature)
            for index in range(self._curvature_samples)
        )
        if sharpest <= 1e-6:
            return lookahead
        return max(min(lookahead, self._curvature_gain / sharpest), self._min_lookahead)

    def step(
        self,
        *,
        x: float,
        z: float,
        heading: float,
        speed: float,
        path: ReferencePath,
        progress: float | None = None,
        aim_shift: float = 0.0,
        aim_shift_at=None,
    ) -> LateralCommand:
        """Compute the steering command for one frame.

        Args:
            x, z: vehicle position in the world ground plane.
            heading: vehicle heading, radians, as `convert.heading_rad` gives it.
            speed: forward speed, m/s. Only its magnitude is used.
            path: the path to follow.
            progress: arc length from the previous step, if there was one.
            aim_shift: metres to the right of the path to aim, as a painted
                lane bounds it (core.control.lane_guard); the errors stay
                the path's own.
            aim_shift_at: more metres to the right to aim, by distance along
                the path at the aim (core.control.turn_line): a line beside
                the path, followed without making it the path.
        """
        nearest = path.nearest(x, z, near=progress)
        lateral = path.lateral_error(x, z, nearest)
        heading_error = _wrap(heading - nearest.heading)

        lookahead = self._lookahead_on(path, nearest.s, speed)
        # Steered from the driven axle (`VehicleGeometry.axle_behind`), the
        # aim `lookahead` on from it; where the truck is along the path is
        # still the given point's.
        behind = self._geometry.axle_behind
        axle_x = x - behind * math.sin(heading)
        axle_z = z - behind * math.cos(heading)
        target_s = nearest.s - behind + lookahead
        at_end = target_s >= path.length
        aim = path.sample_at(target_s)
        aim_x, aim_z = aim.x, aim.z
        if target_s > path.length:
            # Past the end the aim runs on along the path's last heading.
            # Pinned to the end, it came under the truck in the last metres
            # and the wheel swung: +0.72, then -1.0, at a pump near Linz.
            over = target_s - path.length
            aim_x += over * math.sin(aim.heading)
            aim_z += over * math.cos(aim.heading)
        if aim_shift_at is not None:
            aim_shift = aim_shift + aim_shift_at(min(target_s, path.length))
        if aim_shift:
            # Right of a heading h is (-cos h, sin h).
            aim_x -= aim_shift * math.cos(aim.heading)
            aim_z += aim_shift * math.sin(aim.heading)

        forward, left = _to_vehicle_frame(aim_x - axle_x, aim_z - axle_z, heading)
        distance = math.hypot(forward, left)
        if distance < 1e-6:
            # Standing on the aim point: nothing to aim at, so hold straight.
            return LateralCommand(
                steering=0.0,
                wheel_angle=0.0,
                lateral_error=lateral,
                heading_error=heading_error,
                lookahead=lookahead,
                progress=nearest.s,
                at_end=at_end,
            )

        # Pure pursuit: the wheel angle that puts a bicycle model through a
        # point at (distance, alpha) from the rear axle.
        alpha = math.atan2(left, forward)
        wheel_angle = math.atan2(2.0 * self._geometry.wheelbase * math.sin(alpha), distance)
        wheel_angle = max(
            -self._geometry.max_wheel_angle, min(wheel_angle, self._geometry.max_wheel_angle)
        )

        return LateralCommand(
            steering=self._geometry.command_for_angle(wheel_angle),
            wheel_angle=wheel_angle,
            lateral_error=lateral,
            heading_error=heading_error,
            lookahead=lookahead,
            progress=nearest.s,
            at_end=at_end,
        )


def _to_vehicle_frame(dx: float, dz: float, heading: float) -> tuple[float, float]:
    """World offset to (forward, left) in the vehicle frame.

    Forward is `(sin h, cos h)` and left is `(cos h, -sin h)`; both follow from
    the measured convention that `+x` is the left-hand side at heading zero.
    """
    forward = dx * math.sin(heading) + dz * math.cos(heading)
    left = dx * math.cos(heading) - dz * math.sin(heading)
    return forward, left


def _wrap(angle: float) -> float:
    return -((-angle + math.pi) % (2.0 * math.pi) - math.pi)


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    index = bisect.bisect_left([p[0] for p in points], x)
    (x0, y0), (x1, y1) = points[index - 1], points[index]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _invert(points: tuple[tuple[float, float], ...], y: float) -> float:
    if y <= points[0][1]:
        return points[0][0]
    if y >= points[-1][1]:
        return points[-1][0]
    index = bisect.bisect_left([p[1] for p in points], y)
    (x0, y0), (x1, y1) = points[index - 1], points[index]
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
