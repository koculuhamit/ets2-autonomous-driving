"""How a drive went, in numbers: from what was recorded of it.

Everything here reads arrays a drive left behind -- the telemetry
(core.recording), the ACC's log (`acc.npy`, tools/follow.py), the route it
followed -- and nothing live, so a drive is measured the same way however
long ago it ran, and a change to the stack is measured against the drives
before it.

- **lateral**: the truck's distance from the route's lane middle
  (`ReferencePath.lateral_error`), its mean and worst, and the time spent
  more than `OUT_OF_LANE_M` from it. The tractor swings wide in tight
  turns on purpose (`widen_for_trailer`), so a few metres in a turn is not
  necessarily a fault.
- **longitudinal**: hard brakes (`HARD_BRAKE_MPS2`), the worst jerk, and
  the shortest time to collision with the lead the ACC followed.
- **comfort**: steering reversals a minute (`REVERSAL_RAD` either way).
- **safety**: hits -- the speed falling `IMPACT_DROP_MPS` within
  `IMPACT_WITHIN_S`, as core.control.limits.Impacts stops for -- and damage
  taken (the truck's wear, %).
- **interventions per 100 km** over a session (`session_summary`): drives
  that ended with the driver taking over or a safety stop, against the
  distance driven -- the one number to watch.

Distances are the game's kilometres, from the truck's odometer. The world's
metres the truck moves through are a sixteenth to an eighteenth of them on
the drives measured: counted in those, a drive towards Amsterdam of 358 km
read 19 km, and the interventions per 100 km came out about seventeen times
too many.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from core.evaluation.health import memory_growth

__all__ = [
    "DriveMetrics",
    "SessionSummary",
    "drive_metrics",
    "session_summary",
]

OUT_OF_LANE_M: Final = 1.5
"""Further than this from the lane's middle, a 2.55 m truck in a 3.5 m lane
is over the line."""

OFF_ROUTE_M: Final = 20.0
"""Further than this from the route the truck is not on it at all: the drive
planned again on the way (a detour, a turn missed), and the route it was
recorded with is no longer the one it followed. Counted as `off_route_s`,
not as lateral error."""

HARD_BRAKE_MPS2: Final = 3.0
HARD_BRAKE_END_MPS2: Final = 2.0
HARD_BRAKE_FOR_S: Final = 0.3
"""A hard brake is over `HARD_BRAKE_MPS2` for `HARD_BRAKE_FOR_S` or more,
and ends under `HARD_BRAKE_END_MPS2`: one stop is one hard brake, not each
time the deceleration wavers across a line."""

RESAMPLE_S: Final = 0.02
"""Speed is put on an even 50 Hz grid before it is differentiated: the
recording's times are when each block was read, 7 to 120 ms apart, and a
short interval across a whole frame's change in speed read as 18 m/s^2."""
REVERSAL_RAD: Final = math.radians(2.0)
IMPACT_DROP_MPS: Final = 2.5
IMPACT_WITHIN_S: Final = 0.15
SMOOTH_S: Final = 0.5
"""Speed is smoothed over this before it is differentiated: the telemetry's
own noise, differentiated twice, is jerk that nobody felt."""

INTERVENTIONS: Final = frozenset({"disengaged", "unsafe", "interrupted"})
"""How a drive ended (tools/follow.py --outcome) that counts against it."""


@dataclass(frozen=True, slots=True)
class DriveMetrics:
    duration_s: float
    distance_km: float
    """The game's kilometres, from the odometer."""
    mean_speed_kmh: float
    lateral_mean_m: float
    lateral_max_m: float
    out_of_lane_s: float
    off_route_s: float
    hard_brakes: int
    max_deceleration_mps2: float
    max_jerk_mps3: float
    min_ttc_s: float
    min_gap_m: float
    steering_reversals_per_min: float
    impacts: int
    damage_pct: float
    ended: str = ""
    memory_mb_per_h: float = math.nan
    """Resident memory gained an hour (core.evaluation.health); nan unsampled."""

    def as_dict(self) -> dict:
        return asdict(self)


def drive_metrics(
    t: np.ndarray,
    block: np.ndarray,
    path=None,
    acc: np.ndarray | None = None,
    ended: str = "",
    health: np.ndarray | None = None,
    detail: str = "",
) -> DriveMetrics:
    """A drive's metrics from its telemetry (`t`, `block`: core.recording's
    arrays), the `ReferencePath` it followed (None: no lateral numbers), the
    ACC's log (columns: t, speed, lead id, distance, closing speed, ...) and
    how it ended (`ended`, and its `detail`: tools/follow.py --outcome)."""
    t = np.asarray(t, float)
    truck = block["truck"]
    speed = np.asarray(truck["speed"], float)
    position = truck["placement"]["position"]
    x, z = np.asarray(position["x"], float), np.asarray(position["z"], float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    odometer = np.asarray(truck["odometer"], float)
    distance_km = float(max(0.0, odometer[-1] - odometer[0])) if len(t) > 1 else 0.0
    mean_speed = (
        float(np.sum(np.abs(speed[1:]) * np.diff(t)) / duration) * 3.6 if duration > 0 else 0.0
    )

    lateral = _lateral(path, x, z)
    dt = np.diff(t) if len(t) > 1 else np.zeros(0)
    off_route = 0.0
    if len(lateral):
        away = np.nan_to_num(np.abs(lateral)) > OFF_ROUTE_M
        off_route = float(dt[away[1:]].sum())
        lateral = np.where(away, math.nan, lateral)
    if len(lateral) and np.isfinite(lateral).any():
        off = np.abs(lateral)
        lateral_mean, lateral_max = float(np.nanmean(off)), float(np.nanmax(off))
        out_of_lane = float(dt[np.nan_to_num(off[1:]) > OUT_OF_LANE_M].sum())
    else:
        lateral_mean = lateral_max = out_of_lane = math.nan

    even = np.arange(t[0], t[-1], RESAMPLE_S) if duration > RESAMPLE_S else t
    smooth = _smoothed(even, np.interp(even, t, speed))
    acceleration = _derivative(even, smooth)
    jerk = _derivative(even, _smoothed(even, acceleration))
    deceleration = -acceleration
    hard = _hard_brakes(even, deceleration)

    steering = np.asarray(truck["effective_steering"], float)
    wheel = steering * math.radians(40.0)  # about the lock's angle: reversals, not angles
    minutes = duration / 60.0

    min_ttc, min_gap = _lead(acc)
    # As core.control.limits.DamageWatch sums it.
    trailer = block["trailers"][:, 0]
    wear = (
        np.asarray(truck["wear_cabin"], float)
        + np.asarray(truck["wear_chassis"], float)
        + np.asarray(trailer["wear_chassis"], float)
        + np.asarray(trailer["cargo_damage"], float)
    )
    hit, damage_told = _told(detail)
    return DriveMetrics(
        duration_s=duration,
        distance_km=distance_km,
        mean_speed_kmh=mean_speed,
        lateral_mean_m=lateral_mean,
        lateral_max_m=lateral_max,
        out_of_lane_s=out_of_lane,
        off_route_s=off_route,
        hard_brakes=hard,
        max_deceleration_mps2=float(deceleration.max()) if len(deceleration) else 0.0,
        max_jerk_mps3=float(np.abs(jerk).max()) if len(jerk) else 0.0,
        min_ttc_s=min_ttc,
        min_gap_m=min_gap,
        steering_reversals_per_min=_reversals(wheel) / minutes if minutes > 0 else 0.0,
        impacts=max(_impacts(t, speed), int(hit)),
        damage_pct=max(
            float(max(0.0, (wear[-1] - wear[0]) * 100.0)) if len(wear) else 0.0, damage_told
        ),
        ended=ended,
        memory_mb_per_h=memory_growth(health) if health is not None else math.nan,
    )


@dataclass(frozen=True, slots=True)
class SessionSummary:
    drives: int
    distance_km: float
    duration_h: float
    interventions: int
    interventions_per_100km: float
    impacts: int
    hard_brakes: int
    damage_pct: float
    lateral_max_m: float
    out_of_lane_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def session_summary(drives: Iterable[DriveMetrics]) -> SessionSummary:
    """A session's drives together: interventions per 100 km above all."""
    drives = list(drives)
    distance_km = sum(d.distance_km for d in drives)
    interventions = sum(1 for d in drives if d.ended in INTERVENTIONS)
    lateral = [d.lateral_max_m for d in drives if not math.isnan(d.lateral_max_m)]
    return SessionSummary(
        drives=len(drives),
        distance_km=distance_km,
        duration_h=sum(d.duration_s for d in drives) / 3600.0,
        interventions=interventions,
        interventions_per_100km=interventions / distance_km * 100.0 if distance_km else math.nan,
        impacts=sum(d.impacts for d in drives),
        hard_brakes=sum(d.hard_brakes for d in drives),
        damage_pct=sum(d.damage_pct for d in drives),
        lateral_max_m=max(lateral) if lateral else math.nan,
        out_of_lane_s=sum(d.out_of_lane_s for d in drives if not math.isnan(d.out_of_lane_s)),
    )


def _lateral(path, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    if path is None or not len(x):
        return np.zeros(0)
    errors, near = [], None
    for xi, zi in zip(x, z, strict=True):
        sample = path.nearest(float(xi), float(zi), near=near)
        near = sample.s
        # Off either end of the route the error is to its end, not a lane's.
        inside = 0.0 < sample.s < path.length
        errors.append(path.lateral_error(float(xi), float(zi), sample) if inside else math.nan)
    return np.asarray(errors)


def _smoothed(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(t) < 3:
        return values
    rate = (len(t) - 1) / max(float(t[-1] - t[0]), 1e-9)
    width = max(1, round(SMOOTH_S * rate))
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _derivative(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(len(t))
    return np.gradient(values, t)


def _hard_brakes(t: np.ndarray, deceleration: np.ndarray) -> int:
    count, began = 0, None
    for time_, value in zip(t, deceleration, strict=True):
        if began is None and value > HARD_BRAKE_MPS2:
            began = time_
        elif began is not None and value < HARD_BRAKE_END_MPS2:
            count += time_ - began >= HARD_BRAKE_FOR_S
            began = None
    if began is not None and t[-1] - began >= HARD_BRAKE_FOR_S:
        count += 1
    return int(count)


def _reversals(angle: np.ndarray) -> int:
    """Changes of steering direction by at least `REVERSAL_RAD` from the
    furthest the wheel went the other way; the first move sets a direction
    and is not one."""
    if not len(angle):
        return 0
    count, direction, extreme = 0, 0, float(angle[0])
    for value in (float(v) for v in angle[1:]):
        if (direction > 0 and value > extreme) or (direction < 0 and value < extreme):
            extreme = value
            continue
        moved = value - extreme
        if abs(moved) >= REVERSAL_RAD and (direction == 0 or (moved > 0) != (direction > 0)):
            count += direction != 0
            direction, extreme = (1 if moved > 0 else -1), value
    return int(count)


def _told(detail: str) -> tuple[bool, float]:
    """Whether a drive's outcome says it was stopped by a hit, and the damage
    it gives (core.control.limits.Collided): recorded before the fix in
    tools/follow.py, the frame that showed the hit is not in the recording."""
    if not detail.startswith("hit something"):
        return False, 0.0
    found = re.search(r"([0-9.]+) % damage", detail)
    return True, float(found.group(1)) if found else 0.0


def _impacts(t: np.ndarray, speed: np.ndarray) -> int:
    """Times the speed fell `IMPACT_DROP_MPS` within `IMPACT_WITHIN_S`: a hit."""
    count, last = 0, -math.inf
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > IMPACT_WITHIN_S:
            j += 1
        drop = float(np.max(np.abs(speed[j : i + 1])) - abs(speed[i]))
        if drop >= IMPACT_DROP_MPS and t[i] - last > 1.0:
            count += 1
            last = t[i]
    return int(count)


def _lead(acc: np.ndarray | None) -> tuple[float, float]:
    """(shortest time to collision, s; shortest gap, m) to the lead followed."""
    if acc is None or not len(acc):
        return math.inf, math.inf
    acc = np.asarray(acc, float)
    distance, closing = acc[:, 3], acc[:, 4]
    seen = np.isfinite(distance)
    gap = float(distance[seen].min()) if seen.any() else math.inf
    # The gap's rate of change: negative, the lead is being caught up.
    approaching = seen & np.isfinite(closing) & (closing < -0.1)
    if not approaching.any():
        return math.inf, gap
    ttc = float((distance[approaching] / -closing[approaching]).min())
    return ttc, gap
