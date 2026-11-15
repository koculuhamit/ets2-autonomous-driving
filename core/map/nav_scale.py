"""How the game's navigation counts a road: how long, and how fast.

The world is drawn smaller than the Europe it stands for, and the game counts
distance in the Europe's metres: the job's "1,450 km" and the telemetry's
navigation distance are world metres times a scale. Measured over 1,505
stretches of 150 m in twenty recorded drives -- navigation distance lost per
world metre driven -- the scale is

* 19 outside cities,
* 3 inside a city's area (Hamburg, Milano, Calais, Antwerp, Dover: 3.0),
* 15 in Britain (the Dover to London motorway: 15.0),

matching the game's own 1:19, 1:3 and 1:15. A city's area is a rectangle
`width` by `height` centred on its item's position (taken from a corner, the
stretches through Hamburg measured 3 outside it and 19 inside).

That the game's navigation distance can be reproduced from the map is what
lets a route planned here be checked against the game's: equal lengths, the
same route.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_CLASS",
    "GPS_MAX_KMH",
    "SCALE_CITY",
    "SCALE_GBR",
    "SCALE_OPEN",
    "SPEED_CLASSES",
    "SPEED_SHARE",
    "CityArea",
    "class_codes",
    "lane_nav_lengths",
    "lane_scales",
    "nav_speeds",
    "road_side_class",
    "spread_classes",
]

SCALE_OPEN: Final = 19.0
SCALE_CITY: Final = 3.0
SCALE_GBR: Final = 15.0

_SCALE_BY_COUNTRY: Final = {"gbr": SCALE_GBR}


@dataclass(frozen=True, slots=True)
class CityArea:
    """A city's area: a rectangle centred on (x, z)."""

    x: float
    z: float
    width: float
    height: float


def lane_scales(
    lane_owner: np.ndarray,
    owner_country: Mapping[int, int],
    countries: Mapping[int, str],
) -> np.ndarray:
    """The scale outside cities of each lane, by the country of the road or
    prefab it belongs to."""
    by_owner: dict[int, float] = {}
    for owner, country in owner_country.items():
        scale = _SCALE_BY_COUNTRY.get(countries.get(country, ""))
        if scale is not None:
            by_owner[int(owner)] = scale
    scales = np.full(len(lane_owner), SCALE_OPEN, dtype=np.float32)
    if by_owner:
        owners = np.asarray(lane_owner, dtype=np.uint64)
        special = np.array(sorted(by_owner), dtype=np.uint64)
        hit = np.isin(owners, special)
        scales[hit] = [by_owner[int(o)] for o in owners[hit]]
    return scales


def lane_nav_lengths(
    points: np.ndarray,
    lane_start: np.ndarray,
    scales: np.ndarray,
    cities: Iterable[CityArea],
) -> np.ndarray:
    """Navigation metres of each lane: each segment's world length times the
    city scale if its middle is in a city's area, else the lane's own."""
    points = np.asarray(points, dtype=np.float64)
    lane_start = np.asarray(lane_start, dtype=np.int64)
    lanes = len(lane_start) - 1
    # Segment i joins point i and i + 1; those crossing from one lane into
    # the next are dropped below.
    xz = points[:, [0, 2]]
    lengths = np.hypot(*np.diff(xz, axis=0).T)
    middles = 0.5 * (xz[1:] + xz[:-1])
    owner = np.repeat(np.arange(lanes), np.diff(lane_start))[:-1]
    within = np.ones(len(lengths), dtype=bool)
    within[lane_start[1:-1] - 1] = False  # last point of a lane to first of the next

    factor = np.asarray(scales, dtype=np.float64)[owner]
    in_city = _in_cities(middles, list(cities))
    factor[in_city] = SCALE_CITY

    out = np.bincount(owner[within], weights=(lengths * factor)[within], minlength=lanes)
    return out.astype(np.float32)


def _in_cities(xz: np.ndarray, cities: list[CityArea]) -> np.ndarray:
    inside = np.zeros(len(xz), dtype=bool)
    if not cities or not len(xz):
        return inside
    order = np.argsort(xz[:, 0], kind="stable")
    xs = xz[order, 0]
    for city in cities:
        lo = np.searchsorted(xs, city.x - city.width / 2.0, side="left")
        hi = np.searchsorted(xs, city.x + city.width / 2.0, side="right")
        if lo >= hi:
            continue
        candidates = order[lo:hi]
        near = np.abs(xz[candidates, 1] - city.z) <= city.height / 2.0
        inside[candidates[near]] = True
    return inside


# --------------------------------------------------------------------- #
# How fast: what the game's GPS weighs a route by.

SPEED_SHARE: Final = {"motorway": 0.9, "expressway": 0.9, "local_road": 0.8}
"""The share of a road's truck speed limit the GPS reckons on. Measured as
navigation distance lost per second of navigation time, over 1,800 stretches
in recorded drives: 0.900 of the limit on motorways and expressways (72 km/h
at 80, 81 at 90), 0.800 on local roads (48 at 60)."""

GPS_MAX_KMH: Final = 90.0
"""The fastest a limit counts at: a truck's limiter. British motorways and
expressways are 60 mph (96.6 km/h) for trucks, yet the GPS reckons 81.0
km/h on them, nine tenths of 90, over 20 stretches of recorded drives;
their local roads, 50 mph, it reckons at 64.4, eight tenths of 80.5."""

DEFAULT_CLASS: Final = "local_road"
DEFAULT_LIMIT_KMH: Final = 60.0
"""For a lane whose country states no truck limit for its class."""

_RANK: Final = {"motorway": 3, "expressway": 2, "local_road": 1}

SPEED_CLASSES: Final = ("local_road", "expressway", "motorway")
"""A lane's speed class as stored: its index here."""


def class_codes(classes: Iterable[str]) -> np.ndarray:
    """Speed classes as `SPEED_CLASSES` indices."""
    index = {name: code for code, name in enumerate(SPEED_CLASSES)}
    return np.asarray([index.get(c, 0) for c in classes], dtype=np.uint8)


def road_side_class(
    rule: str, lane_types: Iterable[str], speed_classes: Mapping[str, str]
) -> str | None:
    """A road side's speed class: its own rule if it names one, else what
    most of its lanes' traffic lane types say; None if neither says."""
    if rule in _RANK:
        return rule
    counted: dict[str, int] = {}
    for lane_type in lane_types:
        found = speed_classes.get(lane_type)
        if found in _RANK:
            counted[found] = counted.get(found, 0) + 1
    if not counted:
        return None
    return max(counted, key=lambda c: (counted[c], _RANK[c]))


def spread_classes(
    classes: list[str | None],
    successor_start: np.ndarray,
    successors: np.ndarray,
    rounds: int = 8,
) -> list[str]:
    """Classes for the lanes without one -- a junction's -- from the lanes
    they lead to, else the lanes that lead into them; then the default."""
    n = len(classes)
    starts = np.asarray(successor_start).tolist()
    nexts = np.asarray(successors).tolist()
    before: list[list[int]] = [[] for _ in range(n)]
    for lane in range(n):
        for k in range(starts[lane], starts[lane + 1]):
            before[nexts[k]].append(lane)
    out = list(classes)
    for _ in range(rounds):
        changed = False
        for lane in range(n):
            if out[lane] is not None:
                continue
            ahead = [out[nexts[k]] for k in range(starts[lane], starts[lane + 1])]
            ahead = [c for c in ahead if c is not None]
            behind = [out[b] for b in before[lane] if out[b] is not None]
            pick = ahead or behind
            if pick:
                out[lane] = max(pick, key=lambda c: _RANK.get(c, 0))
                changed = True
        if not changed:
            break
    return [c if c is not None else DEFAULT_CLASS for c in out]


def nav_speeds(
    classes: Iterable[str],
    countries: Iterable[int],
    limits: Mapping[int, Mapping[str, object]],
) -> np.ndarray:
    """The speed, km/h, the GPS reckons each lane at: its country's truck
    limit for its class, up to `GPS_MAX_KMH`, times the class's share."""
    out = []
    for speed_class, country in zip(classes, countries, strict=True):
        limit = limits.get(int(country), {}).get(speed_class)
        kmh = float(getattr(limit, "limit", 0.0) or getattr(limit, "max_limit", 0.0) or 0.0)
        if kmh <= 0.0:
            kmh = DEFAULT_LIMIT_KMH
        kmh = min(kmh, GPS_MAX_KMH)
        out.append(kmh * SPEED_SHARE.get(speed_class, SPEED_SHARE[DEFAULT_CLASS]))
    return np.asarray(out, dtype=np.float32)
