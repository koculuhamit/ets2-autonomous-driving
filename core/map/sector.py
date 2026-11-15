"""Reading what the extractor wrote.

One file per extraction run, holding the roads and junctions of the sectors it
was asked for. Coordinates are the game's world `(x, y, z)`, the same ones the
telemetry reports -- measured to be identical, with no transform, see
`docs/results/map_extraction.md`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from core.map import dlc

__all__ = [
    "City",
    "Company",
    "Ferry",
    "FerryLine",
    "LaneMeasurement",
    "MapPoint",
    "Prefab",
    "PrefabLane",
    "Road",
    "RoadLook",
    "Sector",
    "Semaphore",
    "SemaphoreProfile",
    "Signal",
    "TruckSpeedLimit",
]

FORMAT: Final = "ets2ad-map-roads"
FORMAT_VERSION: Final = 8
READABLE_VERSIONS: Final = frozenset({6, 7, 8})
"""7 added the ferries; a 6 is read as having none. 8 added what the GPS
weighs roads by (speed classes, truck speed limits, avoided roads); a 7 is
read as having none of it."""


@dataclass(frozen=True, slots=True)
class RoadLook:
    """How many lanes a road has and where their centres are.

    `lane_offsets` is measured from the junctions the road meets rather than
    derived from a lane width, because the game states no width anywhere
    readable. Positive is left of the road's direction.

    `lane_forward` is measured the same way, and matters more than it looks:
    a one-way road has every lane running the same way regardless of which
    side of the centreline it sits on, and a rule based on the side gets six
    of the fifteen looks surveyed in a city sector wrong.
    """

    lanes_left: int
    lanes_right: int
    road_offset: float
    lane_offsets: tuple[float, ...] | None
    lane_forward: tuple[bool, ...] | None
    lane_agreement: tuple[float, ...] | None
    """Share of junctions that agreed on each lane's direction."""

    lane_types_left: tuple[str, ...] = ()
    """Traffic lane type of each lane left of the centreline
    (`traffic_lane.road.motorway`), which gives its speed class."""

    lane_types_right: tuple[str, ...] = ()

    defined: LaneMeasurement | None = None
    """The layout worked out from the look's definition where it was never
    measured (`defined_layout`); None where that cannot be trusted."""

    left_hand: LaneMeasurement | None = None
    """The same look measured where traffic keeps left, if it is used there.

    Kept apart because pooling Britain with the continent made the motorway
    looks a coin toss -- one came back 54% one way, another backwards.
    """

    def measurement(self, left_hand_traffic: bool) -> LaneMeasurement | None:
        """The lane layout for roads on one side of the road or the other.

        Where a look was only ever measured where traffic keeps right, its
        left-hand layout is the mirror image: every lane moved to the other
        side of the centreline, still running the same way along the road.
        That is not assumed. Of the ten looks measured both ways across
        Europe, all ten are exact mirrors, the worst lane 0.4 cm out. It
        filled 36 British motorway roads that had been left without lanes.
        """
        right = (
            None
            if self.lane_offsets is None
            else LaneMeasurement(self.lane_offsets, self.lane_forward, self.lane_agreement)
        )
        if right is None and self.left_hand is None and self.defined is not None:
            right = self.defined
        if not left_hand_traffic:
            return right
        if self.left_hand is not None:
            return self.left_hand
        return None if right is None else right.mirrored()

    @property
    def lane_count(self) -> int:
        return self.lanes_left + self.lanes_right


@dataclass(frozen=True, slots=True)
class LaneMeasurement:
    """Where a look's lanes are and which way they run, as the junctions say."""

    offsets: tuple[float, ...]
    forward: tuple[bool, ...] | None
    agreement: tuple[float, ...] | None

    def mirrored(self) -> LaneMeasurement:
        """The same lanes on the other side of the centreline."""
        return LaneMeasurement(
            offsets=tuple(-offset for offset in self.offsets),
            forward=self.forward,
            agreement=self.agreement,
        )


@dataclass(frozen=True, slots=True)
class Road:
    """A stretch of road between two map nodes."""

    uid: str
    look: str
    start_node: str
    end_node: str
    ai_vehicles: bool
    """Whether the game's own traffic uses it. False on service roads."""

    left_hand_traffic: bool
    hidden: bool
    points: np.ndarray
    """`(n, 3)` centreline, sampled at the extractor's spacing."""

    dlc_guard: int = 0
    """Which DLC it belongs to; see `core.map.dlc`."""

    country: int = 0
    """Country id, as the map's nodes carry it; see `Sector.countries`."""

    gps_avoid: bool = False
    """A road the game's GPS routes round."""

    rule_left: str = ""
    """A side's own speed class (`motorway`, `local_road`, ...), if it sets one."""

    rule_right: str = ""


@dataclass(frozen=True, slots=True)
class PrefabLane:
    """One drivable path through a junction.

    The geometry is only half of it. Which lanes it leads to, which light
    governs it and which indicator it expects are all stated by the game, and
    all of it is needed to cross a junction the way traffic expects.
    """

    points: np.ndarray
    length: float
    from_node: str | None
    """Uid of the map node it starts at, or None if it starts inside."""

    to_node: str | None
    next_lanes: tuple[int, ...]
    previous_lanes: tuple[int, ...]
    semaphore: int
    """Id of the traffic light governing it, or -1."""

    rule: str
    blinker: str


@dataclass(frozen=True, slots=True)
class Semaphore:
    """A traffic light, where the map says it is."""

    id: int
    type: str
    profile: str
    position: np.ndarray


@dataclass(frozen=True, slots=True)
class SemaphoreProfile:
    """How a kind of traffic light or barrier behaves, from /def/world."""

    name: str
    types: tuple[str, ...]
    """What each of a prefab's semaphores is under this profile, by its id:
    `traffic_light`, `traffic_light_major`, `barrier_manual`, ..."""

    sleep: tuple[int, int] | None
    """Minutes of the day between which the lights are dark, or None. Many
    crossroad lights sleep from 23:30 (1410) to 03:00 (180)."""


@dataclass(frozen=True, slots=True)
class Signal:
    """What stands at the start of a lane: a traffic light or a barrier."""

    kind: str
    """The profile's type for this semaphore, or the semaphore's own type
    when it does not use a profile; empty when unknown."""

    profile: str
    sleep: tuple[int, int] | None = None

    @property
    def is_traffic_light(self) -> bool:
        return self.kind.startswith("traffic_light")

    def asleep(self, minute_of_day: float) -> bool:
        """Whether the light is dark at this time of day."""
        if self.sleep is None:
            return False
        start, end = self.sleep
        minute = minute_of_day % 1440
        if start <= end:
            return start <= minute < end
        return minute >= start or minute < end


@dataclass(frozen=True, slots=True)
class Prefab:
    """A junction, roundabout, service area or company yard."""

    uid: str
    model: str
    node_uids: tuple[str, ...]
    node_positions: np.ndarray
    lanes: tuple[PrefabLane, ...]
    semaphores: tuple[Semaphore, ...]
    placement_residual: float
    """How far the fitted placement missed the prefab's own nodes, metres."""

    dlc_guard: int = 0
    country: int = 0
    light_profile: str | None = None
    """The light timing the map item gives the junction; "" for none, None
    for an extraction that did not read it (`Sector.signal`)."""

    gas_pumps: tuple[np.ndarray, ...] = ()
    """Where a truck stands to fill up, `(3,)` each: a gas station's pump lanes."""

    map_points: tuple[MapPoint, ...] = ()
    """The prefab's map drawing, where it has somewhere to park: the yard."""


@dataclass(frozen=True, slots=True)
class MapPoint:
    """One point of a prefab's map drawing: of a road strip, or a polygon's outline."""

    position: tuple[float, float]
    """World (x, z)."""

    neighbors: tuple[int, ...]
    size: str
    """`Polygon`, or a road's width: `OneWay`, `TwoLanes`, ..."""

    color: str
    """A polygon's: `Light` (paved), `Dark` (built on), `Green`, or `None`."""


@dataclass(frozen=True, slots=True)
class Company:
    """A company depot: where a job starts or ends.

    `company` and `city` are the same ids the telemetry's job carries, which is
    how a job's destination is found on the map.
    """

    company: str
    city: str
    position: np.ndarray
    prefab: str | None
    """Uid of the prefab it belongs to: the yard, and the gates into it."""

    spots: tuple[tuple[str, np.ndarray], ...]
    """Parking spots, as (Load or Unload, position)."""

    spot_details: tuple[tuple[str, np.ndarray], ...] = ()
    """Each spot's (difficulty, `(2,)` direction (x, z)), in `spots`' order;
    empty for an extraction that did not say."""


@dataclass(frozen=True, slots=True)
class City:
    city: str
    position: np.ndarray
    width: float
    height: float
    shown: bool
    """Whether the game shows it on its map."""


@dataclass(frozen=True, slots=True)
class TruckSpeedLimit:
    """A country's truck speed limits for one speed class, km/h; 0 is none."""

    limit: float
    urban_limit: float
    max_limit: float


@dataclass(frozen=True, slots=True)
class Ferry:
    """Where a ship or the Channel Tunnel's train is boarded, in a port."""

    port: str
    """The port's id, as the ferry lines name it: `calais`, `frederikshv`."""

    position: np.ndarray
    unload_offset: np.ndarray
    """Where the truck is put ashore on arriving, from `position`."""

    train: bool


@dataclass(frozen=True, slots=True)
class FerryLine:
    """A crossing the game offers, one way: from one port to another."""

    start: str
    end: str
    distance_km: float
    """As the game states it: in its own kilometres, like the GPS's."""

    minutes: float = 0.0
    """How long the crossing takes, as the game states it."""


@dataclass(frozen=True, slots=True)
class Sector:
    """Everything one extraction run produced."""

    sectors: tuple[tuple[int, int], ...]
    sector_size: int
    looks: dict[str, RoadLook]
    roads: tuple[Road, ...]
    prefabs: tuple[Prefab, ...]
    countries: dict[int, str] = field(default_factory=dict)
    """Country id to ISO 3166 alpha-3, from the game's own definitions."""

    archives: tuple[str, ...] = ()
    """The map archives it was read from: base_map.scs and the map DLCs installed."""

    companies: tuple[Company, ...] = ()
    cities: tuple[City, ...] = ()
    semaphore_profiles: dict[str, SemaphoreProfile] = field(default_factory=dict)
    """By name, with inheritance resolved. Empty for an older extraction."""

    ferries: tuple[Ferry, ...] = ()
    ferry_lines: tuple[FerryLine, ...] = ()

    speed_classes: dict[str, str] = field(default_factory=dict)
    """Traffic lane type to speed class (`traffic_lane.road.local` -> `local_road`)."""

    truck_speed_limits: dict[int, dict[str, TruckSpeedLimit]] = field(default_factory=dict)
    """By country id, then speed class."""

    def signal(self, prefab: Prefab, semaphore: int) -> Signal | None:
        """What semaphore `semaphore` of `prefab` is, or None if it has none.

        A semaphore either uses a profile, which gives its type by the
        semaphore's id, or states its own type (`TrafficLightMinor`, ...)
        and names a profile only for the timing -- and the sleep.
        """
        if semaphore < 0:
            return None
        found = next((s for s in prefab.semaphores if s.id == semaphore), None)
        if found is None:
            # The lane names a semaphore the junction does not have: in
            # city-01, a junction with no light on that approach at all.
            return None
        if found.type.startswith("TrafficLight") and prefab.light_profile == "":
            # A universal junction's descriptor has lights whether the map
            # lights them or not; the map item's profile does. Without one
            # there is no light in the game: in Nynäshamn and Uppsala five
            # junctions like that held the truck for a lamp, some 46 s each,
            # while each light it stopped at and saw change had a profile.
            return None
        profile = self.semaphore_profiles.get(found.profile)
        sleep = profile.sleep if profile is not None else None
        if found.type != "UseProfile":
            return Signal(_snake(found.type), found.profile, sleep)
        if profile is None:
            return Signal("", found.profile)
        if semaphore < len(profile.types):
            kind = profile.types[semaphore]
        else:
            # More semaphores than the profile lists types for -- a toll
            # plaza's twentieth booth: take them for the last kind it has.
            kind = next((t for t in reversed(profile.types) if t != "model_only"), "")
        return Signal(kind, found.profile, sleep)

    def restricted_to(self, owned: Iterable[str]) -> Sector:
        """Only the roads and junctions that exist for someone owning `owned`.

        The map carries every DLC's roads; this is the map a particular player
        drives. Roads into a region they do not own end where the game puts its
        barrier, which is what a planner for that player needs to know.
        """
        owned = frozenset(owned)
        prefabs = tuple(p for p in self.prefabs if dlc.allowed(p.dlc_guard, owned))
        kept = {p.uid for p in prefabs}
        return replace(
            self,
            roads=tuple(r for r in self.roads if dlc.allowed(r.dlc_guard, owned)),
            prefabs=prefabs,
            # A depot exists where its yard does.
            companies=tuple(c for c in self.companies if c.prefab in kept),
        )

    @classmethod
    def load(cls, path: str | Path) -> Sector:
        """Read an extractor output file.

        Raises:
            ValueError: the file is not an extraction, or is a version this
                code does not know how to read.
        """
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != FORMAT:
            raise ValueError(f"not a map extraction: {document.get('format')!r}")
        if document.get("format_version") not in READABLE_VERSIONS:
            raise ValueError(
                f"extraction format version {document.get('format_version')}, "
                f"expected {FORMAT_VERSION}"
            )
        return cls(
            sectors=tuple((int(s[0]), int(s[1])) for s in document["sectors"]),
            sector_size=int(document["sector_size"]),
            looks={name: _look(body, name) for name, body in document["road_looks"].items()},
            roads=tuple(_road(body) for body in document["roads"]),
            prefabs=tuple(_prefab(body) for body in document["prefabs"]),
            countries={int(k): v for k, v in (document.get("countries") or {}).items()},
            archives=tuple(document.get("archives") or ()),
            companies=tuple(_company(body) for body in document.get("companies") or ()),
            cities=tuple(_city(body) for body in document.get("cities") or ()),
            semaphore_profiles=_profiles(document.get("semaphore_profiles") or {}),
            ferries=tuple(
                Ferry(
                    port=body["port"],
                    position=np.asarray(body["position"], dtype=float),
                    unload_offset=np.asarray(body["unload_offset"], dtype=float),
                    train=bool(body["train"]),
                )
                for body in document.get("ferries") or ()
            ),
            ferry_lines=tuple(
                FerryLine(
                    body["from"],
                    body["to"],
                    float(body["distance"] or 0.0),
                    float(body.get("time") or 0.0),
                )
                for body in document.get("ferry_connections") or ()
            ),
            speed_classes=dict(document.get("speed_classes") or {}),
            truck_speed_limits={
                int(country): {
                    speed_class: TruckSpeedLimit(
                        float(limits.get("limit") or 0.0),
                        float(limits.get("urban_limit") or 0.0),
                        float(limits.get("max_limit") or 0.0),
                    )
                    for speed_class, limits in by_class.items()
                }
                for country, by_class in (document.get("truck_speed_limits") or {}).items()
            },
        )


def _snake(name: str) -> str:
    """`TrafficLightMinor` as the profiles spell it: `traffic_light_minor`."""
    return "".join(f"_{c.lower()}" if c.isupper() and i else c.lower() for i, c in enumerate(name))


def _profiles(raw: dict[str, Any]) -> dict[str, SemaphoreProfile]:
    """Profiles with what each inherits filled in.

    A profile states only what it changes: `cr_1x1.italy` inherits
    everything from `cr_1x1.netherlands`, which inherits from `cr_1x1`. A
    sleep window of 0 to 0 is a profile saying its lights never sleep.
    """
    resolved: dict[str, SemaphoreProfile] = {}

    def resolve(name: str, seen: frozenset[str]) -> SemaphoreProfile:
        if name in resolved:
            return resolved[name]
        body = raw.get(name) or {}
        parent_name = body.get("inherited")
        parent = (
            resolve(parent_name, seen | {name})
            if parent_name in raw and parent_name not in seen
            else SemaphoreProfile(name, (), None)
        )
        types = tuple(body.get("types") or ()) or parent.types
        start, end = body.get("sleep_start"), body.get("sleep_end")
        if start is None or end is None:
            sleep = parent.sleep
        else:
            sleep = None if start == end else (int(start), int(end))
        resolved[name] = SemaphoreProfile(name, types, sleep)
        return resolved[name]

    for name in raw:
        resolve(name, frozenset())
    return resolved


def _company(body: dict[str, Any]) -> Company:
    return Company(
        company=body["company"],
        city=body["city"],
        position=np.asarray(body["position"], dtype=float),
        prefab=body.get("prefab"),
        spots=tuple(
            (spot["type"], np.asarray(spot["position"], dtype=float))
            for spot in body.get("spots") or ()
        ),
        spot_details=tuple(
            (spot.get("difficulty", ""), np.asarray(spot["direction"], dtype=float))
            for spot in body.get("spots") or ()
            if "direction" in spot
        ),
    )


def _city(body: dict[str, Any]) -> City:
    return City(
        city=body["city"],
        position=np.asarray(body["position"], dtype=float),
        width=float(body["width"]),
        height=float(body["height"]),
        shown=bool(body["shown"]),
    )


DEFINED_LANE_WIDTH_M: Final = 4.5
DEFINED_FAMILIES: Final = ("look",)
"""Look families whose `road_offset` is where the inner lane's edge is, so a
layout can be worked out from the definition alone: in each of the 13
two-way "look" looks measured, every lane sits at road_offset + (k + 0.5) x
4.5 m. Other families mean something else by it -- a blke look's lanes
straddle it -- and are left unmeasured rather than guessed. Seven British
motorway looks were never measured, having no junction to measure them
at, and left Plymouth cut off from the rest of the map."""


def defined_layout(
    name: str, lanes_left: int, lanes_right: int, road_offset: float
) -> LaneMeasurement | None:
    """A two-way look's lanes from its definition, where its family allows:
    right of the centreline forward, as where traffic keeps right."""
    family = next(
        (f for f in DEFINED_FAMILIES if name.startswith(f) and name[len(f) :][:1].isdigit()), None
    )
    if family is None or not lanes_left or not lanes_right:
        return None
    width = DEFINED_LANE_WIDTH_M
    right = [-(road_offset + (k + 0.5) * width) for k in reversed(range(lanes_right))]
    left = [road_offset + (k + 0.5) * width for k in range(lanes_left)]
    return LaneMeasurement(
        offsets=tuple(right + left),
        forward=tuple([True] * lanes_right + [False] * lanes_left),
        agreement=None,
    )


def _look(body: dict[str, Any], name: str = "") -> RoadLook:
    offsets = body.get("lane_offsets")
    forward = body.get("lane_forward")
    agreement = body.get("lane_agreement")
    left_hand = body.get("left_hand_traffic")
    return RoadLook(
        lanes_left=int(body.get("lanes_left") or 0),
        lanes_right=int(body.get("lanes_right") or 0),
        road_offset=float(body.get("road_offset") or 0.0),
        lane_offsets=tuple(float(o) for o in offsets) if offsets else None,
        lane_forward=tuple(bool(f) for f in forward) if forward else None,
        lane_agreement=tuple(float(a) for a in agreement) if agreement else None,
        lane_types_left=tuple(body.get("lane_types_left") or ()),
        lane_types_right=tuple(body.get("lane_types_right") or ()),
        defined=defined_layout(
            name,
            int(body.get("lanes_left") or 0),
            int(body.get("lanes_right") or 0),
            float(body.get("road_offset") or 0.0),
        ),
        left_hand=_measurement(left_hand) if left_hand else None,
    )


def _measurement(body: dict[str, Any]) -> LaneMeasurement | None:
    offsets = body.get("lane_offsets")
    if not offsets:
        return None
    forward = body.get("lane_forward")
    agreement = body.get("lane_agreement")
    return LaneMeasurement(
        offsets=tuple(float(o) for o in offsets),
        forward=tuple(bool(f) for f in forward) if forward else None,
        agreement=tuple(float(a) for a in agreement) if agreement else None,
    )


def _road(body: dict[str, Any]) -> Road:
    return Road(
        uid=body["uid"],
        look=body["road_look"],
        start_node=body["start_node"],
        end_node=body["end_node"],
        ai_vehicles=bool(body["ai_vehicles"]),
        left_hand_traffic=bool(body["left_hand_traffic"]),
        hidden=bool(body["hidden"]),
        points=np.asarray(body["points"], dtype=float),
        dlc_guard=int(body.get("dlc_guard", 0)),
        country=int(body.get("country", 0)),
        gps_avoid=bool(body.get("gps_avoid", False)),
        rule_left=body.get("rule_left") or "",
        rule_right=body.get("rule_right") or "",
    )


def _prefab(body: dict[str, Any]) -> Prefab:
    nodes = body["nodes"]
    lanes = body.get("lanes") or ()
    return Prefab(
        uid=body["uid"],
        model=body["model"],
        node_uids=tuple(node["uid"] for node in nodes),
        node_positions=np.asarray([node["position"] for node in nodes], dtype=float),
        lanes=tuple(_lane(lane) for lane in lanes),
        semaphores=tuple(
            Semaphore(
                id=int(s["id"]),
                type=s["type"],
                profile=s["profile"],
                position=np.asarray(s["position"], dtype=float),
            )
            for s in body.get("semaphores") or ()
        ),
        placement_residual=float(body.get("placement_residual") or 0.0),
        dlc_guard=int(body.get("dlc_guard", 0)),
        country=int(body.get("country", 0)),
        light_profile=body.get("semaphore_profile"),
        gas_pumps=tuple(np.asarray(p, dtype=float) for p in body.get("gas_pumps") or ()),
        map_points=tuple(
            MapPoint(
                position=(float(m["position"][0]), float(m["position"][1])),
                neighbors=tuple(int(n) for n in m["neighbors"]),
                size=m["size"],
                color=m["color"],
            )
            for m in body.get("map_points") or ()
        ),
    )


def _lane(body: dict[str, Any]) -> PrefabLane:
    return PrefabLane(
        points=np.asarray(body["points"], dtype=float),
        length=float(body["length"]),
        from_node=body["from_node"],
        to_node=body["to_node"],
        next_lanes=tuple(int(i) for i in body["next"] if i >= 0),
        previous_lanes=tuple(int(i) for i in body["previous"] if i >= 0),
        semaphore=int(body["semaphore"]),
        rule=body["rule"],
        blinker=body["blinker"],
    )
