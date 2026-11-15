"""Following the game's own GPS: which way it turns, read off the minimap.

The planner's shortest route and the game's navigation route are not the
same: on the first delivery drive the truck turned left where the game's
GPS turned right, and from then on went its own way, past the yard it had
set out from. The telemetry carries the navigation's distance and time, not
its route. The one place the route is, is on screen: the minimap in the
bottom-right corner draws it in red over the yellow roads.

The minimap is a perspective view, heading-up, from a virtual camera above
and behind the truck: `MinimapCamera`. Its six numbers were fitted by
laying the map's lanes over the minimap's roads in 11 frames of two drives
(city streets in Turin, 0-40 km/h); the zoom did not change with speed.

The route is not followed from the picture. At each fork ahead that the
minimap shows, every way on is drawn into the minimap, and the one lying on
the red is the game's (`gps_turn`). Where that is not the planned one, the
route is planned again through it (`reroute`): the planner still chooses
the lanes, the GPS only which way at each junction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core.map import CompanyPlace, RoadNetwork
from core.planning.route import CHANGE, ONWARD, Route, RouteError, rejoin, reroute_from

__all__ = [
    "MINIMAP",
    "GpsRouter",
    "GpsTurn",
    "MinimapCamera",
    "estimate_zoom",
    "gps_turn",
    "on_route",
    "road_mask",
    "route_mask",
    "unpack_mask",
]

REGION: Final = (1570, 770, 1905, 1040)
"""The minimap in a 1920x1080 frame: left, top, right, bottom."""

ROUTE_HUE: Final = (4, 12)
"""OpenCV hue of the route's red. The roads' yellow is 19-22."""

GREEN_HUE: Final = (50, 70)
ROAD_HUE_MAX: Final = 30
"""Lane arrows, drawn on the route."""

MIN_SATURATION: Final = 150
MIN_VALUE: Final = 150

LOOK_AHEAD_M: Final = 150.0
"""How far along the route forks are looked for."""

BRANCH_M: Final = 50.0
"""How far along each way on is drawn to score it."""

MIN_POINTS: Final = 8
"""Fewer visible points than this and a way on is not judged."""

ON_ROUTE: Final = 0.6
OFF_ROUTE: Final = 0.35
"""A way on is the GPS's if at least `ON_ROUTE` of it lies on the red; the
planned one is overruled only if at most `OFF_ROUTE` of it does."""


@dataclass(frozen=True, slots=True)
class MinimapCamera:
    """Ground in the truck's frame to minimap pixels (in the cropped region)."""

    focal: float
    pitch: float
    height: float
    behind: float
    centre_u: float
    centre_v: float

    @property
    def zoom(self) -> float:
        """How far out the camera is, against the fitted one."""
        return self.height / MINIMAP_HEIGHT

    def zoomed(self, zoom: float) -> MinimapCamera:
        """The camera drawn back along its line of sight `zoom` times as far:
        the same as the ground being shrunk `zoom` times."""
        return replace(self, height=MINIMAP_HEIGHT * zoom, behind=MINIMAP_BEHIND * zoom)

    def project(self, right: NDArray, forward: NDArray) -> tuple[NDArray, NDArray]:
        """`(n, 2)` pixel positions, and which points are in front of the camera."""
        y = np.asarray(forward, float) + self.behind
        depth = y * math.cos(self.pitch) + self.height * math.sin(self.pitch)
        down = self.height * math.cos(self.pitch) - y * math.sin(self.pitch)
        ok = depth > 1.0
        safe = np.where(ok, depth, 1.0)
        u = self.centre_u + self.focal * np.asarray(right, float) / safe
        v = self.centre_v + self.focal * down / safe
        return np.column_stack([u, v]), ok


MINIMAP_HEIGHT: Final = 30.27
MINIMAP_BEHIND: Final = 47.96
MINIMAP: Final = MinimapCamera(221.15, 0.5773, MINIMAP_HEIGHT, MINIMAP_BEHIND, 164.99, 142.52)
"""At the zoom of the Turin drives. The minimap's zoom changes -- in Milan
at night it was 4.2 times further out -- so it is measured as it goes
(`estimate_zoom`)."""

ZOOMS: Final = tuple(float(z) for z in np.exp(np.linspace(np.log(0.7), np.log(8.0), 33)))
ZOOM_RADIUS_M: Final = 900.0


def unpack_mask(packed: bytes) -> NDArray[np.bool_] | None:
    """A mask `route_mask` made and the perception service packed; None if empty."""
    if not packed:
        return None
    left, top, right, bottom = REGION
    shape = (bottom - top, right - left)
    bits = np.unpackbits(np.frombuffer(packed, np.uint8))[: shape[0] * shape[1]]
    return bits.reshape(shape).astype(bool)


def road_mask(image: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Pixels of the minimap, cropped to `REGION`, that any road is drawn on."""
    left, top, right, bottom = REGION
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)
    roads = (hue >= ROUTE_HUE[0]) & (hue <= ROAD_HUE_MAX)
    green = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])
    return bright & (roads | green)


def estimate_zoom(
    network: RoadNetwork, pose: tuple[float, float, float], roads: NDArray[np.bool_]
) -> float | None:
    """The minimap's zoom, from laying the map's shown roads over its roads.

    Each zoom in `ZOOMS` is scored by how far the map's lanes fall from the
    drawn roads and the drawn roads from the lanes; the best wins. Hidden
    roads are left out: the minimap does not draw them. None if too little
    road is drawn to tell.
    """
    if roads.sum() < 500:
        return None
    x, z, course = pose
    pieces = []
    for lane in network.lanes_near(x, z, ZOOM_RADIUS_M):
        if network.lane_hidden[lane]:
            continue
        points = np.asarray(network.lane_points(int(lane)), float)[::2]
        dx, dz = points[:, 0] - x, points[:, 2] - z
        pieces.append(
            np.column_stack(
                [
                    -(dx * math.cos(course) - dz * math.sin(course)),
                    dx * math.sin(course) + dz * math.cos(course),
                ]
            )
        )
    if not pieces:
        return None
    ground = np.vstack(pieces)
    distance = cv2.distanceTransform((~roads).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    best: tuple[float, float] | None = None
    for zoom in ZOOMS:
        camera = MINIMAP.zoomed(zoom)
        uv, ok = camera.project(ground[:, 0], ground[:, 1])
        seen = ok & visible(roads.shape, uv, camera)
        if seen.sum() < 30:
            continue
        ij = uv[seen].astype(int)
        drawn = np.zeros(roads.shape, np.uint8)
        drawn[ij[:, 1], ij[:, 0]] = 255
        drawn = cv2.dilate(drawn, np.ones((3, 3), np.uint8))
        back = cv2.distanceTransform(255 - drawn, cv2.DIST_L2, 3)
        cost = float(np.minimum(distance[ij[:, 1], ij[:, 0]], 20).mean()) + float(
            np.minimum(back[roads], 20).mean()
        )
        if best is None or cost < best[1]:
            best = (zoom, cost)
    return None if best is None else best[0]


def route_mask(image: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Pixels of the minimap, cropped to `REGION`, that the GPS route covers."""
    left, top, right, bottom = REGION
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)
    red = (hue >= ROUTE_HUE[0]) & (hue <= ROUTE_HUE[1])
    green = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1])
    mask = bright & (red | green)
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def visible(mask_shape: tuple[int, int], uv: NDArray, camera: MinimapCamera) -> NDArray[np.bool_]:
    """Points on the minimap that nothing is drawn over."""
    height, width = mask_shape
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < width) & (v >= 30) & (v < height)
    inside &= ~((u < 100) & (v < 95))  # speed and speed limit
    inside &= ~((u > 250) & (v > 230))  # the zoom hint
    inside &= np.hypot(u - camera.centre_u, v - camera.centre_v) > 25  # the truck's arrow
    return inside


def on_route(
    mask: NDArray[np.bool_], right: NDArray, forward: NDArray, camera: MinimapCamera = MINIMAP
) -> float | None:
    """The share of a line, in the truck's frame, lying on the route; None if
    too little of it is on the minimap to tell."""
    uv, ok = camera.project(right, forward)
    seen = ok & visible(mask.shape, uv, camera)
    if seen.sum() < MIN_POINTS:
        return None
    ij = uv[seen].astype(int)
    return float(mask[ij[:, 1], ij[:, 0]].mean())


@dataclass(frozen=True, slots=True)
class GpsTurn:
    """Where the GPS goes another way than the route."""

    index: int
    """`route.lanes[index]` is the lane the fork is at the end of."""

    lane: int
    """The way on the GPS takes."""

    scores: dict[int, float]

    step: str = ONWARD
    """How `route.lanes[index]` leads to `lane`: onward, or a change into
    the lane beside it."""

    avoid: tuple[int, ...] = ()
    """The lanes of the way refused, which the new route must not go back
    through."""


def gps_turn(
    network: RoadNetwork,
    route: Route,
    at: int,
    pose: tuple[float, float, float],
    mask: NDArray[np.bool_],
    camera: MinimapCamera = MINIMAP,
) -> GpsTurn | None:
    """The first fork or choice of lane ahead, from `route.lanes[at]` on,
    where the GPS goes another way than `route`; None where they agree, or
    where it cannot tell."""
    travelled = 0.0
    for i in range(at, len(route.lanes) - 1):
        lane = route.lanes[i]
        if travelled > LOOK_AHEAD_M:
            break
        travelled += float(network.lane_length[lane])
        turn = _fork_choice(network, route, i, pose, mask, camera)
        if turn is None:
            # Only where no fork says otherwise: at a lane that both forks
            # and has a neighbour, changing into the neighbour won over
            # going on through the fork, and the new route came back into
            # the lane from beside it -- then went the other way again.
            turn = _lane_choice(network, route, i, pose, mask, camera)
        if turn is not None:
            return turn
    return None


def _fork_choice(
    network: RoadNetwork,
    route: Route,
    i: int,
    pose: tuple[float, float, float],
    mask: NDArray[np.bool_],
    camera: MinimapCamera,
) -> GpsTurn | None:
    """Where `route.lanes[i]` forks and the GPS takes another branch."""
    x, z, course = pose
    lane = route.lanes[i]
    if route.steps[i] != ONWARD:
        return None
    options = [int(o) for o in network.next_lanes(lane)]
    if len(options) < 2:
        return None
    scores: dict[int, float] = {}
    for option in options:
        line = _way_on(network, option, BRANCH_M * max(1.0, camera.zoom))
        dx, dz = line[:, 0] - x, line[:, 2] - z
        forward = dx * math.sin(course) + dz * math.cos(course)
        right = -(dx * math.cos(course) - dz * math.sin(course))
        score = on_route(mask, right, forward, camera)
        if score is not None:
            scores[option] = score
    planned = route.lanes[i + 1]
    if planned not in scores or len(scores) < 2:
        return None
    best = max(scores, key=scores.__getitem__)
    if best != planned and scores[best] >= ON_ROUTE and scores[planned] <= OFF_ROUTE:
        return GpsTurn(i, best, scores)
    return None


LANE_WAY_M: Final = 600.0
"""How far on two lanes side by side are followed to see where they part."""

DIVERGE_M: Final = 6.0
"""Of two ways side by side, only what lies further than this from the other
is compared: on the minimap two lanes of one road are a pixel apart, and
both lie on the route until the road splits."""

LANE_LATE_M: Final = 60.0
"""Where two lanes part nearer the truck than this, it is too late to move
across: the route is left to its fork, and the GPS to what follows."""


def _lane_choice(
    network: RoadNetwork,
    route: Route,
    i: int,
    pose: tuple[float, float, float],
    mask: NDArray[np.bool_],
    camera: MinimapCamera,
) -> GpsTurn | None:
    """Where the route picks its way by lane rather than at a fork: into
    the lane beside (`route.steps[i]` a change), or on in a lane whose
    neighbour goes elsewhere. On a Danish motorway the route moved into the
    right lane 400 m before it became the exit, which the GPS did not take;
    the lane has no fork, so the forks alone never saw it."""
    lane = route.lanes[i]
    if route.steps[i] == CHANGE:
        onward = [int(o) for o in network.next_lanes(lane)]
        if not onward:
            return None
        end = np.asarray(network.lane_points(lane), float)
        heading = math.atan2(end[-1, 0] - end[-2, 0], end[-1, 2] - end[-2, 2])
        stay = min(onward, key=lambda o: _turn(network, o, heading))
        others = [(stay, ONWARD, np.vstack([end, _way_on(network, stay, LANE_WAY_M)]))]
        planned, planned_lanes = _route_way(network, route, i + 1)
    else:
        others = [
            (int(a), CHANGE, _way_on(network, int(a), LANE_WAY_M))
            for a in network.lanes_alongside(lane)
        ]
        planned, planned_lanes = _route_way(network, route, i + 1)
        planned = np.vstack([np.asarray(network.lane_points(lane), float), planned])
    for other, step, way in others:
        apart_planned = _apart(planned, way)
        apart_other = _apart(way, planned)
        if len(apart_planned) == 0 or len(apart_other) == 0:
            continue
        parting = min(
            float(np.hypot(*(apart_planned[0, [0, 2]] - pose[:2]))),
            float(np.hypot(*(apart_other[0, [0, 2]] - pose[:2]))),
        )
        if parting < LANE_LATE_M:
            continue
        on_planned = _score(apart_planned, pose, mask, camera)
        on_other = _score(apart_other, pose, mask, camera)
        if on_planned is None or on_other is None:
            continue
        if on_other >= ON_ROUTE and on_planned <= OFF_ROUTE:
            refused = _refused(network, planned_lanes, way)
            if step == CHANGE:
                refused = (lane, *refused)  # not back into the lane left
            scores = {route.lanes[i + 1]: on_planned, other: on_other}
            return GpsTurn(i, other, scores, step, refused)
    return None


def _route_way(network: RoadNetwork, route: Route, start: int) -> tuple[NDArray, tuple[int, ...]]:
    """The route's own lanes from `route.lanes[start]`, to `LANE_WAY_M`: their
    points, and the lanes."""
    pieces, lanes = [], []
    covered = 0.0
    for lane in route.lanes[start:]:
        pieces.append(np.asarray(network.lane_points(lane), float))
        lanes.append(lane)
        covered += float(network.lane_length[lane])
        if covered >= LANE_WAY_M:
            break
    return np.vstack(pieces), tuple(lanes)


def _refused(network: RoadNetwork, lanes: tuple[int, ...], other: NDArray) -> tuple[int, ...]:
    """Of the lanes of a way refused, those up to where it parts from `other`
    and on while apart: not any it shares with `other` once they meet again,
    which a new route may well need."""
    refused = []
    parted = False
    for lane in lanes:
        apart = len(_apart(np.asarray(network.lane_points(lane), float), other)) > 0
        if parted and not apart:
            break
        parted |= apart
        refused.append(lane)
    return tuple(refused)


def _apart(line: NDArray, other: NDArray) -> NDArray:
    """The points of `line` further than `DIVERGE_M` from every point of `other`."""
    a, b = line[:, [0, 2]], other[::2, [0, 2]]
    nearest = np.full(len(a), np.inf)
    for start in range(0, len(b), 256):
        piece = b[start : start + 256]
        gaps = np.hypot(a[:, None, 0] - piece[None, :, 0], a[:, None, 1] - piece[None, :, 1])
        nearest = np.minimum(nearest, gaps.min(axis=1))
    return line[nearest > DIVERGE_M]


def _score(
    line: NDArray, pose: tuple[float, float, float], mask: NDArray[np.bool_], camera: MinimapCamera
) -> float | None:
    x, z, course = pose
    dx, dz = line[:, 0] - x, line[:, 2] - z
    forward = dx * math.sin(course) + dz * math.cos(course)
    right = -(dx * math.cos(course) - dz * math.sin(course))
    return on_route(mask, right, forward, camera)


def _way_on(network: RoadNetwork, lane: int, length: float = BRANCH_M) -> NDArray:
    """A lane and, if it is short, the straightest lanes after it, to `length`."""
    pieces = [np.asarray(network.lane_points(lane), float)]
    covered = float(network.lane_length[lane])
    while covered < length:
        options = [int(o) for o in network.next_lanes(lane)]
        if not options:
            break
        end = pieces[-1]
        heading = math.atan2(end[-1, 0] - end[-2, 0], end[-1, 2] - end[-2, 2])

        lane = min(options, key=lambda o, h=heading: _turn(network, o, h))
        pieces.append(np.asarray(network.lane_points(lane), float))
        covered += float(network.lane_length[lane])
    return np.vstack(pieces)


def _turn(network: RoadNetwork, lane: int, heading: float) -> float:
    """How far a lane's start turns from `heading`, radians."""
    p = network.lane_points(lane)
    start = math.atan2(p[1][0] - p[0][0], p[1][2] - p[0][2])
    return abs((start - heading + math.pi) % (2 * math.pi) - math.pi)


def reroute(network: RoadNetwork, route: Route, turn: GpsTurn, depot: CompanyPlace) -> Route:
    """`route` up to the fork, the GPS's way on, and back onto `route` where
    that way meets it again (`rejoin`); if it does not nearby, the shortest
    way from there to the depot's nearest gate.

    Raises:
        RouteError: no gate can be reached that way.
    """
    head = (*route.lanes[: turn.index + 1], turn.lane)
    steps = (*route.steps[: turn.index], turn.step)
    avoid = frozenset(turn.avoid)
    rejoined = rejoin(route, head, steps, turn.index, avoid)
    if rejoined is not None:
        return rejoined
    try:
        return reroute_from(network, head, steps, depot, avoid)
    except RouteError:
        if not avoid:
            raise
    # Kept off every lane of the way refused, the GPS's way found no depot
    # at all, twelve times in a row, and the route stayed on the exit the
    # GPS did not take. Its way first, then whatever is best from there.
    rejoined = rejoin(route, head, steps, turn.index)
    if rejoined is not None:
        return rejoined
    return reroute_from(network, head, steps, depot)


ZOOM_EVERY_S: Final = 5.0

REFUSED_S: Final = 30.0
"""A way the GPS was seen to refuse is not turned back onto for this long:
one odd pair of minimaps is not enough to undo a lane change just made."""

LATE_M: Final = 15.0
"""A fork nearer the truck than this is left as planned: a route changed
there came when the truck was already in the junction, going the planned
way, 8 m from the new route's line, and the drive stopped."""
LOCATE_BEHIND: Final = 2
LOCATE_AHEAD: Final = 12
LOCATE_RADIUS_M: Final = 6.0


@dataclass
class GpsRouter:
    """Keeps a route to the depot turning where the game's GPS turns.

    `update` is called with the newest minimap now and then. A fork where
    the GPS goes another way must be seen so twice running before the route
    is changed, so one odd frame of the minimap does not.
    """

    network: RoadNetwork
    depot: CompanyPlace
    route: Route
    at: int = 0
    """Index in `route.lanes` of the lane the truck is on, as last found."""

    camera: MinimapCamera = MINIMAP
    _pending: tuple[int, int] | None = None
    _refused: dict[int, float] = field(default_factory=dict)
    """Lanes of ways refused, and when: not turned back onto for `REFUSED_S`."""
    _zoom_at: float | None = None

    def locate(self, x: float, z: float) -> int:
        """Where along the route the truck is, as an index into its lanes.

        Near where it was last found first; failing that, anywhere on the
        route. A drive resumed half way along a route starts far beyond the
        first lanes, and searching only near them, the router never found
        the truck and checked the forks at the route's start for the rest of
        the drive: twice the truck went on past where the GPS turned off.
        """
        lanes = self.route.lanes
        near = range(max(0, self.at - LOCATE_BEHIND), min(len(lanes), self.at + LOCATE_AHEAD))
        for candidates in (near, range(len(lanes))):
            best: tuple[int, float] | None = None
            for i in candidates:
                projection = self.network.project(lanes[i], x, z)
                if projection.overshoot < 1.0 and (best is None or projection.distance < best[1]):
                    best = (i, projection.distance)
            if best is not None and best[1] < LOCATE_RADIUS_M:
                self.at = best[0]
                break
        return self.at

    def _to_fork(self, turn: GpsTurn, pose: tuple[float, float, float]) -> float:
        """Metres from the truck to the fork of `turn`, along the route."""
        lanes = self.route.lanes
        here = self.network.project(lanes[self.at], pose[0], pose[1])
        distance = float(self.network.lane_length[lanes[self.at]]) - here.s
        for i in range(self.at + 1, turn.index + 1):
            if self.route.steps[i - 1] == ONWARD:
                distance += float(self.network.lane_length[lanes[i]])
        return distance

    def update(
        self,
        pose: tuple[float, float, float],
        mask: NDArray[np.bool_],
        roads: NDArray[np.bool_] | None = None,
        now: float = 0.0,
    ) -> Route | None:
        """A new route if the GPS has been seen to turn another way, else None.

        `roads` (`road_mask`) lets the minimap's zoom be measured, every
        `ZOOM_EVERY_S`; without it the zoom last measured, or the fitted one.

        Raises:
            RouteError: the GPS's way leads nowhere the depot can be reached from.
        """
        if roads is not None and (self._zoom_at is None or now - self._zoom_at >= ZOOM_EVERY_S):
            zoom = estimate_zoom(self.network, pose, roads)
            if zoom is not None:
                self.camera = MINIMAP.zoomed(zoom)
                self._zoom_at = now
        at = self.locate(pose[0], pose[1])
        turn = gps_turn(self.network, self.route, at, pose, mask, self.camera)
        if (
            turn is None
            or self._to_fork(turn, pose) < LATE_M
            or now - self._refused.get(turn.lane, -math.inf) < REFUSED_S
        ):
            self._pending = None
            return None
        key = (self.route.lanes[turn.index], turn.lane)
        if self._pending != key:
            self._pending = key
            return None
        self._pending = None
        self.route = reroute(self.network, self.route, turn, self.depot)
        for lane in turn.avoid:
            self._refused[lane] = now
        return self.route
