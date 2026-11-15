"""Vehicles followed from frame to frame, and the one to follow.

A detection is one frame's opinion. Following a car needs more: that the car
in this frame is the one from the last, how far away it is without the
frame-to-frame jitter of a box edge a few pixels below the horizon, and how
fast that distance is closing. Tracks give all three.

Association is by box overlap, greedy on the best pairs first -- at the
detector's rate a vehicle moves a few pixels between frames, and nothing
cleverer is needed until measurements say otherwise. A track is confirmed
after `CONFIRM_HITS` sightings, so a single-frame false detection never
becomes a lead vehicle, and forgotten when unseen for `FORGET_AFTER_S`.

Distance is the median over the last `HISTORY_S` of sightings, each carried
forward to now at the closing speed -- a plain median is half a second old,
2.5 m too far for a car closing at 5 m/s. Lateral position is the median of
the last three: in a bend the whole scene sweeps sideways as the truck
turns, and a second-long median trailed the lane by 2 m at 60 m. Closing
speed does not come from differencing it: 100 m ahead, a box's bottom edge
sits 24 px below the horizon, and a pixel of cab pitch is 4 m. It comes from
how fast the box grows instead -- a box's width in pixels is `focal * width
/ distance`, so `d ln(width_px) / dt = -closing / distance`, and the cab
pitching does not change a box's width. Measured on a held-out drive,
differencing distances made a car 168 m ahead of a truck at 90 km/h drive
backwards at 152 km/h.

Growth alone is fooled the other way: a car coming out from behind another,
or into the frame, has a box that widens while it stays where it is. Run in
the shadow of a hand drive, a box going from 49 to 91 px in 0.3 s at a
steady 22.7 m became a car closing at 46 m/s, and a hard brake. So both
are measured -- growth, and the trend of the ground distance -- each as the
median of pairwise slopes, which one jumping box does not move; they must
agree on the direction, and the smaller is taken. A track younger than
`SETTLED_S` or with fewer than `SETTLED_SIGHTINGS` sightings reports no
closing at all.

The lead vehicle is the nearest confirmed track in the truck's path, with a
width a vehicle can have seen from behind. The path is the lane ahead as the
map has it (`Path`); without one, straight ahead -- which, where a motorway
splits, puts the gore sign in front of the truck and makes it a stopped car.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from core.perception.camera import GroundCamera
from core.perception.detection import Detection, Kind

__all__ = ["CrossingFilter", "Path", "Track", "Tracker", "lead_vehicle", "plausible"]

MATCH_IOU: Final = 0.3
CONFIRM_HITS: Final = 3
FORGET_AFTER_S: Final = 0.5
HISTORY_S: Final = 1.0
CROSSING_WINDOW_S: Final = 0.6
CROSSING_WITHIN_M: Final = 40.0
"""Crossing is only judged this near. Further out, a few metres of distance
error move a bend's lane sideways under a lead, and on a motorway drive that
alone dropped the lead 39 times in five minutes; the junction crossings that
needed catching were all within 33 m."""
STALE_MEASUREMENT_S: Final = 0.3
"""A track still seen, but not measured for this long -- its foot hidden by
the dashboard -- has no distance worth following. In the shadow of a city
drive such a track kept its last estimate, 14.3 m and closing at 7 m/s,
while the car it was crossed out of view, and became a hard brake."""
SETTLED_S: Final = 0.25
SETTLED_SIGHTINGS: Final = 5

MAX_VEHICLE_WIDTH_M: Final = 20.0
"""Wider than any vehicle seen from any side: a tractor and trailer broadside
is about 16.5 m. A tunnel portal seen as a truck came out at 38 m."""

MAX_LEAD_WIDTH_M: Final = 6.0
"""A vehicle ahead in the same lane shows its back, perhaps a little of its
side in a bend: a trailer's 2.55 m, plus 13.6 m x sin(10 deg)."""

CLIPPED_REACH_M: Final = 1.5
"""A vehicle cut off by the side of the view (`clipped`) is in the way only
where its inner edge -- the side of it in view -- comes within this of the
lane's middle: the truck's half width and a little. Its middle is not
known, and the middle of what is in view lies towards the lane: coaches,
tankers and caravans overtaking in the next lane, seen beside the cab's
pillar at 10 m, were taken for vehicles ahead about 11 times an hour, and
braked the truck at speed for a moment each."""

PATH_HALF_WIDTH_M: Final = 2.0
"""Of the truck's own lane, about its line of sight: half of 4.5 m, less a margin."""

CUT_MARGIN_PX: Final = 12.0
"""How near the bottom of the view a box's foot may be and still be the road.
The dashboard's top edge sits about 10 px above the crop and moves with the
seat: a car crossing close in front was hidden by it at 695 px, 5 px above the
cut. The price: nothing nearer than about 9.5 m gets a ground distance, which
is closer than any gap kept while moving, but not than a standstill gap."""

MAX_CROSSING_MPS: Final = 1.5
"""Moving across the lane faster than this, a vehicle is crossing the road,
not leading on it: at a junction, run in the shadow of a city drive, cars
crossing in front made three of four hard brakes the driver never made.
Crossing traffic is for junction logic to handle, not following.

Measured against the lane (`CrossingFilter`), not the camera: in a bend
everything sweeps across the image as the truck turns, and a first version
measuring against the camera dropped a third of the leads on a motorway."""

YIELD_CORRIDOR_M: Final = 2.5
"""Either side of the lane's middle, the room a crossing vehicle takes from the truck."""

BROADSIDE_M: Final = {"car": 3.5, "motorcycle": 1.5, "truck": 7.0, "bus": 7.0}
"""Seen at least this wide, a vehicle shows its side: it is going across the
road, not along it. From behind, a car is 1.8-2.0 m and a lorry 2.55 m, plus
a little of the side in a bend (4.5 m or 13.6 m x sin 10 deg); broadside, a
car is 4.5 m and a lorry 10-16 m. On the A38 into Plymouth cars going the
same way in the next lane, swept sideways in the bends of long slip-road
junctions, were given way to at 7 m/s^2 seven times; each showed its back."""

ALONG_SHARE: Final = 0.5
MIN_ALONG_MPS: Final = 5.0
"""A vehicle moving across the lane but along the road at least this share
of the truck's speed, and this fast, either way, is on the road, not
crossing it: a car coming down a slip road onto the main road, as fast as
the truck, or a lorry on the other carriageway coming the other way,
broadside to the camera as it passes close. Out of Folkestone and past it
such were given way to as if standing where they were seen, and braked the
truck at 7 m/s^2; one in the lane is followed like any other. A car
crossing a junction moves along the road at next to nothing."""

YIELD_MARGIN_S: Final = 1.5
TRUCK_LENGTH_M: Final = 17.0
MOVE_OFF_MPS2: Final = 1.0

CUT_IN_WITHIN_M: Final = 60.0
CUT_IN_WINDOW_S: Final = 1.0
CUT_IN_SETTLED_S: Final = 0.5
CUT_IN_SIGHTINGS: Final = 4
CUT_IN_MPS: Final = 0.5
CUT_IN_EDGE_M: Final = 1.5
CUT_IN_HORIZON_S: Final = 2.5
CUT_IN_COMING_MPS: Final = 4.0
CUT_IN_COMING_SHARE: Final = 0.3
"""A vehicle once read as coming towards the truck faster than the larger of
these -- 4 m/s, or 30 % of the truck's speed, as core.control.following tells
one oncoming -- is not cutting in, however it moves across the view: in a bend
the oncoming lane closes on the lane's middle, its closing speed lags down to
the truck's own, and cars going by were braked for hard, at up to 4.8 m/s^2
at 51 km/h. Standing or slow is not enough: the lorry moving across at Torino
read as all but standing, and it had to be followed. One in the lane's middle
is followed whatever it was read as."""

CUT_IN_BESIDE_M: Final = 6.0
"""A vehicle beside the lane -- its middle within `CUT_IN_BESIDE_M` of the
lane's, the next lane -- within `CUT_IN_WITHIN_M`, moving towards its
middle at `CUT_IN_MPS` or more over the last `CUT_IN_WINDOW_S`, is cutting
in (`CutInFilter`) if its near side will be within `CUT_IN_EDGE_M` of the
middle -- the truck's half width and a little -- before the truck gets to
it, or within `CUT_IN_HORIZON_S`. It is followed whatever its middle and
its width (`lead_vehicle`).

Out of Torino a lorry from a slip road moved across into the truck's lane
40 m ahead, at 0.9 m/s; its middle stayed 3-6 m off the lane's, and angled
it was over 6 m wide: it was never the lead, and was hit at 80 km/h. A
lorry going straight in the next lane drifts towards the middle by a
quarter of that as it is overtaken: the far end of its side, nearer the
middle of the image, is taken for as near as its back. And a lorry parked
27 m off the road, whose box jumped from its side to its back, moved
towards the lane by 3 m in a frame: the truck braked from 70 to 41 km/h.
A car overtaking alongside, 10 m ahead and as fast as the truck, its feet
just above the dashboard where distances are least sure, crept towards the
middle by a few tenths of a metre a second: within 4 s it would have been
in the lane, and it was followed. Within 2.5 s it is not; the lorry out of
Torino was in it within 1.5."""

MAX_LEAD_DISTANCE_M: Final = 120.0
"""Beyond this nothing is a lead. Following at 90 km/h with a 3 s gap is 75 m;
and 150 m out the box's foot is 16 px below the horizon, where a pixel is
9 m and a bend's lateral is anybody's guess."""

VEHICLES: Final = frozenset({Kind.CAR, Kind.TRUCK, Kind.BUS, Kind.MOTORCYCLE})


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    last_seen: float
    hits: int = 1
    kinds: Counter = field(default_factory=Counter)
    sightings: deque = field(default_factory=deque)
    """(time, distance, lateral, width, width in pixels) for the last `HISTORY_S`."""

    @property
    def kind(self) -> Kind:
        return self.kinds.most_common(1)[0][0]

    @property
    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS

    @property
    def measurement_age(self) -> float:
        """Seconds between the last sighting and the last one that measured anything."""
        if not self.sightings:
            return float("inf")
        return self.last_seen - self.sightings[-1][0]

    def estimate(self) -> tuple[float, float, float, float] | None:
        """(distance, closing speed, lateral, width) now; closing speed is
        negative when the gap shrinks. None when never seen on the road."""
        if not self.sightings:
            return None
        t, distance, lateral, width, pixels = (
            np.array(v) for v in zip(*self.sightings, strict=True)
        )
        typical = float(np.median(distance))
        closing = 0.0
        if len(t) >= SETTLED_SIGHTINGS and np.ptp(t) >= SETTLED_S and np.all(pixels > 0):
            from_growth = -typical * _median_slope(t, np.log(pixels))
            from_ground = _median_slope(t, distance)
            if from_growth * from_ground > 0:
                closing = float(np.sign(from_growth) * min(abs(from_growth), abs(from_ground)))
        now = float(np.median(distance + closing * (t[-1] - t)))
        return now, closing, float(np.median(lateral[-3:])), float(np.median(width))


class Tracker:
    def __init__(self, camera: GroundCamera) -> None:
        self.camera = camera
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def update(
        self, detections: Sequence[Detection], now: float, cut_row: float | None = None
    ) -> list[Track]:
        """Fold in one frame's detections, taken at `now` (seconds).

        `cut_row` is the bottom of what the detector was shown. A box reaching
        it is cut off there, and where it meets the road is not in view: it
        still counts as a sighting, but gives no distance. Taken as one, a car
        crossing close in front had its box doubled in width at the cut and
        became a hard brake.
        """
        vehicles = [d for d in detections if d.kind in VEHICLES]
        pairs = sorted(
            (
                (_iou(track.box, found.box), t, d)
                for t, track in enumerate(self.tracks)
                for d, found in enumerate(vehicles)
            ),
            reverse=True,
        )
        taken_tracks: set[int] = set()
        taken_found: set[int] = set()
        for overlap, t, d in pairs:
            if overlap < MATCH_IOU:
                break
            if t in taken_tracks or d in taken_found:
                continue
            taken_tracks.add(t)
            taken_found.add(d)
            self._see(self.tracks[t], vehicles[d], now, cut_row)
        for d, found in enumerate(vehicles):
            if d not in taken_found:
                track = Track(next(self._ids), found.box, now, hits=0)
                self._see(track, found, now, cut_row)
                self.tracks.append(track)
        self.tracks = [t for t in self.tracks if now - t.last_seen <= FORGET_AFTER_S]
        return self.tracks

    def _see(self, track: Track, found: Detection, now: float, cut_row: float | None) -> None:
        track.box = found.box
        track.last_seen = now
        track.hits += 1
        track.kinds[found.kind] += 1
        if cut_row is not None and found.box[3] >= cut_row - CUT_MARGIN_PX:
            return
        where = self.camera.locate(found.box)
        if where is not None:
            track.sightings.append((now, *where, found.box[2] - found.box[0]))
        while track.sightings and now - track.sightings[0][0] > HISTORY_S:
            track.sightings.popleft()


@dataclass(frozen=True)
class Path:
    """The middle of the lane ahead, as seen from the camera.

    `ahead` and `right` are metres forward of and to the right of the eye,
    sorted by `ahead`. Past its end the path is taken to go on straight.
    """

    ahead: np.ndarray
    right: np.ndarray

    def __post_init__(self) -> None:
        if len(self.ahead) < 2 or np.any(np.diff(self.ahead) <= 0):
            raise ValueError("a path needs two or more points, in order ahead")

    @classmethod
    def from_truck_frame(
        cls, ahead: Sequence[float], right: Sequence[float], camera: GroundCamera
    ) -> Path:
        """From points relative to the truck's centreline, as the map gives them."""
        ahead = np.asarray(ahead, float)
        keep = np.concatenate([[True], np.diff(ahead) > 0]) & (ahead > 0)
        # Keep only the stretch that runs away from the truck: a path that
        # turns back past 90 degrees has no single lateral for a distance.
        keep &= np.maximum.accumulate(ahead) == ahead
        return cls(ahead[keep], np.asarray(right, float)[keep] - camera.seat_right_m)

    def right_at(self, ahead: float) -> float:
        return float(np.interp(ahead, self.ahead, self.right))


def lead_vehicle(
    tracks: Sequence[Track],
    path: Path | None = None,
    exclude: frozenset[int] = frozenset(),
    cutting_in: frozenset[int] = frozenset(),
) -> tuple[Track, tuple[float, float, float, float]] | None:
    """The vehicle to follow: nearest confirmed one in the path, of a width a
    vehicle ahead can have, or `cutting_in` (`CutInFilter`); and not in
    `exclude` (say, crossing). With its estimate, or None when the road is clear."""
    best = None
    for track in tracks:
        if not track.confirmed or track.measurement_age > STALE_MEASUREMENT_S:
            continue
        estimate = track.estimate()
        if estimate is None:
            continue
        distance, _, lateral, width = estimate
        if distance > MAX_LEAD_DISTANCE_M:
            continue
        middle = 0.0 if path is None else path.right_at(distance)
        clipped = getattr(track, "clipped", 0)
        if clipped:
            # Cut off on that side: only its inner edge is where it is.
            inner = (lateral - middle) - clipped * width / 2.0
            if -clipped * inner < -CLIPPED_REACH_M:
                continue
        in_path = abs(lateral - middle) <= PATH_HALF_WIDTH_M and width <= MAX_LEAD_WIDTH_M
        if not in_path and track.id not in cutting_in:
            continue
        if track.id in exclude:
            continue
        if best is None or distance < best[1][0]:
            best = (track, estimate)
    return best


PULLING_IN_REACH_M: Final = 1.0
PULLING_AWAY_MPS: Final = 1.0
"""See `CrossingFilter.pulling_in`. One drawing away faster than
`PULLING_AWAY_MPS` is not in the truck's way: on a motorway a tanker moving
out of the next lane ahead, 3.7 m/s faster, would have been followed 13 m
off."""

STEADY_SLACK_M: Final = 0.3
STEADY_APPROACH_M: Final = 1.0
"""A vehicle closing on the lane steadily: over the window, never more than
`STEADY_SLACK_M` further from its middle than the sighting before, and at
least `STEADY_APPROACH_M` nearer in all (`CrossingMotion.steady`)."""


@dataclass(frozen=True)
class CrossingMotion:
    """Of a track moving across the lane, for `CrossingFilter.in_the_way`."""

    offset: float
    """From the lane's middle, metres, positive to the right."""
    rate: float
    width: float
    distance: float
    closing: float
    broadside: bool
    """Seen at least `BROADSIDE_M` wide: showing its side."""
    steady: bool
    """Closing on the lane at every sighting of the window, not a box that
    jumped: by Metz a lorry pulling out from a side road came from 10.6 m to
    3.6 m off in 0.7 s, every sighting nearer."""


class CrossingFilter:
    """Which tracks are moving across the lane rather than along it.

    Each track's offset from the middle of the lane is followed over the last
    `CROSSING_WINDOW_S`; the lane and the track are seen from the same camera,
    so the truck turning moves both and cancels out. Only a vehicle showing
    its side (`BROADSIDE_M`) is crossing; one seen narrower, at an angle, is
    still one to give way to where asked (`in_the_way`, `any_width`).
    """

    def __init__(self) -> None:
        self._offsets: dict[int, deque] = {}
        self._motion: dict[int, CrossingMotion] = {}
        """Of each track moving across the lane, for `in_the_way`."""
        self._where: dict[int, tuple[float, float]] = {}
        """Each track's latest offset from the lane's middle, and width."""
        self._pulled_in: set[int] = set()
        """Tracks found pulling in (`pulling_in`), kept while they reach the lane."""

    def crossing(self, tracks: Sequence[Track], path: Path | None, now: float) -> frozenset[int]:
        found = set()
        alive = set()
        self._motion = {}
        for track in tracks:
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, closing, lateral, width = estimate
            alive.add(track.id)
            if distance > CROSSING_WITHIN_M:
                self._offsets.pop(track.id, None)
                continue
            broadside = width >= BROADSIDE_M.get(str(track.kind), 0.0)
            middle = 0.0 if path is None else path.right_at(distance)
            history = self._offsets.setdefault(track.id, deque())
            history.append((now, lateral - middle))
            self._where[track.id] = (lateral - middle, width)
            while history and now - history[0][0] > CROSSING_WINDOW_S:
                history.popleft()
            if len(history) >= SETTLED_SIGHTINGS and history[-1][0] - history[0][0] >= SETTLED_S:
                t, offset = (np.array(v) for v in zip(*history, strict=True))
                rate = _median_slope(t, offset)
                if abs(rate) > MAX_CROSSING_MPS:
                    if broadside:
                        found.add(track.id)
                    away = np.abs(offset)
                    steady = bool(
                        np.all(np.diff(away) <= STEADY_SLACK_M)
                        and away[0] - away[-1] >= STEADY_APPROACH_M
                    )
                    self._motion[track.id] = CrossingMotion(
                        float(offset[-1]), rate, width, distance, closing, broadside, steady
                    )
        for gone in set(self._offsets) - alive:
            del self._offsets[gone]
        for gone in set(self._where) - alive:
            del self._where[gone]
        return frozenset(found)

    def pulling_in(self, speed: float) -> frozenset[int]:
        """Of the tracks moving across the lane, those joining the traffic
        (`joining`, the same way only) that close on the lane steadily
        (`CrossingMotion.steady`) and reach within `PULLING_IN_REACH_M` of its
        middle, however wide they are seen: to be followed. Out of a side road
        in Norway a lorry pulled into the truck's lane 30 m ahead, seen from
        its corner 9 to 15 m wide -- no vehicle ahead is that wide
        (`MAX_LEAD_WIDTH_M`), and one joining the traffic is not crossing it --
        and was hit from behind at 41 km/h."""
        joining = self.joining(speed, oncoming=False)
        found = {
            track_id
            for track_id, motion in self._motion.items()
            if track_id in joining
            and motion.steady
            and motion.closing < PULLING_AWAY_MPS
            and abs(motion.offset) - motion.width / 2.0 <= PULLING_IN_REACH_M
        }
        # Once in, it stops moving across, and is still seen at an angle:
        # kept while it reaches the lane.
        for track_id in self._pulled_in:
            where = self._where.get(track_id)
            if where is not None and abs(where[0]) - where[1] / 2.0 <= PULLING_IN_REACH_M:
                found.add(track_id)
        self._pulled_in = found
        return frozenset(found)

    def joining(self, speed: float, *, oncoming: bool = True) -> frozenset[int]:
        """Of the tracks found crossing, those moving along the road with the
        traffic, and with `oncoming` the other way too (`ALONG_SHARE`,
        `MIN_ALONG_MPS`): joining the traffic or meeting it, not crossing it.
        Where the route turns across the oncoming lanes, a car coming the
        other way is one to give way to: out of Mannheim, taken for one only
        meeting the traffic, it was hit."""
        least = max(ALONG_SHARE * speed, MIN_ALONG_MPS)
        return frozenset(
            track_id
            for track_id, motion in self._motion.items()
            if (abs(speed + motion.closing) if oncoming else speed + motion.closing) >= least
        )

    def in_the_way(
        self,
        speed: float,
        *,
        oncoming: bool = True,
        joining_too: bool = False,
        any_width: bool = False,
        steady_only: bool = False,
    ) -> tuple[int, float] | None:
        """The nearest vehicle crossing the lane that will be in it when the
        truck gets there, as (track id, distance); None if all are clear.

        A crossing vehicle is not one to follow, but it is one to give way
        to: at a junction in Dover a lorry turning across was the lead, then
        found crossing and dropped, and the truck drove on into it. Each
        crossing vehicle is carried across the lane at the rate it moves, and
        the time it is in the lane -- its whole seen width, `YIELD_CORRIDOR_M`
        either side of the middle -- set against the time the truck, from its
        speed and moving off at `MOVE_OFF_MPS2`, is at that distance; within
        `YIELD_MARGIN_S` of each other, it is in the way. With `joining_too`,
        one moving along the road (`joining`) counts as well: into a
        roundabout, a lorry coming round the ring moves partly the truck's way,
        and was hit. With `any_width`, a vehicle not showing its side counts
        too; with `steady_only`, only one closing on the lane steadily
        (`CrossingMotion.steady`).
        """
        nearest: tuple[int, float] | None = None
        joining = frozenset() if joining_too else self.joining(speed, oncoming=oncoming)
        for track_id, motion in self._motion.items():
            if track_id in joining:
                continue
            if not (motion.broadside or any_width) or (steady_only and not motion.steady):
                continue
            offset, rate, width = motion.offset, motion.rate, motion.width
            distance = motion.distance
            near, far = offset - width / 2.0, offset + width / 2.0
            if rate > 0.0:
                t_in, t_out = (-YIELD_CORRIDOR_M - far) / rate, (YIELD_CORRIDOR_M - near) / rate
            else:
                t_in, t_out = (YIELD_CORRIDOR_M - near) / rate, (-YIELD_CORRIDOR_M - far) / rate
            if t_out < 0.0:
                continue  # already across and away
            arrive = _time_to(distance, speed)
            gone_by = _time_to(distance + TRUCK_LENGTH_M, speed)
            in_the_lane = (
                max(t_in, 0.0) - YIELD_MARGIN_S <= gone_by and t_out + YIELD_MARGIN_S >= arrive
            )
            if in_the_lane and (nearest is None or distance < nearest[1]):
                nearest = (track_id, distance)
        return nearest


class CutInFilter:
    """Which tracks beside the lane are moving into it ahead of the truck
    (`CUT_IN_MPS`). Like `CrossingFilter`, each one's offset from the middle
    of the lane is followed, so the truck turning cancels out.

    With the truck's `speed`, one once read as coming towards it
    (`CUT_IN_COMING_MPS`) never counts: it is followed only once in the lane."""

    def __init__(self) -> None:
        self._offsets: dict[int, deque] = {}
        self._coming: set[int] = set()

    def cutting_in(
        self,
        tracks: Sequence[Track],
        path: Path | None,
        now: float,
        speed: float | None = None,
        beside: float = CUT_IN_BESIDE_M,
    ) -> frozenset[int]:
        found = set()
        alive = set()
        for track in tracks:
            if not track.confirmed or track.measurement_age > STALE_MEASUREMENT_S:
                continue
            estimate = track.estimate()
            if estimate is None:
                continue
            distance, closing, lateral, width = estimate
            alive.add(track.id)
            if (
                speed is not None
                and closing < 0.0
                and speed + closing < -max(CUT_IN_COMING_MPS, CUT_IN_COMING_SHARE * speed)
            ):
                self._coming.add(track.id)
            if (
                track.id in self._coming
                or distance > CUT_IN_WITHIN_M
                or width > MAX_VEHICLE_WIDTH_M
            ):
                self._offsets.pop(track.id, None)
                continue
            middle = 0.0 if path is None else path.right_at(distance)
            offset = lateral - middle
            if abs(offset) > beside:
                self._offsets.pop(track.id, None)
                continue
            history = self._offsets.setdefault(track.id, deque())
            history.append((now, offset))
            while history and now - history[0][0] > CUT_IN_WINDOW_S:
                history.popleft()
            if len(history) < CUT_IN_SIGHTINGS or history[-1][0] - history[0][0] < CUT_IN_SETTLED_S:
                continue
            t, offsets = (np.array(v) for v in zip(*history, strict=True))
            towards = -math.copysign(1.0, offset) * _median_slope(t, offsets)
            if towards < CUT_IN_MPS:
                continue
            edge = abs(offset) - width / 2.0
            arrive = distance / -closing if closing < 0.0 else math.inf
            if max(edge - CUT_IN_EDGE_M, 0.0) / towards <= min(arrive, CUT_IN_HORIZON_S):
                found.add(track.id)
        for gone in set(self._offsets) - alive:
            del self._offsets[gone]
        self._coming &= {track.id for track in tracks}
        return frozenset(found)


def _time_to(distance: float, speed: float) -> float:
    """Seconds for the truck to cover `distance` from `speed`, gathering speed
    at `MOVE_OFF_MPS2` -- from a standstill, 4.5 s to the other side of a lane."""
    a = MOVE_OFF_MPS2
    return (-speed + float(np.sqrt(speed * speed + 2.0 * a * max(distance, 0.0)))) / a


def plausible(track: Track) -> bool:
    """Whether a track's size is one a vehicle can have, from any side."""
    estimate = track.estimate()
    return estimate is None or estimate[3] <= MAX_VEHICLE_WIDTH_M


def _median_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen: the median of the slopes between every pair of points."""
    i, j = np.triu_indices(len(x), k=1)
    run = x[j] - x[i]
    keep = run > 1e-9
    return float(np.median((y[j] - y[i])[keep] / run[keep])) if keep.any() else 0.0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    across = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    down = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = across * down
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0
