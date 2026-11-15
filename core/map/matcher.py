"""Which lane the truck is on, followed through time rather than guessed afresh.

Taking the nearest lane at every sample is accurate about distance and wrong
about lanes. Laid over the taught drive it put the truck within 0.84 m of a lane
in the median, and 32 of its 89 lane-to-lane steps were between lanes the
network does not connect -- 28 of them inside junctions, where the through,
turning and merging lanes run a metre or two apart and the nearest one changes
from sample to sample. A position alone cannot say which of them the truck is
on. Where it was a moment ago can.

So this keeps a handful of hypotheses -- lane, distance along it, and a score
-- and moves each one only where the network lets a vehicle go: further along
its lane, into the lanes that follow, or across into a lane alongside. Each
candidate is scored by how well the measured position and heading fit it, and
the best few survive. It is a beam search over the lane graph, which is the
usual way to do map matching and the reason it works: a hypothesis cannot jump
between two lanes the road does not join.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from core.map.network import Projection, RoadNetwork

__all__ = ["LaneMatch", "LaneMatcher"]

#: How far off a lane a position can plausibly be while on it, one sigma. Lane
#: centres are measured to decimetres; the rest is the vehicle using its lane.
LATERAL_SIGMA_M: Final = 1.5

#: Heading disagreement at one sigma. Generous, because a truck changing lanes
#: or turning in a junction points across its lane for a few seconds.
HEADING_SIGMA_RAD: Final = math.radians(25.0)

#: A lane change costs this much log-likelihood, so it is believed only when the
#: position keeps saying so. About what a 2.5 sigma lateral miss costs.
LANE_CHANGE_PENALTY: Final = 3.0

#: How far across a vehicle can be seen to move into a lane alongside that the
#: graph does not offer as a lane change: a lane and a bit.
SIDEWAYS_RADIUS_M: Final = 6.0

#: The extra price of such a move, over a listed lane change. The graph only
#: offers lane changes on roads; inside junctions two approach lanes can run
#: side by side for 40 m, and a truck in the other one has to be followed there.
SIDEWAYS_PENALTY: Final = LANE_CHANGE_PENALTY + 1.0

#: How many hypotheses survive each step.
BEAM: Final = 8

#: Where to look when starting, or starting again.
SEARCH_RADIUS_M: Final = 15.0

#: If even the best hypothesis is this far off, it is lost: start again.
LOST_DISTANCE_M: Final = 12.0

#: Room beyond a lane's end before a hypothesis must have moved on from it.
SLACK_M: Final = 10.0


@dataclass(frozen=True, slots=True)
class LaneMatch:
    """Where the matcher believes the vehicle is."""

    lane: int
    s: float
    lateral: float
    """Signed distance from the lane's line, positive to its left."""

    heading_error: float
    """Vehicle heading minus lane heading, radians, wrapped."""

    confidence: float
    """Share of the belief on this hypothesis, 0 to 1."""


@dataclass(slots=True)
class _Hypothesis:
    lane: int
    s: float
    score: float
    projection: Projection
    heading_error: float


@dataclass
class LaneMatcher:
    """Follows a vehicle along the lane graph, one position at a time.

    matcher = LaneMatcher(network)
    for x, z, heading in positions:
        match = matcher.update(x, z, heading)
    """

    network: RoadNetwork
    beam: int = BEAM
    _hypotheses: list[_Hypothesis] = field(default_factory=list)
    _last: tuple[float, float] | None = None

    def reset(self) -> None:
        self._hypotheses = []
        self._last = None

    def update(self, x: float, z: float, heading: float) -> LaneMatch | None:
        """Take one position and heading; return the best belief, or None.

        `heading` is the direction of travel in the `atan2(dx, dz)` frame --
        `core.bridge.convert.course_rad`, not `heading_rad`.
        None means nothing on the network fits: off-road, or in a yard the
        extraction does not cover. The next update starts afresh.
        """
        travelled = 0.0 if self._last is None else math.hypot(x - self._last[0], z - self._last[1])
        self._last = (x, z)

        candidates = (
            self._advance(x, z, heading, travelled)
            if self._hypotheses
            else self._start(x, z, heading)
        )
        if not candidates:
            self._hypotheses = []
            return None

        candidates.sort(key=lambda h: h.score, reverse=True)
        best = candidates[0]
        if best.projection.distance > LOST_DISTANCE_M:
            self._hypotheses = []
            return None

        kept: dict[int, _Hypothesis] = {}
        for hypothesis in candidates:
            if hypothesis.lane not in kept:
                kept[hypothesis.lane] = hypothesis
            if len(kept) == self.beam:
                break
        self._hypotheses = list(kept.values())
        for hypothesis in self._hypotheses:
            hypothesis.score -= best.score

        weights = [math.exp(h.score) for h in self._hypotheses]
        return LaneMatch(
            lane=best.lane,
            s=best.s,
            lateral=best.projection.lateral,
            heading_error=best.heading_error,
            confidence=weights[0] / sum(weights),
        )

    # ------------------------------------------------------------------ #

    def _start(self, x: float, z: float, heading: float) -> list[_Hypothesis]:
        lanes = self.network.lanes_near(x, z, SEARCH_RADIUS_M)
        return [self._score(int(lane), x, z, heading, 0.0) for lane in lanes]

    def _advance(self, x: float, z: float, heading: float, travelled: float) -> list[_Hypothesis]:
        alongside = [int(lane) for lane in self.network.lanes_near(x, z, SIDEWAYS_RADIUS_M)]
        candidates: list[_Hypothesis] = []
        for hypothesis in self._hypotheses:
            reachable = self._reachable(hypothesis, travelled)
            for lane in alongside:
                reachable.setdefault(lane, SIDEWAYS_PENALTY)
            for lane, penalty in reachable.items():
                candidates.append(self._score(lane, x, z, heading, hypothesis.score - penalty))
        return candidates

    def _reachable(self, hypothesis: _Hypothesis, travelled: float) -> dict[int, float]:
        """Lanes a vehicle on this hypothesis could be on after `travelled` metres.

        Along its lane and through the lanes that follow while there is
        distance left to cover, plus one lane change from any of them.
        """
        reach = travelled + SLACK_M
        found: dict[int, float] = {hypothesis.lane: 0.0}
        frontier = [(hypothesis.lane, hypothesis.projection.length - hypothesis.s)]
        while frontier:
            lane, remaining = frontier.pop()
            if remaining >= reach:
                continue
            for following in self.network.next_lanes(lane):
                following = int(following)
                if following not in found:
                    found[following] = 0.0
                    frontier.append(
                        (following, remaining + float(self.network.lane_length[following]))
                    )
        for lane in list(found):
            for alongside in self.network.lanes_alongside(lane):
                found.setdefault(int(alongside), LANE_CHANGE_PENALTY)
        return found

    def _score(self, lane: int, x: float, z: float, heading: float, prior: float) -> _Hypothesis:
        projection = self.network.project(lane, x, z)
        error = _wrap(heading - projection.heading)
        miss = math.hypot(projection.lateral, projection.overshoot)
        likelihood = -0.5 * (miss / LATERAL_SIGMA_M) ** 2 - 0.5 * (error / HEADING_SIGMA_RAD) ** 2
        return _Hypothesis(lane, projection.s, prior + likelihood, projection, error)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
