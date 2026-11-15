"""The road network: loading it, turning it into lanes, and joining them up.

Built against synthetic extractions rather than the game's, so the tests say
what the code does rather than what one sector of Italy happens to contain.
The figures from the real thing are in `docs/results/map_extraction.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.map import LaneGraph, Sector


def straight(
    start: tuple[float, float], end: tuple[float, float], count: int = 5
) -> list[list[float]]:
    """A road centreline from one point to another, at y = 0."""
    xs = np.linspace(start[0], end[0], count)
    zs = np.linspace(start[1], end[1], count)
    return [[float(x), 0.0, float(z)] for x, z in zip(xs, zs, strict=True)]


def extraction(
    *,
    roads: list[dict] | None = None,
    prefabs: list[dict] | None = None,
    looks: dict[str, dict] | None = None,
) -> dict:
    return {
        "format": "ets2ad-map-roads",
        "format_version": 6,
        "sector_size": 4000,
        "sample_spacing": 2.0,
        "sectors": [[0, 0]],
        "node_count": 0,
        "item_count": 0,
        "road_looks": looks
        or {
            "two_way": {
                "lanes_left": 1,
                "lanes_right": 1,
                "road_offset": 0.0,
                "lane_offsets": [-2.25, 2.25],
                "lane_forward": [True, False],
                "lane_agreement": [1.0, 1.0],
                "lane_samples": [12, 12],
            }
        },
        "roads": roads or [],
        "prefabs": prefabs or [],
    }


def road(uid: str, points: list[list[float]], look: str = "two_way", **kwargs) -> dict:
    body = {
        "uid": uid,
        "road_look": look,
        "length": 100.0,
        "start_node": f"{uid}-a",
        "end_node": f"{uid}-b",
        "ai_vehicles": True,
        "left_hand_traffic": False,
        "hidden": False,
        "points": points,
    }
    body.update(kwargs)
    return body


def write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_a_road_survives_the_round_trip(tmp_path: Path) -> None:
    sector = Sector.load(
        write(tmp_path, extraction(roads=[road("r1", straight((0, 0), (0, 100)))]))
    )
    assert len(sector.roads) == 1
    assert sector.roads[0].uid == "r1"
    assert sector.roads[0].points.shape == (5, 3)
    assert sector.looks["two_way"].lane_count == 2


def test_another_format_is_refused(tmp_path: Path) -> None:
    document = extraction()
    document["format"] = "something-else"
    with pytest.raises(ValueError, match="not a map extraction"):
        Sector.load(write(tmp_path, document))


def test_a_future_version_is_refused(tmp_path: Path) -> None:
    """Silently misreading a changed layout is worse than not reading it."""
    document = extraction()
    document["format_version"] = 99
    with pytest.raises(ValueError, match="format version 99"):
        Sector.load(write(tmp_path, document))


# --------------------------------------------------------------------------- #
# Lanes from a centreline
# --------------------------------------------------------------------------- #


def test_lanes_sit_at_their_measured_offsets(tmp_path: Path) -> None:
    """A road running along +z has its left lane towards +x."""
    sector = Sector.load(
        write(tmp_path, extraction(roads=[road("r1", straight((0, 0), (0, 100)))]))
    )
    graph = LaneGraph.build(sector)

    assert len(graph.lanes) == 2
    offsets = sorted(lane.offset for lane in graph.lanes)
    assert offsets == [-2.25, 2.25]
    left = next(lane for lane in graph.lanes if lane.offset > 0)
    assert left.points[:, 0] == pytest.approx(2.25)


def test_the_measured_direction_decides_which_way_a_lane_runs(tmp_path: Path) -> None:
    sector = Sector.load(
        write(tmp_path, extraction(roads=[road("r1", straight((0, 0), (0, 100)))]))
    )
    graph = LaneGraph.build(sector)

    forward = next(lane for lane in graph.lanes if lane.offset < 0)
    backward = next(lane for lane in graph.lanes if lane.offset > 0)
    assert forward.start[2] < forward.end[2]
    assert backward.start[2] > backward.end[2]


def test_a_one_way_road_runs_one_way_on_both_sides(tmp_path: Path) -> None:
    """The lane's side of the centreline does not decide its direction.

    Six of the fifteen road looks surveyed in a city sector are one-way with
    lanes either side of the centreline, and a rule based on the side gets
    every one of them wrong.
    """
    looks = {
        "one_way": {
            "lanes_left": 0,
            "lanes_right": 2,
            "road_offset": 0.0,
            "lane_offsets": [-2.25, 2.25],
            "lane_forward": [True, True],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [20, 20],
        }
    }
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="one_way")], looks=looks
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    assert all(lane.start[2] < lane.end[2] for lane in graph.lanes)


def test_without_a_measurement_the_side_decides(tmp_path: Path) -> None:
    looks = {
        "unsurveyed": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 0.0,
            "lane_offsets": [-2.25, 2.25],
            "lane_forward": None,
            "lane_agreement": None,
            "lane_samples": None,
        }
    }
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="unsurveyed")], looks=looks
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    forward = next(lane for lane in graph.lanes if lane.offset < 0)
    assert forward.start[2] < forward.end[2]


def test_a_look_with_no_measured_offsets_yields_no_lanes(tmp_path: Path) -> None:
    """Better a hole in the network than lanes invented at a guessed width."""
    looks = {
        "unknown": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 0.0,
            "lane_offsets": None,
            "lane_forward": None,
            "lane_agreement": None,
            "lane_samples": None,
        }
    }
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="unknown")], looks=looks
    )
    assert LaneGraph.build(Sector.load(write(tmp_path, document))).lanes == []


def unmeasured(lanes_left: int, lanes_right: int, road_offset: float) -> dict:
    return {
        "lanes_left": lanes_left,
        "lanes_right": lanes_right,
        "road_offset": road_offset,
        "lane_offsets": None,
        "lane_forward": None,
        "lane_agreement": None,
        "lane_samples": None,
    }


def test_an_unmeasured_look_family_motorway_takes_its_lanes_from_its_definition(
    tmp_path: Path,
) -> None:
    """Seven British motorway looks were never measured and left Plymouth cut
    off; in their family every measured look has its lanes at road_offset +
    (k + 0.5) x 4.5 m."""
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="look4")],
        looks={"look4": unmeasured(3, 3, 5.0)},
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    offsets = sorted(lane.offset for lane in graph.lanes)
    assert offsets == pytest.approx([-16.25, -11.75, -7.25, 7.25, 11.75, 16.25])
    forward = [lane for lane in graph.lanes if lane.start[2] < lane.end[2]]
    assert all(lane.offset < 0 for lane in forward), "traffic keeps right"


def test_where_traffic_keeps_left_the_defined_lanes_are_mirrored(tmp_path: Path) -> None:
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="look4", left_hand_traffic=True)],
        looks={"look4": unmeasured(3, 3, 5.0)},
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    forward = [lane for lane in graph.lanes if lane.start[2] < lane.end[2]]
    assert len(forward) == 3
    assert all(lane.offset > 0 for lane in forward)


def test_an_unmeasured_one_way_look_is_not_guessed(tmp_path: Path) -> None:
    """Where one-way looks put their lanes differs from look to look."""
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), look="look17")],
        looks={"look17": unmeasured(3, 0, 0.0)},
    )
    assert LaneGraph.build(Sector.load(write(tmp_path, document))).lanes == []


def test_two_roads_of_different_medians_meet_without_a_step(tmp_path: Path) -> None:
    """A British motorway's median widens from 4 m to 30 m between two
    pieces sharing a node: 13 m apart, the lanes did not join, and Plymouth
    was cut off. Each lane now leaves where the other road's left off."""
    narrow = {
        "lanes_left": 1,
        "lanes_right": 1,
        "road_offset": 2.0,
        "lane_offsets": [-4.25, 4.25],
        "lane_forward": [True, False],
        "lane_agreement": [1.0, 1.0],
        "lane_samples": [9, 9],
    }
    wide = dict(narrow, road_offset=15.0, lane_offsets=[-17.25, 17.25])
    first = road("r1", straight((0, 0), (0, 100)), look="narrow")
    second = road("r2", straight((0, 100), (0, 300)), look="wide")
    second["start_node"] = first["end_node"]
    document = extraction(roads=[first, second], looks={"narrow": narrow, "wide": wide})
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    ahead = [i for i, lane in enumerate(graph.lanes) if lane.owner == "r1" and lane.start[2] < 50]
    onward = [i for i, lane in enumerate(graph.lanes) if lane.owner == "r2" and lane.end[2] > 250]
    assert len(ahead) == len(onward) == 1
    assert graph.successors[ahead[0]] == [onward[0]]
    joined = graph.lanes[onward[0]]
    assert joined.start[0] == pytest.approx(graph.lanes[ahead[0]].end[0], abs=0.05)
    assert joined.end[0] == pytest.approx(-17.25, abs=0.05), "its own offset by its end"


def test_a_road_goes_from_the_lanes_of_the_one_before_to_its_own(tmp_path: Path) -> None:
    """Near Exeter a British motorway piece with lanes 7.25 m out runs on from
    one at 4.25 m and into one at 17.25 m. Eased at both ends, its lane was
    12.8 m out 60% along, where the paint was 5.9 m out, and the truck drove
    the hard shoulder. It eases from the one before; the one after eases on."""
    narrow = {
        "lanes_left": 1,
        "lanes_right": 1,
        "road_offset": 2.0,
        "lane_offsets": [-4.25, 4.25],
        "lane_forward": [True, False],
        "lane_agreement": [1.0, 1.0],
        "lane_samples": [9, 9],
    }
    middle = dict(narrow, road_offset=5.0, lane_offsets=[-7.25, 7.25])
    wide = dict(narrow, road_offset=15.0, lane_offsets=[-17.25, 17.25])
    first = road("r1", straight((0, 0), (0, 100)), look="narrow")
    between = road("r9", straight((0, 100), (0, 350), count=11), look="middle")  # largest uid
    last = road("r2", straight((0, 350), (0, 600)), look="wide")
    between["start_node"] = first["end_node"]
    last["start_node"] = between["end_node"]
    document = extraction(
        roads=[first, between, last], looks={"narrow": narrow, "middle": middle, "wide": wide}
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    (lane,) = [x for x in graph.lanes if x.owner == "r9" and x.start[2] < x.end[2]]
    at_60 = lane.points[np.argmin(np.abs(lane.points[:, 2] - 250.0))]
    u = 0.6
    assert at_60[0] == pytest.approx(-(4.25 + 3.0 * u * u * (3 - 2 * u)), abs=0.1)
    assert lane.end[0] == pytest.approx(-7.25, abs=0.05), "its own lanes by its end"
    (onward,) = [x for x in graph.lanes if x.owner == "r2" and x.start[2] < x.end[2]]
    assert onward.start[0] == pytest.approx(-7.25, abs=0.05)
    assert onward.end[0] == pytest.approx(-17.25, abs=0.05)


# --------------------------------------------------------------------------- #
# Joining
# --------------------------------------------------------------------------- #


def test_two_roads_meeting_end_to_end_are_connected(tmp_path: Path) -> None:
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100))),
            road("r2", straight((0, 100), (0, 200))),
        ]
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    def lane(owner: str, forward: bool) -> int:
        return next(
            i
            for i, candidate in enumerate(graph.lanes)
            if candidate.owner == owner and (candidate.offset < 0) == forward
        )

    assert graph.successors[lane("r1", forward=True)] == [lane("r2", forward=True)]
    # The oncoming carriageway joins the same way round, from r2 back into r1.
    assert graph.successors[lane("r2", forward=False)] == [lane("r1", forward=False)]
    # And never across: no lane leads into the opposite direction.
    assert lane("r2", forward=False) not in graph.successors[lane("r1", forward=True)]


def test_roads_that_do_not_meet_are_not_connected(tmp_path: Path) -> None:
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100))),
            road("r2", straight((0, 140), (0, 240))),
        ]
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    assert all(not targets for targets in graph.successors.values())


def test_a_junction_states_its_own_connections(tmp_path: Path) -> None:
    """Inside a prefab the links come from the descriptor, not from geometry."""
    prefab = {
        "uid": "p1",
        "model": "cross",
        "nodes": [{"uid": "n0", "position": [0.0, 0.0, 0.0]}],
        "placement_residual": 0.001,
        "semaphores": [],
        "lanes": [
            {
                "length": 10.0,
                "from_node": "n0",
                "from_offset": -2.25,
                "to_node": None,
                "to_offset": None,
                "next": [1],
                "previous": [],
                "semaphore": 3,
                "rule": "cross_end",
                "blinker": "RightBlinker",
                "points": straight((0, 0), (0, 10), count=3),
            },
            {
                "length": 10.0,
                "from_node": None,
                "from_offset": None,
                "to_node": None,
                "to_offset": None,
                "next": [],
                "previous": [0],
                "semaphore": -1,
                "rule": "",
                "blinker": "NoBlinker",
                "points": straight((0, 10), (10, 10), count=3),
            },
        ],
    }
    graph = LaneGraph.build(Sector.load(write(tmp_path, extraction(prefabs=[prefab]))))

    assert graph.successors[0] == [1]
    assert graph.lanes[0].semaphore == 3
    assert graph.lanes[0].blinker == "RightBlinker"


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_nearest_finds_the_lane_under_a_position(tmp_path: Path) -> None:
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)))])
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    found = graph.nearest(-2.25, 50.0)
    assert found is not None
    index, distance = found
    assert graph.lanes[index].offset == pytest.approx(-2.25)
    assert distance < 0.5


def test_nearest_says_nothing_rather_than_guessing(tmp_path: Path) -> None:
    """Off the network is a real answer; the yards are not all mapped."""
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)))])
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    assert graph.nearest(500.0, 500.0, radius=20.0) is None


def test_a_route_follows_the_lanes(tmp_path: Path) -> None:
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100))),
            road("r2", straight((0, 100), (0, 200))),
            road("r3", straight((0, 200), (0, 300))),
        ]
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    start = next(i for i, lane in enumerate(graph.lanes) if lane.owner == "r1" and lane.offset < 0)
    goal = next(i for i, lane in enumerate(graph.lanes) if lane.owner == "r3" and lane.offset < 0)

    route = graph.route(start, goal)
    assert route is not None
    assert [graph.lanes[i].owner for i in route] == ["r1", "r2", "r3"]


def test_a_route_against_the_traffic_is_not_found(tmp_path: Path) -> None:
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100))),
            road("r2", straight((0, 100), (0, 200))),
        ]
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    forward = next(
        i for i, lane in enumerate(graph.lanes) if lane.owner == "r2" and lane.offset < 0
    )
    oncoming = next(
        i for i, lane in enumerate(graph.lanes) if lane.owner == "r1" and lane.offset > 0
    )

    assert graph.route(forward, oncoming) is None


def test_a_lane_is_its_own_route(tmp_path: Path) -> None:
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)))])
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    assert graph.route(0, 0) == [0]


# --------------------------------------------------------------------------- #
# Which side the traffic keeps to
# --------------------------------------------------------------------------- #


def mirrored_look() -> dict:
    """A two-way look measured on both sides of the Channel."""
    return {
        "two_way": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 0.0,
            "lane_offsets": [-2.25, 2.25],
            "lane_forward": [True, False],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [40, 40],
            "left_hand_traffic": {
                "lane_offsets": [-2.25, 2.25],
                "lane_forward": [False, True],
                "lane_agreement": [1.0, 1.0],
                "lane_samples": [30, 30],
            },
        }
    }


def test_a_left_hand_road_uses_what_was_measured_on_left_hand_roads(tmp_path: Path) -> None:
    """Pooling Britain with the continent made motorway looks a coin toss.

    The same look runs the other way round where traffic keeps left, so it is
    measured there separately and used there.
    """
    document = extraction(
        roads=[road("r1", straight((0, 0), (0, 100)), left_hand_traffic=True)],
        looks=mirrored_look(),
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    left = next(lane for lane in graph.lanes if lane.offset > 0)
    right = next(lane for lane in graph.lanes if lane.offset < 0)
    assert left.start[2] < left.end[2]  # keeps left, so the left lane runs forward
    assert right.start[2] > right.end[2]


def test_a_right_hand_road_is_not_affected_by_the_left_hand_measurement(tmp_path: Path) -> None:
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)))], looks=mirrored_look())
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    right = next(lane for lane in graph.lanes if lane.offset < 0)
    assert right.start[2] < right.end[2]


def test_a_left_hand_road_with_no_left_hand_measurement_is_mirrored(tmp_path: Path) -> None:
    """Measured both ways, all ten looks were exact mirrors; so a missing one is mirrored.

    The continent's layout, keep-right, has the forward lane at -2.25. Moved
    to the other side of the centreline for a road that keeps left, it is at
    +2.25 and still runs forward.
    """
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)), left_hand_traffic=True)])
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    left = next(lane for lane in graph.lanes if lane.offset > 0)
    right = next(lane for lane in graph.lanes if lane.offset < 0)
    assert left.offset == pytest.approx(2.25)
    assert left.start[2] < left.end[2]
    assert right.start[2] > right.end[2]


def test_a_look_measured_nowhere_still_gets_no_lanes_on_either_side(tmp_path: Path) -> None:
    """Mirroring needs something to mirror: better a hole than a width guessed."""
    looks = {
        "unknown": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 0.0,
            "lane_offsets": None,
            "lane_forward": None,
            "lane_agreement": None,
            "lane_samples": None,
        }
    }
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100)), look="unknown", left_hand_traffic=True),
            road("r2", straight((50, 0), (50, 100)), look="unknown"),
        ],
        looks=looks,
    )
    assert len(LaneGraph.build(Sector.load(write(tmp_path, document))).lanes) == 0


# --------------------------------------------------------------------------- #
# Joining by the node two roads share
# --------------------------------------------------------------------------- #


def test_lanes_that_do_not_line_up_are_joined_through_their_shared_node(tmp_path: Path) -> None:
    """Where a road's look changes, its lanes move sideways at the node.

    Here from 2.25 m to 3.25 m out: a metre, twice the geometric tolerance.
    Across Europe 1562 visible road lanes ended like that, between 1 and 6 m
    from where they should have continued. The node the two roads share says
    they continue; the nearest lane pointing the same way says into which.
    """
    looks = {
        "narrow": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 0.0,
            "lane_offsets": [-2.25, 2.25],
            "lane_forward": [True, False],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [10, 10],
        },
        "wide": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 1.0,
            "lane_offsets": [-3.25, 3.25],
            "lane_forward": [True, False],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [10, 10],
        },
    }
    document = extraction(
        roads=[
            road("r1", straight((0, 0), (0, 100)), look="narrow", end_node="shared"),
            road("r2", straight((0, 100), (0, 200)), look="wide", start_node="shared"),
        ],
        looks=looks,
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    def lane(owner: str, forward: bool) -> int:
        return next(
            i
            for i, candidate in enumerate(graph.lanes)
            if candidate.owner == owner and (candidate.offset < 0) == forward
        )

    assert graph.successors[lane("r1", forward=True)] == [lane("r2", forward=True)]
    assert graph.successors[lane("r2", forward=False)] == [lane("r1", forward=False)]


def test_a_dead_end_road_is_not_turned_around(tmp_path: Path) -> None:
    """The oncoming lane starts 4.5 m away at the same node, and is not a way out.

    Most of what looked like broken links across Europe was this: 2352 roads
    that simply end. A lane graph that joined them to their own oncoming lane
    would be routing a truck into U-turns on roads that stop.
    """
    document = extraction(roads=[road("r1", straight((0, 0), (0, 100)))])
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    forward = next(i for i, lane in enumerate(graph.lanes) if lane.offset < 0)
    assert not graph.successors.get(forward)


# --------------------------------------------------------------------------- #
# Changing lanes
# --------------------------------------------------------------------------- #


def motorway_look() -> dict:
    """Two lanes each way, carriageways a metre apart."""
    return {
        "motorway": {
            "lanes_left": 2,
            "lanes_right": 2,
            "road_offset": 1.0,
            "lane_offsets": [-7.75, -3.25, 3.25, 7.75],
            "lane_forward": [True, True, False, False],
            "lane_agreement": [1.0, 1.0, 1.0, 1.0],
            "lane_samples": [20, 20, 20, 20],
        }
    }


def lane_at(graph: LaneGraph, owner: str, offset: float) -> int:
    return next(
        i
        for i, lane in enumerate(graph.lanes)
        if lane.owner == owner and lane.offset == pytest.approx(offset)
    )


def test_lanes_running_the_same_way_side_by_side_can_be_changed_between(tmp_path: Path) -> None:
    """Without this, a lane the junction ahead does not feed is a dead end."""
    document = extraction(
        roads=[road("m1", straight((0, 0), (0, 100)), look="motorway")], looks=motorway_look()
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    outer, inner = lane_at(graph, "m1", -7.75), lane_at(graph, "m1", -3.25)
    assert graph.lane_changes[outer] == [inner]
    assert graph.lane_changes[inner] == [outer]


def test_a_lane_change_never_crosses_into_oncoming_traffic(tmp_path: Path) -> None:
    """-3.25 and +3.25 are 6.5 m apart and run opposite ways: not a lane change."""
    document = extraction(
        roads=[road("m1", straight((0, 0), (0, 100)), look="motorway")], looks=motorway_look()
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    inner_forward, inner_back = lane_at(graph, "m1", -3.25), lane_at(graph, "m1", 3.25)
    assert inner_back not in graph.lane_changes[inner_forward]
    assert inner_forward not in graph.lane_changes[inner_back]


def test_a_two_lane_road_has_no_lane_changes(tmp_path: Path) -> None:
    """One lane each way: the only lane alongside is the oncoming one."""
    graph = LaneGraph.build(
        Sector.load(write(tmp_path, extraction(roads=[road("r1", straight((0, 0), (0, 100)))])))
    )
    assert not any(graph.lane_changes.values())


def test_a_one_way_road_changes_lanes_across_its_centre_line(tmp_path: Path) -> None:
    """On a one-way road the lanes either side of the centreline run together."""
    looks = {
        "one_way": {
            "lanes_left": 0,
            "lanes_right": 2,
            "road_offset": 0.0,
            "lane_offsets": [-2.25, 2.25],
            "lane_forward": [True, True],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [20, 20],
        }
    }
    document = extraction(
        roads=[road("o1", straight((0, 0), (0, 100)), look="one_way")], looks=looks
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))
    assert graph.lane_changes[lane_at(graph, "o1", -2.25)] == [lane_at(graph, "o1", 2.25)]


def single_look() -> dict:
    return {
        "single": {
            "lanes_left": 1,
            "lanes_right": 1,
            "road_offset": 1.0,
            "lane_offsets": [-3.25, 3.25],
            "lane_forward": [True, False],
            "lane_agreement": [1.0, 1.0],
            "lane_samples": [20, 20],
        }
    }


def test_a_route_changes_lanes_to_reach_a_lane_nothing_leads_into(tmp_path: Path) -> None:
    """The case that fragmented the network.

    The road before feeds only the inner lane of the motorway, so the outer
    lane has no way in. Joining ends -- by geometry or through the node --
    only ever gives a lane somewhere to go; it never gives one a way in. A
    lane change does.
    """
    document = extraction(
        roads=[
            road("s1", straight((0, 0), (0, 100)), look="single", end_node="n1"),
            road("m1", straight((0, 100), (0, 200)), look="motorway", start_node="n1"),
        ],
        looks={**motorway_look(), **single_look()},
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    before = lane_at(graph, "s1", -3.25)
    inner, outer = lane_at(graph, "m1", -3.25), lane_at(graph, "m1", -7.75)
    assert graph.successors[before] == [inner]
    assert not graph.predecessors.get(outer)
    assert graph.route(before, outer) == [before, inner, outer]


def test_a_lane_that_is_dropped_merges_into_the_one_that_continues(tmp_path: Path) -> None:
    """The other way round needs no lane change: the node joins the ending lane on."""
    document = extraction(
        roads=[
            road("m1", straight((0, 0), (0, 100)), look="motorway", end_node="n1"),
            road("s1", straight((0, 100), (0, 200)), look="single", start_node="n1"),
        ],
        looks={**motorway_look(), **single_look()},
    )
    graph = LaneGraph.build(Sector.load(write(tmp_path, document)))

    outer, inner = lane_at(graph, "m1", -7.75), lane_at(graph, "m1", -3.25)
    onward = lane_at(graph, "s1", -3.25)
    assert graph.successors[inner] == [onward]
    assert graph.successors[outer] == [onward]


# --------------------------------------------------------------------------- #
# Traffic lights and barriers
# --------------------------------------------------------------------------- #


def _light_junction(semaphores: list[dict], lane_semaphore: int) -> dict:
    return {
        "uid": "p1",
        "model": "cross",
        "nodes": [{"uid": "n0", "position": [0.0, 0.0, 0.0]}],
        "placement_residual": 0.001,
        "semaphores": semaphores,
        "lanes": [
            {
                "length": 10.0,
                "from_node": "n0",
                "from_offset": -2.25,
                "to_node": None,
                "to_offset": None,
                "next": [],
                "previous": [],
                "semaphore": lane_semaphore,
                "rule": "",
                "blinker": "NoBlinker",
                "points": straight((0, 0), (0, 10), count=3),
            }
        ],
    }


PROFILES = {
    "cr_1x1": {
        "inherited": None,
        "types": ["traffic_light_minor", "traffic_light_major", "model_only"],
        "sleep_start": 1410,
        "sleep_end": 180,
    },
    "cr_1x1.netherlands": {
        "inherited": "cr_1x1",
        "types": [],
        "sleep_start": None,
        "sleep_end": None,
    },
    "cr_1x1.italy": {
        "inherited": "cr_1x1.netherlands",
        "types": [],
        "sleep_start": None,
        "sleep_end": None,
    },
    "cr_1x1.denmark": {"inherited": "cr_1x1", "types": [], "sleep_start": 0, "sleep_end": 0},
    "ab_toll": {
        "inherited": None,
        "types": ["barrier_manual", "barrier_automatic"],
        "sleep_start": None,
        "sleep_end": None,
    },
}


def _signal_of(tmp_path: Path, semaphore: dict, lane_semaphore: int, light_profile=None):
    junction = _light_junction([semaphore], lane_semaphore)
    if light_profile is not None:
        junction["semaphore_profile"] = light_profile
    document = extraction(prefabs=[junction])
    document["semaphore_profiles"] = PROFILES
    return LaneGraph.build(Sector.load(write(tmp_path, document))).lanes[0].signal


def _semaphore(id_: int, profile: str, type_: str = "UseProfile") -> dict:
    return {"id": id_, "type": type_, "profile": profile, "position": [0.0, 0.0, 0.0]}


def test_a_profile_says_what_a_semaphore_is_by_its_id(tmp_path: Path) -> None:
    signal = _signal_of(tmp_path, _semaphore(1, "cr_1x1"), 1)
    assert signal.kind == "traffic_light_major"
    assert signal.is_traffic_light
    assert signal.sleep == (1410, 180)


def test_a_profile_inherits_what_it_does_not_state(tmp_path: Path) -> None:
    signal = _signal_of(tmp_path, _semaphore(0, "cr_1x1.italy"), 0)
    assert signal.kind == "traffic_light_minor"
    assert signal.sleep == (1410, 180), "two levels up"


def test_a_sleep_of_zero_to_zero_is_a_light_that_never_sleeps(tmp_path: Path) -> None:
    signal = _signal_of(tmp_path, _semaphore(0, "cr_1x1.denmark"), 0)
    assert signal.sleep is None
    assert not signal.asleep(0)


def test_a_semaphore_with_its_own_type_still_sleeps_by_its_profile(tmp_path: Path) -> None:
    signal = _signal_of(tmp_path, _semaphore(4, "cr_1x1", "TrafficLightMinor"), 4)
    assert signal.kind == "traffic_light_minor"
    assert signal.sleep == (1410, 180)


def test_a_junction_the_map_gives_no_light_profile_has_no_light(tmp_path: Path) -> None:
    """Five universal junctions in Nynäshamn and Uppsala, lights in their
    descriptors and none in the game, held the truck for a lamp."""
    unlit = _semaphore(4, "2ph", "TrafficLightMajor")
    assert _signal_of(tmp_path, unlit, 4, light_profile="") is None
    assert _signal_of(tmp_path, unlit, 4, light_profile="yza9zh0433o").is_traffic_light
    assert _signal_of(tmp_path, unlit, 4).is_traffic_light, "an older extraction, not read"


def test_a_light_its_own_profile_describes_needs_no_map_profile(tmp_path: Path) -> None:
    """Milan's junctions name a profile per semaphore, and their lamps light."""
    signal = _signal_of(tmp_path, _semaphore(1, "cr_1x1"), 1, light_profile="")
    assert signal.kind == "traffic_light_major"


def test_a_toll_booth_beyond_the_profile_list_is_still_a_barrier(tmp_path: Path) -> None:
    signal = _signal_of(tmp_path, _semaphore(7, "ab_toll"), 7)
    assert signal.kind == "barrier_automatic"
    assert not signal.is_traffic_light


def test_a_lane_without_a_semaphore_has_no_signal(tmp_path: Path) -> None:
    assert _signal_of(tmp_path, _semaphore(0, "cr_1x1"), -1) is None


def test_a_semaphore_the_junction_does_not_have_is_no_signal(tmp_path: Path) -> None:
    assert _signal_of(tmp_path, _semaphore(0, "cr_1x1"), 2) is None


def test_a_sleep_window_can_run_past_midnight() -> None:
    from core.map import Signal

    signal = Signal("traffic_light", "cr_1x1", (1410, 180))
    assert signal.asleep(1430) and signal.asleep(0) and signal.asleep(179)
    assert not signal.asleep(180) and not signal.asleep(720) and not signal.asleep(1409)
    assert signal.asleep(1440 * 9 + 60), "any day"
    assert Signal("traffic_light", "x", (60, 120)).asleep(90)
    assert not Signal("traffic_light", "x", (60, 120)).asleep(130)
