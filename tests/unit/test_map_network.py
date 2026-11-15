"""The compiled road network: what the stack opens instead of building.

Everything here is checked against `LaneGraph`, which it is compiled from:
the cache is a storage format, not a second implementation, and the one way
it could quietly go wrong is by answering differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.map import Lane, LaneGraph, MapCacheError, RoadNetwork, Sector


def line(start: tuple[float, float], end: tuple[float, float], spacing: float = 2.0) -> list:
    """A centreline at the extractor's 2 m spacing."""
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    count = max(2, round(length / spacing) + 1)
    xs = np.linspace(start[0], end[0], count)
    zs = np.linspace(start[1], end[1], count)
    return [[float(x), 0.0, float(z)] for x, z in zip(xs, zs, strict=True)]


LOOK = {
    "lanes_left": 1,
    "lanes_right": 1,
    "road_offset": 0.0,
    "lane_offsets": [-2.25, 2.25],
    "lane_forward": [True, False],
    "lane_agreement": [1.0, 1.0],
    "lane_samples": [10, 10],
    "left_hand_traffic": None,
}


def uid(value: int) -> str:
    """Uids are 16 hex digits in the extraction, and come back that way."""
    return f"{value:016X}"


def road(name: int, points: list, start: int, end: int) -> dict:
    return {
        "uid": uid(name),
        "road_look": "two_way",
        "length": 0.0,
        "start_node": uid(start),
        "end_node": uid(end),
        "ai_vehicles": True,
        "left_hand_traffic": False,
        "hidden": False,
        "points": points,
    }


def graph_of(tmp_path: Path, roads: list[dict]) -> LaneGraph:
    document = {
        "format": "ets2ad-map-roads",
        "format_version": 6,
        "sector_size": 4000,
        "sample_spacing": 2.0,
        "sectors": [[0, 0]],
        "node_count": 0,
        "item_count": 0,
        "road_looks": {"two_way": LOOK},
        "roads": roads,
        "prefabs": [],
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return LaneGraph.build(Sector.load(path))


@pytest.fixture
def chain(tmp_path: Path) -> LaneGraph:
    """Three roads end to end.

    Straight on at each node, because a node has one heading that both roads
    share: a road cannot kink at a node, and sharp turns are junctions.
    """
    return graph_of(
        tmp_path,
        [
            road(0xA1, line((0, 0), (0, 100)), 0x10, 0x11),
            road(0xA2, line((0, 100), (0, 200)), 0x11, 0x12),
            road(0xA3, line((0, 200), (0, 300)), 0x12, 0x13),
        ],
    )


def saved(graph: LaneGraph, directory: Path) -> RoadNetwork:
    RoadNetwork.compile(graph).save(directory)
    return RoadNetwork.open(directory)


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


def test_every_lane_comes_back_as_it_went_in(chain: LaneGraph, tmp_path: Path) -> None:
    network = saved(chain, tmp_path / "cache")

    assert len(network) == len(chain.lanes)
    for index, lane in enumerate(chain.lanes):
        back = network.lane(index)
        assert back.points == pytest.approx(lane.points, abs=1e-3)
        assert back.offset == pytest.approx(lane.offset)
        assert back.owner == lane.owner
        assert back.start_node == lane.start_node
        assert back.end_node == lane.end_node


def test_what_stands_at_a_lane_start_comes_back(tmp_path: Path) -> None:
    from core.map import Lane, Signal

    graph = LaneGraph()
    light = Signal("traffic_light_major", "cr_1x1", (1410, 180))
    toll = Signal("barrier_manual", "ab_toll")
    for index, signal in enumerate((None, light, toll, light)):
        graph.lanes.append(
            Lane(
                points=np.array([[0.0, 0.0, 10.0 * index], [0.0, 0.0, 10.0 * index + 5.0]]),
                source="prefab",
                owner=f"{index + 1:016X}",
                semaphore=-1 if signal is None else index,
                signal=signal,
            )
        )
    network = saved(graph, tmp_path / "cache")

    assert [network.signal(i) for i in range(4)] == [None, light, toll, light]
    assert network.lane(1).signal == light
    assert len(network.signals) == 2, "one table entry per kind of signal"


def test_the_connections_come_back_as_they_went_in(chain: LaneGraph, tmp_path: Path) -> None:
    network = saved(chain, tmp_path / "cache")
    for index in range(len(chain.lanes)):
        assert sorted(int(i) for i in network.next_lanes(index)) == sorted(
            chain.successors.get(index, [])
        )


def test_open_maps_rather_than_reads(chain: LaneGraph, tmp_path: Path) -> None:
    """The point of the cache: opening it costs nothing until it is used. A
    plain array over the mapped file: indexing a memmap goes through Python."""
    network = saved(chain, tmp_path / "cache")
    assert isinstance(network.points.base, np.memmap)
    assert type(network.points) is np.ndarray


def test_an_opened_network_goes_to_another_process_as_where_it_is(
    chain: LaneGraph, tmp_path: Path
) -> None:
    """Spawned on Windows, the GPS worker was handed a copy of the whole map,
    279 MB, beside the one mapped in the loop's process; opened from the same
    files, the two share its pages."""
    import pickle

    opened = saved(chain, tmp_path / "cache")
    sent = pickle.dumps(opened)
    assert len(sent) < 1_000
    again = pickle.loads(sent)
    assert isinstance(again.points.base, np.memmap)
    assert np.array_equal(again.points, opened.points)
    assert again.rules == opened.rules
    compiled = RoadNetwork.compile(chain)
    assert np.array_equal(pickle.loads(pickle.dumps(compiled)).points, compiled.points)


# --------------------------------------------------------------------------- #
# Queries agree with the graph
# --------------------------------------------------------------------------- #


def test_nearest_agrees_with_the_graph(chain: LaneGraph, tmp_path: Path) -> None:
    network = saved(chain, tmp_path / "cache")
    for x, z in [(-2.0, 50.0), (2.0, 150.0), (-1.0, 250.0), (3.0, 299.0)]:
        expected = chain.nearest(x, z)
        found = network.nearest(x, z)
        assert found is not None and expected is not None
        assert found[0] == expected[0]


def test_nearest_is_by_the_line_not_by_the_points(tmp_path: Path) -> None:
    """The case that picked the wrong lane: nearest point on one, nearest line on another.

    Lane A runs along x = 0 with points every 2 m; lane B runs 2 m to its side
    with its points shifted 1 m along. A query 0.9 m from A's line, midway
    between two of A's points, is 1.35 m from A's nearest point and 1.1 m from
    B's -- so choosing the lane by its nearest point says B, and B is 0.2 m
    further away than A. It happened in 3 of 2000 queries across Europe.
    """
    along = np.arange(0.0, 41.0, 2.0)
    a = np.column_stack([np.zeros_like(along), np.zeros_like(along), along])
    b = np.column_stack([np.full_like(along, 2.0), np.zeros_like(along), along + 1.0])
    graph = LaneGraph()
    graph.lanes.append(Lane(points=a, source="road", owner=uid(0xA)))
    graph.lanes.append(Lane(points=b, source="road", owner=uid(0xB)))
    network = saved(graph, tmp_path / "cache")

    found = network.nearest(0.9, 11.0)

    assert found is not None
    assert found[0] == 0
    assert found[1] == pytest.approx(0.9, abs=1e-3)


def test_nearest_says_nothing_far_from_the_network(chain: LaneGraph, tmp_path: Path) -> None:
    network = saved(chain, tmp_path / "cache")
    assert network.nearest(5000.0, 5000.0) is None


def test_routes_agree_with_the_graph(chain: LaneGraph, tmp_path: Path) -> None:
    network = saved(chain, tmp_path / "cache")

    def forward(name: int) -> int:
        return next(
            i for i, lane in enumerate(chain.lanes) if lane.owner == uid(name) and lane.offset < 0
        )

    route = network.route(forward(0xA1), forward(0xA3))
    assert route is not None
    assert route == chain.route(forward(0xA1), forward(0xA3))
    assert [network.lane(i).owner for i in route] == [uid(0xA1), uid(0xA2), uid(0xA3)]


# --------------------------------------------------------------------------- #
# Refusing what it cannot trust
# --------------------------------------------------------------------------- #


def test_a_directory_without_a_manifest_is_not_a_cache(tmp_path: Path) -> None:
    """An interrupted save leaves arrays and no manifest; that is not a cache."""
    with pytest.raises(MapCacheError, match="no map cache"):
        RoadNetwork.open(tmp_path)


def test_another_version_is_refused(chain: LaneGraph, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    RoadNetwork.compile(chain).save(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["format_version"] = 99
    (directory / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(MapCacheError, match="rebuild it"):
        RoadNetwork.open(directory)


def test_a_missing_array_is_reported_by_name(chain: LaneGraph, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    RoadNetwork.compile(chain).save(directory)
    (directory / "successors.npy").unlink()

    with pytest.raises(MapCacheError, match=r"successors\.npy"):
        RoadNetwork.open(directory)


def test_lane_changes_come_back_as_they_went_in(tmp_path: Path) -> None:
    look = dict(LOOK)
    look.update(
        lanes_left=0,
        lanes_right=2,
        lane_offsets=[-2.25, 2.25],
        lane_forward=[True, True],
        lane_agreement=[1.0, 1.0],
        lane_samples=[10, 10],
    )
    document = {
        "format": "ets2ad-map-roads",
        "format_version": 6,
        "sector_size": 4000,
        "sample_spacing": 2.0,
        "sectors": [[0, 0]],
        "node_count": 0,
        "item_count": 0,
        "road_looks": {"two_way": look},
        "roads": [road(0xC1, line((0, 0), (0, 50)), 0x30, 0x31)],
        "prefabs": [],
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    graph = LaneGraph.build(Sector.load(path))
    network = saved(graph, tmp_path / "cache")

    assert [int(i) for i in network.lanes_alongside(0)] == graph.lane_changes[0] == [1]
    assert [int(i) for i in network.lanes_alongside(1)] == graph.lane_changes[1] == [0]
