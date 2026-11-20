"""Small hand-made lane networks, for tests that need a map but not the game's."""

from __future__ import annotations

import itertools
import math

import numpy as np

from core.map import Lane, LaneGraph, RoadNetwork


def polyline(*corners: tuple[float, float], spacing: float = 2.0) -> np.ndarray:
    """Points every `spacing` metres along straight pieces between corners, at y = 0."""
    points = []
    for (ax, az), (bx, bz) in itertools.pairwise(corners):
        count = max(2, round(math.hypot(bx - ax, bz - az) / spacing) + 1)
        for t in np.linspace(0.0, 1.0, count)[:-1]:
            points.append((ax + t * (bx - ax), 0.0, az + t * (bz - az)))
    points.append((corners[-1][0], 0.0, corners[-1][1]))
    return np.array(points)


def network(
    lanes: list[np.ndarray], successors=(), changes=(), public=None, prefabs=()
) -> RoadNetwork:
    """A compiled network of these lanes, joined as stated and no other way.
    Every lane carries the game's traffic, as a road does, unless `public`
    names the ones that do: the others are a yard's. Those in `prefabs` are
    a junction's lanes, the others a road's."""
    graph = LaneGraph()
    for index, points in enumerate(lanes):
        graph.lanes.append(
            Lane(
                points=points,
                source="prefab" if index in prefabs else "road",
                owner=f"{index + 1:016X}",
                ai_vehicles=public is None or index in public,
            )
        )
    for a, b in successors:
        graph.connect(a, b)
    for a, b in changes:
        graph.lane_changes[a].append(b)
        graph.lane_changes[b].append(a)
    return RoadNetwork.compile(graph)
