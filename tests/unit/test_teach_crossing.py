"""Teaching a border crossing from a manual drive."""

from __future__ import annotations

import numpy as np
import pytest

from core.map import Lane, LaneGraph, RoadNetwork
from core.planning.route import plan
from tests.unit.lane_networks import polyline
from tools.teach_crossing import crossings_through


def _border() -> RoadNetwork:
    graph = LaneGraph()
    graph.lanes.append(Lane(points=polyline((0, 0), (0, 1000)), source="road", owner="0A"))
    graph.lanes.append(
        Lane(
            points=polyline((0, 1000), (30, 1150), (0, 1300)),
            source="road",
            owner="0B",
            ai_vehicles=False,
        )
    )
    graph.lanes.append(Lane(points=polyline((0, 1300), (0, 2300)), source="road", owner="0C"))
    graph.connect(0, 1)
    graph.connect(1, 2)
    return RoadNetwork.compile(graph)


def test_the_driven_part_through_the_customs_stretch_is_taught() -> None:
    route = plan(_border(), start=(0.0, 10.0, 0.0), through=[(0.0, 2290.0)])
    driven = np.column_stack([np.full(400, -4.0), np.zeros(400), np.linspace(900, 1400, 400)])
    taught = crossings_through(driven, route)
    assert len(taught) == 1
    points = np.asarray(taught[0]["points"])
    assert points[0, 2] < 1010 and points[-1, 2] > 1290
    assert np.allclose(points[:, 0], -4.0)


def test_a_drive_that_went_elsewhere_teaches_nothing() -> None:
    route = plan(_border(), start=(0.0, 10.0, 0.0), through=[(0.0, 2290.0)])
    elsewhere = np.column_stack([np.full(50, 500.0), np.zeros(50), np.linspace(900, 1400, 50)])
    assert crossings_through(elsewhere, route) == []


def test_a_passage_is_the_line_driven_between_two_moments(monkeypatch) -> None:
    """Out of a roundabout in Bern: the turn driven by hand, from second 205
    to 223 of the recording, standing still before and after."""
    from types import SimpleNamespace

    import tools.teach_crossing as teach

    seconds = np.arange(0.0, 300.0, 0.05)
    z = np.clip((seconds - 205.0) * 5.0, 0.0, 90.0)
    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    position = np.zeros(len(seconds), dtype=dtype)
    position["z"] = z
    telemetry = {"t": 1000.0 + seconds, "block": {"truck": {"placement": {"position": position}}}}
    monkeypatch.setattr(teach, "open_recording", lambda _: SimpleNamespace(telemetry=telemetry))
    line = teach.driven_line("recording", (205.0, 223.0))
    assert line[0, 2] == pytest.approx(0.0, abs=2.5)
    assert line[-1, 2] == pytest.approx(90.0, abs=4.0)  # kept every 2 m, smoothed
    assert np.all(np.diff(line[:, 2]) > 0), "only the turn, not the standing either side"
