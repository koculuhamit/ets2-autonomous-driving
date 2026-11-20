"""Stopping before the turn onto a depot's own road, when that road is long."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.localization.path import ReferencePath
from core.map import CompanyPlace
from core.planning.route import SERVICE_STOP_BEFORE_M, Route, service_road_stop
from tests.unit.lane_networks import network, polyline
from tools.follow import service_stop_on


def kassel(service_m: float):
    """A public road 100 m long, a junction 10 m across, then the depot's
    own road `service_m` long to the gate at its end."""
    road = polyline((0, 0), (0, 100))
    junction = polyline((0, 100), (0, 110))
    service = polyline((0, 110), (0, 110 + service_m))
    net = network(
        [road, junction, service], successors=[(0, 1), (1, 2)], public=(0, 1), prefabs=(1,)
    )
    gate = (0.0, 110.0 + service_m + 2.0)
    return net, CompanyPlace("kaarfor", "kassel", gate, (gate,), ())


def test_a_long_road_of_the_depot_s_own_is_not_turned_onto() -> None:
    """In Kassel the truck hit pallets 10 m past the turn: it stops on the
    public road, short of the junction."""
    net, _ = kassel(170.0)
    assert service_road_stop(net, [0, 1, 2]) == pytest.approx((0.0, 100.0 - SERVICE_STOP_BEFORE_M))


def test_a_short_way_in_is_driven_to_the_gate() -> None:
    net, _ = kassel(30.0)
    assert service_road_stop(net, [0, 1, 2]) is None


def test_a_route_begun_on_the_depot_s_road_is_driven_to_its_end() -> None:
    """No public road on it to stop on: begun in the depot's grounds."""
    net, _ = kassel(170.0)
    assert service_road_stop(net, [2]) is None
    assert service_road_stop(net, [1, 2]) is None, "only the junction before it"


def _path(net, lanes):
    points = np.vstack([net.lane_points(lane)[:, [0, 2]] for lane in lanes])
    return ReferencePath(points, spacing=1.0)


def test_the_stop_is_marked_only_on_a_route_that_ends_at_the_depot() -> None:
    net, depot = kassel(170.0)
    route = Route(net, (0, 1, 2), ("onward", "onward"))
    path = _path(net, [0, 1, 2])
    stop = service_stop_on(route, path, depot)
    assert stop == pytest.approx(100.0 - SERVICE_STOP_BEFORE_M, abs=1.0)
    elsewhere = CompanyPlace("tree_et", "berlin", (5000.0, 0.0), ((5000.0, 0.0),), ())
    assert service_stop_on(route, path, elsewhere) is None, "its gate is not on the network"
    assert service_stop_on(route, path, None) is None
    ahead = SimpleNamespace(network=net, lanes=(0,))
    assert service_stop_on(ahead, _path(net, [0]), depot) is None, "ends on the public road"
