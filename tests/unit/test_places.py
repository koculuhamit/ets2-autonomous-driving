"""Depots and cities: finding a job's destination on the map, and a route to it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.map import CompanyPlace, Places, Sector
from core.planning.route import RouteError, plan_to_depot
from tests.unit.lane_networks import network, polyline


def extraction(tmp_path: Path, *, guard: int = 0) -> Sector:
    document = {
        "format": "ets2ad-map-roads",
        "format_version": 6,
        "sector_size": 4000,
        "sample_spacing": 2.0,
        "sectors": [[0, 0]],
        "node_count": 0,
        "item_count": 0,
        "road_looks": {},
        "roads": [],
        "prefabs": [
            {
                "uid": "00000000000000AA",
                "model": "yard",
                "dlc_guard": guard,
                "country": 0,
                "nodes": [
                    {"uid": "01", "position": [10.0, 0.0, 20.0]},
                    {"uid": "02", "position": [30.0, 0.0, 20.0]},
                ],
                "placement_residual": 0.0,
                "lanes": [],
                "semaphores": [],
            }
        ],
        "companies": [
            {
                "company": "tree_et",
                "city": "berlin",
                "position": [20.0, 0.0, 40.0],
                "prefab": "00000000000000AA",
                "spots": [{"type": "Unload", "difficulty": "Easy", "position": [25.0, 0.0, 45.0]}],
            }
        ],
        "cities": [
            {
                "city": "berlin",
                "position": [0.0, 0.0, 0.0],
                "width": 900,
                "height": 900,
                "shown": True,
            }
        ],
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return Sector.load(path)


def test_a_depot_is_found_by_the_ids_a_job_uses(tmp_path: Path) -> None:
    places = Places.from_sector(extraction(tmp_path))
    depot = places.company("tree_et", "berlin")
    assert depot is not None
    assert depot.position == (20.0, 40.0)
    assert depot.spots == (("Unload", (25.0, 45.0)),)


def test_a_depot_s_gates_are_its_yard_s_nodes(tmp_path: Path) -> None:
    depot = Places.from_sector(extraction(tmp_path)).company("tree_et", "berlin")
    assert depot.gates == ((10.0, 20.0), (30.0, 20.0))


def test_a_depot_not_on_this_map_is_none_not_an_error(tmp_path: Path) -> None:
    """The destination may be in a DLC the map was not built with."""
    places = Places.from_sector(extraction(tmp_path))
    assert places.company("tree_et", "paris") is None
    assert places.city("berlin").shown


def test_places_survive_the_round_trip(tmp_path: Path) -> None:
    places = Places.from_sector(extraction(tmp_path))
    places.save(tmp_path)
    assert Places.open(tmp_path) == places


def test_a_depot_in_a_dlc_not_owned_is_not_on_the_map(tmp_path: Path) -> None:
    """A depot exists where its yard does, and the yard is guarded like any prefab."""
    sector = extraction(tmp_path, guard=4)  # Italy
    assert Places.from_sector(sector.restricted_to(set())).companies == ()
    assert len(Places.from_sector(sector.restricted_to({"italy"})).companies) == 1


def depot_at(*gates: tuple[float, float]) -> CompanyPlace:
    return CompanyPlace("tree_et", "berlin", gates[0], gates, ())


def test_a_route_to_a_depot_goes_to_the_gate_nearest_by_road() -> None:
    """A yard has two gates here: one 100 m on, one at the end of a 600 m detour.

    The far gate is listed first, so taking the first gate that can be reached
    would choose it.
    """
    approach = polyline((0, 0), (0, 100))
    short = polyline((0, 100), (0, 200))
    detour = polyline((0, 100), (300, 100), (300, 400))
    net = network([approach, short, detour], successors=[(0, 1), (0, 2)])

    route = plan_to_depot(net, (0.0, 10.0, 0.0), depot_at((300.0, 395.0), (0.0, 195.0)))
    assert route.lanes == (0, 1)


def test_a_depot_with_no_gate_on_the_network_is_refused() -> None:
    net = network([polyline((0, 0), (0, 100))])
    with pytest.raises(RouteError, match="no gate of tree_et in berlin"):
        plan_to_depot(net, (0.0, 10.0, 0.0), depot_at((5000.0, 5000.0)))


def test_a_gate_set_back_from_the_road_is_reached_by_the_road_nearest_it() -> None:
    """In Turin the yard's gate was 31.7 m from the road; its driveway is the
    yard's, not the network's."""
    net = network([polyline((0, 0), (0, 200))])
    assert plan_to_depot(net, (0.0, 10.0, 0.0), depot_at((45.0, 150.0))).lanes == (0,)
    with pytest.raises(RouteError):
        plan_to_depot(net, (0.0, 10.0, 0.0), depot_at((75.0, 150.0)))
