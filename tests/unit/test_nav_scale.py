"""How the game's navigation counts a road: its distance and its speed."""

from __future__ import annotations

import numpy as np
import pytest

from core.map.nav_scale import (
    SCALE_CITY,
    SCALE_GBR,
    SCALE_OPEN,
    CityArea,
    lane_nav_lengths,
    lane_scales,
    nav_speeds,
    road_side_class,
    spread_classes,
)
from core.map.sector import TruckSpeedLimit
from tests.unit.lane_networks import network, polyline


def test_a_lane_counts_nineteen_times_its_length_outside_cities() -> None:
    net = network([polyline((0, 0), (0, 100))])
    scales = np.full(1, SCALE_OPEN, dtype=np.float32)
    (nav,) = lane_nav_lengths(net.points, net.lane_start, scales, [])
    assert nav == pytest.approx(100.0 * SCALE_OPEN, rel=1e-4)


def test_the_part_in_a_city_area_counts_three_times() -> None:
    """The area is centred on its position: from z = 50 to 150 here."""
    net = network([polyline((0, 0), (0, 100))])
    scales = np.full(1, SCALE_OPEN, dtype=np.float32)
    city = CityArea(x=0.0, z=100.0, width=40.0, height=100.0)
    (nav,) = lane_nav_lengths(net.points, net.lane_start, scales, [city])
    assert nav == pytest.approx(50.0 * SCALE_OPEN + 50.0 * SCALE_CITY, rel=0.03)


def test_lanes_are_counted_apart() -> None:
    net = network([polyline((0, 0), (0, 100)), polyline((500, 0), (500, 20))])
    scales = np.full(2, SCALE_OPEN, dtype=np.float32)
    nav = lane_nav_lengths(net.points, net.lane_start, scales, [])
    assert nav == pytest.approx([100.0 * SCALE_OPEN, 20.0 * SCALE_OPEN], rel=1e-4)


def test_britain_counts_fifteen() -> None:
    owners = np.array([0xA, 0xB], dtype=np.uint64)
    scales = lane_scales(owners, {0xA: 10, 0xB: 5}, {10: "gbr", 5: "deu"})
    assert scales.tolist() == [SCALE_GBR, SCALE_OPEN]


def test_a_side_s_own_rule_names_its_class() -> None:
    classes = {"traffic_lane.road.local": "local_road"}
    assert road_side_class("motorway", ["traffic_lane.road.local"], classes) == "motorway"
    assert road_side_class("", ["traffic_lane.road.local"], classes) == "local_road"
    assert road_side_class("wire", [], classes) is None


def test_a_junction_takes_the_class_of_the_road_it_leads_to() -> None:
    # 0 -> 1 -> 2: a motorway, a junction lane, a local road.
    successor_start = np.array([0, 1, 2, 2])
    successors = np.array([1, 2])
    assert spread_classes(["motorway", None, "local_road"], successor_start, successors) == [
        "motorway",
        "local_road",
        "local_road",
    ]


def test_the_gps_reckons_nine_tenths_of_a_motorway_limit_and_eight_of_a_local_one() -> None:
    limits = {
        5: {
            "motorway": TruckSpeedLimit(80.0, 80.0, 80.0),
            "local_road": TruckSpeedLimit(60.0, 50.0, 60.0),
        }
    }
    speeds = nav_speeds(["motorway", "local_road", "expressway"], [5, 5, 5], limits)
    assert speeds.tolist() == pytest.approx([72.0, 48.0, 60.0 * 0.9])


def test_a_limit_over_a_truck_s_limiter_counts_as_the_limiter() -> None:
    """Britain's truck limits, 60 and 50 mph: the GPS reckoned 81.0 and 64.4
    km/h on its motorways and local roads."""
    limits = {
        10: {
            "motorway": TruckSpeedLimit(96.56064, 96.56064, 96.56064),
            "local_road": TruckSpeedLimit(80.4672, 48.28032, 80.4672),
        }
    }
    speeds = nav_speeds(["motorway", "local_road"], [10, 10], limits)
    assert speeds.tolist() == pytest.approx([81.0, 64.4], abs=0.05)
