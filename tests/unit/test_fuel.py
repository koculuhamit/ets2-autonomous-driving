"""Gas station pumps as places on lanes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.map.fuel import Pump, load_pumps, pumps_on, save_pumps
from tests.unit.lane_networks import network, polyline


class Station:
    """A prefab as far as `pumps_on` reads it."""

    def __init__(self, uid: str, pumps) -> None:
        self.uid = uid
        self.gas_pumps = tuple(np.asarray(p, dtype=float) for p in pumps)


def forecourt(joined: bool = True):
    """A road into a station's lane (item 2) and on out of it; or the lane
    alone on open ground, beside a road it does not join."""
    lanes = [
        polyline((0, -20), (10, 0)),  # item 1: the way in
        polyline((10, 0), (10, 60)),  # item 2: the station
        polyline((10, 60), (10, 100)),  # item 3: the way out
        polyline((0, 0), (0, 100)),  # item 4: a road beside
    ]
    return network(lanes, successors=[(0, 1), (1, 2)] if joined else [])


def test_a_pump_is_where_its_station_s_lane_passes() -> None:
    station = Station("0000000000000002", [(10.3, 0.0, 40.0)])
    (pump,) = pumps_on(forecourt(), [station])
    assert pump.lane == 1
    assert pump.s == pytest.approx(40.0, abs=0.1)


def test_a_pump_off_its_station_s_lanes_is_not_one_to_drive_to() -> None:
    assert pumps_on(forecourt(), [Station("0000000000000002", [(4.0, 0.0, 40.0)])]) == []
    assert pumps_on(forecourt(), [Station("0000000000000009", [(10.0, 0.0, 40.0)])]) == []


def test_a_pump_island_no_road_leads_to_is_not_one_to_drive_to() -> None:
    """Out of Oslo every pump within 5 km stood on lanes joined to nothing."""
    station = Station("0000000000000002", [(10.3, 0.0, 40.0)])
    assert pumps_on(forecourt(joined=False), [station]) == []


def test_pumps_come_back_as_they_went(tmp_path) -> None:
    pumps = [Pump(3, 12.5, (1.0, 2.0)), Pump(7, 0.0, (-5.0, 6.0))]
    save_pumps(tmp_path, pumps)
    assert load_pumps(tmp_path) == pumps
    assert load_pumps(tmp_path / "none") == []


def test_no_stations_no_pumps() -> None:
    assert pumps_on(replace(forecourt()), []) == []


def test_a_station_s_forecourt_and_the_service_road_past_it_are_kept_out_of() -> None:
    """Near Innsbruck the way went by a service road through a station beside
    the motorway, and the truck hit the pumps' island at 49 km/h."""
    from types import SimpleNamespace

    from core.map.fuel import station_lanes

    lanes = [
        polyline((0, 0), (0, 300)),  # 0: the motorway, the game's traffic on it
        polyline((12, 0), (12, 300)),  # 1: the service road past the pumps
        polyline((40, 0), (40, 300)),  # 2: another quiet road, far from them
    ]
    net = network(lanes, successors=[])
    ai = np.asarray(net.lane_ai).copy()
    ai[:] = [True, False, False]
    object.__setattr__(net, "lane_ai", ai)
    station = SimpleNamespace(uid="00000000000000AB", gas_pumps=[np.array([13.0, 0.0, 150.0])])
    assert station_lanes(net, [station]).tolist() == [1]
