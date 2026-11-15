"""The driver's panel and the mission's status file."""

from __future__ import annotations

from core.mission.status import read_status, write_status
from tools.panel import render

TRUCK = {
    "speed_kmh": 84.0,
    "limit_kmh": 80.0,
    "gear": 12,
    "beacon": True,
    "parking_brake": False,
    "fuel_range_km": 640.0,
    "nav_km": 123.4,
    "damage_pct": 0.4,
    "id": "scania.s_2016",
    "calibration": "the Actros's camera",
    "paused": False,
}


def test_the_status_is_written_whole_and_read_back(tmp_path) -> None:
    path = tmp_path / "status.json"
    assert read_status(path) is None
    write_status(path, "driving", "driving", job=("rt_log", "bern"))
    status = read_status(path)
    assert status["state"] == "driving" and status["job"] == ["rt_log", "bern"]
    assert not (tmp_path / "status.json.new").exists()


def test_the_panel_shows_the_mission_and_the_truck() -> None:
    text = render({"state": "driving", "message": "driving", "at": 0.0}, TRUCK, 10.0)
    assert "DRIVING: driving" in text
    assert "84 km/h" in text and "limit 80" in text and "OVER" in text
    assert "ENGAGED" in text and "123.4 km" in text
    assert "scania.s_2016: the Actros's camera" in text


def test_without_the_mission_or_the_game_it_says_so() -> None:
    text = render(None, None, 0.0)
    assert "not running" in text and "no telemetry" in text


def test_an_old_word_from_a_waiting_mission_is_dated() -> None:
    text = render({"state": "arrived", "message": "parked", "at": 0.0}, TRUCK, 600.0)
    assert "10 min ago" in text
