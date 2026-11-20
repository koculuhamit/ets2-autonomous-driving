"""Summing measured drives by period of development."""

from __future__ import annotations

import json
import math

import pytest

from tools.analysis.phases import main, period


def _drive(km: float, minutes: float, *, ended="end_of_route", hits=0, brakes=0, hardest=3.0):
    return {
        "distance_km": km,
        "duration_s": minutes * 60.0,
        "ended": ended,
        "impacts": hits,
        "hard_brakes": brakes,
        "max_deceleration_mps2": hardest,
    }


DRIVES = {
    "20261001-100000/drive-01": _drive(500.0, 30.0, hits=1, brakes=6, hardest=6.0),
    "20261001-120000/drive-01": _drive(1.0, 1.0, ended="disengaged", hardest=9.0),
    "20261020-100000/drive-01": _drive(100.0, 30.0, brakes=1, hardest=4.0),
}


def test_a_period_takes_the_sessions_dated_within_it() -> None:
    summed, _ = period("October", DRIVES, "20261001", "20261009")
    assert summed.drives == 2
    assert summed.distance_km == pytest.approx(501.0)
    assert summed.deliveries == 1
    assert summed.hits_per_1000km == pytest.approx(1000.0 / 501.0)
    assert summed.hard_brakes_per_h == pytest.approx(6 / (31 / 60))


def test_a_test_start_does_not_set_the_hardest_deceleration() -> None:
    _, hardest = period("October", DRIVES, "20261001", "20261009")
    assert hardest == 6.0


def test_an_empty_period_is_not_a_number() -> None:
    summed, hardest = period("none", DRIVES, "20250101", "20250102")
    assert summed.drives == 0
    assert math.isnan(summed.hits_per_1000km) and math.isnan(hardest)


def test_it_prints_a_table_per_period_and_per_day(tmp_path, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"drives": DRIVES, "summary": {}}), encoding="utf-8")
    assert main([str(report)]) == 0
    out = capsys.readouterr().out
    assert "| 20261001 | 2 | 501 |" in out
    assert "| 20261020 | 1 | 100 |" in out
