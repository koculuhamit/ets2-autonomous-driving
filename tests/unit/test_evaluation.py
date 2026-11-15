"""Measuring drives: synthetic telemetry with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE
from core.evaluation import (
    HealthSampler,
    drive_metrics,
    memory_growth,
    session_summary,
)
from core.evaluation.metrics import _reversals
from core.localization.path import ReferencePath

GAME_KM_PER_WORLD_KM = 17.0
"""About what the odometer ran against the world's metres on the drives measured."""


def drive(speed: np.ndarray, dt: float = 0.02, offset: np.ndarray | None = None):
    """Straight up +z at `speed`, `offset` metres to the side of x = 0, the
    odometer running `GAME_KM_PER_WORLD_KM` times the world's distance."""
    t = np.arange(len(speed)) * dt
    blocks = np.zeros(len(speed), dtype=BLOCK_DTYPE)
    truck = blocks["truck"]
    truck["speed"] = speed
    z = np.concatenate([[0.0], np.cumsum(speed[:-1] * dt)])
    truck["placement"]["position"]["z"] = z
    truck["placement"]["position"]["x"] = 0.0 if offset is None else offset
    truck["odometer"] = 50_000.0 + z / 1000.0 * GAME_KM_PER_WORLD_KM
    return t, blocks


def test_distance_speed_and_a_lane_kept() -> None:
    t, blocks = drive(np.full(3000, 20.0))  # 60 s at 72 km/h
    path = ReferencePath(np.array([[0.0, -50.0], [0.0, 2000.0]]))
    metrics = drive_metrics(t, blocks, path=path, ended="end_of_route")
    assert metrics.distance_km == pytest.approx(1.1996 * GAME_KM_PER_WORLD_KM, abs=0.02), (
        "the game's km, from the odometer, not the world's metres"
    )
    assert metrics.mean_speed_kmh == pytest.approx(72.0, abs=0.1), "the speed the truck showed"
    assert metrics.lateral_max_m == pytest.approx(0.0, abs=1e-6)
    assert metrics.hard_brakes == 0 and metrics.impacts == 0


def test_out_of_the_lane_and_off_the_route_are_told_apart() -> None:
    offset = np.zeros(3000)
    offset[1000:1500] = 2.0  # 10 s over the line
    offset[2000:2250] = 40.0  # 5 s on some other road
    t, blocks = drive(np.full(3000, 20.0), offset=offset)
    path = ReferencePath(np.array([[0.0, -50.0], [0.0, 2000.0]]))
    metrics = drive_metrics(t, blocks, path=path)
    assert metrics.out_of_lane_s == pytest.approx(10.0, abs=0.1)
    assert metrics.off_route_s == pytest.approx(5.0, abs=0.1)
    assert metrics.lateral_max_m == pytest.approx(2.0)


def test_one_stop_is_one_hard_brake() -> None:
    speed = np.concatenate([np.full(500, 20.0), np.maximum(20.0 - 5.0 * np.arange(300) * 0.02, 0)])
    speed = np.concatenate([speed, np.zeros(200)])
    t, blocks = drive(speed)
    metrics = drive_metrics(t, blocks)
    assert metrics.hard_brakes == 1
    assert metrics.max_deceleration_mps2 == pytest.approx(5.0, abs=0.3)


def test_a_hit_is_the_speed_gone_at_once() -> None:
    speed = np.concatenate([np.full(200, 6.0), np.zeros(200)])
    t, blocks = drive(speed)
    assert drive_metrics(t, blocks).impacts == 1


def test_a_drive_stopped_by_a_hit_counts_it_though_the_recording_does_not_show_it() -> None:
    """On a delivery towards Amsterdam the drive stopped "hit something:
    0.2 % damage"; the frame that showed it was never recorded, and the
    report gave no hit and no damage."""
    t, blocks = drive(np.full(300, 18.0))
    stopped = drive_metrics(t, blocks, ended="unsafe", detail="hit something: 0.2 % damage")
    assert stopped.impacts == 1 and stopped.damage_pct == pytest.approx(0.2)
    lost = drive_metrics(t, blocks, ended="unsafe", detail="hit something: 9 km/h lost at once")
    assert lost.impacts == 1 and lost.damage_pct == 0.0
    other = drive_metrics(t, blocks, ended="unsafe", detail="no view of the road")
    assert other.impacts == 0 and other.damage_pct == 0.0


def test_the_lead_s_time_to_collision_and_gap() -> None:
    t, blocks = drive(np.full(100, 20.0))
    acc = np.array(
        [
            # t, speed, lead id, distance, closing (the gap's rate), ...
            [0.0, 20.0, 1, 40.0, -4.0, 0, 0, 0],
            [0.1, 20.0, 1, 30.0, -10.0, 0, 0, 0],
            [0.2, 20.0, 1, 25.0, 2.0, 0, 0, 0],
            [0.3, 20.0, 0, np.nan, np.nan, 0, 0, 0],
        ]
    )
    metrics = drive_metrics(t, blocks, acc=acc)
    assert metrics.min_ttc_s == pytest.approx(3.0)
    assert metrics.min_gap_m == pytest.approx(25.0)


def test_steering_reversals_are_counted_past_a_threshold() -> None:
    swing = np.radians([0, 3, 0, 3, 0, 3, 0.0])
    assert _reversals(swing) == 5
    assert _reversals(np.radians([0, 1, 0, 1, 0.0])) == 0


def test_interventions_per_100_km() -> None:
    t, blocks = drive(np.full(5000, 20.0))  # 2 km
    drives = [
        drive_metrics(t, blocks, ended="end_of_route"),
        drive_metrics(t, blocks, ended="disengaged"),
    ]
    summary = session_summary(drives)
    km = 4.0 * GAME_KM_PER_WORLD_KM
    assert summary.distance_km == pytest.approx(km, abs=0.1)
    assert summary.interventions == 1
    assert summary.interventions_per_100km == pytest.approx(100.0 / km, rel=0.01)


def test_memory_growth_leaves_out_the_loading() -> None:
    hours = np.linspace(0.0, 2.0, 100)
    resident = 500.0 + 30.0 * hours  # 30 MB an hour
    resident[:10] += np.linspace(-400, 0, 10)  # loading the model
    samples = np.column_stack([hours * 3600.0, resident, np.zeros(100)])
    assert memory_growth(samples) == pytest.approx(30.0, rel=0.01)
    assert math.isnan(memory_growth(samples[:3]))


def test_the_sampler_measures_this_process() -> None:
    sampler = HealthSampler(every_s=60.0)
    sampler.sample()
    sampler.sample()
    samples = sampler.stop()
    assert samples.shape == (2, 3)
    assert (samples[:, 1] > 10.0).all(), "a Python process is more than 10 MB"
