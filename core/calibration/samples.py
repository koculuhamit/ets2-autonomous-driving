"""Measurements out of system-identification runs (tools/sysid.py), for fitting.

A run's phases are labelled with their command (tools/analyse_sysid.py
splits them into segments: `phase`, `command`, `t`, `block`); these read
what the truck did under each.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = ["brake_samples", "steering_samples"]

SETTLED_S: Final = 0.8
"""The last of a held steering step, when the wheels have got there."""

MIN_SAMPLES: Final = 8
TURN: Final = 2.0 * math.pi
"""The telemetry's wheel angles are in turns."""


def steering_samples(segments) -> list[tuple[float, float]]:
    """(applied `effective_steering`, the steered wheels' mean angle,
    radians) at the end of each held step ("steer" phases)."""
    found = []
    for seg in (s for s in segments if s.phase == "steer"):
        settled = seg.t >= seg.t[-1] - SETTLED_S
        if settled.sum() < MIN_SAMPLES:
            continue
        truck = seg.block["truck"]
        wheels = truck["wheels"]
        steerable = np.asarray(wheels["steerable"][-1], bool)
        if not steerable.any():
            continue
        angles = np.abs(np.asarray(wheels["steering"][settled], float)[:, steerable])
        applied = float(np.asarray(truck["effective_steering"][settled], float).mean())
        sign = 1.0 if applied >= 0.0 else -1.0
        found.append((applied, sign * float(angles.mean()) * TURN))
    return found


def brake_samples(segments) -> list[tuple[float, float]]:
    """(brake command, deceleration m/s^2) of each braking run ("brake"
    phases), while the brake was applied and the truck still moving."""
    found = []
    for seg in (s for s in segments if s.phase == "brake"):
        truck = seg.block["truck"]
        applied = np.asarray(truck["effective_brake"], float)
        speed = np.asarray(truck["speed"], float)
        band = (applied > seg.command * 0.8) & (speed > 0.5)
        if band.sum() < MIN_SAMPLES:
            continue
        slope = float(np.polyfit(seg.t[band] - seg.t[band][0], speed[band], 1)[0])
        found.append((seg.command, -slope))
    return found
