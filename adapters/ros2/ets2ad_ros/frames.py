"""From the game's coordinates to ROS's (REP 103), and back.

The game's world is x east, y up, z south, metres, and a vehicle's heading is
in rotations, its forward direction `(-sin h, -cos h)` in the (x, z) plane
(core.bridge.convert, measured). ROS's map frame is x east, y north, z up,
and a vehicle's base_link has x forward, y left, z up.

    ros (x, y, z) = game (x, -z, y)          a proper rotation: no mirror
    yaw = heading * 2 pi + pi / 2            0 east, counter-clockwise

Pitch: the game's is negative nose-down; ROS's pitch about base_link's y
(left) axis is positive nose-down, so the sign turns over. Roll is left at
zero: the game's sign for it has not been measured.

Standard library and Python 3.10 only: both the stack and the bridge import it.
"""

from __future__ import annotations

import math

TURN = 2.0 * math.pi


def position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """A game position as a point in ROS's map frame."""
    return (x, -z, y)


def game_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """The inverse of `position`."""
    return (x, z, -y)


def yaw(heading_rotations: float) -> float:
    """The ROS yaw of a game heading, wrapped to (-pi, pi]."""
    angle = heading_rotations * TURN + math.pi / 2.0
    return math.atan2(math.sin(angle), math.cos(angle))


def pitch(game_pitch_rotations: float) -> float:
    return -game_pitch_rotations * TURN


def quaternion(yaw_rad: float, pitch_rad: float = 0.0) -> tuple[float, float, float, float]:
    """(x, y, z, w) of yaw about z then pitch about the new y, roll zero."""
    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def ahead_of(
    origin: tuple[float, float, float], yaw_rad: float, ahead: float, left: float
) -> tuple[float, float, float]:
    """A point `ahead` metres forward of and `left` metres left of `origin`,
    on the ground plane of a vehicle facing `yaw_rad` (ROS map frame)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (origin[0] + ahead * c - left * s, origin[1] + ahead * s + left * c, origin[2])
