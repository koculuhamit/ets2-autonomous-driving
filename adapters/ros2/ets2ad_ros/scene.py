"""What the bridge draws, worked out from the stack's messages without ROS.

The node (ets2ad_ros.bridge) only copies these numbers into messages, so
everything that can be wrong about geometry is here, where the stack's tests
reach it.

Frames:
    map           the game's world in ROS axes (ets2ad_ros.frames)
    base_link     the truck: its placement in the game, on the ground under
                  it, x forward, y left, z up
    trailer       the first trailer, the same way
    eye           the driver's eye, where the view and the distances come from
    eye_optical   the eye in camera axes (z forward, x right, y down), pitched
                  down so that the horizon falls on the calibrated row

Standard library and Python 3.10 only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import frames

TRUCK_FRONT_OVERHANG_M = 1.4
TRUCK_REAR_OVERHANG_M = 1.0
TRUCK_WIDTH_M = 2.5
TRUCK_HEIGHT_M = 3.8
CAB_LENGTH_M = 2.4
CHASSIS_HEIGHT_M = 1.2
TRAILER_AHEAD_OF_KINGPIN_M = 1.2
TRAILER_REAR_OVERHANG_M = 1.5
TRAILER_WIDTH_M = 2.55
TRAILER_HEIGHT_M = 4.0
STALE_MEASUREMENT_S = 0.3
"""A track not measured for longer is coasting on its last estimate; the
stack stops following it then (core.perception.tracking), and it is not drawn."""

VEHICLE_HEIGHT_M = {
    "car": 1.5,
    "truck": 3.8,
    "bus": 3.3,
    "motorcycle": 1.4,
    "bicycle": 1.6,
    "person": 1.8,
}
VEHICLE_LENGTH_M = {
    "car": 4.5,
    "truck": 12.0,
    "bus": 12.0,
    "motorcycle": 2.2,
    "bicycle": 1.8,
    "person": 0.6,
}


@dataclass(frozen=True)
class Pose:
    position: tuple
    """(x, y, z) in the parent frame."""
    orientation: tuple
    """Quaternion (x, y, z, w)."""


@dataclass(frozen=True)
class Box:
    """A box in some frame: its centre, its yaw, and its size."""

    frame: str
    centre: tuple
    yaw: float
    size: tuple
    """(length, width, height)."""


def vehicle_pose(placement: dict) -> Pose:
    x, y, z = placement["p"]
    yaw = frames.yaw(placement["heading"])
    return Pose(frames.position(x, y, z), frames.quaternion(yaw, frames.pitch(placement["pitch"])))


def yaw_of(placement: dict) -> float:
    return frames.yaw(placement["heading"])


def _axles(wheels: list) -> tuple:
    """(ahead of the frontmost axle, ahead of the rearmost) in the vehicle's
    own frame: the game's vehicle space has z backwards."""
    if not wheels:
        return (0.0, 0.0)
    ahead = [-wheel[2] for wheel in wheels]
    return (max(ahead), min(ahead))


def truck_boxes(wheels: list) -> tuple:
    """The cab, over the front axle, and the chassis under the trailer."""
    front, rear = _axles(wheels)
    front, rear = front + TRUCK_FRONT_OVERHANG_M, rear - TRUCK_REAR_OVERHANG_M
    cab = Box(
        "base_link",
        (front - CAB_LENGTH_M / 2.0, 0.0, TRUCK_HEIGHT_M / 2.0),
        0.0,
        (CAB_LENGTH_M, TRUCK_WIDTH_M, TRUCK_HEIGHT_M),
    )
    chassis = Box(
        "base_link",
        ((front + rear) / 2.0, 0.0, CHASSIS_HEIGHT_M / 2.0),
        0.0,
        (front - rear, TRUCK_WIDTH_M - 0.2, CHASSIS_HEIGHT_M),
    )
    return cab, chassis


def trailer_box(trailer: dict) -> Box:
    _, rear = _axles(trailer.get("wheels", []))
    front = -trailer.get("kingpin_z", 0.0) + TRAILER_AHEAD_OF_KINGPIN_M
    rear -= TRAILER_REAR_OVERHANG_M
    return Box(
        "trailer",
        ((front + rear) / 2.0, 0.0, TRAILER_HEIGHT_M / 2.0),
        0.0,
        (front - rear, TRAILER_WIDTH_M, TRAILER_HEIGHT_M),
    )


def eye(wheels: list, camera: dict) -> tuple:
    """Where the eye sits in base_link. Over the front axle, as in a cab-over
    truck: telemetry does not say how far forward the seat is."""
    front, _ = _axles(wheels)
    return (front, -camera["seat_right_m"], camera["height_m"])


def eye_optical(camera: dict) -> Pose:
    """eye -> eye_optical: camera axes, pitched down to put the horizon on
    the calibrated row of the full frame."""
    rows = camera["size"][1]
    down = math.atan2(rows / 2.0 - camera["horizon_row"], camera["focal_px"])
    # Rotate base axes to optical (x right, y down, z forward), then pitch
    # about the optical x axis.
    base = _quaternion_multiply((-0.5, 0.5, -0.5, 0.5), _axis_angle((1.0, 0.0, 0.0), -down))
    return Pose((0.0, 0.0, 0.0), base)


def camera_info(camera: dict, width: int, height: int) -> dict:
    """K and P of the view as sent, scaled from the calibrated frame; the
    principal point in the middle, the pitch being in eye_optical."""
    full_width, full_height = camera["size"]
    sx, sy = width / full_width, height / full_height
    fx, fy = camera["focal_px"] * sx, camera["focal_px"] * sy
    cx = camera["centre_col"] * sx
    cy = height / 2.0
    return {
        "width": width,
        "height": height,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def trusted(track: dict) -> bool:
    """Whether the stack itself would act on a track: confirmed by several
    frames, of a size a vehicle can have, and measured just now. The others
    -- a box seen once, a sign read as a car, a car gone a moment ago -- were
    drawn too, and the view filled with vehicles that were not there."""
    return (
        bool(track.get("confirmed", True))
        and bool(track.get("plausible", True))
        and float(track.get("age", 0.0)) <= STALE_MEASUREMENT_S
    )


def object_boxes(objects: dict, wheels: list) -> list:
    """Each vehicle perception tracks and the stack would act on, as a box on
    the ground in base_link. `ahead` is to the vehicle's back from the eye,
    so the box reaches on from there."""
    camera = objects["eye"]
    eye_ahead, eye_left, _ = eye(wheels, camera)
    boxes = []
    for track in objects["tracks"]:
        if not trusted(track):
            continue
        kind = track["kind"]
        length = VEHICLE_LENGTH_M.get(kind, 4.5)
        height = VEHICLE_HEIGHT_M.get(kind, 2.0)
        ahead = eye_ahead + track["ahead"] + length / 2.0
        left = eye_left - track["right"]
        boxes.append(
            (
                track,
                Box(
                    "base_link",
                    (ahead, left, height / 2.0),
                    0.0,
                    (length, max(track["width"], 0.5), height),
                ),
            )
        )
    return boxes


def route_points(route: dict) -> list:
    return [frames.position(*point) for point in route["points"]]


def lane_lines(lanes: dict) -> list:
    return [[frames.position(*point) for point in lane] for lane in lanes["lanes"]]


LIGHT_WORDS = {"amber": "yellow"}
"""How a light's colour reads in the view: the driver calls amber yellow."""


def hud(pose: dict, status: dict | None, objects: dict | None) -> str:
    speed = pose["speed"] * 3.6
    lines = [f"{speed:5.1f} km/h"]
    if pose.get("limit"):
        lines[0] += f"  (limit {pose['limit'] * 3.6:.0f})"
    if status:
        lines.append(status["state"])
    if objects:
        light = objects.get("light", "unknown")
        if light != "unknown":
            lines.append(f"light: {LIGHT_WORDS.get(light, light)}")
        if objects.get("mirror"):
            side = "right" if objects["mirror"][0] > 0 else "left"
            lines.append(f"mirror ({side}): {objects['mirror'][1]:.2f}")
    if pose.get("paused"):
        lines.append("paused")
    return "\n".join(lines)


def _axis_angle(axis: tuple, angle: float) -> tuple:
    s = math.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0))


def _quaternion_multiply(a: tuple, b: tuple) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q: tuple, v: tuple) -> tuple:
    """Vector `v` turned by quaternion `q`."""
    conjugate = (-q[0], -q[1], -q[2], q[3])
    x, y, z, _ = _quaternion_multiply(_quaternion_multiply(q, (v[0], v[1], v[2], 0.0)), conjugate)
    return (x, y, z)
