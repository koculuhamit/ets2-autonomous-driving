"""Tell the ROS 2 bridge what the stack knows, without ever waiting for it.

    feed = Feed()                 # does nothing until a bridge is listening
    tap = Tap(feed)
    tap.pose(now, snapshot.block) # every turn of the loop
    ...
    feed.close()

The drive's loop must not be slowed by a bridge that is slow, stuck or
absent, so `Feed.send` only puts the message in a mailbox, one slot per kind,
and returns; a thread of its own connects to the bridge's socket
(ets2ad_ros.wire.socket_path), and sends what is in the mailbox. A newer
message of a kind replaces one not yet sent: a slow bridge sees fewer poses,
never older ones. With no bridge listening the thread tries again every few
seconds and the messages are dropped.

`Tap` turns the stack's own objects into messages (ets2ad_ros.wire) and keeps
each kind to its rate. Nothing here imports ROS.
"""

from __future__ import annotations

import contextlib
import math
import socket
import threading
import time
from pathlib import Path

import numpy as np

from adapters.ros2.ets2ad_ros import wire
from core.bridge.layout import HeaderFlag

RETRY_S = 2.0
SEND_TIMEOUT_S = 0.5

POSE_EVERY_S = 1.0 / 30.0
OBJECTS_EVERY_S = 0.1
IMAGE_EVERY_S = 0.2
LANES_EVERY_S = 2.0
LANES_RADIUS_M = 400.0
"""Lanes with a point this near are sent whole. Wide: a motorway's lane can
have points hundreds of metres apart, and a narrow search misses the very
lane the truck is on."""
LANE_SPACING_M = 2.0
ROUTE_SPACING_M = 1.0


class Feed:
    """A mailbox and the thread that empties it into the bridge's socket."""

    def __init__(self, path: Path | None = None, *, retry_s: float = RETRY_S) -> None:
        self.path = path or wire.socket_path()
        self._retry_s = retry_s
        self._mailbox: dict[str, wire.Message] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._socket: socket.socket | None = None
        self.connected = False
        self.sent = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="ros-feed", daemon=True)
        self._thread.start()

    def send(self, message: wire.Message) -> None:
        with self._lock:
            if message.kind in self._mailbox:
                self.dropped += 1
            self._mailbox[message.kind] = message
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        self._thread.join(timeout=2.0)
        self._disconnect()

    def _run(self) -> None:
        next_try = 0.0
        while not self._closed:
            self._wake.wait(timeout=self._retry_s)
            self._wake.clear()
            if self._closed:
                break
            if self._socket is None:
                if time.monotonic() < next_try:
                    self._take()
                    continue
                if not self._connect():
                    next_try = time.monotonic() + self._retry_s
                    self._take()
                    continue
            for message in self._take():
                try:
                    self._socket.sendall(wire.encode(message))
                    self.sent += 1
                except (OSError, wire.WireError):
                    self._disconnect()
                    next_try = time.monotonic() + self._retry_s
                    break

    def _take(self) -> list[wire.Message]:
        with self._lock:
            # Kinds in a fixed order: a route before the pose that is on it.
            taken = [self._mailbox.pop(kind) for kind in wire.KINDS if kind in self._mailbox]
        return taken

    def _connect(self) -> bool:
        if not hasattr(socket, "AF_UNIX") or not self.path.exists():
            return False
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        candidate.settimeout(SEND_TIMEOUT_S)
        try:
            candidate.connect(str(self.path))
        except OSError:
            candidate.close()
            return False
        self._socket = candidate
        self.connected = True
        return True

    def _disconnect(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
        self._socket = None
        self.connected = False


def _round(values, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _thinned(points: np.ndarray, spacing: float) -> np.ndarray:
    """Every point at least `spacing` from the one kept before it, and the last."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    kept = [0]
    for index in range(1, len(points) - 1):
        if np.hypot(*(points[index, [0, 2]] - points[kept[-1], [0, 2]])) >= spacing:
            kept.append(index)
    kept.append(len(points) - 1)
    return points[kept]


def _placement(placement) -> dict:
    position = placement["position"]
    orientation = placement["orientation"]
    return {
        "p": _round([position["x"], position["y"], position["z"]]),
        "heading": float(orientation["heading"]),
        "pitch": float(orientation["pitch"]),
    }


def _wheels(wheels) -> list:
    count = int(wheels["count"])
    return [
        _round([*wheels["position"][i].tolist(), wheels["radius"][i], wheels["steerable"][i]], 3)
        for i in range(count)
    ]


def pose_fields(block) -> dict:
    """The truck and trailer as the bridge draws them. Wheel positions are in
    the vehicle's own space (x right, y up from the ground, z back)."""
    truck = block["truck"]
    fields = {
        "truck": _placement(truck["placement"]),
        "wheels": _wheels(truck["wheels"]),
        "speed": float(truck["speed"]),
        "steering": float(truck["effective_steering"]),
        "throttle": float(truck["effective_throttle"]),
        "brake": float(truck["effective_brake"]),
        "gear": int(truck["gear_displayed"]),
        "blinker": int(truck["indicator_right"]) - int(truck["indicator_left"]),
        "beacon": bool(truck["light_beacon"]),
        "hazard": bool(truck["hazard_warning"]),
        "odometer_km": float(truck["odometer"]),
        "paused": bool(int(block["header"]["flags"]) & HeaderFlag.PAUSED),
    }
    limit = float(block["navigation"]["speed_limit"])
    fields["limit"] = limit if limit > 0.0 else None
    trailer = block["trailers"][0]
    if int(trailer["connected"]):
        fields["trailer"] = _placement(trailer["placement"])
        fields["trailer"]["wheels"] = _wheels(trailer["wheels"])
        fields["trailer"]["kingpin_z"] = float(trailer["hook_position"][2])
    return fields


def route_fields(points: np.ndarray, height: float = 0.0) -> dict:
    """A route's line, game (x, y, z), thinned to `ROUTE_SPACING_M`. A line
    of (x, z) -- the drive's path -- is drawn at `height`."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and points.shape[1] == 2:
        points = np.column_stack([points[:, 0], np.full(len(points), height), points[:, 1]])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"route points of shape {points.shape}, not (n, 2) or (n, 3)")
    return {"points": _round(_thinned(points, ROUTE_SPACING_M), 2)}


def lanes_fields(network, x: float, z: float, radius: float = LANES_RADIUS_M) -> dict:
    lanes = []
    for lane in network.lanes_near(x, z, radius):
        points = _thinned(network.lane_points(int(lane)), LANE_SPACING_M)
        lanes.append(_round(points, 2))
    return {"lanes": lanes, "centre": _round([x, z], 1), "radius": radius}


def objects_fields(report, camera, *, lead_id: int | None = None) -> dict:
    """What perception sees, from the eye: `ahead` along the view, `right`
    across it, metres. `camera` places the eye in the truck."""
    tracks = [
        {
            "id": int(track.id),
            "kind": str(getattr(track.kind, "name", track.kind)).lower(),
            "ahead": round(float(track.distance), 2),
            "right": round(float(track.lateral), 2),
            "width": round(float(track.width), 2),
            "closing": round(float(track.closing), 2),
            "confirmed": bool(track.confirmed),
            "plausible": bool(getattr(track, "plausible", True)),
            "age": round(float(getattr(track, "measurement_age", 0.0)), 3),
        }
        for track in report.tracks
        if math.isfinite(track.distance) and math.isfinite(track.lateral)
    ]
    mirror = report.mirror_far or report.mirror
    return {
        "tracks": tracks,
        "lead": lead_id,
        "light": str(getattr(report.light, "name", report.light)).lower(),
        "mirror": None if mirror is None else [int(mirror[0]), round(float(mirror[1]), 3)],
        "covered": bool(report.covered),
        "eye": camera_fields(camera),
    }


def camera_fields(camera) -> dict:
    return {
        "focal_px": float(camera.focal_px),
        "centre_col": float(camera.centre_col),
        "horizon_row": float(camera.horizon_row),
        "height_m": float(camera.height_m),
        "seat_right_m": float(camera.seat_right_m),
        "size": [int(v) for v in getattr(camera, "frame_size", (1920, 1080))],
    }


class Tap:
    """The stack's side of the bridge: each kind at its own rate."""

    def __init__(self, feed: Feed) -> None:
        self.feed = feed
        self._last: dict[str, float] = {}
        self._route_id: int | None = None

    def _due(self, kind: str, now: float, every: float) -> bool:
        if now - self._last.get(kind, -math.inf) < every:
            return False
        self._last[kind] = now
        return True

    def pose(self, now: float, block) -> None:
        if self._due(wire.POSE, now, POSE_EVERY_S):
            self.feed.send(wire.Message(wire.POSE, now, pose_fields(block)))

    def route(self, now: float, points: np.ndarray, height: float = 0.0) -> None:
        """Sent when the line is a new one (by identity), and again every
        few seconds for a bridge that came up after it."""
        new = id(points) != self._route_id
        if new or self._due(wire.ROUTE, now, 5.0):
            self._route_id = id(points)
            self._last[wire.ROUTE] = now
            self.feed.send(wire.Message(wire.ROUTE, now, route_fields(points, height)))

    def lanes(self, now: float, network, x: float, z: float) -> None:
        if network is not None and self._due(wire.LANES, now, LANES_EVERY_S):
            self.feed.send(wire.Message(wire.LANES, now, lanes_fields(network, x, z)))

    def objects(self, now: float, report, camera, *, lead_id: int | None = None) -> None:
        if report is not None and self._due(wire.OBJECTS, now, OBJECTS_EVERY_S):
            fields = objects_fields(report, camera, lead_id=lead_id)
            self.feed.send(wire.Message(wire.OBJECTS, now, fields))

    def image(self, now: float, jpeg: bytes, camera, width: int, height: int) -> None:
        if jpeg and self._due(wire.IMAGE, now, IMAGE_EVERY_S):
            fields = {"camera": camera_fields(camera), "width": width, "height": height}
            self.feed.send(wire.Message(wire.IMAGE, now, fields, bytes(jpeg)))

    def status(self, now: float, state: str, line: str = "") -> None:
        self.feed.send(wire.Message(wire.STATUS, now, {"state": state, "line": line}))
