"""The ROS 2 adapter's shared half: framing, axes, and what the bridge draws."""

from __future__ import annotations

import ast
import math
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.ros2 import feed as feed_module
from adapters.ros2.ets2ad_ros import frames, scene, wire
from core.bridge.layout import BLOCK_DTYPE

PACKAGE = Path(feed_module.__file__).parent / "ets2ad_ros"

CAMERA = {
    "focal_px": 819.0,
    "centre_col": 960.1,
    "horizon_row": 438.3,
    "height_m": 2.881,
    "seat_right_m": -0.86,
    "size": [1920, 1080],
}
# The recorded truck: front axle 1.81 m ahead of its placement, rear 2.04 m behind.
WHEELS = [
    [-1.05, 0.51, -1.81, 0.506, 1],
    [1.05, 0.51, -1.81, 0.506, 1],
    [-0.92, 0.51, 2.04, 0.506, 0],
    [0.92, 0.51, 2.04, 0.506, 0],
]


def test_a_message_survives_the_wire_however_it_is_split() -> None:
    first = wire.Message(wire.POSE, 1.5, {"speed": 12.25, "truck": {"p": [1.0, 2.0, 3.0]}})
    second = wire.Message(wire.IMAGE, 2.0, {"width": 640}, b"\xff\xd8jpeg\xff\xd9")
    stream = wire.encode(first) + wire.encode(second)
    decoder = wire.Decoder()
    got = []
    for start in range(0, len(stream), 7):
        got += decoder.feed(stream[start : start + 7])
    assert got == [first, second]


def test_another_stream_is_refused() -> None:
    with pytest.raises(wire.WireError, match="another version"):
        wire.Decoder().feed(b"E2R0" + bytes(8))
    with pytest.raises(ValueError):
        wire.encode(wire.Message(wire.POSE, 0.0, {"speed": math.nan}))


def test_the_shared_modules_keep_to_python_3_10_and_the_standard_library() -> None:
    """The bridge runs on Humble's Python 3.10, without the stack's packages."""
    allowed = {"__future__", "json", "os", "struct", "dataclasses", "pathlib", "math"}
    for name in ("wire.py", "frames.py", "scene.py", "__init__.py"):
        tree = ast.parse((PACKAGE / name).read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= allowed, name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed, (name, node.module)


def test_game_axes_turn_into_ros_axes_without_a_mirror() -> None:
    assert frames.position(1.0, 2.0, 3.0) == (1.0, -3.0, 2.0)
    assert frames.game_position(*frames.position(1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    # Heading 0 drives along -z, which is north: yaw pi/2.
    assert frames.yaw(0.0) == pytest.approx(math.pi / 2)
    # A quarter turn: forward (-sin h, -cos h) = (-1, 0) is west.
    assert abs(frames.yaw(0.25)) == pytest.approx(math.pi)


@pytest.mark.parametrize("heading", [0.0, 0.1, 0.25, 0.5, 0.71, 0.99])
def test_yaw_points_where_the_truck_drives(heading: float) -> None:
    h = heading * 2 * math.pi
    forward_game = (-math.sin(h), 0.0, -math.cos(h))
    fx, fy, _ = frames.position(*forward_game)
    yaw = frames.yaw(heading)
    assert (math.cos(yaw), math.sin(yaw)) == pytest.approx((fx, fy), abs=1e-9)
    q = frames.quaternion(yaw)
    assert scene.rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((fx, fy, 0.0), abs=1e-9)


def test_downhill_pitches_the_nose_down() -> None:
    q = frames.quaternion(0.0, frames.pitch(-0.01))
    assert scene.rotate(q, (1.0, 0.0, 0.0))[2] < 0.0


def test_the_eye_looks_forward_and_a_little_down() -> None:
    q = scene.eye_optical(CAMERA).orientation
    forward = scene.rotate(q, (0.0, 0.0, 1.0))
    down = math.atan2(540.0 - 438.3, 819.0)
    assert forward == pytest.approx((math.cos(down), 0.0, -math.sin(down)), abs=1e-9)
    assert scene.rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


def test_the_horizon_falls_on_the_calibrated_row() -> None:
    """A point far ahead on the eye's level projects on horizon_row."""
    info = scene.camera_info(CAMERA, 1920, 1080)
    q = scene.eye_optical(CAMERA).orientation
    inverse = (-q[0], -q[1], -q[2], q[3])
    x, y, z = scene.rotate(inverse, (1000.0, 0.0, 0.0))
    fx, _, cx, _, fy, cy = info["k"][:6]
    assert (fx * x / z + cx, fy * y / z + cy) == pytest.approx((960.1, 438.3), abs=0.01)
    assert scene.camera_info(CAMERA, 640, 360)["k"][0] == pytest.approx(819.0 / 3)


def test_the_truck_and_its_trailer_are_drawn_round_their_axles() -> None:
    cab, chassis = scene.truck_boxes(WHEELS)
    assert chassis.size[0] == pytest.approx(1.81 + 2.04 + 1.4 + 1.0)
    assert chassis.centre[0] == pytest.approx((1.81 + 1.4 - 2.04 - 1.0) / 2)
    assert cab.centre[0] + cab.size[0] / 2 == pytest.approx(1.81 + 1.4), "the cab is the front"
    assert chassis.size[2] < cab.size[2]
    trailer = scene.trailer_box(
        {"wheels": [[-0.82, 0.51, z, 0.5, 0] for z in (2.95, 4.31, 5.63)], "kingpin_z": -3.44}
    )
    assert trailer.size[0] == pytest.approx(3.44 + 1.2 + 5.63 + 1.5)


def test_a_vehicle_ahead_is_placed_from_the_eye() -> None:
    objects = {
        "eye": CAMERA,
        "tracks": [{"id": 4, "kind": "car", "ahead": 20.0, "right": 1.0, "width": 1.8}],
    }
    [(track, box)] = scene.object_boxes(objects, WHEELS)
    assert track["id"] == 4
    # Eye over the front axle, 0.86 m left of the middle; the car 1 m right of the eye.
    assert box.centre[:2] == pytest.approx((1.81 + 20.0 + 2.25, 0.86 - 1.0))
    assert box.size == pytest.approx((4.5, 1.8, 1.5))


def test_only_tracks_the_stack_would_act_on_are_drawn() -> None:
    base = {"kind": "car", "ahead": 20.0, "right": 0.0, "width": 1.8}
    tracks = [
        {**base, "id": 1, "confirmed": True, "plausible": True, "age": 0.05},
        {**base, "id": 2, "confirmed": False, "plausible": True, "age": 0.0},
        {**base, "id": 3, "confirmed": True, "plausible": False, "age": 0.0},
        {**base, "id": 4, "confirmed": True, "plausible": True, "age": 0.8},
        {**base, "id": 5},
    ]
    boxes = scene.object_boxes({"eye": CAMERA, "tracks": tracks}, WHEELS)
    assert [track["id"] for track, _ in boxes] == [1, 5], "5: a replay's lead, no flags"


def _block() -> np.ndarray:
    block = np.zeros((), dtype=BLOCK_DTYPE)
    truck = block["truck"]
    truck["placement"]["position"]["x"] = 11005.6
    truck["placement"]["position"]["z"] = -27696.6
    truck["placement"]["orientation"]["heading"] = 0.0714
    truck["speed"] = 12.5
    truck["indicator_left"] = 1
    truck["wheels"]["count"] = 4
    for index, wheel in enumerate(WHEELS):
        truck["wheels"]["position"][index] = tuple(wheel[:3])
        truck["wheels"]["radius"][index] = wheel[3]
        truck["wheels"]["steerable"][index] = wheel[4]
    block["navigation"]["speed_limit"] = 13.89
    block["header"]["flags"] = 1 << 1
    return block


def test_the_pose_is_read_from_telemetry() -> None:
    fields = feed_module.pose_fields(_block())
    assert fields["truck"]["p"] == pytest.approx([11005.6, 0.0, -27696.6])
    assert fields["wheels"][0] == pytest.approx(WHEELS[0], abs=1e-3)
    assert fields["blinker"] == -1
    assert fields["limit"] == pytest.approx(13.89, abs=1e-3)
    assert fields["paused"] is True
    assert "trailer" not in fields
    wire.encode(wire.Message(wire.POSE, 0.0, fields))


def test_a_route_is_thinned_but_keeps_its_ends() -> None:
    points = np.column_stack([np.linspace(0, 10, 101), np.zeros(101), np.zeros(101)])
    sent = feed_module.route_fields(points)["points"]
    assert sent[0] == [0.0, 0.0, 0.0] and sent[-1] == [10.0, 0.0, 0.0]
    assert len(sent) == 11
    with pytest.raises(ValueError):
        feed_module.route_fields(np.zeros((3, 4)))


def test_objects_carry_tracks_light_and_mirror() -> None:
    track = SimpleNamespace(
        id=3,
        kind="car",
        distance=25.0,
        lateral=-0.4,
        width=1.9,
        closing=1.2,
        confirmed=True,
        plausible=False,
        measurement_age=0.12,
    )
    gone = SimpleNamespace(
        id=4, kind="car", distance=math.nan, lateral=0.0, width=1.9, closing=0.0, confirmed=False
    )
    report = SimpleNamespace(
        tracks=(track, gone), light="red", mirror=None, mirror_far=(1, 0.04), covered=False
    )
    camera = SimpleNamespace(**{k: v for k, v in CAMERA.items() if k != "size"})
    fields = feed_module.objects_fields(report, camera, lead_id=3)
    assert [t["id"] for t in fields["tracks"]] == [3]
    assert fields["tracks"][0]["plausible"] is False and fields["tracks"][0]["age"] == 0.12
    assert fields["mirror"] == [1, 0.04] and fields["light"] == "red" and fields["lead"] == 3
    assert fields["eye"]["size"] == [1920, 1080]
    text = scene.hud({"speed": 12.5, "limit": 13.89}, {"state": "driving"}, fields)
    assert "45.0 km/h" in text and "limit 50" in text and "mirror (right)" in text
    # Amber reads as the driver calls it.
    yellow = scene.hud({"speed": 5.0}, None, {**fields, "light": "amber"})
    assert "light: yellow" in yellow and "amber" not in yellow


def test_the_feed_drops_messages_with_no_bridge(tmp_path: Path) -> None:
    feed = feed_module.Feed(tmp_path / "nobody.sock", retry_s=0.05)
    started = time.perf_counter()
    for index in range(1000):
        feed.send(wire.Message(wire.POSE, float(index), {"n": index}))
    assert time.perf_counter() - started < 0.5, "send never waits"
    time.sleep(0.15)
    assert not feed.connected and feed.sent == 0
    feed.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets in this Python")
def test_the_feed_sends_the_newest_of_each_kind_to_a_bridge(tmp_path: Path) -> None:
    path = tmp_path / "bridge.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    server.settimeout(3.0)
    feed = feed_module.Feed(path, retry_s=0.05)
    feed.send(wire.Message(wire.STATUS, 0.0, {"state": "ready", "line": ""}))
    connection, _ = server.accept()
    connection.settimeout(3.0)
    decoder = wire.Decoder()
    got: list[wire.Message] = []
    deadline = time.monotonic() + 3.0
    while not got and time.monotonic() < deadline:
        got += decoder.feed(connection.recv(65536))
    assert got[0].kind == wire.STATUS
    tap = feed_module.Tap(feed)
    block = _block()
    tap.pose(1.0, block)
    tap.pose(1.001, block)  # sooner than its rate: not sent
    tap.route(1.0, np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    deadline = time.monotonic() + 3.0
    kinds: list[str] = []
    while len(kinds) < 2 and time.monotonic() < deadline:
        kinds += [m.kind for m in decoder.feed(connection.recv(65536))]
    assert sorted(kinds) == [wire.POSE, wire.ROUTE]
    feed.close()
    connection.close()
    server.close()


def test_the_drive_path_is_drawn_at_the_truck_height() -> None:
    sent = feed_module.route_fields(np.array([[0.0, 0.0], [5.0, -5.0]]), height=7.5)["points"]
    assert sent == [[0.0, 7.5, 0.0], [5.0, 7.5, -5.0]]


class _Taken:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: self.calls.append(name)


def test_a_turn_of_the_drive_tells_the_bridge_everything_it_has() -> None:
    from tools.follow import tell_ros

    tap = _Taken()
    path = SimpleNamespace(points=np.zeros((4, 2)))
    report = SimpleNamespace(tracks=())
    following = SimpleNamespace(lead_id=None, decision=SimpleNamespace(reason="free road"))
    tell_ros(tap, 1.0, _block(), path, None, report, object(), following)
    assert tap.calls == ["pose", "route", "lanes", "objects", "status"]
    tap.calls.clear()
    tell_ros(tap, 1.0, _block(), None, None, None, None, None)
    assert tap.calls == ["pose", "lanes", "status"]


def test_mission_passes_ros_on_to_each_drive() -> None:
    from tools.mission import drive_arguments

    args = SimpleNamespace(duration=10, time_gap=1.5, keep_frames=0.0, allow_cpu=False, ros=True)
    assert "--ros" in drive_arguments(args)
    assert "--ros" not in drive_arguments(SimpleNamespace(**{**vars(args), "ros": False}))


def test_the_route_is_drawn_at_the_road_s_heights_the_near_pass_where_it_crosses_itself() -> None:
    """Drawn at the truck's own height all along, the route floated over the
    lanes downhill and under them uphill, by up to 8.5 m 200 m on."""
    from adapters.ros2.feed import heights_along

    # Up a hill, round a loop, and over the road it came along, 8 m higher.
    s = np.arange(0.0, 400.0, 2.0)
    angle = np.clip((s - 100.0) / 200.0, 0.0, 1.0) * 2.0 * np.pi
    x = np.where(s < 100.0, 0.0, 30.0 * np.sin(angle))
    z = np.where(s < 100.0, s, 100.0 + 30.0 * (1.0 - np.cos(angle)) - (s >= 300.0) * (s - 300.0))
    y = 0.04 * s
    ground = np.column_stack([x, y, z])
    path = ground[:, [0, 2]]
    heights = heights_along(path, ground, 0, 0.0)
    assert np.abs(heights - y).max() < 0.2
    # Started from the upper pass, where the route goes over its own start.
    upper = int(np.argmin(np.abs(s - 330.0)))
    from_above = heights_along(path, ground, upper, y[upper])
    assert abs(from_above[upper] - y[upper]) < 0.2


def test_the_tap_sends_the_route_about_the_truck_at_the_road_s_heights() -> None:
    from adapters.ros2.feed import ROUTE_AHEAD_M, Tap

    class Recorder:
        def __init__(self) -> None:
            self.sent = []

        def send(self, message) -> None:
            self.sent.append(message)

    s = np.arange(0.0, 6000.0, 1.0)
    ground = np.column_stack([np.zeros_like(s), 0.05 * s, s])
    tap = Tap(Recorder())
    tap.route(1.0, ground[:, [0, 2]], height=50.0, progress=1000.0, ground=ground)
    points = np.asarray(tap.feed.sent[-1].fields["points"])
    assert points[:, 2].max() <= 1000.0 + ROUTE_AHEAD_M + 1.0
    assert np.abs(points[:, 1] - 0.05 * points[:, 2]).max() < 0.1
