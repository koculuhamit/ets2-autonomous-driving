"""The ROS 2 node: listens for the stack's messages and publishes them.

    ros2 launch ets2ad_ros ets2ad.launch.py          # the node and RViz2
    ros2 run ets2ad_ros bridge                        # the node alone

then, in the stack, `tools/mission.py --ros` (live) or `tools/ros_replay.py
<drive folder>` (a recorded drive).

Topics (frames in ets2ad_ros.scene):
    /tf                                   map -> base_link -> eye -> eye_optical,
                                          map -> trailer
    /ets2/odom                            nav_msgs/Odometry
    /ets2/route                           nav_msgs/Path
    /ets2/detections                      vision_msgs/Detection3DArray
    /ets2/markers                         visualization_msgs/MarkerArray: the vehicles seen
    /ets2/truck                           visualization_msgs/MarkerArray: the truck, its
                                          trailer, the light, a line of text
    /ets2/lanes                           visualization_msgs/MarkerArray: the map's lanes
    /ets2/camera/image_raw/compressed     sensor_msgs/CompressedImage (JPEG)
    /ets2/camera/camera_info              sensor_msgs/CameraInfo
    /ets2/state                           std_msgs/String: the stack's state, JSON

The node only listens and publishes: it sends nothing to the game.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import queue
import socket
import stat
import threading
from pathlib import Path as FilePath

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from . import scene, wire

LIGHT_COLOURS = {
    "red": (1.0, 0.1, 0.1),
    "amber": (1.0, 0.7, 0.0),
    "green": (0.1, 0.9, 0.2),
}


def _colour(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


class Bridge(Node):
    def __init__(self) -> None:
        super().__init__("ets2ad_bridge")
        self.declare_parameter("socket", str(wire.socket_path()))
        path = self.get_parameter("socket").get_parameter_value().string_value
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tf = TransformBroadcaster(self)
        self.odom = self.create_publisher(Odometry, "/ets2/odom", 10)
        self.route = self.create_publisher(Path, "/ets2/route", latched)
        self.detections = self.create_publisher(Detection3DArray, "/ets2/detections", 10)
        self.markers = self.create_publisher(MarkerArray, "/ets2/markers", 10)
        self.truck = self.create_publisher(MarkerArray, "/ets2/truck", 10)
        self.lane_markers = self.create_publisher(MarkerArray, "/ets2/lanes", latched)
        self.image = self.create_publisher(CompressedImage, "/ets2/camera/image_raw/compressed", 5)
        self.camera_info = self.create_publisher(CameraInfo, "/ets2/camera/camera_info", 5)
        self.state = self.create_publisher(String, "/ets2/state", latched)

        self._inbox: queue.Queue[wire.Message] = queue.Queue(maxsize=256)
        self._pose: dict | None = None
        self._objects: dict | None = None
        self._status: dict | None = None
        self._camera: dict | None = None
        self._stopping = threading.Event()
        self._server = self._listen(path)
        threading.Thread(target=self._accept, daemon=True).start()
        self.create_timer(1.0 / 60.0, self._drain)
        self.get_logger().info(f"listening for the stack on {path}")

    # The socket, on a thread of its own.

    def _listen(self, path: str) -> socket.socket:
        where = FilePath(path)
        if where.exists():
            if not stat.S_ISSOCK(where.stat().st_mode):
                raise SystemExit(f"{path} is there and is not a socket; not touching it")
            where.unlink()
        where.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Only this user may connect: the socket is made 0600.
        old = os.umask(0o177)
        try:
            server.bind(path)
        finally:
            os.umask(old)
        server.listen(1)
        server.settimeout(0.5)
        self._path = path
        return server

    def _accept(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.get_logger().info("the stack is connected")
            self._read(connection)
            self.get_logger().info("the stack went away; waiting for it")

    def _read(self, connection: socket.socket) -> None:
        decoder = wire.Decoder()
        connection.settimeout(0.5)
        with connection:
            while not self._stopping.is_set():
                try:
                    data = connection.recv(1 << 20)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not data:
                    return
                try:
                    messages = decoder.feed(data)
                except (wire.WireError, ValueError) as error:
                    self.get_logger().error(f"dropping the connection: {error}")
                    return
                for message in messages:
                    with contextlib.suppress(queue.Full):
                        self._inbox.put_nowait(message)

    def close(self) -> None:
        self._stopping.set()
        with contextlib.suppress(OSError):
            self._server.close()
        with contextlib.suppress(OSError):
            FilePath(self._path).unlink()

    # Publishing, on the executor's thread.

    def _drain(self) -> None:
        while True:
            try:
                message = self._inbox.get_nowait()
            except queue.Empty:
                return
            handler = getattr(self, f"_on_{message.kind}", None)
            if handler is None:
                continue
            try:
                handler(message)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                self.get_logger().warning(f"a {message.kind} message not drawn: {error!r}")

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _transform(self, parent: str, child: str, pose: scene.Pose, stamp) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in pose.position
        )
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in pose.orientation)
        return t

    def _on_pose(self, message: wire.Message) -> None:
        pose = message.fields
        self._pose = pose
        stamp = self._stamp()
        truck = scene.vehicle_pose(pose["truck"])
        transforms = [self._transform("map", "base_link", truck, stamp)]
        if "trailer" in pose:
            trailer = scene.vehicle_pose(pose["trailer"])
            transforms.append(self._transform("map", "trailer", trailer, stamp))
        if self._camera is not None:
            eye = scene.Pose(scene.eye(pose["wheels"], self._camera), (0.0, 0.0, 0.0, 1.0))
            transforms.append(self._transform("base_link", "eye", eye, stamp))
            optical = scene.eye_optical(self._camera)
            transforms.append(self._transform("eye", "eye_optical", optical, stamp))
        self.tf.sendTransform(transforms)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        p = odom.pose.pose
        p.position.x, p.position.y, p.position.z = (float(v) for v in truck.position)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
            float(v) for v in truck.orientation
        )
        odom.twist.twist.linear.x = float(pose["speed"])
        self.odom.publish(odom)
        self._draw(stamp)

    def _on_route(self, message: wire.Message) -> None:
        path = Path()
        path.header.stamp = self._stamp()
        path.header.frame_id = "map"
        for x, y, z in scene.route_points(message.fields):
            stamped = PoseStamped()
            stamped.header = path.header
            stamped.pose.position.x, stamped.pose.position.y = float(x), float(y)
            stamped.pose.position.z = float(z) + 0.1
            stamped.pose.orientation.w = 1.0
            path.poses.append(stamped)
        self.route.publish(path)

    def _on_lanes(self, message: wire.Message) -> None:
        marker = Marker()
        marker.header.stamp = self._stamp()
        marker.header.frame_id = "map"
        marker.ns, marker.id = "lanes", 0
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.15
        marker.color = _colour(0.6, 0.6, 0.65, 0.8)
        for line in scene.lane_lines(message.fields):
            for a, b in itertools.pairwise(line):
                marker.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                marker.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
        self.lane_markers.publish(MarkerArray(markers=[marker]))

    def _on_objects(self, message: wire.Message) -> None:
        self._objects = message.fields
        self._camera = message.fields["eye"]
        if self._pose is None:
            return
        stamp = self._stamp()
        array = Detection3DArray()
        array.header.stamp = stamp
        array.header.frame_id = "base_link"
        for track, box in scene.object_boxes(message.fields, self._pose["wheels"]):
            detection = Detection3D()
            detection.header = array.header
            detection.id = str(track["id"])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = track["kind"]
            hypothesis.hypothesis.score = 1.0 if track.get("confirmed") else 0.5
            detection.results.append(hypothesis)
            centre = detection.bbox.center
            centre.position.x, centre.position.y, centre.position.z = (float(v) for v in box.centre)
            centre.orientation.w = 1.0
            detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = (
                float(v) for v in box.size
            )
            array.detections.append(detection)
        self.detections.publish(array)

    def _on_image(self, message: wire.Message) -> None:
        stamp = self._stamp()
        self._camera = message.fields["camera"]
        image = CompressedImage()
        image.header.stamp = stamp
        image.header.frame_id = "eye_optical"
        image.format = "jpeg"
        image.data = message.blob
        self.image.publish(image)
        info = scene.camera_info(
            message.fields["camera"], message.fields["width"], message.fields["height"]
        )
        msg = CameraInfo()
        msg.header = image.header
        msg.width, msg.height = info["width"], info["height"]
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        msg.k = [float(v) for v in info["k"]]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [float(v) for v in info["p"]]
        self.camera_info.publish(msg)

    def _on_status(self, message: wire.Message) -> None:
        self._status = message.fields
        self.state.publish(String(data=json.dumps({"t": message.t, **message.fields})))

    # The markers that follow the truck.

    def _box_marker(self, box: scene.Box, ns: str, ident: int, colour, stamp) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = box.frame
        marker.ns, marker.id = ns, ident
        marker.type = Marker.CUBE
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            float(v) for v in box.centre
        )
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in box.size)
        marker.color = colour
        marker.lifetime = Duration(sec=0, nanosec=500_000_000)
        return marker

    def _draw(self, stamp) -> None:
        pose = self._pose
        # Two arrays: what the truck is and says, which a camera view from
        # inside the cab must not draw over the road, and what it sees.
        cab, chassis = scene.truck_boxes(pose["wheels"])
        own = [
            self._box_marker(cab, "truck", 0, _colour(0.15, 0.45, 0.9, 0.95), stamp),
            self._box_marker(chassis, "truck", 1, _colour(0.2, 0.25, 0.35, 0.95), stamp),
        ]
        markers = []
        if "trailer" in pose:
            own.append(
                self._box_marker(
                    scene.trailer_box(pose["trailer"]),
                    "trailer",
                    0,
                    _colour(0.75, 0.75, 0.8, 0.8),
                    stamp,
                )
            )
        objects = self._objects
        if objects is not None:
            for track, box in scene.object_boxes(objects, pose["wheels"]):
                lead = track["id"] == objects.get("lead")
                colour = _colour(1.0, 0.5, 0.0, 0.9) if lead else _colour(0.9, 0.9, 0.2, 0.7)
                markers.append(self._box_marker(box, "vehicles", track["id"], colour, stamp))
            light = LIGHT_COLOURS.get(objects.get("light", ""))
            if light is not None:
                sphere = Marker()
                sphere.header.stamp = stamp
                sphere.header.frame_id = "base_link"
                sphere.ns, sphere.id, sphere.type = "light", 0, Marker.SPHERE
                sphere.pose.position.x, sphere.pose.position.z = 6.0, 5.5
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 1.2
                sphere.color = _colour(*light)
                sphere.lifetime = Duration(sec=0, nanosec=500_000_000)
                own.append(sphere)
        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = "base_link"
        text.ns, text.id, text.type = "hud", 0, Marker.TEXT_VIEW_FACING
        # Above the cab and ahead of it, clear of the truck seen from behind.
        text.pose.position.x = 14.0
        text.pose.position.z = 9.0
        text.pose.orientation.w = 1.0
        text.scale.z = 1.6
        text.color = _colour(1.0, 1.0, 1.0)
        text.text = scene.hud(pose, self._status, objects)
        own.append(text)
        self.truck.publish(MarkerArray(markers=own))
        self.markers.publish(MarkerArray(markers=markers))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
