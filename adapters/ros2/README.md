# ROS 2 bridge

The drive as ROS 2 topics, to watch in RViz2 or record with `ros2 bag`: the
truck and its trailer, the route, the map's lanes round the truck, the
vehicles perception tracks, the traffic light's colour, and the view of the
road with all of that drawn over it.

The bridge only listens. It sends nothing to the game, and the drive does not
wait for it: with no bridge running, or a slow one, the stack drops what it
would have sent.

## How it fits

```
stack (Python 3.11+, no ROS)                 ets2ad_ros (ROS 2 Humble, Python 3.10)
tools/mission.py --ros ──► feed.py ──Unix socket──► bridge ──► topics ──► RViz2
tools/ros_replay.py ───────┘
```

ROS 2 Humble's rclpy is built for Ubuntu 22.04's Python 3.10, and the stack
needs 3.11, so they run as two processes. `ets2ad_ros/wire.py` is the format
between them; it, `frames.py` and `scene.py` keep to Python 3.10 and the
standard library, and the stack's tests cover them.

## Setting up (Ubuntu 22.04)

Install ROS 2 Humble (`ros-humble-desktop`) and the message packages the node
uses:

```bash
sudo apt install ros-humble-desktop ros-humble-vision-msgs \
    ros-humble-compressed-image-transport python3-colcon-common-extensions
```

Build the package into a workspace outside the repository:

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws && cd ~/ros2_ws
colcon build --symlink-install --base-paths ~/code_hub/ets2_linuxtest/adapters/ros2
```

## Running

In one terminal, the bridge and RViz2:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ets2ad_ros ets2ad.launch.py
```

In another, the stack, either driving:

```bash
./ets2ad.sh --ros            # or: uv run python tools/mission.py --ros
```

or playing a recorded drive back, with no game running:

```bash
uv run python tools/ros_replay.py data/missions/<mission>/drive-01 --speed 2
```

The socket is `$XDG_RUNTIME_DIR/ets2ad-ros.sock`, which only the user can
open; `ETS2AD_ROS_SOCKET` names another.

RViz2 started from VS Code's terminal, when VS Code is the snap, dies with
`symbol lookup error: /snap/core20/.../libpthread.so.0`: the snap's
environment leaks into it. Start it from an ordinary terminal, or first
`unset GTK_PATH GTK_EXE_PREFIX GIO_MODULE_DIR LOCPATH`.

## Topics

| Topic | Type | |
|---|---|---|
| `/tf` | | `map` → `base_link` → `eye` → `eye_optical`, `map` → `trailer` |
| `/ets2/odom` | `nav_msgs/Odometry` | the truck, from telemetry |
| `/ets2/route` | `nav_msgs/Path` | the line the truck drives (latched) |
| `/ets2/lanes` | `visualization_msgs/MarkerArray` | the map's lanes within 400 m (latched) |
| `/ets2/detections` | `vision_msgs/Detection3DArray` | vehicles perception tracks, in `base_link` |
| `/ets2/markers` | `visualization_msgs/MarkerArray` | the vehicles perception tracks, as boxes |
| `/ets2/truck` | `visualization_msgs/MarkerArray` | truck, trailer, the light's colour, a line of text |
| `/ets2/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | the view of the road, JPEG, up to 5 Hz |
| `/ets2/camera/camera_info` | `sensor_msgs/CameraInfo` | the calibrated camera |
| `/ets2/state` | `std_msgs/String` | the stack's state, JSON |

`map` is the game's world in ROS axes (x east, y north, z up), so positions
are the game's own. The eye is placed over the front axle: telemetry does not
say how far forward the driver sits, so the boxes of vehicles ahead can sit a
metre or so off.

## What a replay shows

A recording keeps the vehicle the truck followed and the light's colour, not
every track, so a replay draws only those. A view of the road is in a
recording only if it was driven with `--keep-frames`.
