"""The bridge and RViz2 with the drive's view.

ros2 launch ets2ad_ros ets2ad.launch.py [rviz:=false] [socket:=/path/to.sock]
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("ets2ad_ros")
    runtime = os.environ.get("XDG_RUNTIME_DIR") or str(Path.home() / ".cache" / "ets2ad")
    default_socket = os.environ.get("ETS2AD_ROS_SOCKET", str(Path(runtime) / "ets2ad-ros.sock"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("socket", default_value=default_socket),
            Node(
                package="ets2ad_ros",
                executable="bridge",
                name="ets2ad_bridge",
                output="screen",
                parameters=[{"socket": LaunchConfiguration("socket")}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", str(Path(share) / "rviz" / "ets2ad.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
