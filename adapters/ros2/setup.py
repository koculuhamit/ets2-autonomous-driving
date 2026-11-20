"""ament_python build of the bridge node: `colcon build --base-paths adapters/ros2`."""

from pathlib import Path

from setuptools import setup

PACKAGE = "ets2ad_ros"

setup(
    name=PACKAGE,
    version="2.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", sorted(str(p) for p in Path("launch").glob("*.launch.py"))),
        (f"share/{PACKAGE}/rviz", sorted(str(p) for p in Path("rviz").glob("*.rviz"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="The ETS2 autopilot's view of the drive as ROS 2 topics.",
    license="MIT",
    entry_points={"console_scripts": [f"bridge = {PACKAGE}.bridge:main"]},
)
