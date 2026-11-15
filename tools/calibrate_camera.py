#!/usr/bin/env python3
"""Measure the interior camera from recorded drives, for distances on the road.

    python tools/calibrate_camera.py data/map/installed data/runs/traffic-01 \\
        data/runs/traffic-02 --out configs/camera.json

Needs recordings made with --images, and the compiled map they were driven on.
Four measurements, all from the recordings (core.perception.camera explains
the geometry):

- **horizon and straight-ahead column**: where the lines of the truck's lane
  meet, on stretches that are straight for a second either side.
- **eye height**: how fast those lines spread apart per row, against the lane
  width the map gives at that spot.
- **a first focal length**, from turns: how far the scenery above the horizon
  moved sideways against the yaw the telemetry reports. Biased, because the
  game turns the camera into bends by itself; only a starting point.
- **the distance scale** (focal x height), from straight driving: how far
  the road moved seen from above, against how far the truck went. The
  focal length written out is this scale over the height.
- **where the eye sits across the cab**: the lane's lines put the lane's
  middle `(b_left + b_right) / 2 * height` metres to the camera's right, and
  the map says how far the truck is from the lane's middle.

Each is the median over many frames, and the spread between recordings is
printed: the camera moves when the player looks around, and that shows up
here as recordings that disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.map import RoadNetwork  # noqa: E402
from core.perception.camera import (  # noqa: E402
    GroundCamera,
    bird_eye_maps,
    focal_from_yaw,
    fov_deg,
    vanishing_point,
)
from core.perception.lanes import lane_lines  # noqa: E402
from core.recording import open_recording  # noqa: E402

SCENERY = (290, 430, 700, 1550)
"""Rows and columns above the horizon, below the cab roof, clear of pillars
and mirror insets: far away, so a yaw moves it and a few metres of travel
do not."""

DASHBOARD_ROW = 680
"""Below this the view is the dashboard, which does not move with the road."""

MIN_YAW_DEG = 0.25
STRAIGHT_DEG_PER_S = 0.5
MIN_SPEED = 8.0


def headings(recording) -> np.ndarray:
    """Truck heading at each image's time, radians, unwrapped."""
    telemetry = recording.telemetry
    turns = np.asarray(telemetry["block"]["truck"]["placement"]["orientation"]["heading"], float)
    return np.interp(recording.frames["t"], telemetry["t"], np.unwrap(turns * 2 * np.pi))


def telemetry_at(recording, field_path: list[str]) -> np.ndarray:
    values = recording.telemetry["block"]
    for name in field_path:
        values = values[name]
    return np.interp(recording.frames["t"], recording.telemetry["t"], np.asarray(values, float))


def scenery(image: np.ndarray) -> np.ndarray:
    top, bottom, left, right = SCENERY
    return cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.float32)


def lane_width(network: RoadNetwork, x: float, z: float) -> tuple[float, float] | None:
    """The lane's width there, and how far left of its middle the truck is."""
    near = network.nearest(x, z)
    if near is None:
        return None
    lane = near[0]
    own = network.project(lane, x, z).lateral
    gaps = [abs(network.project(int(o), x, z).lateral - own) for o in network.lanes_alongside(lane)]
    return (float(np.median(gaps)), own) if gaps else None


def measure(path: Path, network: RoadNetwork, every: int) -> dict:
    recording = open_recording(path)
    heading = headings(recording)
    speed = telemetry_at(recording, ["truck", "speed"])
    x = telemetry_at(recording, ["truck", "placement", "position", "x"])
    z = telemetry_at(recording, ["truck", "placement", "position", "z"])
    t = recording.frames["t"]

    shifts, yaws = [], []
    window = cv2.createHanningWindow(scenery(recording.image(0)).shape[::-1], cv2.CV_32F)
    previous = None
    for i in range(len(t)):
        band = scenery(recording.image(i))
        if previous is not None:
            yaw = heading[i] - heading[i - 1]
            if abs(yaw) > np.radians(MIN_YAW_DEG) and band.std() > 15:
                (dx, _), response = cv2.phaseCorrelate(previous, band, window)
                if response > 0.2:
                    shifts.append(dx)
                    yaws.append(yaw)
        previous = band

    points, widenings, widths, middles, truck_left = [], [], [], [], []
    for i in range(0, len(t), every):
        span = (t > t[i] - 1.0) & (t < t[i] + 1.0)
        rate = np.degrees(np.ptp(heading[span])) / 2.0
        if rate > STRAIGHT_DEG_PER_S or abs(speed[i]) < MIN_SPEED:
            continue
        left, right = lane_lines(recording.image(i))
        lane = lane_width(network, x[i], z[i])
        if left is None or right is None or lane is None:
            continue
        points.append(vanishing_point([left, right]))
        widenings.append(right.b - left.b)
        widths.append(lane[0])
        middles.append((left.b + right.b) / 2.0)
        truck_left.append(lane[1])

    focal, used = focal_from_yaw(shifts, yaws) if len(shifts) > 10 else (float("nan"), 0)
    column, row = np.median(np.array(points), axis=0) if points else (np.nan, np.nan)
    return {
        "recording": path.name,
        "focal_px": focal,
        "yaw_pairs": used,
        "horizon_row": float(row),
        "centre_col": float(column),
        "straight_frames": len(points),
        "horizon_spread_px": float(np.subtract(*np.percentile([p[1] for p in points], [75, 25])))
        if points
        else float("nan"),
        "widening": float(np.median(widenings)) if widenings else float("nan"),
        "lane_width_m": float(np.median(widths)) if widths else float("nan"),
        # Per frame: the lane's middle, in eye heights to the camera's right,
        # and the truck's distance left of that middle, in metres.
        "middles": middles,
        "truck_left": truck_left,
        "shifts": shifts,
        "yaws": yaws,
    }


def road_motion(path: Path, camera: GroundCamera) -> list[float]:
    """For neighbouring images on straight road: how far the road moved seen
    from above through `camera`, over how far the truck actually went.

    1.0 means the camera's scale is right; 1.1, that it puts things 10 %
    too far away.
    """
    recording = open_recording(path)
    heading = headings(recording)
    speed = telemetry_at(recording, ["truck", "speed"])
    t = recording.frames["t"]
    nearest = camera.distance(DASHBOARD_ROW) + 0.5
    step = 0.05
    maps = bird_eye_maps(camera, (nearest, nearest + 32.0), (-3.0, 3.0), step)
    window = None
    ratios, previous = [], None
    for i in range(len(t)):
        ends = min(i + 5, len(t) - 1), max(i - 5, 0)
        straight = abs(heading[ends[0]] - heading[ends[1]]) < np.radians(0.5)
        if not straight or speed[i] < 10.0:
            previous = None
            continue
        grey = cv2.cvtColor(recording.image(i), cv2.COLOR_BGR2GRAY).astype(np.float32)
        above = cv2.remap(grey, *maps, cv2.INTER_LINEAR)
        above = cv2.morphologyEx(above, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
        if window is None:
            window = cv2.createHanningWindow(above.shape[::-1], cv2.CV_32F)
        if previous is not None and above.std() > 3:
            (_, shift), response = cv2.phaseCorrelate(previous, above, window)
            travelled = 0.5 * (speed[i] + speed[i - 1]) * (t[i] - t[i - 1])
            if response > 0.15:
                ratios.append(-shift * step / travelled)
        previous = above
    return ratios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path, help="compiled network, from tools/build_map.py")
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="write the camera here as JSON")
    parser.add_argument(
        "--every", type=int, default=3, help="look for lane lines in every Nth image"
    )
    args = parser.parse_args()

    network = RoadNetwork.open(args.cache)
    results = [measure(path, network, args.every) for path in args.recordings]
    for r in results:
        height = r["lane_width_m"] / r["widening"]
        print(
            f"{r['recording']}: focal {r['focal_px']:.0f} px ({r['yaw_pairs']} yaw pairs), "
            f"horizon row {r['horizon_row']:.1f} (IQR {r['horizon_spread_px']:.1f}), "
            f"centre {r['centre_col']:.1f}, height {height:.2f} m "
            f"({r['straight_frames']} straight frames, lane {r['lane_width_m']:.2f} m)"
        )

    yaw_focal, used = focal_from_yaw(
        [s for r in results for s in r["shifts"]], [y for r in results for y in r["yaws"]]
    )
    horizon = float(np.nanmedian([r["horizon_row"] for r in results]))
    centre = float(np.nanmedian([r["centre_col"] for r in results]))
    height = float(
        np.nanmedian([r["lane_width_m"] / r["widening"] for r in results if r["straight_frames"]])
    )
    print(
        f"\nfrom turns: focal {yaw_focal:.0f} px ({used} yaw pairs); horizon row "
        f"{horizon:.1f}, centre {centre:.1f}, height {height:.2f} m"
    )

    # The scale, from the road moving. Pairs where the view locked onto
    # something that did not move come out near zero; keep those near the
    # current estimate and refine twice.
    camera = GroundCamera(yaw_focal, horizon, centre, height)
    ratios = [q for path in args.recordings for q in road_motion(path, camera)]
    scale = 1.0
    for _ in range(2):
        near = [q for q in ratios if abs(q / scale - 1.0) < 0.25]
        scale = float(np.median(near))
    focal = yaw_focal / scale
    print(
        f"from the road moving: scale {scale:.3f} over {len(near)} of {len(ratios)} pairs "
        f"(IQR {np.percentile(near, 25):.3f}-{np.percentile(near, 75):.3f})"
    )
    camera = GroundCamera(focal, horizon, centre, height)
    # The lane's middle is m * height to the camera's right and truck_left
    # to the truck's right, so the camera sits truck_left - m * height to the
    # right of the truck's centreline.
    seat = float(
        np.median(
            [
                left - middle * height
                for r in results
                for middle, left in zip(r["middles"], r["truck_left"], strict=True)
            ]
        )
    )
    print(f"eye {abs(seat):.2f} m {'left' if seat < 0 else 'right'} of the truck's centreline")
    print(
        f"\ncamera: focal {focal:.0f} px (horizontal FOV {fov_deg(focal, 1920):.1f} deg), "
        f"horizon row {horizon:.1f}, centre {centre:.1f}, height {height:.2f} m, "
        f"50 m ahead is row {horizon + focal * height / 50:.0f}"
    )
    if args.out:
        camera = GroundCamera(focal, horizon, centre, height, seat)
        document = {
            **camera.to_json(),
            "frame_size": [1920, 1080],
            "measured_from": [r["recording"] for r in results],
            "yaw_focal_px": round(yaw_focal, 1),
            "road_motion_pairs": len(near),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
