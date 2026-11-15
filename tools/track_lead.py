#!/usr/bin/env python3
"""Follow the vehicle ahead through a recording, and check it makes sense.

    python tools/track_lead.py data/runs/traffic-02 --model data/models/dfine-nano-ets2

Runs the detector, the tracker and the lead-vehicle choice over every image
of a recording, as the stack would live, and prints what it followed. The
path the lead must be on is the lane ahead from the map, planned from where
the telemetry puts the truck; `--straight` uses straight ahead instead, to
compare.

It also runs adaptive cruise in the shadow (core.control.following): with
the truck's own speed from the recording, what speed following would have
asked for, and when it would have braked hard. A hard brake where the
driver did not brake is the number that matters most before letting the
stack follow anything: each one is a phantom stop on an open road. Without
ground truth for other vehicles, the check is physics: the lead's own speed
-- the truck's speed from telemetry plus the rate the gap changes -- cannot
be negative, and on a motorway is rarely zero. A lead standing still at
motorway speed is either a jam or a phantom, and is worth a look.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge import convert  # noqa: E402
from core.control.following import GapPolicy, Lead, follow  # noqa: E402
from core.map import RoadNetwork  # noqa: E402
from core.perception.camera import GroundCamera  # noqa: E402
from core.perception.detection import Detector  # noqa: E402
from core.perception.tracking import CrossingFilter, Tracker, lead_vehicle  # noqa: E402
from core.perception.tracking import Path as LanePath  # noqa: E402
from core.planning import route as routing  # noqa: E402
from core.recording import open_recording  # noqa: E402

PATH_AHEAD_M = 200.0


def lane_ahead(network: RoadNetwork, block, camera: GroundCamera) -> LanePath | None:
    """The lane ahead of the truck from the map, seen from the driver's seat."""
    position = block["truck"]["placement"]["position"]
    x, z = float(position["x"]), float(position["z"])
    course = convert.course_rad(block)
    try:
        planned = routing.plan_ahead(
            network, (x, z, course), PATH_AHEAD_M, stop_before_lights=False
        )
    except routing.RouteError:
        return None
    points = planned.points()
    dx, dz = points[:, 0] - x, points[:, 2] - z
    # Forward is (sin c, cos c) and left is (cos c, -sin c) for a course c in
    # the atan2(dx, dz) frame; see core.control.lateral._to_vehicle_frame.
    ahead = dx * np.sin(course) + dz * np.cos(course)
    right = -(dx * np.cos(course) - dz * np.sin(course))
    try:
        return LanePath.from_truck_frame(ahead, right, camera)
    except ValueError:
        return None


DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "dfine-nano-ets2"
DEFAULT_CAMERA = REPO_ROOT / "configs" / "camera.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--map", type=Path, default=REPO_ROOT / "data" / "map" / "installed")
    parser.add_argument("--straight", action="store_true", help="ignore the map: straight ahead")
    args = parser.parse_args()

    camera = GroundCamera.from_json(json.loads(args.camera.read_text(encoding="utf-8")))
    detector = Detector.open(args.model)
    tracker = Tracker(camera)
    recording = open_recording(args.recording)
    t = np.asarray(recording.frames["t"], float)
    ego = np.interp(
        t,
        recording.telemetry["t"],
        np.asarray(recording.telemetry["block"]["truck"]["speed"], float),
    )

    network = None if args.straight else RoadNetwork.open(args.map)
    blocks = recording.telemetry["block"]
    telemetry_index = np.clip(np.searchsorted(recording.telemetry["t"], t) - 1, 0, None)

    brake = np.interp(
        t,
        recording.telemetry["t"],
        np.asarray(recording.telemetry["block"]["truck"]["effective_brake"], float),
    )
    policy = GapPolicy()
    crossing_filter = CrossingFilter()
    shadow = []  # following target, urgent deceleration
    rows = []  # time, ego speed, lead id, distance, closing, lateral, width
    no_path = 0
    for i in range(len(t)):
        image = recording.image(i)
        cut_row = detector.view.windscreen.pixels(image.shape[1], image.shape[0])[3]
        tracker.update(detector.detect(image), float(t[i]), cut_row)
        path = None
        if network is not None:
            path = lane_ahead(network, blocks[telemetry_index[i]], camera)
            no_path += path is None
        crossing = crossing_filter.crossing(tracker.tracks, path, float(t[i]))
        found = lead_vehicle(tracker.tracks, path, crossing)
        decision = follow(None if found is None else Lead(found[1][0], found[1][1]), ego[i], policy)
        shadow.append(
            (
                np.nan if decision.target_speed is None else decision.target_speed,
                decision.urgent_deceleration or 0.0,
            )
        )
        if found is None:
            rows.append((t[i], ego[i], 0, np.nan, np.nan, np.nan, np.nan))
        else:
            track, (distance, closing, lateral, width) = found
            rows.append((t[i], ego[i], track.id, distance, closing, lateral, width))
    a = np.array(rows)
    has = a[:, 2] > 0
    own = a[:, 1] + a[:, 4]  # the lead's own speed
    switches = int(np.sum(np.diff(a[:, 2]) != 0))
    moving = has & (a[:, 1] > 15.0)
    print(f"{len(a)} images, a lead in {has.sum()} ({has.mean():.0%}), {switches} changes of lead")
    if network is not None:
        print(f"no lane ahead from the map in {no_path} images: straight ahead there")
    if has.any():
        print(
            f"lead distance: median {np.nanmedian(a[has, 3]):.0f} m, "
            f"5-95% {np.nanpercentile(a[has, 3], 5):.0f}-{np.nanpercentile(a[has, 3], 95):.0f} m"
        )
    if moving.any():
        print(
            f"above 54 km/h, the lead's own speed: median {np.median(own[moving]) * 3.6:.0f} km/h; "
            f"below 0 in {np.sum(own[moving] < -2.0)} images, "
            f"below 10 km/h in {np.sum(own[moving] < 2.8)} of {moving.sum()}"
        )
        for i in np.flatnonzero(moving & (own < 2.8))[::10][:15]:
            print(
                f"  image {i}: lead {int(a[i, 2])} at {a[i, 3]:.0f} m, "
                f"own speed {own[i] * 3.6:.0f} km/h, width {a[i, 6]:.1f} m, lateral {a[i, 5]:.1f} m"
            )

    target, urgent = np.array(shadow).T
    hard = urgent > 0
    events = np.flatnonzero(hard & ~np.concatenate([[False], hard[:-1]]))
    print(f"\nadaptive cruise in the shadow, time gap {policy.time_gap_s:.1f} s:")
    capped = np.isfinite(target)
    if capped.any():
        slower = capped & (target < ego - 1.0)
        print(
            f"  would have slowed below the driver's speed in {slower.sum()} images "
            f"({slower.mean():.0%}), by a median {np.median((ego - target)[slower]) * 3.6:.0f} km/h"
            if slower.any()
            else "  never wanted to go slower than the driver"
        )
    print(f"  hard braking: {len(events)} times, {hard.sum()} images")
    for e in events[:15]:
        # Did the driver brake within a second either side?
        near = (t > t[e] - 1.0) & (t < t[e] + 1.0)
        driver = "the driver braked too" if brake[near].max() > 0.05 else "THE DRIVER DID NOT"
        print(
            f"    image {e}: {a[e, 3]:.0f} m, closing {-a[e, 4] * 3.6:.0f} km/h, "
            f"asks {urgent[e]:.1f} m/s2 -- {driver}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
