#!/usr/bin/env python3
"""Play a recorded drive to the ROS 2 bridge, as if it were being driven.

    ros2 launch ets2ad_ros ets2ad.launch.py              # in a ROS 2 shell
    uv run python tools/ros_replay.py data/missions/<mission>/drive-01 [--speed 4]

The truck and its trailer come from the recorded telemetry, the route from
the plan beside the recording (`<drive>.npz`) and from the routes it took on
the way (`routes/taken-*.npz`), the map's lanes from the map the plan names
(or --map). The recording keeps only the vehicle the truck followed and the
light's colour, not every track, so that is what is drawn of perception. A
recording made with --keep-frames shows its views too.

Nothing is sent to the game: the game need not be running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.ros2.feed import Feed, Tap  # noqa: E402
from core.perception.camera import GroundCamera  # noqa: E402
from core.perception.lights import LightState  # noqa: E402
from core.recording.format import IMAGES_DIR, image_name  # noqa: E402
from core.recording.reader import open_recording  # noqa: E402

CAMERA_FILE = REPO_ROOT / "configs" / "camera.json"


def plan_of(drive: Path) -> Path:
    return drive.parent / f"{drive.name}.npz"


def routes_taken(drive: Path, network) -> list[tuple[float, np.ndarray]]:
    """(drive time, line) of each route the drive took after its plan."""
    taken = []
    if network is None:
        return taken
    for path in sorted((drive / "routes").glob("taken-*.npz")):
        at = float(path.stem.split("-", 1)[1])
        lanes = np.load(path)["lanes"]
        if len(lanes):
            points = np.concatenate([network.lane_points(int(lane)) for lane in lanes])
            taken.append((at, points))
    return taken


def lead_at(acc: np.ndarray | None, t: float):
    """The vehicle the truck followed at drive time `t`, from acc.npy:
    (time, speed, lead id, distance, closing, ...)."""
    if acc is None or len(acc) == 0:
        return None, None
    row = acc[min(np.searchsorted(acc[:, 0], t, side="right"), len(acc)) - 1]
    if row[2] == 0 or not np.isfinite(row[3]):
        return None, None
    track = SimpleNamespace(
        id=int(row[2]),
        kind="car",
        distance=float(row[3]),
        lateral=0.0,
        width=1.9,
        closing=float(row[4]) if np.isfinite(row[4]) else 0.0,
        confirmed=True,
    )
    return track, int(row[2])


def light_at(lights: np.ndarray | None, t: float) -> LightState:
    """The filtered light at drive time `t`, from lights.npy: (time,
    distance, filtered state, seen state, ...), within half a second."""
    if lights is None or len(lights) == 0:
        return LightState.UNKNOWN
    index = np.searchsorted(lights[:, 0], t, side="right") - 1
    if index < 0 or t - lights[index, 0] > 0.5:
        return LightState.UNKNOWN
    return list(LightState)[int(lights[index, 2])]


def view_at(recording, t0: float, t: float) -> int | None:
    """The index of the last view kept at or before drive time `t`, within
    two seconds; None if there is none."""
    frames = recording.frames
    if len(frames) == 0:
        return None
    index = int(np.searchsorted(frames["t"] - t0, t, side="right")) - 1
    if index < 0 or t - (frames["t"][index] - t0) > 2.0:
        return None
    return index


def _load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drive", type=Path, help="a drive's folder, with telemetry.bin")
    parser.add_argument("--speed", type=float, default=1.0, help="times real time")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the drive")
    parser.add_argument("--map", type=Path, help="the map cache; default: the plan's")
    parser.add_argument("--loop", action="store_true", help="play again from the start")
    args = parser.parse_args()

    drive = args.drive.resolve()
    recording = open_recording(drive)
    telemetry = recording.telemetry
    if len(telemetry) == 0:
        raise SystemExit(f"{drive}: no telemetry")
    camera = GroundCamera.from_json(json.loads(CAMERA_FILE.read_text(encoding="utf-8")))

    plan = plan_of(drive)
    route = None
    map_dir = args.map
    if plan.is_file():
        with np.load(plan) as saved:
            route = np.asarray(saved["points"], dtype=float)
            if map_dir is None and "metadata" in saved.files:
                map_dir = Path(json.loads(str(saved["metadata"])).get("cache", ""))
    network = None
    if map_dir is not None and (map_dir / "manifest.json").is_file():
        from core.map import RoadNetwork

        network = RoadNetwork.open(map_dir)
    elif map_dir is not None:
        print(f"no map at {map_dir}: lanes and later routes are not drawn", file=sys.stderr)
    taken = routes_taken(drive, network)
    acc = _load(drive / "acc.npy")
    lights = _load(drive / "lights.npy")
    outcome = drive / "outcome.json"
    ended = json.loads(outcome.read_text())["reason"] if outcome.is_file() else "unknown"

    feed = Feed()
    tap = Tap(feed)
    print(f"to {feed.path}: start the bridge if it is not running", file=sys.stderr)
    times = telemetry["t"] - telemetry["t"][0]
    try:
        while True:
            index = int(np.searchsorted(times, args.start))
            began = time.perf_counter() - args.start / args.speed
            told = -np.inf
            shown = None
            while index < len(telemetry):
                t = float(times[index])
                wait = began + t / args.speed - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                block = telemetry["block"][index]
                position = block["truck"]["placement"]["position"]
                x, z = float(position["x"]), float(position["z"])
                line = route
                for at, points in taken:
                    if at <= t:
                        line = points
                if line is not None:
                    tap.route(t, line)
                tap.pose(t, block)
                tap.lanes(t, network, x, z)
                track, lead = lead_at(acc, t)
                report = SimpleNamespace(
                    tracks=(track,) if track else (),
                    light=light_at(lights, t),
                    mirror=None,
                    mirror_far=None,
                    covered=False,
                )
                tap.objects(t, report, camera, lead_id=lead)
                view = view_at(recording, float(telemetry["t"][0]), t)
                if view is not None and view != shown:
                    shown = view
                    frame = recording.frames[view]
                    jpeg = (drive / IMAGES_DIR / image_name(view)).read_bytes()
                    tap.image(t, jpeg, camera, int(frame["width"]), int(frame["height"]))
                if t - told >= 1.0:
                    told = t
                    tap.status(t, f"replay x{args.speed:g}: {t:3.0f} s of {times[-1]:.0f}")
                # Every frame at 60 Hz is more than any display wants.
                index += max(1, round(args.speed))
            tap.status(float(times[-1]), f"replay ended: {ended}")
            if not args.loop:
                break
            args.start = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.2)
        feed.close()
    print(f"{feed.sent} messages sent", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
