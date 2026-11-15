#!/usr/bin/env python3
"""Check camera distances against a trailer whose position is known.

    python tools/distance_check.py data/runs/trailer-distance-01

The drive: record, drop the trailer, drive off, turn, and come back towards
it. The telemetry keeps the dropped trailer's position, and always has the
truck's, so every image has a true distance. The detector finds the trailer,
the camera model says how far it is, and the two are compared.

The true distance is between the truck's and the trailer's origins, along
the truck's heading. The camera sees where the trailer's nearest part meets
the ground, and sits somewhere forward of the truck's origin; neither
offset is known, so a few metres of the error at any distance may be those.
Over a range of 30-70 m, the offset and a scale error cannot be told apart
by a fit; the table gives measured over true by distance instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge import convert  # noqa: E402
from core.perception.camera import GroundCamera  # noqa: E402
from core.perception.detection import Detector, Kind  # noqa: E402
from core.recording import open_recording  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "dfine-nano-ets2"
DEFAULT_CAMERA = REPO_ROOT / "configs" / "camera.json"
BINS = (0.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 150.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument(
        "--max-width-m", type=float, default=4.0, help="wider boxes are not the trailer's end"
    )
    parser.add_argument(
        "--from-behind", action="store_true", help="also fit horizon and scale to the trailer"
    )
    args = parser.parse_args()

    camera = GroundCamera.from_json(json.loads(args.camera.read_text(encoding="utf-8")))
    detector = Detector.open(args.model, threshold=0.3)
    recording = open_recording(args.recording)
    blocks = recording.telemetry["block"]
    connected = np.asarray(blocks["trailers"]["connected"][:, 0]).astype(bool)
    positions = blocks["trailers"]["placement"]["position"][:, 0]
    dropped = np.flatnonzero(connected[:-1] & ~connected[1:])
    if len(dropped):
        at = int(dropped[0])
    elif not connected.any() and np.ptp(positions["x"]) < 0.01 and np.ptp(positions["z"]) < 0.01:
        at = 0  # dropped before the recording began, and it has not moved since
    else:
        print("no trailer standing still in this recording", file=sys.stderr)
        return 1
    tx, tz = float(positions["x"][at]), float(positions["z"][at])
    dropped_t = float(recording.telemetry["t"][at])
    print(f"trailer standing from {dropped_t:.1f} s, at ({tx:.1f}, {tz:.1f})")

    index = np.clip(np.searchsorted(recording.telemetry["t"], recording.frames["t"]) - 1, 0, None)
    rows = []  # true ahead, true across, measured distance, measured lateral, width, score
    for i, j in enumerate(index):
        if recording.frames["t"][i] < dropped_t + 1.0:
            continue
        block = blocks[j]
        position = block["truck"]["placement"]["position"]
        course = convert.course_rad(block)
        dx, dz = tx - float(position["x"]), tz - float(position["z"])
        ahead = dx * np.sin(course) + dz * np.cos(course)
        right = -(dx * np.cos(course) - dz * np.sin(course)) - camera.seat_right_m
        if ahead < 5.0 or abs(right) > 0.5 * ahead:
            continue
        # The trailer is the box on its bearing. Matching by where the camera
        # places a box instead picked the neighbouring trailers in a yard.
        bearing = camera.centre_col + camera.focal_px * right / ahead
        best = None
        for found in detector.detect(recording.image(i)):
            if found.kind not in (Kind.TRUCK, Kind.CAR, Kind.BUS):
                continue
            if not found.box[0] - 10 <= bearing <= found.box[2] + 10:
                continue
            where = camera.locate(found.box)
            if where is None or where[2] > args.max_width_m:
                continue
            if best is None or found.score > best[1]:
                best = (where, found.score, found.box[3])
        if best is not None:
            (distance, lateral, width), score, bottom = best
            rows.append((ahead, right, distance, lateral, width, score, bottom))

    if len(rows) < 10:
        print(f"only {len(rows)} images with the trailer found ahead", file=sys.stderr)
        return 1
    a = np.array(rows)
    true, measured = a[:, 0], a[:, 2]
    ratio = measured / true
    # A box on the bearing can still be something else in front of or behind
    # the trailer; those are off by a factor, not a few percent.
    sane = np.abs(ratio - np.median(ratio)) < 0.3
    print(
        f"{sane.sum()} images with the trailer found ({(~sane).sum()} set aside as another "
        f"object), true {true[sane].min():.0f}-{true[sane].max():.0f} m, "
        f"trailer width seen: median {np.median(a[sane, 4]):.2f} m\n"
    )
    print("true distance     n   measured / true   median error   p90 |error|")
    for low, high in itertools.pairwise(BINS):
        inside = sane & (true >= low) & (true < high)
        if inside.sum() < 3:
            continue
        err = measured[inside] - true[inside]
        print(
            f"{low:4.0f}-{high:<4.0f} m {inside.sum():5d}   {np.median(ratio[inside]):13.3f}"
            f"   {np.median(err):+10.1f} m  {np.percentile(np.abs(err), 90):9.1f} m"
        )

    if args.from_behind:
        fit_camera(a[sane], blocks[at], camera)
    return 0


def fit_camera(a: np.ndarray, block, camera: GroundCamera) -> None:
    """The horizon and scale the trailer says the camera has, approached from behind.

    From behind, the box's foot is where the trailer's rearmost wheels touch
    the road, and the telemetry has those wheels in the trailer's frame (+z
    is rearwards). The eye is taken to be over the truck's front axle, which
    the telemetry also has: in a cab-over truck the driver sits about there,
    give or take half a metre. Then `1 / distance = (row - horizon) / scale`,
    a straight line in the box's foot row.
    """
    trailer_wheels = block["trailers"][0]["wheels"]
    rear = float(np.max(trailer_wheels["position"]["z"][: int(trailer_wheels["count"])]))
    truck_wheels = block["truck"]["wheels"]
    front = -float(np.min(truck_wheels["position"]["z"][: int(truck_wheels["count"])]))
    true = a[:, 0] - rear - front
    rows = a[:, 6]
    slope, intercept = np.polyfit(rows, 1.0 / true, 1)
    scale, horizon = 1.0 / slope, -intercept / slope
    ratio = a[:, 2] / true
    print(
        f"\nfrom behind: rear wheels {rear:.2f} m behind the trailer's origin, eye taken "
        f"{front:.2f} m ahead of the truck's; measured / true: median {np.median(ratio):.3f}"
    )
    print(
        f"what the trailer says: horizon row {horizon:.1f} (calibrated {camera.horizon_row:.1f}), "
        f"scale {scale:.0f} px m (calibrated {camera.focal_px * camera.height_m:.0f})"
    )
    fitted = scale / (rows - horizon)
    print(f"distance error with those: RMS {np.sqrt(np.mean((fitted - true) ** 2)):.2f} m")


if __name__ == "__main__":
    sys.exit(main())
