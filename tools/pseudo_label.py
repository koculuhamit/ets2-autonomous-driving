#!/usr/bin/env python3
"""Label recorded frames with a large, slow detector, to train a small, fast one.

    python tools/pseudo_label.py data/datasets/ets2-01 data/runs/traffic-01 data/runs/traffic-02
    python tools/pseudo_label.py data/datasets/events --snapshots data/missions

The detector that runs while driving has 20-odd milliseconds and misses
vehicles past about 100 m. The largest D-FINE takes 180 ms a frame and finds
them (docs/results/vehicle_detection.md). Offline, time is free: it labels
every recorded frame, and the small one is fine-tuned on those labels.

Each frame gets two looks from the teacher -- the whole windscreen, and the
stretch of road around the horizon at twice the magnification, where distant
vehicles are a few pixels wide -- merged into one box per object. A box the
zoomed look cut at its edge is left to the wide look, which saw all of it.

`--snapshots` labels the black box's pictures instead (`events/*/` under
each drive of a mission, at half size): taken braking hard and waiting at
lights, all over the map, they show signal heads the recordings do not.
Their "recording" is the mission session, to hold out by.

Writes `labels.json`, COCO style, with every box's score kept. The training
script picks its own threshold; the labels are not truth, only the best
guess available without labelling by hand, and are spot-checked by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.perception.detection import (  # noqa: E402
    DEFAULT_CAB_VIEW,
    CabView,
    Detection,
    Detector,
    Kind,
    Rect,
    merge,
)
from core.recording import open_recording  # noqa: E402

DEFAULT_TEACHER = REPO_ROOT / "data" / "models" / "dfine-xlarge-o2c"
DATASET_FORMAT = "ets2ad-labels-1"

HORIZON = Rect(640 / 1920, 336 / 1080, 1312 / 1920, 560 / 1080)
"""The road around the horizon, for the default cab view: 672x224 pixels of a
1080p frame, shown to a 448x1344 model at twice its size."""

EDGE_PX = 3.0


def cut_by(found: Detection, rect: Rect, width: int, height: int) -> bool:
    """Whether a box reaches the edge of the look it came from."""
    left, top, right, bottom = rect.pixels(width, height)
    x0, y0, x1, y1 = found.box
    return (
        x0 - left < EDGE_PX or y0 - top < EDGE_PX or right - x1 < EDGE_PX or bottom - y1 < EDGE_PX
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="dataset directory to write labels.json in")
    parser.add_argument("recordings", type=Path, nargs="*", help="recordings made with --images")
    parser.add_argument(
        "--snapshots", type=Path, nargs="*", default=[], help="mission folders' event pictures"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--floor", type=float, default=0.3, help="drop boxes less sure than this")
    parser.add_argument(
        "--every", type=int, default=2, help="label every Nth image; neighbours are near-copies"
    )
    args = parser.parse_args()

    wide = Detector.open(args.teacher, view=DEFAULT_CAB_VIEW, threshold=args.floor)
    zoom = Detector.open(
        args.teacher,
        view=CabView(windscreen=HORIZON, covered=DEFAULT_CAB_VIEW.covered),
        threshold=args.floor,
    )
    print(f"teacher {wide.spec.source} on {wide.provider}")

    kinds = list(Kind)
    images, annotations = [], []
    started = time.perf_counter()
    for name, file_name, image, extra in _frames(args):
        height, width = image.shape[:2]
        near = wide.detect(image)
        far = [d for d in zoom.detect(image) if not cut_by(d, HORIZON, width, height)]
        image_id = len(images)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "recording": name,
                **extra,
                "width": width,
                "height": height,
            }
        )
        for found in merge(near, far):
            x0, y0, x1, y1 = found.box
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": kinds.index(found.kind),
                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                    "area": round((x1 - x0) * (y1 - y0), 1),
                    "iscrowd": 0,
                    "score": round(found.score, 3),
                }
            )
        if len(images) % 100 == 0:
            rate = len(images) / (time.perf_counter() - started)
            print(f"\r{len(images)} images, {len(annotations)} boxes, {rate:.1f}/s", end="")

    args.output.mkdir(parents=True, exist_ok=True)
    document = {
        "info": {
            "format": DATASET_FORMAT,
            "made": datetime.now(UTC).isoformat(),
            "teacher": wide.spec.source,
            "teacher_sha256": wide.spec.sha256,
            "floor": args.floor,
            "every": args.every,
            "recordings": [str(p) for p in args.recordings],
            "snapshots": [str(p) for p in args.snapshots],
            "cab_view": "DEFAULT_CAB_VIEW",
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": str(kind)} for i, kind in enumerate(kinds)],
    }
    (args.output / "labels.json").write_text(json.dumps(document), encoding="utf-8")
    print(f"\n{len(images)} images, {len(annotations)} boxes -> {args.output / 'labels.json'}")
    return 0


def _frames(args):
    """(recording name, file name, BGR image, more about it) of each frame
    to label: every `--every`th of each recording, every event picture."""
    import cv2

    for recording_path in args.recordings:
        recording = open_recording(recording_path)
        for index in range(0, len(recording.frames), args.every):
            extra = {"frame": index, "t": float(recording.frames["t"][index])}
            file_name = str(recording_path / "images" / f"{index:06d}.jpg")
            yield recording_path.name, file_name, recording.image(index), extra
    for missions in args.snapshots:
        for picture in sorted(missions.glob("*/drive-*/events/*/*.jpg")):
            image = cv2.imread(str(picture), cv2.IMREAD_COLOR)
            if image is None:
                continue
            session = picture.relative_to(missions).parts[0]
            yield session, str(picture), image, {}


if __name__ == "__main__":
    sys.exit(main())
