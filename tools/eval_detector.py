#!/usr/bin/env python3
"""Score detectors against labels, on recordings they were not trained on.

    python tools/eval_detector.py data/datasets/ets2-01/labels.json \\
        --recordings traffic-02 --models data/models/dfine-nano data/models/dfine-nano-ets2

Runs each model on the labelled frames of the named recordings and reports
precision, recall, and recall by box height -- the stand-in for distance.
Vehicles, matched regardless of kind -- or, with `--kind lights`, traffic
lights' heads -- reference boxes at or above `--floor`. The reference is
the teacher's labels, so this says how close a model comes to the teacher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.perception.detection import Detector, Kind  # noqa: E402
from core.perception.evaluation import Score  # noqa: E402

VEHICLES = {Kind.CAR, Kind.TRUCK, Kind.BUS, Kind.MOTORCYCLE}
GROUPS = {"vehicles": VEHICLES, "lights": {Kind.TRAFFIC_LIGHT}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument(
        "--recordings", nargs="+", required=True, help="recording names to score on"
    )
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--floor", type=float, default=0.5, help="reference boxes less sure are dropped"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="the models' own threshold")
    parser.add_argument("--kind", choices=sorted(GROUPS), default="vehicles")
    args = parser.parse_args()

    document = json.loads(args.labels.read_text(encoding="utf-8"))
    kinds = list(Kind)
    wanted = set(args.recordings)
    images = [im for im in document["images"] if im["recording"] in wanted]
    if not images:
        print(f"no labelled images from {sorted(wanted)}", file=sys.stderr)
        return 1
    wanted_kinds = GROUPS[args.kind]
    reference: dict[int, list] = {im["id"]: [] for im in images}
    for a in document["annotations"]:
        keep = a["image_id"] in reference and a["score"] >= args.floor
        if keep and kinds[a["category_id"]] in wanted_kinds:
            x, y, w, h = a["bbox"]
            reference[a["image_id"]].append((x, y, x + w, y + h))
    total = sum(len(v) for v in reference.values())
    print(f"{len(images)} images, {total} reference {args.kind}\n")

    for model in args.models:
        detector = Detector.open(model, threshold=args.threshold)
        score, took = Score(), []
        for image in images:
            frame = cv2.imread(image["file_name"], cv2.IMREAD_COLOR)
            before = time.perf_counter()
            found = detector.detect(frame)
            took.append(time.perf_counter() - before)
            score.add([d.box for d in found if d.kind in wanted_kinds], reference[image["id"]])
        print(
            f"{model.name}: precision {score.precision:.3f}, recall {score.recall:.3f}, "
            f"{np.median(took) * 1000:.1f} ms median"
        )
        for (low, high), (hit, count) in sorted(score.by_height.items()):
            span = f"{low:.0f}-{high:.0f} px" if np.isfinite(high) else f"{low:.0f}+ px"
            print(f"    {span:>12s} tall: {hit:5d}/{count:<5d} found ({hit / count:.0%})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
