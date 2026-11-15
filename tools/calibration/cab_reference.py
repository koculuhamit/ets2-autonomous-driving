#!/usr/bin/env python3
"""Record what the dashboard looks like from the cab, for core.perception.cab.

    python tools/calibration/cab_reference.py data/missions/*/drive-*/events/* data/runs/*/images

Takes every JPEG frame under the folders given -- black boxes, recorded
images -- all of them from the interior camera, and saves the mean of their
dashboards' edges to configs/cab_dashboard.npy. Run again for another truck.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.perception.cab import CAB_REFERENCE_FILE, dashboard_edges  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=CAB_REFERENCE_FILE)
    parser.add_argument("--every", type=int, default=4, help="use every Nth frame of a folder")
    args = parser.parse_args()
    edges = []
    for folder in args.folders:
        for frame in sorted(folder.glob("*.jpg"))[:: args.every]:
            image = cv2.imread(str(frame))
            if image is not None:
                edges.append(dashboard_edges(image))
    if not edges:
        print("no frames found", file=sys.stderr)
        return 1
    mean = np.mean(edges, axis=0)
    reference = ((mean - mean.mean()) / mean.std()).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, reference)
    print(f"{len(edges)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
