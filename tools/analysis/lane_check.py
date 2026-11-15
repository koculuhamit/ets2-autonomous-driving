#!/usr/bin/env python3
"""Where does the map's lane part from the painted one? Checked on drives.

    python tools/analysis/lane_check.py data/follow/job8-11

tools/follow.py logs, for each view of the road, the painted lines of the
truck's lane and the middle of the path it drives, all in metres right of the
driver's eye `LANE_CHECK_M` ahead (lanes.npy). Where both lines are seen, the
painted lane's middle less the path's is how far the map's lane is off the
paint. Stretches where that stays over `--over` metres are listed with where
they are, worst first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.recording.format import TELEMETRY_RECORD_DTYPE  # noqa: E402

LANE_WIDTHS_M = (2.5, 5.0)
"""A pair of lines this far apart is a lane; wider or narrower, a misread."""


def offsets(lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(time, painted middle less the path's middle) where a lane was seen."""
    t, left, right, path = lanes.T
    width = right - left
    seen = np.isfinite(width) & (width > LANE_WIDTHS_M[0]) & (width < LANE_WIDTHS_M[1])
    seen &= np.isfinite(path)
    return t[seen], ((left + right) / 2.0 - path)[seen]


def stretches(t: np.ndarray, off: np.ndarray, over: float, least_s: float = 2.0):
    """Runs of |off| > `over` lasting `least_s`, smoothed over a second: (start, end, median)."""
    if len(t) == 0:
        return []
    order = np.argsort(t)
    t, off = t[order], off[order]
    smooth = np.array([np.median(off[(t >= x - 0.5) & (t <= x + 0.5)]) for x in t])
    bad = np.abs(smooth) > over
    found, start = [], None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        if (not b or i == len(bad) - 1) and start is not None:
            end = i if b else i - 1
            if t[end] - t[start] >= least_s:
                found.append((t[start], t[end], float(np.median(smooth[start : end + 1]))))
            start = None
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drive", type=Path)
    parser.add_argument("--over", type=float, default=0.6, help="metres off to list")
    args = parser.parse_args()

    lanes = np.load(args.drive / "lanes.npy")
    records = np.memmap(args.drive / "telemetry.bin", TELEMETRY_RECORD_DTYPE, "r")
    times = np.asarray(records["t"]) - float(records["t"][0])
    t, off = offsets(lanes)
    print(
        f"{args.drive}: lane seen in {len(t)} of {len(lanes)} views; painted middle less "
        f"path's: median {np.median(off):+.2f} m, 90% within {np.percentile(np.abs(off), 90):.2f} m"
    )
    found = sorted(stretches(t, off, args.over), key=lambda s: -abs(s[2]))
    for start, end, median in found:
        i = int(np.searchsorted(times, start))
        position = records[min(i, len(records) - 1)]["block"]["truck"]["placement"]["position"]
        print(
            f"  t={start:7.1f}-{end:7.1f} s  {median:+.2f} m  "
            f"at ({float(position['x']):.0f}, {float(position['z']):.0f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
