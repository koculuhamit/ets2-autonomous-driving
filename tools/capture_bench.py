#!/usr/bin/env python3
"""How fast the game window can be captured, and how much of that is new.

    python tools/capture_bench.py --seconds 5

Grabs the game window back to back for a while and reports:

  - how long a grab takes, and the interval between grabs;
  - how many of the grabs differ from the one before -- the game draws at its
    own rate, and a grab faster than that returns the same picture twice, so
    this is the rate at which *new* images arrive, which is the one that
    matters to perception.

Needs an X11 session with the game running and its window uncovered.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pal import GAME_WINDOW_TITLE  # noqa: E402
from pal.linux import LinuxScreenCapture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--title", default=GAME_WINDOW_TITLE)
    args = parser.parse_args()

    stamps, durations, digests = [], [], []
    with LinuxScreenCapture(args.title) as capture:
        where = capture.geometry
        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            before = time.perf_counter()
            frame = capture.grab()
            durations.append(time.perf_counter() - before)
            stamps.append(frame.timestamp)
            # A checksum of a stripe across the middle is enough to tell a new
            # picture from a repeat, and much cheaper than the whole frame.
            middle = frame.image[frame.image.shape[0] // 2 :: 37]
            digests.append(zlib.crc32(middle.tobytes()))

    grabs = len(stamps)
    elapsed = stamps[-1] - stamps[0]
    interval = np.diff(stamps) * 1000
    took = np.array(durations) * 1000
    new = 1 + sum(1 for a, b in itertools.pairwise(digests) if a != b)

    print(f"window {where.width}x{where.height} at ({where.x}, {where.y}), {args.seconds:.0f} s")
    print(f"grabs          {grabs} ({grabs / elapsed:.0f} per second)")
    print(
        f"grab takes     median {np.median(took):.1f} ms, p95 {np.percentile(took, 95):.1f} ms, "
        f"max {took.max():.1f} ms"
    )
    print(
        f"interval       median {np.median(interval):.1f} ms, p95 "
        f"{np.percentile(interval, 95):.1f} ms, max {interval.max():.1f} ms"
    )
    print(f"new pictures   {new} ({new / elapsed:.0f} per second: the game's own frame rate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
