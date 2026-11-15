#!/usr/bin/env python3
"""Teach the planner a driver's way through a border's customs.

    python tools/teach_crossing.py data/runs/border-it-ch-01 data/routes/job3-08.npz

At some borders the map's lanes are not the way through: at the
Italian-Swiss border the only mapped way round went through the customs
parking into an inspection hall, and the lanes between the booths are not
in the map at all. Drive through by hand while tools/record.py records,
then give this the recording and a route that crosses there: each customs
stretch of the route (`Route.quiet_stretches`) the recording passes
through is saved, as driven, to configs/crossings.json, and from then on
every route through that stretch takes the driven line instead
(`Route.taught_crossings`).

    python tools/teach_crossing.py data/runs/bern-roundabout-01 --passage 62 80

Anywhere else the map's lanes lead over what is not road -- out of a
roundabout in Bern, over an island -- `--passage FROM TO` saves the line
driven between those seconds of the recording as a passage: every route
that passes its two ends, going its way, takes it there.
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

from core.planning.route import CROSSINGS_FILE, PASSAGE  # noqa: E402
from core.recording import open_recording  # noqa: E402
from tools.follow import saved_route  # noqa: E402

#: The recording must pass this close to a stretch's ends to have driven it, m.
PASSED_WITHIN_M = 10.0

#: The driven line is kept every this many metres, lightly smoothed.
SPACING_M = 2.0
SMOOTHING = 5


def driven_line(recording: Path, between: tuple[float, float] | None = None) -> np.ndarray:
    """The truck's positions through a recording, `(n, 3)`, every `SPACING_M`;
    with `between`, only from and to those seconds of it."""
    telemetry = open_recording(recording).telemetry
    position = telemetry["block"]["truck"]["placement"]["position"]
    points = np.column_stack([np.asarray(position[axis], float) for axis in ("x", "y", "z")])
    if between is not None:
        seconds = np.asarray(telemetry["t"], float) - float(telemetry["t"][0])
        points = points[(seconds >= between[0]) & (seconds <= between[1])]
    step = np.hypot(*np.diff(points[:, [0, 2]], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(step)])
    keep = np.searchsorted(s, np.arange(0.0, s[-1], SPACING_M))
    line = points[np.unique(keep)]
    if len(line) > SMOOTHING:
        kernel = np.ones(SMOOTHING) / SMOOTHING
        for axis in (0, 2):
            padded = np.pad(line[:, axis], SMOOTHING, mode="edge")
            line[:, axis] = np.convolve(padded, kernel, mode="same")[SMOOTHING:-SMOOTHING]
    return line


def crossings_through(line: np.ndarray, route) -> list[dict]:
    """The customs stretches of `route` that `line` drives through, each with
    the part of `line` from its entry to its exit."""
    taught = []
    for first, last in route.quiet_stretches():
        entry = np.asarray(route.network.lane_points(route.lanes[first])[0], float)[[0, 2]]
        leave = np.asarray(route.network.lane_points(route.lanes[last])[-1], float)[[0, 2]]
        to_entry = np.hypot(*(line[:, [0, 2]] - entry).T)
        i0 = int(np.argmin(to_entry))
        to_exit = np.hypot(*(line[i0:, [0, 2]] - leave).T)
        i1 = i0 + int(np.argmin(to_exit))
        if to_entry[i0] > PASSED_WITHIN_M or to_exit[i1 - i0] > PASSED_WITHIN_M or i1 <= i0:
            continue
        taught.append(
            {
                "entry": entry.round(2).tolist(),
                "exit": leave.round(2).tolist(),
                "points": line[i0 : i1 + 1].round(2).tolist(),
            }
        )
    return taught


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path, help="a manual drive through the border")
    parser.add_argument("route", type=Path, nargs="?", help="a route from plan_route.py through it")
    parser.add_argument(
        "--passage",
        type=float,
        nargs=2,
        metavar=("FROM", "TO"),
        help="save the line driven between these seconds of the recording as a passage",
    )
    parser.add_argument("--output", type=Path, default=CROSSINGS_FILE)
    args = parser.parse_args()

    if args.passage is not None:
        line = driven_line(args.recording, tuple(args.passage))
        if len(line) < 5:
            print("the truck hardly moved between those seconds", file=sys.stderr)
            return 1
        return _save(
            [
                {
                    "kind": PASSAGE,
                    "entry": line[0, [0, 2]].round(2).tolist(),
                    "exit": line[-1, [0, 2]].round(2).tolist(),
                    "points": line.round(2).tolist(),
                }
            ],
            args.output,
            args.recording,
        )
    if args.route is None:
        parser.error("give a route, or --passage FROM TO")
    route = saved_route(args.route)
    if route is None:
        print(f"{args.route} is not a planned route", file=sys.stderr)
        return 2
    taught = crossings_through(driven_line(args.recording), route)
    if not taught:
        print("the recording drives through none of the route's customs stretches")
        return 1
    return _save(taught, args.output, args.recording)


def _save(taught: list[dict], output: Path, recording: Path) -> int:
    """Add `taught` to the crossings file, each replacing one taught before
    from the same place."""
    document = {"crossings": []}
    if output.is_file():
        document = json.loads(output.read_text(encoding="utf-8"))
    for crossing in taught:
        # A crossing taught again replaces the one before.
        document["crossings"] = [
            old
            for old in document["crossings"]
            if np.hypot(*(np.subtract(old["entry"], crossing["entry"]))) > PASSED_WITHIN_M
        ] + [crossing | {"source": str(recording)}]
        print(
            f"taught: {len(crossing['points'])} points from {crossing['entry']} "
            f"to {crossing['exit']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline: in text mode Windows would write the file with CRLF.
    output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(document['crossings'])} crossings in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
