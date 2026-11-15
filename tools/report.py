#!/usr/bin/env python3
"""Measure recorded drives: per drive, and a session's together.

    python tools/report.py data/missions/20261004-205330
    python tools/report.py data/missions/*            # every session
    python tools/report.py data/missions/* --json report.json --markdown report.md

A session is a folder of drives as tools/mission.py writes them -- each a
recording (`drive-NN/`) beside the route it followed (`drive-NN.npz`), with
the ACC's log and how it ended (`outcome.json`). The numbers are
core.evaluation's: lateral distance from the lane's middle, hard brakes,
jerk, time to collision, steering reversals, hits, damage; and over the
sessions the interventions per 100 km. Drives too short to say anything
(`MIN_DURATION_S`) are left out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation import DriveMetrics, drive_metrics, session_summary  # noqa: E402
from core.recording.format import RecordingFormatError  # noqa: E402
from core.recording.reader import open_recording  # noqa: E402

MIN_DURATION_S = 5.0
SPACING_M = 2.0


def drives_in(session: Path) -> list[Path]:
    """The recorded drives of a session folder, in order; or the folder
    itself, when it is a drive."""
    if (session / "manifest.json").is_file():
        return [session]
    return sorted(p for p in session.glob("drive-*") if (p / "manifest.json").is_file())


def measure(drive: Path) -> DriveMetrics | None:
    try:
        recording = open_recording(drive)
    except (RecordingFormatError, OSError, ValueError) as exc:
        print(f"skipping {drive}: {exc}", file=sys.stderr)
        return None
    if len(recording.telemetry) < 2 or recording.duration < MIN_DURATION_S:
        return None
    acc = np.load(drive / "acc.npy") if (drive / "acc.npy").is_file() else None
    health = np.load(drive / "health.npy") if (drive / "health.npy").is_file() else None
    ended = detail = ""
    if (drive / "outcome.json").is_file():
        outcome = json.loads((drive / "outcome.json").read_text(encoding="utf-8"))
        ended, detail = outcome.get("reason", ""), outcome.get("detail", "")
    return drive_metrics(
        recording.telemetry["t"],
        recording.telemetry["block"],
        path=_route(drive),
        acc=acc,
        ended=ended,
        health=health,
        detail=detail,
    )


def _route(drive: Path):
    """The route the drive followed, as the drive steered by it; None for none."""
    route = drive.with_suffix(".npz")
    if not route.is_file():
        return None
    from tools.follow import load_path

    try:
        return load_path(route, SPACING_M)
    except (OSError, ValueError, KeyError) as exc:
        print(f"{drive.name}: no route ({exc})", file=sys.stderr)
        return None


def _cell(value) -> str:
    if isinstance(value, float):
        return "-" if math.isinf(value) or math.isnan(value) else f"{value:.2f}"
    return str(value)


def markdown(rows: list[tuple[str, DriveMetrics]], summary) -> str:
    names = list(DriveMetrics.__dataclass_fields__)
    lines = ["| drive | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for name, metrics in rows:
        values = metrics.as_dict()
        lines.append(f"| {name} | " + " | ".join(_cell(values[n]) for n in names) + " |")
    lines.append("")
    lines += [f"- **{key}**: {_cell(value)}" for key, value in summary.as_dict().items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, help="write every number here")
    parser.add_argument("--markdown", type=Path, help="write the tables here")
    args = parser.parse_args()

    rows: list[tuple[str, DriveMetrics]] = []
    for session in args.sessions:
        for drive in drives_in(session):
            metrics = measure(drive)
            if metrics is None:
                continue
            name = f"{session.name}/{drive.name}" if drive != session else drive.name
            rows.append((name, metrics))
            print(
                f"{name}: {metrics.distance_km:.1f} km {metrics.duration_s / 60:.0f} min "
                f"lateral max {_cell(metrics.lateral_max_m)} m, "
                f"out of lane {metrics.out_of_lane_s:.0f} s, "
                f"off the route {metrics.off_route_s:.0f} s, "
                f"hard brakes {metrics.hard_brakes}, jerk {metrics.max_jerk_mps3:.1f}, "
                f"TTC {_cell(metrics.min_ttc_s)} s, hits {metrics.impacts}, "
                f"damage {metrics.damage_pct:.1f} %, memory {_cell(metrics.memory_mb_per_h)} MB/h, "
                f"ended {metrics.ended or '?'}"
            )
    if not rows:
        print("no drives found", file=sys.stderr)
        return 1
    summary = session_summary(m for _, m in rows)
    print(
        f"\n{summary.drives} drives, {summary.distance_km:.0f} km, {summary.duration_h:.1f} h: "
        f"{summary.interventions} interventions, "
        f"{_cell(summary.interventions_per_100km)} per 100 km; {summary.impacts} hits, "
        f"{summary.hard_brakes} hard brakes, {summary.damage_pct:.1f} % damage"
    )
    if args.json:
        data = {
            "drives": {name: m.as_dict() for name, m in rows},
            "summary": summary.as_dict(),
        }
        args.json.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(rows, summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
