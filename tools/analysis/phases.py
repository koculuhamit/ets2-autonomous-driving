#!/usr/bin/env python3
"""Sum the drives `tools/report.py --json` measured, by period of development.

    python tools/report.py <sessions...> --json report.json
    python tools/analysis/phases.py report.json

Prints a Markdown table: for each period, the drives and the game's
kilometres in it, the deliveries driven to the end of the route, hits per
1000 km, hard brakes per hour, and the median of each drive's hardest
deceleration. Then the same for each day. docs/results/README.md was made
with it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PHASES: Final = (
    ("Sep 30 - Oct 9: development", "20260930", "20261009"),
    ("Nov 14 - 16: release 2.0.0", "20261114", "20261116"),
    ("Nov 18 - 20: towards 2.1.0", "20261118", "20261120"),
)
"""Inclusive by the session's date, which starts its name: `20261120-134316`."""

LONG_DRIVE_S: Final = 240.0
"""A drive's hardest deceleration counts only from drives this long: a test
start of a minute says nothing about how the truck is driven."""


@dataclass(frozen=True, slots=True)
class Period:
    name: str
    drives: int
    distance_km: float
    hours: float
    deliveries: int
    hits: int
    hard_brakes: int

    @property
    def hits_per_1000km(self) -> float:
        return self.hits / self.distance_km * 1000.0 if self.distance_km else math.nan

    @property
    def hard_brakes_per_h(self) -> float:
        return self.hard_brakes / self.hours if self.hours else math.nan


def period(name: str, drives: dict[str, dict], first: str, last: str) -> tuple[Period, float]:
    """The drives whose session is dated from `first` to `last` (YYYYMMDD),
    summed; and the median of the long ones' hardest deceleration, m/s^2."""
    chosen = [m for key, m in drives.items() if first <= key[:8] <= last]
    hardest = [
        m["max_deceleration_mps2"]
        for m in chosen
        if m["duration_s"] >= LONG_DRIVE_S and math.isfinite(m["max_deceleration_mps2"])
    ]
    summed = Period(
        name=name,
        drives=len(chosen),
        distance_km=sum(m["distance_km"] for m in chosen),
        hours=sum(m["duration_s"] for m in chosen) / 3600.0,
        deliveries=sum(m["ended"] == "end_of_route" for m in chosen),
        hits=sum(m["impacts"] for m in chosen),
        hard_brakes=sum(m["hard_brakes"] for m in chosen),
    )
    return summed, statistics.median(hardest) if hardest else math.nan


def table(rows: list[tuple[Period, float]]) -> str:
    lines = [
        "| | drives | km | hours | deliveries | hits / 1000 km | hard brakes / h "
        "| hardest deceleration, median |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, hardest in rows:
        lines.append(
            f"| {p.name} | {p.drives} | {p.distance_km:.0f} | {p.hours:.1f} | {p.deliveries} "
            f"| {p.hits_per_1000km:.1f} | {p.hard_brakes_per_h:.1f} | {hardest:.1f} m/s² |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="the JSON tools/report.py --json wrote")
    args = parser.parse_args(argv)
    drives = json.loads(args.report.read_text(encoding="utf-8"))["drives"]
    print(table([period(name, drives, first, last) for name, first, last in PHASES]))
    print()
    days = sorted({key[:8] for key in drives})
    print(table([period(day, drives, day, day) for day in days]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
