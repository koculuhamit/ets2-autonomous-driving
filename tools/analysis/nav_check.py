#!/usr/bin/env python3
"""Does the route planned from the map go the game's way? Checked on drives.

    python tools/analysis/nav_check.py data/follow/job5-06 data/follow/job5-07
    python tools/analysis/nav_check.py data/follow/job*

For each recorded drive, a minute in: the route `core.planning.game_route`
plans from where the truck was, to the job's depot, by the model that fits
the game's navigation distance best; which model that was, and how the
route's length compares with the game's. Then, every `--every` seconds on,
how far the game's distance and the route's still differ: a difference that
stays is the destination's yard, one that changes as the truck drives is a
junction where the route and the game part.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.convert import course_rad  # noqa: E402
from core.map import Places, RoadNetwork  # noqa: E402
from core.planning.game_route import GameRouter  # noqa: E402
from core.planning.nav_field import MODELS  # noqa: E402
from core.planning.route import RouteError  # noqa: E402
from core.recording.format import TELEMETRY_RECORD_DTYPE  # noqa: E402


def _text(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drives", nargs="+", type=Path, help="recorded drives (data/follow/...)")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "map" / "installed")
    parser.add_argument("--every", type=float, default=120.0, help="seconds between checks")
    args = parser.parse_args()

    network = RoadNetwork.open(args.cache)
    places = Places.open(args.cache)
    for drive in args.drives:
        records = np.memmap(drive / "telemetry.bin", TELEMETRY_RECORD_DTYPE, "r")
        if len(records) < 600:
            continue
        blocks, times = records["block"], np.asarray(records["t"])
        first = int(np.searchsorted(times, 60.0))
        job = blocks["job"][first]
        depot = places.company(_text(job["destination_company"]), _text(job["destination_city"]))
        source = places.company(_text(job["source_company"]), _text(job["source_city"]))
        if depot is None:
            print(f"{drive}: the depot is not on this map")
            continue

        def pose(i: int, blocks=blocks) -> tuple[float, float, float]:
            position = blocks["truck"]["placement"]["position"][i]
            return float(position["x"]), float(position["z"]), course_rad(blocks[i])

        game = float(blocks["navigation"]["distance"][first])
        try:
            router = GameRouter.plan(
                network,
                depot,
                pose(first),
                origin=source.position if source is not None else None,
                nav_distance=game,
            )
        except RouteError as error:
            print(f"{drive}: no route: {error}")
            continue
        model = MODELS[[id(f) for f in router._fields].index(id(router.field))]
        ours = router.nav_distance(pose(first))
        print(
            f"{drive}: to {depot.company} in {depot.city}, local roads x{model.local_factor:g}: "
            f"game {game / 1000:.1f} km, route {ours / 1000:.1f} km ({(ours - game) / 1000:+.1f})"
        )
        for i in range(first, len(records), 6):
            t = float(times[i])
            router.update(pose(i), now=t, nav_distance=float(blocks["navigation"]["distance"][i]))
            if t - float(times[first]) >= args.every:
                first = i
                game = float(blocks["navigation"]["distance"][i])
                ours = router.nav_distance(pose(i))
                print(f"    t={t:6.0f} s: {(ours - game) / 1000:+.1f} km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
