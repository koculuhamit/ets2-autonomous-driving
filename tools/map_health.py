#!/usr/bin/env python3
"""How much of a compiled road network can reach how much of the rest.

    python tools/map_health.py data/map/installed --extraction europe.json

A route planner needs more than lanes that exist: it needs them joined up.
This takes the lane nearest a known-good position as the seed, finds every
lane reachable from it and every lane that can reach it -- following both
successors and lane changes, as a route would -- and classes each visible
road lane:

    core     reachable from the seed and back again
    enter    reachable, but no way back: a road that ends, a missed join
    leave    can reach the seed, cannot be reached from it
    island   neither

broken down by country, because the failures found so far have all been
regional. Written after doing this by hand four times.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.map import RoadNetwork  # noqa: E402

#: On the taught loop in northern Italy, which is known to be driveable.
DEFAULT_SEED = (-6500.0, 29100.0)


def reach(network: RoadNetwork, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Lanes reachable from `seed`, and lanes that can reach it."""
    n = len(network)
    forward = [
        np.concatenate([network.next_lanes(i), network.lanes_alongside(i)]) for i in range(n)
    ]
    backward: list[list[int]] = [[] for _ in range(n)]
    for source, targets in enumerate(forward):
        for target in targets:
            backward[int(target)].append(source)

    def search(neighbours) -> np.ndarray:
        seen = np.zeros(n, dtype=bool)
        seen[seed] = True
        queue = deque([seed])
        while queue:
            lane = queue.popleft()
            for other in neighbours(lane):
                other = int(other)
                if not seen[other]:
                    seen[other] = True
                    queue.append(other)
        return seen

    return search(lambda i: forward[i]), search(lambda i: backward[i])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--extraction", type=Path, required=True, help="the JSON the cache was built from"
    )
    parser.add_argument("--seed", type=float, nargs=2, default=DEFAULT_SEED, metavar=("X", "Z"))
    parser.add_argument("--min-lanes", type=int, default=300, help="smallest country to list")
    args = parser.parse_args()

    network = RoadNetwork.open(args.cache)
    document = json.loads(args.extraction.read_text(encoding="utf-8"))
    roads = {int(r["uid"], 16): r for r in document["roads"]}
    countries = document.get("countries", {})

    hit = network.nearest(*args.seed)
    if hit is None:
        parser.error(f"no lane near {args.seed}")
    down, up = reach(network, hit[0])

    classes: dict[str, Counter] = defaultdict(Counter)
    owners = np.asarray(network.lane_owner)
    for lane in np.flatnonzero(np.asarray(network.lane_source) == 0):
        road = roads.get(int(owners[lane]))
        if road is None or road["hidden"]:
            continue
        where = countries.get(str(road["country"]), "none")
        if road["left_hand_traffic"]:
            where += " (left-hand)"
        kind = (
            "core"
            if down[lane] and up[lane]
            else "enter"
            if down[lane]
            else "leave"
            if up[lane]
            else "island"
        )
        classes[where][kind] += 1
        classes["ALL"][kind] += 1

    print(f"{args.cache}: {len(network)} lanes, seeded at {tuple(args.seed)}")
    print(f"{'':16s} {'lanes':>7s} {'core':>6s} {'enter':>6s} {'leave':>6s} {'island':>6s}")
    order = sorted(classes, key=lambda c: (c != "ALL", -sum(classes[c].values())))
    for where in order:
        counts = classes[where]
        total = sum(counts.values())
        if where != "ALL" and total < args.min_lanes:
            continue
        shares = [100 * counts[k] / total for k in ("core", "enter", "leave", "island")]
        print(f"{where:16s} {total:7d} " + " ".join(f"{s:6.1f}" for s in shares))
    return 0


if __name__ == "__main__":
    sys.exit(main())
