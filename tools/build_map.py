#!/usr/bin/env python3
"""Compile a map extraction into the cache the driving stack opens.

    python tools/build_map.py extraction.json data/map/europe
    python tools/build_map.py extraction.json data/map/mine --dlc installed --game "$ETS2"
    python tools/build_map.py extraction.json data/map/it --dlc italy,france

`--dlc` says whose map this is. The extraction holds every DLC's roads,
because the game ships them all; `all` (the default) keeps every released
DLC's, for a cache someone else will use, `installed` keeps what a game
directory has, and a list keeps exactly those. See `core.map.dlc`.

The extraction comes from `tools/map_extract` (C#, see ADR-0006). This builds
the lane graph from it -- about 20 s and 2 GB for all of Europe -- and writes
the arrays the stack maps in instead, which open in well under a second.
Run it again whenever the game's map changes; nothing else needs it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.map import LaneGraph, Places, RoadNetwork, Sector, dlc  # noqa: E402
from core.map.ferries import link_ferries, save_links  # noqa: E402
from core.map.fuel import STATION_LANES_FILE, pumps_on, save_pumps, station_lanes  # noqa: E402
from core.map.nav_scale import (  # noqa: E402
    CityArea,
    class_codes,
    lane_nav_lengths,
    lane_scales,
    nav_speeds,
    road_side_class,
    spread_classes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("extraction", type=Path, help="JSON written by tools/map_extract")
    parser.add_argument("output", type=Path, help="cache directory to write")
    parser.add_argument(
        "--dlc",
        default="all",
        help="whose map: 'all' released DLCs, 'installed' in --game, or a list",
    )
    parser.add_argument("--game", type=Path, help="game directory, for --dlc installed")
    args = parser.parse_args()

    try:
        owned = dlc.parse_profile(args.dlc, args.game)
    except ValueError as error:
        parser.error(str(error))
    print(f"DLCs: {', '.join(sorted(owned)) or 'none, base game only'}")

    started = time.perf_counter()
    everything = Sector.load(args.extraction)
    sector = everything.restricted_to(owned)
    loaded = time.perf_counter()
    print(
        f"read {len(everything.roads)} roads, {len(everything.prefabs)} prefabs "
        f"in {loaded - started:.1f} s; kept {len(sector.roads)} and {len(sector.prefabs)}"
    )

    graph = LaneGraph.build(sector)
    built = time.perf_counter()
    print(f"built {len(graph.lanes)} lanes in {built - loaded:.1f} s")

    network = RoadNetwork.compile(graph)
    speeds, classes, avoid = gps_weights(network, sector)
    network = dataclasses.replace(
        network,
        lane_nav_length=nav_lengths(network, sector),
        lane_nav_speed=speeds,
        lane_speed_class=classes,
        lane_gps_avoid=avoid,
    )
    network.save(
        args.output,
        source={
            "extraction": str(args.extraction),
            "sectors": len(sector.sectors),
            "roads": len(sector.roads),
            "prefabs": len(sector.prefabs),
            "dlc_profile": args.dlc,
            "dlcs": sorted(owned),
            # The profile can only keep what the extraction read: a DLC's map
            # is partly in base_map.scs and partly in its own archive, so a
            # cache for someone else's DLCs has to be extracted where they are.
            "archives": list(everything.archives),
        },
    )
    places = Places.from_sector(sector)
    places.save(args.output)
    links = link_ferries(network, sector.ferries, sector.ferry_lines)
    save_links(args.output, links)
    print(f"{len(sector.ferries)} ferry ports, {len(links)} ferry links")
    pumps = pumps_on(network, sector.prefabs)
    save_pumps(args.output, pumps)
    kept_out = station_lanes(network, sector.prefabs)
    np.save(args.output / STATION_LANES_FILE, kept_out, allow_pickle=False)
    print(f"{len(pumps)} gas pumps to drive to; {len(kept_out)} station lanes kept out of")
    done = time.perf_counter()
    size = sum(f.stat().st_size for f in args.output.iterdir()) / 1e6
    print(
        f"wrote {len(network)} lanes, {len(network.successors)} connections, "
        f"{len(network.points)} points, {len(places.companies)} depots -> {args.output} "
        f"({size:.0f} MB) in {done - built:.1f} s"
    )
    return 0


def nav_lengths(network: RoadNetwork, sector: Sector) -> np.ndarray:
    """Each lane's length as the game's navigation counts it."""
    owners = {int(road.uid, 16): road.country for road in sector.roads}
    owners.update({int(prefab.uid, 16): prefab.country for prefab in sector.prefabs})
    cities = [
        CityArea(float(c.position[0]), float(c.position[2]), c.width, c.height)
        for c in sector.cities
    ]
    scales = lane_scales(network.lane_owner, owners, sector.countries)
    return lane_nav_lengths(network.points, network.lane_start, scales, cities)


def gps_weights(network: RoadNetwork, sector: Sector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the game's GPS weighs each lane by: the speed it reckons on, its
    speed class, and whether it is on a road the GPS routes round."""
    roads = {int(road.uid, 16): road for road in sector.roads}
    countries = {int(prefab.uid, 16): prefab.country for prefab in sector.prefabs}
    classes: list[str | None] = []
    lane_countries: list[int] = []
    avoid = np.zeros(len(network), dtype=bool)
    for lane in range(len(network)):
        owner = int(network.lane_owner[lane])
        road = roads.get(owner)
        if road is None:
            classes.append(None)
            lane_countries.append(countries.get(owner, 0))
            continue
        look = sector.looks.get(road.look)
        left = float(network.lane_offset[lane]) > 0.0
        side = road_side_class(
            road.rule_left if left else road.rule_right,
            (look.lane_types_left if left else look.lane_types_right) if look else (),
            sector.speed_classes,
        )
        other = road_side_class(
            road.rule_right if left else road.rule_left,
            (look.lane_types_right if left else look.lane_types_left) if look else (),
            sector.speed_classes,
        )
        classes.append(side or other)
        lane_countries.append(road.country)
        avoid[lane] = road.gps_avoid
    spread = spread_classes(classes, network.successor_start, network.successors)
    speeds = nav_speeds(spread, lane_countries, sector.truck_speed_limits)
    return speeds, class_codes(spread), avoid


if __name__ == "__main__":
    sys.exit(main())
