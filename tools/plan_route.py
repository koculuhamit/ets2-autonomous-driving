#!/usr/bin/env python3
"""Plan a route on the game's own road network, for tools/follow.py to drive.

    python tools/plan_route.py data/map/installed route.npz --live --to X Z
    python tools/plan_route.py data/map/installed route.npz --from X Z COURSE --to X Z
    python tools/plan_route.py data/map/installed loop.npz --along data/routes/teach-01
    python tools/plan_route.py data/map/installed ahead.npz --live --ahead 400
    python tools/plan_route.py data/map/installed job.npz --live --to-job
    python tools/plan_route.py data/map/installed job.npz --live --to-job --game
    python tools/plan_route.py data/map/installed depot.npz --live --to-company tree_et berlin

`--game` plans the route the game's GPS takes -- the quickest as the game
reckons time -- with the lanes chosen for it, 30 km of it at a time
(core.planning.game_route), and says how its length compares with the
game's own navigation distance: for follow.py --game-route.

`--live` takes the start from the running game: where the truck is and which
way it is moving. `--to-job` takes the destination from the job the truck is
on, and `--to-company` names a depot by the ids a job uses; both route to the
gate of the depot's yard. `--ahead` needs no destination: it follows the road
in front of the truck, straightest way on at junctions, and ends before the
first traffic light -- the route to drive a planned route with for the first
time. `--through` adds waypoints before `--to`. `--along` plans
through points every `--every` metres of a recorded drive, which is how a
route planned from the map is compared with one that was driven.

Then:

    python tools/follow.py route.npz --simulate
    python tools/follow.py route.npz --distance 300
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import os
import sys
from pathlib import Path

# Before numpy: on Windows OpenBLAS commits a buffer for each CPU thread as it
# is imported -- 360 MB in each of the drive's processes, where Linux commits
# nothing until it is used -- and one thread is as quick for what is done here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.bridge import convert  # noqa: E402
from core.map import Places, RoadNetwork  # noqa: E402
from core.planning import route as routing  # noqa: E402
from core.planning.destination import GPS_COMPANY, gps_places, place_record  # noqa: E402
from core.planning.game_route import GameRouter  # noqa: E402


def live_block():
    from core.bridge.reader import TelemetryReader

    with TelemetryReader() as reader:
        return reader.wait_for_frame(timeout=5.0).block


def pose_of(block) -> tuple[float, float, float]:
    position = block["truck"]["placement"]["position"]
    return float(position["x"]), float(position["z"]), convert.course_rad(block)


def job_destination(block) -> tuple[str, str] | None:
    """Where the job's GPS route goes, as (company, city): to its trailer
    first (`convert.job_target`); None with no job."""
    return convert.job_target(block)


def recorded(directory: Path, every: float) -> tuple[tuple[float, float, float], list]:
    """A start pose and waypoints every `every` metres along a recorded drive."""
    from core.localization.path import ReferencePath
    from core.recording.reader import open_recording

    recording = open_recording(directory)
    blocks = recording.telemetry["block"]
    position = blocks["truck"]["placement"]["position"]
    speed = np.asarray(blocks["truck"]["speed"], dtype=float)
    first = int(np.argmax(np.abs(speed) > 1.0))
    start = (
        float(position["x"][first]),
        float(position["z"][first]),
        convert.course_rad(blocks[first]),
    )
    path = ReferencePath(
        np.column_stack([position["x"].astype(float), position["z"].astype(float)]), spacing=1.0
    )
    stops = [*np.arange(every, path.length - every / 2, every), path.length - 5.0]
    return start, [(path.sample_at(s).x, path.sample_at(s).z) for s in stops]


GPS_ELSEWHERE: int = 3
"""The exit status when the job's place is not where the game's GPS goes:
tools/mission.py then plans to where it goes (--to-gps)."""

RED_LINE_AHEAD_M = 3000.0
"""Of the road ahead, planned for a drive that follows the red line."""

ELSEWHERE_MIN_M = 30_000.0
ELSEWHERE_SHARE = 0.5
"""The game's way to the job's place, by the map, this much longer or
shorter than the game's own distance -- and no less than `ELSEWHERE_MIN_M`
-- goes somewhere else. In Bremen, a job delivered, its places still told,
the GPS set to a trailer 7.6 km off: the job's way was 1021 km, and the
truck set off for the old job's port."""


def gps_elsewhere(theirs_m: float, ours_m: float) -> bool:
    """Whether the game's GPS, `theirs_m` to go, goes somewhere other than
    the way planned to the job's place, `ours_m` as the game counts it."""
    return abs(ours_m - theirs_m) > max(ELSEWHERE_MIN_M, ELSEWHERE_SHARE * theirs_m)


@functools.lru_cache(maxsize=1)
def open_network(cache: str) -> RoadNetwork:
    """The network of a cache, opened once in a process that plans again
    and again (tools/follow.py --standby): its depot's fields are kept for
    the network they were weighed on (core.planning.game_route.depot_fields)."""
    return RoadNetwork.open(cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path, help="compiled network, from tools/build_map.py")
    parser.add_argument("output", type=Path, help="route file to write (.npz)")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--live", action="store_true", help="start where the truck is now")
    start.add_argument("--from", dest="origin", type=float, nargs=3, metavar=("X", "Z", "COURSE"))
    start.add_argument("--along", type=Path, help="recorded drive to plan along")
    parser.add_argument("--through", type=float, nargs=2, action="append", default=[])
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--to", type=float, nargs=2, metavar=("X", "Z"))
    destination.add_argument("--to-company", nargs=2, metavar=("COMPANY", "CITY"))
    destination.add_argument(
        "--to-job", action="store_true", help="the destination of the job in the game"
    )
    destination.add_argument(
        "--to-gps",
        action="store_true",
        help="wherever the game's GPS goes, found from its distance (core.planning.destination): "
        "a job's trailer not hitched yet, or a place marked on the map; implies --game",
    )
    destination.add_argument(
        "--ahead", type=float, metavar="METRES", help="the road ahead, ending before a light"
    )
    parser.add_argument(
        "--through-lights",
        action="store_true",
        help="with --ahead, carry on through traffic lights (follow.py --acc stops for them)",
    )
    parser.add_argument(
        "--game",
        action="store_true",
        help="with --to-job or --to-company: the route the game's GPS takes, lanes chosen "
        "for it (core.planning.game_route), for follow.py --game-route",
    )
    parser.add_argument("--every", type=float, default=150.0, help="waypoint spacing for --along")
    args = parser.parse_args(argv)

    network = open_network(str(Path(args.cache).resolve()))
    block = live_block() if args.live or args.to_job or args.to_gps else None
    depot = None
    alternatives: list = []
    if args.to_gps:
        args.game, args.live = True, True
        theirs = float(block["navigation"]["distance"])
        if theirs <= 0.0:
            parser.error("the game's GPS has no route")
        found = gps_places(network, Places.open(args.cache), pose_of(block), theirs)
        if not found:
            print("nowhere on the map is as far as the game's GPS says", file=sys.stderr)
            return 1
        depot, alternatives = found[0], found[1:]
        print(
            f"the GPS's {theirs / 1000:.1f} km fit {len(found)} places: driving to "
            f"{depot.company} {depot.city}"
            + (
                f", and if not, {', '.join(f'{a.company} {a.city}' for a in alternatives[:4])}"
                if alternatives
                else ""
            )
        )
    if args.to_job or args.to_company:
        if args.to_job:
            named = job_destination(block)
            if named is None:
                parser.error("the truck has no job")
        else:
            named = tuple(args.to_company)
        depot = Places.open(args.cache).company(*named)
        if depot is None:
            print(f"{named[0]} in {named[1]} is not on this map", file=sys.stderr)
            return 1

    if args.along:
        pose, waypoints = recorded(args.along, args.every)
    else:
        if args.to is None and depot is None and args.ahead is None:
            parser.error("give a destination: --to, --to-company, --to-job or --ahead")
        pose = pose_of(block) if args.live else tuple(args.origin)
        waypoints = [*map(tuple, args.through)]
        if args.to is not None:
            waypoints.append(tuple(args.to))

    try:
        if args.ahead is not None:
            planned = routing.plan_ahead(
                network, pose, args.ahead, stop_before_lights=not args.through_lights
            )
        elif args.to_gps and depot.company == GPS_COMPANY:
            # A place marked on the map, driven to by its red line
            # (follow.py --gps-only): the road ahead to begin with. The way
            # planned to a point guessed from the distance took 2.2 s of the
            # 10 a drive there took to begin, and was not driven.
            planned = routing.plan_ahead(network, pose, RED_LINE_AHEAD_M, stop_before_lights=False)
            print("a place marked on the map: its red line is followed")
        elif depot is not None and args.game:
            theirs = float(block["navigation"]["distance"]) if block is not None else None
            source = None
            if block is not None and int(block["job"]["job_active"]):
                job = block["job"]
                source = Places.open(args.cache).company(
                    bytes(job["source_company"]).split(b"\0", 1)[0].decode(),
                    bytes(job["source_city"]).split(b"\0", 1)[0].decode(),
                )
            router = GameRouter.plan(
                network,
                depot,
                pose,
                origin=source.position if source is not None else None,
                nav_distance=theirs if theirs else None,
                alternatives=alternatives,
            )
            for note in router.notes:
                print(note)
            planned = router.route
            ours = router.nav_distance(pose)
            if theirs:
                print(
                    f"the game's navigation: {theirs / 1000:.1f} km to go; this route and the "
                    f"game's way on from its end: {ours / 1000:.1f} km "
                    f"({(ours - theirs) / 1000:+.1f} km)"
                )
                if args.to_job and gps_elsewhere(theirs, ours):
                    print(
                        f"the game's GPS goes elsewhere: {theirs / 1000:.1f} km to go, the job's "
                        f"place {ours / 1000:.1f} km",
                        file=sys.stderr,
                    )
                    return GPS_ELSEWHERE
            else:
                print(f"{ours / 1000:.1f} km to go, as the game counts it")
        elif depot is not None:
            planned = routing.plan_to_depot(network, pose, depot, through=waypoints)
        else:
            planned = routing.plan(network, pose, waypoints)
    except routing.RouteError as error:
        print(f"no route: {error}", file=sys.stderr)
        return 1

    crossing = None
    if planned.ferry is not None:
        # The truck drives to the ship; the crossing is the game's, from its
        # menu, and the rest is planned again from the other side.
        board = planned.lanes[planned.ferry]
        crossing = next(
            link
            for link in network.ferry_links[board]
            if link.land == planned.lanes[planned.ferry + 1]
        )
        planned = planned.to_ferry()

    hazards = planned.hazards()
    routing.save(
        planned,
        args.output,
        cache=str(args.cache),
        start=list(pose),
        waypoints=[list(w) for w in waypoints],
        hazards=dataclasses.asdict(hazards),
        crossing=None
        if crossing is None
        else {"from": crossing.start, "to": crossing.end, "train": crossing.train},
        destination=place_record(depot) if args.to_gps else None,
        alternatives=[place_record(a) for a in alternatives],
    )
    points = planned.points()
    straight = math.hypot(points[-1, 0] - points[0, 0], points[-1, 2] - points[0, 2])
    print(
        f"{len(planned.lanes)} lanes, {planned.lane_changes} lane changes, "
        f"{planned.length:.0f} m ({straight:.0f} m as the crow flies) -> {args.output}"
    )
    if crossing is not None:
        by = "the train" if crossing.train else "the ferry"
        print(
            f"  ends where the truck boards {by} from {crossing.start} to {crossing.end}: "
            "take it from the game's menu, then plan again from the other side"
        )
    # Say what the map knows is on the way: what the stack will stop for, and
    # what it cannot handle yet.
    first = (
        f", the first after {hazards.first_traffic_light_m:.0f} m"
        if hazards.first_traffic_light_m is not None
        else ""
    )
    print(
        f"on the way: {hazards.junctions} junctions, {hazards.left_turns} left and "
        f"{hazards.right_turns} right turns, {hazards.traffic_lights} traffic lights{first}, "
        f"{hazards.barriers} barriers"
    )
    if hazards.traffic_lights:
        print("  traffic lights are stopped for with follow.py --acc, which sees them")
    if hazards.barriers:
        print(
            "  barriers (toll gates, level crossings) are slowed for, not stopped at: "
            "drive with a mod that makes them passable"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
