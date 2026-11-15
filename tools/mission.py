#!/usr/bin/env python3
"""Deliver the job: engage with the truck's beacon, and drive to the depot.

    python tools/mission.py

Runs until stopped (Ctrl+C), job after job. Switch the truck's beacon on
(O, by default) in the game and the truck drives the game's GPS route
(tools/follow.py --game-route) -- to the depot, or to a port or train
terminal on the way, where it stops with its hazard lights on and sounds its
horn twice. Take the crossing from the game's menu, drive off, and switch
the beacon off and on: the truck plans again from the other side and drives
on. The driver fills up: with the tank low the desktop says so, and with the
beacon off and on after filling up the truck drives on. At the depot's gate
the truck stops with its hazard lights on: the driver parks and delivers.
Switch the beacon off at any time and the controls are yours
at once.

The driver hitches the trailer and drives out of the yard: the map's lanes
stop at a depot's gate, and no route starts inside. With no job told -- a
place marked on the map -- a route on the game's GPS is driven to all the
same: where it ends is found from its distance
(core.planning.destination), and the truck stops there for the driver. A
paused game pauses the drive.

What happens, and why the truck stopped, is printed here, and nowhere
else: nothing pops up on the desktop and nothing sounds there (the
driver's word, 14 November); the drives are recorded under --output, one folder
each, with their black box.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Before numpy: on Windows OpenBLAS commits a buffer for each CPU thread as it
# is imported -- 360 MB in each of the drive's processes, where Linux commits
# nothing until it is used -- and one thread is as quick for what is done here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge import convert  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from core.map import RoadNetwork  # noqa: E402
from core.mission import (  # noqa: E402
    DriveEnd,
    Mission,
    Observation,
    State,
    how_it_ended,
)
from core.mission.status import STATUS_FILE, write_status  # noqa: E402
from pal.base import PalError  # noqa: E402
from tools.plan_route import GPS_ELSEWHERE  # noqa: E402

LOOK_EVERY_S = 0.2
RETRY_S = 2.0


def observation(snapshot) -> Observation:
    block = snapshot.block
    destination = convert.job_target(block)
    position = block["truck"]["placement"]["position"]
    return Observation(
        paused=bool(snapshot.is_paused),
        job=destination,
        beacon=bool(int(block["truck"]["light_beacon"])),
        position=(float(position["x"]), float(position["z"])),
        nav_distance=float(block["navigation"]["distance"]),
    )


def boarding_points(network: RoadNetwork) -> list[tuple[float, float]]:
    """Where every ship and train on the map is boarded: its lanes' ends."""
    points = []
    for board in network.ferry_links:
        end = network.lane_points(board)[-1]
        points.append((float(end[0]), float(end[2])))
    return points


def to_nearest(points: list[tuple[float, float]], at: tuple[float, float]) -> float:
    return min((math.dist(p, at) for p in points), default=math.inf)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *command], cwd=REPO_ROOT, check=False, **kwargs)


def drive_arguments(args) -> list[str]:
    """What tools/follow.py is told for every drive."""
    return [
        "--acc",
        "--game-route",
        "--beacon",
        "--distance",
        "10000000",
        "--duration",
        str(args.duration),
        "--time-gap",
        str(args.time_gap),
        "--yes",
        *(["--keep-frames", str(args.keep_frames)] if args.keep_frames > 0.0 else []),
        *(["--allow-cpu"] if args.allow_cpu else []),
    ]


def outcome_of(folder: Path) -> tuple[str | None, str]:
    """How a drive ended, as tools/follow.py wrote it in `folder`."""
    outcome = folder / "outcome.json"
    if not outcome.is_file():
        return None, ""
    written = json.loads(outcome.read_text(encoding="utf-8"))
    return written.get("reason"), written.get("detail", "")


STANDBY_EVERY_S = 10.0
"""A standby that ended of itself is started again no sooner than this."""


@functools.cache
def private_dir() -> Path:
    """A directory only this user can enter (`tempfile.mkdtemp`, mode 0700),
    made for this run of the mission and removed when it ends: the word file
    between the mission and its standby, and the standby's scratch. Under a
    name anyone could guess in the shared /tmp, another account on the
    machine could make the word file first and bring the standby down."""
    folder = Path(tempfile.mkdtemp(prefix="ets2ad-"))
    atexit.register(shutil.rmtree, folder, True)
    return folder


class Standby:
    """tools/follow.py made ready for the next drive before the beacon is
    on (`follow.standby_until_go`): its model loaded and held, the depot's
    ways weighed. Switched on, the beacon took 17 s to start the truck, the
    planning and the making ready all after it; from standby the route is
    planned and the truck driven within a few seconds."""

    def __init__(self, args) -> None:
        self.started_at = time.monotonic()
        self.word_file = private_dir() / "standby.word"
        self.word_file.unlink(missing_ok=True)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "tools/follow.py",
                "--standby",
                str(self.word_file),
                "--cache",
                str(args.cache),
                *drive_arguments(args),
            ],
            cwd=REPO_ROOT,
        )

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def _say(self, word: str) -> None:
        """Into the word file whole: written aside, then put in its place."""
        aside = self.word_file.with_suffix(".writing")
        aside.write_text(word, encoding="utf-8")
        aside.replace(self.word_file)

    def go(self, folder: Path, to_gps: bool) -> tuple[str | None, str]:
        """Plan from where the truck is and drive, the drive's files in
        `folder`; how it ended, as `drive` says."""
        self._say(f"go {'gps' if to_gps else 'job'} {folder}")
        self.process.wait()
        return outcome_of(folder)

    def cancel(self) -> None:
        """Give up waiting."""
        with contextlib.suppress(OSError):
            self._say("stop")
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.process.kill()


def drive(args, folder: Path, to_gps: bool = False) -> tuple[str | None, str]:
    """Plan from where the truck is and drive it -- to the job's depot, or,
    `to_gps`, to where the game's GPS goes, found from its distance; how the
    drive ended, as tools/follow.py wrote it (None for nothing), and a detail."""
    route = folder.with_suffix(".npz")
    planned = _plan(args, route, to_gps)
    if planned.returncode == GPS_ELSEWHERE and not to_gps:
        # The job's place is not where the game's GPS goes -- a job done,
        # its places still told; a spot marked on the map: go where it goes.
        print("  the game's GPS goes elsewhere than the job: driving where it goes")
        planned = _plan(args, route, True)
    if planned.returncode != 0:
        if "cannot be reached from" in planned.stderr:
            # In a depot's yard, where the map's lanes do not go.
            return "no_route", "in a yard: drive out onto the road, then switch the beacon on"
        lines = planned.stderr.strip().splitlines()
        return "no_route", lines[-1] if lines else "no route"
    arguments = drive_arguments(args)
    if "a place marked on the map" in planned.stdout:
        # Its red line followed, as tools/follow.py --standby does.
        arguments = ["--gps-only" if a == "--game-route" else a for a in arguments]
    run(
        [
            "tools/follow.py",
            str(route),
            *arguments,
            "--output",
            str(folder),
            "--outcome",
            str(folder / "outcome.json"),
        ]
    )
    return outcome_of(folder)


def _plan(args, route: Path, to_gps: bool) -> subprocess.CompletedProcess:
    planned = run(
        [
            "tools/plan_route.py",
            str(args.cache),
            str(route),
            "--live",
            "--to-gps" if to_gps else "--to-job",
            "--game",
        ],
        capture_output=True,
        text=True,
    )
    for line in planned.stdout.splitlines():
        print(f"  {line}")
    return planned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "map" / "installed")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "missions")
    parser.add_argument("--time-gap", type=float, default=1.5, help="ACC time gap, s")
    parser.add_argument(
        "--duration", type=float, default=6 * 3600.0, help="longest single drive, s"
    )
    parser.add_argument(
        "--keep-frames",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="record a view of the road this often while driving, for training (follow.py)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="drive with perception on the CPU, three times slower, where there is no "
        "NVIDIA GPU (the release's launcher passes it then)",
    )
    args = parser.parse_args()

    ports = boarding_points(RoadNetwork.open(args.cache))
    session = args.output / datetime.now().strftime("%Y%m%d-%H%M%S")
    mission = Mission()
    told = ""
    drives = 0
    where = (0.0, 0.0)
    standby: Standby | None = None
    print("switch the truck's beacon on in the game to drive; Ctrl+C here to quit")
    while True:
        try:
            with TelemetryReader() as reader:
                while True:
                    seen = observation(reader.read())
                    where = seen.position
                    start = mission.observe(seen)
                    if mission.message != told:
                        told = mission.message
                        print(f"[{mission.state.value}] {told}", flush=True)
                        write_status(
                            STATUS_FILE,
                            mission.state.value,
                            told,
                            job=seen.job,
                            drives=drives,
                            session=str(session),
                        )
                    if not start:
                        if mission.state is not State.DRIVING and (
                            standby is None
                            or (
                                not standby.alive
                                and time.monotonic() - standby.started_at >= STANDBY_EVERY_S
                            )
                        ):
                            standby = Standby(args)
                        time.sleep(LOOK_EVERY_S)
                        continue
                    drives += 1
                    folder = session / f"drive-{drives:02d}"
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    if standby is not None and standby.alive:
                        reason, detail = standby.go(folder, to_gps=mission.to_gps)
                    else:
                        reason, detail = drive(args, folder, to_gps=mission.to_gps)
                    standby = None
                    snapshot = reader.read()
                    where = observation(snapshot).position
                    if reason == "no_route":
                        how = DriveEnd.NO_ROUTE
                    else:
                        nav = float(snapshot.block["navigation"]["distance"])
                        how, detail = how_it_ended(
                            reason,
                            detail,
                            nav,
                            to_nearest(ports, where),
                        )
                    mission.drive_ended(how, where, detail)
        except KeyboardInterrupt:
            if standby is not None:
                standby.cancel()
            print("\nstopped")
            return 0
        except (PalError, TelemetryError, OSError) as exc:
            if mission.state is State.DRIVING:
                mission.drive_ended(DriveEnd.LOST_GAME, where, str(exc))
            waiting = f"waiting for the game: {exc}"
            if told != waiting:
                told = waiting
                print(f"[{mission.state.value}] {told}", flush=True)
                write_status(STATUS_FILE, mission.state.value, told, drives=drives)
            time.sleep(RETRY_S)


if __name__ == "__main__":
    sys.exit(main())
