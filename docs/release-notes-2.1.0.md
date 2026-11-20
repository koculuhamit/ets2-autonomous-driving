# ETS2 autopilot 2.1.0

Smoother and steadier on the road, with a ROS 2 view of what it sees. Before
release the developer drove deliveries on Linux until three in a row went
through without a fault to fix: to Düsseldorf, Salzburg and Linz, 307 km.
Since 2.0.0 it has driven 1712 km of the game's in 28 drives.

This release was driven on Ubuntu 22.04 only. Its Windows build is made and
checked by CI, as 2.0.0's was, but no delivery has been driven with it on
Windows yet.

Install: [Windows 11](install-windows.md) · [Linux](install-linux.md). Coming
from 2.0.0, run the launcher (or `tools/setup.py`) once with the game closed:
the plugin has changed, and the new one has to be installed before the
retarder can be used.

## New

- **ROS 2 view.** `tools/mission.py --ros` streams the truck, its route, the
  vehicles it tracks, the light it reads and the camera's view to ROS 2
  Humble, for RViz (`adapters/ros2`, with a launch file and an RViz layout).
  `tools/ros_replay.py` plays a recorded drive into the same view.
- **Retarder.** On trucks that have one, slowing down that wants more than
  lifting off and less than the brake's least is done on the retarder, a step
  at a time; how strong a step is, is learned while driving.
- **Brakes learned.** How hard the brake must be pressed is learned from the
  truck's own braking on each drive, not guessed from the load: going by
  the load, a loaded Scania was braked up to 1.7 times as hard as asked.
- **New trucks learned on the way.** A truck driven for the first time has
  its dashboard and windscreen learned in its first minute or two; 15 trucks
  are known now.
- **Tight turns taken wide when the way is clear.** Where keeping off the
  mouth of the road turned into would put the trailer over the kerb, the
  truck goes round wide while the camera sees no one waiting there.

## Changed

- No crawling round turns and roundabouts: their bends set the speed, and a
  vehicle that drives round a bend out of view is not stopped for.
- Gentler stops: a stop for a light, a junction or traffic is braked for as
  soon as the truck is on its curve, not left to the last metres; a vehicle
  seen late may be stopped for a little past the line instead of hard.
- At a red light the truck no longer speeds up towards it, and stops where
  it can still see the light's lamps, so it sees the green.
- Small, far signal heads are read whole when their lamps are too small to
  pick out: a red is seen from about 40 m at sunset.
- A light the map has and the road does not is no longer stopped at when no
  lamp is in view.
- The parking brake is set on a slope at once, on the flat after 3 s, so
  short stops do not pull it.
- The game's camera moved out of the cab for a moment slows the truck; it
  stops only if the camera stays out for 8 s.
- A car coming fast the other way round a bend is not taken for one
  standing in the truck's lane.

## Fixed

- After about 9 s on the brake at rest, the game put the truck in reverse.
- A pedal shown late by the game was taken for the driver's foot, and the
  truck was handed over for nothing.
- When the game stopped showing frames for a minute (its window put away),
  the truck came back with nobody driving it. It is now braked until the
  game shows it again, then stopped and handed over.

## Known limits

The limits of [2.0.0](release-notes-2.0.0.md) still hold, and:

- Not driven on Windows in this release.
- Obstacles that are not vehicles, such as cones and barriers, are not seen.
- A vehicle is now and then seen as two or three tracks in quick succession,
  and the speed it is followed at swings for a few seconds.
- The last moment of a stop can still be felt: below walking pace the brake
  holds the truck firmly.
