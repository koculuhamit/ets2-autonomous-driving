# Latency budget: from the game's picture to the truck's wheels

**Date:** 2026-11-20
**Game:** ETS2 1.61, Linux native, 1920×1080
**Machine:** i7-10750H, RTX 2060 6 GB
**Data:** the drives of 20 November 2026 (`tools/mission.py --ros`), about
508 000 control steps; the bench measurements linked below

The stack drives on two loops of very different speed. Steering and speed
run on the game's telemetry, which arrives every frame. Vehicles and traffic
lights come from the camera, through a detector, and arrive later. The two
are kept apart so that the slow one never holds up the fast one
(`core/perception/service.py`).

## The fast loop: telemetry to command

| step | median | p95 | measured in |
|---|---|---|---|
| the game publishes telemetry | every 16.6 ms | | [input_path](input_path.md) |
| one turn of the control loop | 7.7 ms | 10.1 ms | `acc.npy`, these drives |
| a command takes effect in the game: steering | 27.9 ms | 34.1 ms max | [input_path](input_path.md) |
| a command takes effect in the game: throttle | 25.2 ms | 37.8 ms max | [input_path](input_path.md) |
| a command takes effect in the game: brake | 15.4 ms | 35.2 ms max | [input_path](input_path.md) |

From a telemetry frame to the wheels turning is about **50 ms**: up to one
frame waiting, one turn of the loop, and the game picking the command up on
its next frames. Steering follows the map's lane, whose position is known
exactly, so this is the loop that keeps the truck in its lane.

## The slow loop: camera to command

| step | median | p95 | measured in |
|---|---|---|---|
| grabbing a frame of the game's window | 4.2 ms | 11.4 ms | [capture_bench](capture_bench.md) |
| detecting vehicles and signal heads (D-FINE nano, GPU) | 19–22 ms | about 30 ms | [vehicle_detection](vehicle_detection.md) |
| a frame grabbed to its report ready (detection, tracking, lamps, lanes) | **55.7 ms** | 77.6 ms | these drives |
| between two reports | 55.5 ms (18 a second) | 84.6 ms | these drives |
| how old the report is when the control loop uses it | **81.4 ms** | 118.7 ms | these drives |
| the command then takes effect in the game | 15–28 ms | | as above |

From a picture to the brake acting is about **110 ms** at the median and
**165 ms** at p95: the report's age, one turn of the loop and the game's
input latency. At 90 km/h the truck covers 2.8 m and 4.1 m in that time. The gap the following controller keeps, and the decelerations it
plans with, already allow for it. A vehicle cutting in is the case where it
is felt.

Of the 56 ms a report takes, the detector is about 20 ms. The rest is
everything else the perception process does with the same frame: tracking,
reading the light's lamps, finding the lanes, learning a new truck's
dashboard, and encoding the JPEG the ROS 2 view shows. The process is
allowed 30 reports a second and gets 18, so a faster detector alone would
not change much; doing less with each frame would.

## What the ROS 2 view costs

With `--ros` the perception process also encodes each picture for RViz.
The report's age went from a median of 73 ms to 83 ms and its p95 from
104 ms to 118 ms (measured on 16 November). The telemetry loop is not
affected: the stack hands messages to the ROS 2 bridge without waiting
(`adapters/ros2/feed.py`).

## How these were measured

- **Report age.** Each report carries `time.perf_counter()` from when its
  frame was grabbed. That clock is shared by every process on the machine.
  The control loop logs the age of the report it used on every turn
  (column 8 of `acc.npy`). Where the age drops, a new report has come in.
  The age at that moment is the frame-to-report time, and the time between
  two drops is the report interval.
- **Command latency.** The time from writing a command to the first change
  in the game's telemetry, with a threshold, over repeated steps. How a slew
  was first mistaken for a 130 ms latency is in [input_path](input_path.md).
- **Not measured:** the game's own time from telemetry to the picture on the
  screen. The game draws at 60–80 frames a second, so it is one frame or two.
