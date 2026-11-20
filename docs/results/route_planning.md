# Planning a route on the map, and following it

**Date:** 2026-09-23
**Network:** `data/map/installed` — base game + Scandinavia
**Code:** `core/planning/route.py`, `tools/plan_route.py`, `tools/follow.py`

Until now the truck could only follow a line it had been driven along first.
This is the other half: a route planned on the game's own lanes, from where the
truck is to where it is told, drawn as one line a controller can follow.

## What a route is

A sequence of lanes and how each leads to the next -- onwards along the road,
or across into a lane alongside. `plan()` takes a pose and a list of
positions to pass through, finds the start lane with the map matcher, and
searches the lane graph leg by leg.

`Route.points()` draws it: lanes end to end, and each lane change as a
smoothstep across 50 m, finishing early rather than late. The lane sequence
alone would say "jump 4.5 m sideways", which no controller should be told.

## Checked against the loop that was driven

Planned through a waypoint every 150 m of the taught loop, starting where the
truck started:

| | planned | taught by hand |
|---|---|---|
| length | 992 m | 976 m |
| lanes | 59 | -- |
| lane changes | 1 | -- |

The two lines agree to a median of 0.72 m, which is the difference between a
lane centre and where a person drives in it. They part in two places. For 94 m
near the start the hand-driven loop went through the forecourt of the petrol
station found earlier in the extraction, and the plan stays on the road; both
are routes. And on one two-lane stretch the plan changes lane where the hand
drive did not, 4 m apart for a hundred metres.

### Simulated

The same closed-loop simulation as before -- bicycle model, measured steering
and pedal maps -- with the gradient now taken from the lane heights in the map
rather than from the recorded pitch:

| | planned route | taught route |
|---|---|---|
| lateral error, max | **0.75 m** | 0.86 m |
| lateral error, RMS | **0.20 m** | 0.24 m |
| over 0.5 m | 4 % | 7 % |
| lateral acceleration, max | 2.87 m/s² | 2.80 m/s² |

The planned line is easier to follow than the driven one, which is what lane
centres from a map should be: smooth where a hand-driven line wanders.

## Two things that went wrong on the way

**The nearest lane to a waypoint is not the one to aim for.** The first plan
of the loop was 1878 m: it went round twice. The last waypoint sits inside a
junction where lanes overlap, and the lane nearest to it, 1.1 m away, belonged
to a branch going the other way -- reaching it meant going round again, 1004 m
for a leg of 70. One 1.5 m away was straight ahead. Every lane near a waypoint
is now a candidate, scored as the cost to reach it plus 20 m per metre it
misses the waypoint by.

**Scoring candidates one search at a time was slow.** 5.5 s for the loop, most
of it spent proving that candidates on the wrong branch cannot be reached,
which takes a search of everything reachable. One A* outwards from the start
now scores candidates as it reaches them and stops when nothing queued could
do better: 21 ms for the loop, with the same route.

| Route | road | planning |
|---|---|---|
| taught loop, 7 waypoints | 992 m | 20 ms |
| Italy → Germany | 77 km | 0.62 s |
| Germany → Sweden | 56 km | 0.62 s |
| France → Norway | 119 km | 2.16 s |

## To a job's destination

A job names its destination the way the telemetry carries it: a company id
and a city id, `tree_et` in `berlin`, and no position. The extractor now also
reads every company depot and city area from the map -- 625 depots and 348
city areas, 517 depots on this installation's map -- with the same ids, and
the prefab that is each depot's yard. The yard's control nodes are its gates
onto the road; there are two or more.

`plan_to_depot()` routes to the gate with the shortest route, and not into the
yard: a yard's own lanes are mostly islands, unjoined to the road network, and
backing a trailer into a spot is a different problem.

From the start of the taught loop in northern Italy to 60 depots picked at
random:

| | |
|---|---|
| routes found | 50 of 60 |
| of those not found, in Britain | 7 |
| found on the continent | **50 of 53** |
| road distance, median / longest | 64 km / 148 km |
| planning, median / slowest | 0.8 s / 7.6 s |

Britain cannot be reached without the ferry or the tunnel. The slowest plans
are the failures: proving a gate unreachable means searching everything
reachable, once per gate.

## What the route cannot see

Nothing in the stack perceives anything yet: no traffic light, no vehicle
ahead. The map, though, knows where every light is and which lanes it
governs, so `Route.hazards()` counts junctions, turns and lights before a
route is driven, and `tools/plan_route.py` prints them.

Run over the taught loop, it found what had not been counted before: the loop
the truck had already driven autonomously passes **six traffic lights**, the
first 207 m in. Those runs went through them blind.

`plan_ahead()` exists for that reason. It needs no destination -- it follows
the road in front of the truck, the straightest way on at each junction --
and it ends before the first lane a light governs, and where the road ends.
From the start of the loop that is 206 m; the loop is in a town. It is the
route to drive a planned route with for the first time, on a road chosen for
having no lights on it.

## Driven

2026-09-23. The truck on an autostrada in northern Italy, stationary, with a
kilometre of empty road ahead and no traffic light on it. From where it
stood:

    python tools/plan_route.py data/map/installed ahead-01.npz --live --ahead 800
    python tools/follow.py ahead-01.npz --simulate
    python tools/follow.py ahead-01.npz --speed 40 --distance 850

The route was 872 m over 14 lanes, 7 of them motorway junction pieces, no
turns, no lights, nothing tighter than a 169 m radius. It had never been
driven, by hand or otherwise: every metre of it came from the map.

| | simulated | driven |
|---|---|---|
| distance | 852 m | 820 m, to the end of the route |
| lateral error, max | 0.20 m | **0.36 m** |
| lateral error, RMS | 0.05 m | **0.089 m** |
| over 0.3 m | -- | 1.3 % |
| damage | -- | none |

Matched against the network all the way, the truck was on one of the planned
lanes in **795 samples of 795**. It started 0.38 m off the lane, took it up
within the first few metres, held 40 km/h for four fifths of the run and
stopped at the end.

The real run was about twice the simulated error, which the simulation had
not been before -- on the taught loop it was within 15 % either way. At
these sizes that is a few centimetres of tyre and actuator the model does not
have; it is not a reason to distrust it, and it is a reason to keep driving
what it predicts.

This is the first time the stack drove somewhere it was told rather than
somewhere it was shown.

### At 60 km/h

Everything before had been driven at 40 km/h or less. From where the first
run stopped, a second route of 1678 m planned the same way, driven at 60, with
950 m of the kilometre of clear road as the budget:

| | simulated | driven |
|---|---|---|
| lateral error at 55 km/h and above, max | 0.20 m | **0.43 m** |
| lateral error at 55 km/h and above, RMS | 0.07 m | **0.206 m** |
| on a planned lane | -- | 671 of 671 |
| damage | -- | none |

The budget stopped it at 950 m, 942 m from where it started. At cruise there
is no steady offset -- the mean is +0.07 m -- but the error follows the road's
curvature, a correlation of −0.52. Lateral error is positive to the left and
curvature positive in a left bend, so a negative correlation puts the truck on
the **outside** of every bend: it runs wide, it does not cut the corner.

*This paragraph first said the opposite -- "the truck sits inside every bend,
pure pursuit cutting corners" -- which is what pure pursuit is known for and
not what the sign says.*

The simulation under-predicts the error by a factor of two to three, at 40 as
at 60. Two explanations were tested in simulation, before any more driving,
and neither holds:

- **A slower steering actuator.** Regressing the game's applied steering rate
  on the command in the recorded runs gives about 0.26/s at 20--40 km/h
  (r = 0.8--0.9) against the 0.75/s the simulation assumes. Put into the
  simulation, it makes the error slightly *smaller*: 0.17 m instead of 0.20.
- **The truck turning less than the kinematic model says.** Measured yaw rate
  is 96.4 % of the model's in this run, and running wide is what that would
  do. It accounts for 0.20 → 0.22 m of the 0.43.

Closed-loop data mixes the controller into everything it shows, so the next
step is an experiment rather than another regression: `tools/follow.py
--excite` adds steering doublets of known size -- one amplitude up for 0.3 s,
down for 0.3 s, every 5 s -- while the controller holds the lane, which gives
the actuator a known input to answer.

The experiment was run the same afternoon, at 60 km/h over 1000 m of
autostrada, and it settled the actuator: 0.19 /s per unit command at 50--60
km/h, 0.25 at walking pace, linear in amplitude
(`docs/results/input_path.md`). The steering position loop had been dividing
by 0.75, so at road speed it ran four times slower than designed -- a time
constant near half a second rather than 125 ms.

That looked like the answer. It is not, or not in simulation: with the measured
actuator as the plant, the loop corrected to its designed speed is *worse* on
all three planned routes -- 0.17 → 0.20 m, 0.14 → 0.18 m and 0.26 → 0.43 m
at 60 km/h -- than the slow loop that has been driving. A slow steering loop
filters what pure pursuit asks for, and here that helps. The simulation still
under-predicts the real error two and a half times, so it is not the judge of
this either; the corrected loop -- since 2026-09-24 the default, with the game's pull back to centre compensated (steering_centring.md) -- was then `--steering-schedule`, and the
default stays the loop that has been driven. Deciding between them takes the
game: the same road, both loops, more than one run each, because run to run
the error has varied from 0.27 m to 0.43 m at 60.

### A lane change

Same autostrada, a route planned to stay in lane for 390 m and then move one
lane right, driven at 60 km/h. Left to itself the planner put the change on
the very first lane, with the truck still standing: a lane change costs the
same wherever it happens and the search took the first. A waypoint in the
current lane 350 m on put it where it was meant to be tested, at speed.

Drawn over 50 m, the move would have been 3 s at 60 km/h and 3 m/s² of
lateral acceleration at its peak -- a smoothstep across 4.5 m in time T peaks
at 6 × 4.5 / T² -- harder than any bend on that road, and 2.6 m/s² in
simulation. It is now drawn over 120 m, about 7 s and 0.5 m/s².

| | |
|---|---|
| lane, before and after | offset +2.25 → **−2.25**, one lane right |
| lateral acceleration during the change | **0.43 m/s²** |
| lateral error during the change, max / RMS | 0.56 m / 0.17 m |
| on a planned lane | 617 of 617 |
| damage | none |

The worst error of the run, 0.71 m, is not the lane change but the bend just
before it, a 135 m radius taken at 60 km/h: the truck running wide, as it does.

### The first attempt at 60 did not move

It commanded 43 % throttle for 2.5 s and the stuck check stopped it, blaming
an obstacle. The telemetry said otherwise: the throttle the game applied was
zero throughout and the steering never moved, and the plugin logged nothing
for the run -- the game was not polling the stack's input device. A harmless
steering nudge a minute later went through. The likeliest cause is window
focus: the game polls input devices only while its window has it, and
telemetry keeps flowing regardless.

Standing still, that looks like an obstacle. Moving, nothing would have
noticed, and a truck whose inputs are ignored coasts on. `SafetyMonitor` now
stops a run when a pedal is commanded and none is applied for 1.5 s -- twice
the 0.78 s the automatic gearbox cuts the throttle for at a shift. Replayed
over the recordings, it fires at 1.57 s in the failed run and never in the
four good ones.

## Using it

    python tools/plan_route.py data/map/installed ahead.npz --live --ahead 400
    python tools/plan_route.py data/map/installed route.npz --live --to X Z
    python tools/plan_route.py data/map/installed job.npz --live --to-job
    python tools/follow.py route.npz --simulate
    python tools/follow.py route.npz --distance 300

`--live` plans from where the truck is in the running game. The next step is
the last one: drive a planned route.
