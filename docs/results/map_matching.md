# Knowing which lane the truck is on

**Date:** 2026-09-23
**Network:** `data/map/installed` — base game + Scandinavia, 470 186 lanes
**Recordings:** the taught loop (`teach-01`, driven by hand) and the three
autonomous runs over it (`data/follow/`), sampled every fourth frame
**Code:** `core/map/matcher.py`

## Why a matcher

Everything a planner does with the map starts from "which lane am I on".
Taking the nearest lane at every sample answers "how far from a lane" well and
"which lane" badly. With the heading used to throw out lanes pointing the
wrong way, it still produced lane-to-lane steps that the network does not
connect:

| Recording | lane changes seen | not connected |
|---|---|---|
| teach-01 | 78 | **21** |
| follow, first 300 m | 28 | 5 |
| follow, s = 425 … 968 | 47 | 12 |
| follow, s = 0 … 424 | 28 | 5 |

Almost all of them in junctions, where through, turning and merging lanes run
a metre or two apart and the nearest one changes from sample to sample. A
position cannot say which of them the truck is on. Where it was a moment ago
can.

## What it does

A beam search over the lane graph. It keeps up to eight hypotheses -- a lane, a
distance along it, a score -- and moves each only where a vehicle can go:
along its lane, into the lanes that follow while there is distance to cover,
or across into a lane alongside, at a price. Each candidate is scored by how
well the position (σ = 1.5 m) and heading (σ = 25°) fit it; the best survive.
A hypothesis cannot jump between two lanes the road does not join.

## The first version, and the junction that beat it

| Recording | transitions | not connected | lost |
|---|---|---|---|
| teach-01 | 59 | 3 | 2 |
| follow, first 300 m | 25 | 1 | 0 |
| follow, s = 425 … 968 | 32 | 1 | 1 |
| follow, s = 0 … 424 | 25 | 1 | 0 |

The losses were at the same junction on every lap, at (−6635, 29286). A wider
beam did nothing -- six unconnected steps at 8, 16 and 32 hypotheses -- so the
right answer was not being crowded out; it was not reachable.

Tracing the truck through the junction said why. It drives dead straight along
x = −6635.1, and the lane that fits best the whole way, 1.0 m off and 0° out,
is a straight-through lane nothing the matcher was on leads to. Inside the
junction two approach lanes run side by side for 40 m, 4.7 m apart, and the
truck was in the other one. The graph offers lane changes on roads only --
changing lanes inside a junction is not something a route should plan -- but a
truck does it, and a matcher has to follow what the truck does.

So the matcher also considers any lane within 6 m of the position, at the
price of a lane change plus one: a move sideways into a lane the graph does not
join, believed only when the position keeps saying so. The graph is unchanged.

## Now

Over all four recordings together:

| | |
|---|---|
| lane-to-lane steps along the graph | 137 |
| sideways into a lane alongside | 3 |
| corrections at a fork | 4 |
| lost | **0** |
| lateral error to the matched lane, median | 0.53 m |
| lateral error, 95th percentile | 2.74 m |
| time per update, mean / 99th percentile | 1.64 ms / 3.78 ms |

The four corrections are two junctions, once per lap each: the matcher takes
the wrong branch of a fork where both start at the same point, and moves
across to the right one when the branches part -- 48° apart by then, which is
why they do not count as sideways. That is what an online filter should do
with a fork it cannot yet tell apart; a route being followed would not have
the ambiguity, because the route says which branch.

## Tests

`tests/unit/test_map_matcher.py` builds each failure as a small network: a
fork whose branches start together, a lane change, a brief swerve past the
midline that should not become one, two junction lanes side by side with no
lane change between them, and the oncoming lane. The last two were checked by
disabling what they test: without sideways moves the side-by-side test fails,
and with lane changes free the swerve test does.
