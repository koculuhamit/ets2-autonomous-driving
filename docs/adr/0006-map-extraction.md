# ADR-0006: Where the route comes from

- **Status:** Accepted
- **Date:** 2026-09-22

## Context

The stack can follow a path. It cannot yet be told one without being driven it
first: teach-and-repeat was a way to build and tune a lateral controller with
no map, and it worked, but the goal is different. The goal is that the route
marked on the game's own map -- set by hand or by taking a job -- is driven
autonomously from a keypress.

The telemetry SDK does not expose that route. It gives distance and time
remaining to the destination, the navigation speed limit, and the job's
endpoints. Where the road goes between here and there is not in it. Nor is
anything else that matters for driving among traffic: other vehicles, traffic
lights and signs are absent from telemetry entirely.

So the route has to come from somewhere else, and there are three somewheres.

## Options considered

### Option A — Read the route off the screen

Open the in-game map, screenshot it, find the red line, transform map pixels to
world coordinates.

- No archive parsing, and a first result in days rather than weeks.
- Fragile in the way screen-scraping is always fragile: a UI change, a zoom
  level, a colour-blind mode, a different resolution.
- Yields a line and nothing else. No lane count, no junction geometry, no
  speed limits, no knowledge of which way is one-way. A line is enough to
  steer along and not enough to drive by.

### Option B — Extract the road network from the game's archives

Parse the `.scs` archives, build the road network, and route on it with our own
planner. The destination is known from telemetry; the path to it is ours to
compute.

- Everything perception and behaviour will later need comes from the same
  source: lanes, junction shapes, prefab geometry, speed limits, one-way flags.
- The map is static, so it is parsed once into a cache and never touched again
  while driving.
- It is the larger job, and the format is undocumented.

### Option C — Stay with taught routes and build perception first

- Postpones the problem rather than solving it, and perception developed
  against one hand-driven loop is perception developed against one road.

## Decision

**Option B.** The archives are the source, and a planner of our own runs on
what comes out of them.

Parsing is done with **TruckLib**, as an offline tool, exactly as ADR-0001
anticipated when it allowed C# for this and nothing else. `dotnet-sdk-10.0`
is installed per-user under `~/.local/share/dotnet` -- no root, and the same
pattern `uv` already follows. The tool lives in `tools/map_extract/` and is
run by hand; nothing in a driving loop ever calls it, and the Python side reads
only its output.

The alternative, writing our own HashFS v2 and map-format parsers, was weighed
seriously -- half of HashFS v2 was already reverse-engineered by hand earlier
in this project, and the result would be dependency-free and impressive. It was
rejected on schedule rather than on merit: the interesting problem here is
driving, not archive archaeology, and TruckLib's map format support is kept
current with the game in a way ours would not be.

## Consequences

- A build-time dependency on .NET, for a tool that runs when the game updates
  its map and at no other time.
- The cache format is ours, so the Python side never learns what an `.mbd` is.
- If TruckLib falls behind a game version, extraction stops working while
  driving does not -- the cache is already on disk. That is the right way round.
- Prefab interiors -- junctions, roundabouts, company yards -- are not in the
  map file. They come from `.ppd` descriptors, which is a second step.
