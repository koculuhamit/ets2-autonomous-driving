# Telemetry bridge — first verification

**Date:** 2026-09-20
**Game:** ETS2 1.61.1.0 (Steam buildid 25346574), Linux native, X11
**Host:** Ubuntu 22.04.5, i7-10750H, RTX 2060 6 GB
**Plugin:** layout v1, 4080-byte block, SCS SDK 1.14

First end-to-end run of the plugin against the real game: does the block
appear, is it consistent, is it fast enough, and is it *correct*.

## Does it appear

```
$ ls -l /dev/shm/ets2ad_telemetry
-rw------- 1 user user 4080 ...
```

4080 bytes, matching `sizeof(TelemetryBlock)` exactly. The reader's magic,
layout version and block size checks all passed on open.

## Is it correct

The strongest available check is an independent one. The game writes the
truck's spawn position into its own log:

```
00:01:05.580 : [actor] Initialized at [-6500.0;55.2;29077.4]
```

Reading the block gives `[-6499.99, 55.2, 29077.4]`. The coordinate decode,
the double-precision position layout and the field offsets are all confirmed
against a source that knows nothing about our struct.

Derived quantities that came out plausible on the first try:

| Quantity | Value | Derived from |
|---|---|---|
| Truck | `mercedes/vehicle.mercedes.actros2014` | configuration event |
| Wheelbase | 3.761 m | reported wheel positions, steered vs powered |
| Wheel count | 4 | configuration event |
| Trailer | `vehicle.scs.box`, connected | configuration event |
| Hitch angle | −16.4° at rest, −53°…+12° while manoeuvring | truck − trailer heading |

## Is it fast enough

Sampled for 5 s while the game ran:

| Metric | Value |
|---|---|
| Plugin publish rate | **59.9 Hz** |
| Reads taken | 495 |
| Reads needing a seqlock retry | **0 (0.0 %)** |
| Plugin steady clock vs reader wall clock | 10 ms drift over 5 s (0.2 %) |

59.9 Hz matches the game's frame rate, so the plugin publishes once per frame
as designed and adds no rate limit of its own.

Zero retries across 495 reads is the number that matters for the seqlock: the
writer holds the counter odd only for one 4 KB memcpy, so a reader almost never
lands inside it. The retry path is still needed — it is a race, not an
impossibility — but it costs nothing in practice. The reader reports
`attempts` per snapshot so a regression here shows up rather than being
absorbed silently.

The two clocks agreeing to 0.2 % means `write_time` is usable for measuring
the game's real publish interval, which the latency budget will need.

## Are the values live

Sampled at 50 Hz for 10 s while driving:

| Field | Range observed | Notes |
|---|---|---|
| `speed` | 0.1 → 11.0 km/h | |
| `engine_rpm` | 699 → 1621 | |
| `gear` | 1 → 5 | |
| `input_steering` | −1.00 → +1.00 | full lock both ways |
| `effective_steering` | −1.00 → **+0.83** | see below |
| `wheels.steering[0]` | ±0.10 rotations (±36°) | |
| `heading` | 40.4° → 113.1° | |
| `hitch angle` | −53.4° → +12.4° | |
| `brake` | 0.0 throughout | not used in this run |

**`input_steering` reaches ±1.0 while `effective_steering` only reaches +0.83.**
That gap is the game's own input filtering and rate limiting, and it is exactly
why the block carries both. Phase 3 measures that relationship rather than
assuming the commanded value is the applied one; the controllers are then
designed against what the vehicle actually does.

## What this run did not cover

- **Actuation.** Nothing has been written *into* the game yet. ADR-0002 is
  still open and the Input API spike has not run.
- **Coordinate conventions.** Positions match the game's own log, but the
  heading sign and zero direction have only been eyeballed, not verified
  against a known bearing. Until that is done, `heading` is trusted for
  differences and not for absolutes.
- **The hitch angle sign convention.** The magnitude is plausible and it wraps
  correctly across the samples taken, but which sign corresponds to which way
  the trailer is folded has not been pinned down. Phase 8 depends on this and
  it gets its own check.
- **Long runs.** Five and ten seconds. Reconnection after the game exits, and
  behaviour across a `sdk reload`, are untested.

## Finding: the shared memory block outlives the game

When the game was closed, `/dev/shm/ets2ad_telemetry` **remained**, 4080 bytes,
holding the last frame. No process had it mapped; the name was simply never
unlinked, because ETS2 1.61.1.0 on Linux exited without calling
`scs_telemetry_shutdown`.

A reader attaching afterwards saw a perfectly well-formed block — correct
magic, correct layout version, correct size — and the monitor printed it:

```
-0.0 km/h  gear  +0  rpm 550  steer +0.015  thr 0.00 brk 1.00  hdg 178.2°  0.0 Hz
```

A truck standing still with the brake applied. Entirely plausible, entirely
wrong, and nothing in the data says so. This is the worst failure mode the
bridge can have, and it was found by accident rather than by design — the game
happened to be closed while the monitor was still being tested.

Note that the plugin does unlink on shutdown, and does clear any leftover
segment before creating a new one, so a fresh game session is always clean. The
problem is only for a reader that starts while no game is running. Nothing the
plugin does can fix that: a process killed outright runs no cleanup at all, and
a hung game keeps its segment mapped and current-looking indefinitely.

### Fix: freshness is checked, not assumed

Two additions to `TelemetryReader`:

- **`stale_after` (default 1 s).** `read()` tracks the publish counter and
  raises `StaleTelemetryError` once it has stood still for longer than that.
  The counter, not `write_time` — a dead plugin's last timestamp would keep
  looking recent forever. At 60 Hz, one second is sixty missed frames: far
  beyond any plausible hitch, short enough that a controller notices before it
  has driven anywhere on bad data.
- **`wait_for_frame()`.** A single read cannot distinguish fresh from stale:
  one sample of a counter says nothing about whether it is moving. This blocks
  until the counter advances, so what it returns is known to be current.
  Anything about to act on telemetry starts here. `telemetry_monitor.py` now
  calls it before printing anything, and against the dead block it says:

  ```
  cannot read telemetry: no frame published within 1.0s; the game is not
  running. The shared memory block outlives the game, so its contents are
  last-known, not current.
  ```

A paused game also stops publishing, so it trips the same check. The
distinction does not matter to a caller — in both cases, do not drive — but the
error message names which it was, because it matters when debugging.

## Incidental finding: the game does not log SDK messages

ETS2 1.61 on Linux routes none of the SDK's `log()` output into
`game.log.txt`, and it does not appear in Steam's console logs either —
searched both after a load that demonstrably succeeded. A plugin that fails
during `scs_telemetry_init` would therefore leave no trace whatsoever, and the
reader's only symptom would be "no shared memory", which is equally consistent
with the game simply not running.

The plugin now writes its own diagnostics to
`$XDG_RUNTIME_DIR/ets2ad_plugin.log`, else `$TMPDIR` or `/tmp`
(`%TEMP%\ets2ad_plugin.log` on Windows), truncated at each init, and the
reader's "segment does not exist" error points at it — distinguishing "never
loaded" from "loaded and failed".
