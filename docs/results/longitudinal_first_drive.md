# First drive — longitudinal actuators

**Date:** 2026-09-20
**Game:** ETS2 1.61.1.0, Linux native
**Vehicle:** Mercedes Actros 2014, box trailer attached, automatic gearbox
**Site:** flat company yard, straight line, no steering input
**Tool:** `tools/longitudinal_probe.py --target-speed 15 --throttle 0.4 --brake 0.4`

The first run in which the stack moved the vehicle. Its purpose was narrower
than it sounds: to settle whether `abackward` brakes, and to get a first look
at the longitudinal response.

## Result

| Phase | Speed | Duration | Mean acceleration | `effective_brake` |
|---|---|---|---|---|
| Accelerate, throttle 0.4 | 0 → 15.0 km/h | 5.9 s | +0.70 m/s² | — |
| Coast, throttle released | 15.0 → 14.7 km/h | 1.5 s | −0.06 m/s² | 0.00 |
| Brake, `abackward` 0.4 | 14.7 → 0.4 km/h | 0.7 s | **−5.46 m/s²** | **0.40** |

1876 samples, peak 15.9 km/h, about 21 m travelled. The truck stopped under
control and took no damage; cargo damage stayed at 0.

## `abackward` is the brake

This was the open question from the input spike. At a standstill the game reads
`abackward` as a request to move backwards and engages reverse, which says
nothing about braking and briefly suggested our axis mapping was wrong.

Moving, it is unambiguous: `effective_brake` tracks the command exactly (0.40
commanded, 0.40 applied) and the truck decelerates. The earlier reading was a
standstill artefact, and the axis mapping was never in doubt once the raw
`input_*` channels were consulted instead of the effective ones.

## Braking is stronger than expected

−5.46 m/s² at 40 % brake command is about 0.56 g. For a combination vehicle
that is firm — and it is the *mean* over the stop, so the peak is higher.

Two things follow. Phase 6's comfort limits will be doing real work rather than
rubber-stamping whatever the game does; and emergency braking has considerably
more authority available than a comfortable ACC should ever use. The gap
between those two is worth knowing early.

The trailer was empty for this run. Loaded braking will differ and has to be
measured separately.

## What this is *not*

**It is not a throttle-to-acceleration map.** +0.70 m/s² is the mean across a
0 → 15 km/h pull that ran through to gear 5, so it is dominated by the gearbox,
not by the throttle. A usable map needs steady-state acceleration held at a
fixed throttle in a fixed gear, repeated across gear, speed, load and gradient.
That is Phase 3, and this run is its first data point rather than a substitute
for it.

**Coast drag is unmeasured.** −0.06 m/s² over 1.5 s at 15 km/h is barely above
the noise, and a coast-down worth fitting needs a much higher starting speed
and a much longer run.

## Still open

- Brake command to deceleration, across the command range. Needed before any
  comfort limit means anything.
- Everything above, loaded.
- Gradient effects, from the pitch channel.
- Engine brake and retarder, untouched so far.
