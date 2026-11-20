# First autonomous route following

**Date:** 2026-09-22
**Game:** ETS2 1.61.1.0, Linux native
**Vehicle:** Mercedes Actros 2014, empty box trailer
**Route:** `data/routes/teach-01` — a 976 m loop driven by hand, closing to 3 m
**Tool:** `tools/follow.py --speed 40 --distance 300 --start 0`

The first run in which the stack steered. Teach-and-repeat: the path is a
recorded manual drive, so the truck follows the line it was shown. No map, no
perception.

## Result

300 m of the loop, 415 m of arc length, 47 s.

| | |
|---|---|
| Lateral error, max | **0.702 m** |
| Lateral error, RMS | **0.220 m** |
| Over 0.5 m | 8.1 % of the run |
| Heading error, max / RMS | 8.30° / 2.16° |
| Lateral acceleration, max | 2.81 m/s² |
| Wheel angle used, max | 12.0° |
| Speed error, RMS (after launch) | 3.42 km/h |
| Damage | none |

The worst point is s = 202 m, a 20 m radius corner taken at 24.4 km/h.

## The simulation was right

Predicted before driving, from a bicycle model behind the measured steering
actuator, the measured throttle and brake maps, and the gradient recorded on
the taught drive:

| | Simulated | Measured |
|---|---|---|
| Lateral error, max | 0.86 m | 0.702 m |
| Lateral error, RMS | 0.24 m | 0.220 m |

Within 20 %, and pessimistic, which is the direction to be wrong in. That is
the return on Phase 3: the parameters in that model are measurements, not
estimates, and the model is worth something because of it.

## What the lookahead scan bought

The first simulation of this route gave 1.57 m of error, concentrated in the
tightest seventh where the radius falls to 10 m. Shortening the lookahead fixed
it in simulation — 0.18 m at 0.8 s and a 4 m floor — and that is precisely the
trap the longitudinal gains already demonstrated: a noise-free model rewards
ever-shorter lookaheads without limit, and 4 m is shorter than the wheelbase,
putting the aim point at the front axle of a vehicle towing a trailer.

The actual problem was that a fixed *time* ahead is too far into a tight
corner. Twenty metres of lookahead on a 10 m radius puts the aim point more
than halfway round, and the controller drives the chord. So the lookahead now
scans its window for the sharpest curvature and caps itself at
`gain / curvature`, never below the floor. Scanning rather than reading the
curvature underfoot is the part that matters — otherwise the lookahead only
shortens once the corner has been entered, too late to have aimed anywhere
useful.

1.57 m → 0.86 m in simulation, and the worst point moved off the tight corner
entirely.

## Speed

Held at 40.0 km/h on the open sections. Excluding the launch — 15 s from rest
to 38 km/h, which is the vehicle rather than the controller — speed error is
3.42 km/h RMS with a −1.27 mean. The −12.5 km/h outlier is a corner exit: the
profile returns to 40 as soon as the curvature relaxes, and the truck needs
time to get there. Nothing to fix in the controller; it could be smoothed by
planning the acceleration out of a corner the way the deceleration into one
already is.

## A broken analysis, worth recording

The first pass over this recording reported 227 m of lateral error and 180° of
heading error. The run was fine; the analysis was not. It called
`path.nearest()` without seeding `near`, so on a closed loop the first query
locked onto the wrong branch and every subsequent one followed it.

That parameter exists for exactly this, and there is a test for it. The
controller passes its own progress correctly; the throwaway analysis script did
not, and a throwaway script is where this kind of thing survives.

## The tractor's error is not the vehicle's error

The controller steers the tractor, and the tractor is the shorter half. Reading
the trailer's position out of the same recording:

| Radius | Tractor | Trailer | Off-tracking |
|---|---|---|---|
| > 200 m | +0.05 m | −0.18 m | −0.21 m |
| 50–200 m | +0.04 m | −0.49 m | −0.79 m |
| 20–50 m | −0.43 m | −1.07 m | −0.88 m |
| < 20 m | −0.65 m | −1.50 m | −1.09 m |

Worst trailer error **2.12 m**, against 0.70 m for the tractor. Worst
off-tracking 2.44 m. Maximum hitch angle 28.4°.

Two metres is most of a lane. So the number that matters for lane keeping is
the trailer's, and everything reported above it is optimistic.

The geometry needed to predict this is in the telemetry: the hook position and
the wheel positions give an effective trailer length of **8.138 m** for this
combination, and the steady-state offset on a curve is
`R - sqrt(R^2 - L^2)`.

Checked against the run, with the curvature taken at the *trailer's* own
position rather than the tractor's, and the sign being `sign(curvature)` --
the trailer cuts to the inside:

| Radius | Measured | Predicted |
|---|---|---|
| < 20 m | −1.50 m | −1.87 m |
| 20–50 m | −1.07 m | −1.32 m |
| 50–200 m | −0.49 m | −0.36 m |
| > 200 m | −0.18 m | −0.01 m |

Correlation 0.56, and a fit of `measured = 0.69 x predicted - 0.18`. The 0.69
is the transient: off-tracking is a steady-state quantity and these corners are
not long enough to develop it. The −0.18 is a constant offset present even on
straights, unexplained.

**Nothing steers on this yet, deliberately.** A correlation of 0.56 is enough
to say the model has the right shape and not enough to justify shifting the
aim point by the number it produces; a wrong compensation makes the trailer
worse, not better. Establishing it properly needs a constant-radius run long
enough for the off-tracking to settle, which is a measurement rather than a
guess. `TrailerGeometry` is in place, measured from telemetry, and
`follow.py` now reports the trailer's error alongside the tractor's on every
run.

## The whole loop

Driven in two runs, 2026-09-22, in the order the truck happened to be sitting
in: s = 425 to the end, then s = 0 to 425. 967 m of the 976, at a 40 km/h
ceiling, no damage and no intervention.

| | s = 425 … 968 | s = 0 … 424 |
|---|---|---|
| distance | 543 m in 63 s | 424 m in 49 s |
| tractor, worst | 0.91 m at s = 968 | 0.70 m at s = 203 |
| tractor, RMS | 0.287 m | 0.219 m |
| trailer, worst | 2.22 m at s = 707 | 2.07 m at s = 217 |
| trailer, RMS | 0.719 m | 0.580 m |
| hitch, worst | 25.2° | 26.0° |
| lateral acceleration, realized | 2.58 m/s² | 2.48 m/s² |

The second half is the harder one: every corner tighter than 25 m is in it,
including the 10 m radius at s = 970 that had only ever been simulated. It is
also where the worst error is, 0.91 m in that corner at 18 km/h -- and the
simulation said 0.83. Which is the second useful thing this run said:

**the simulation is within about 15%, in either direction.** On the first
300 m it predicted 0.86 m and the truck held 0.70; here it predicted 0.83 and
the truck used 0.91. It is a good enough tuning surface and not a safety
argument, and nothing should be signed off on it that has not also been driven.

### The speed profile holds its limit, and the first measurement of it did not

First reported here as a 30% overshoot: 3.24 m/s² against a planned 2.5. That
number was wrong, and wrong in an instructive way.

It was computed as `v² · κ`, taking `v` from the telemetry and `κ` from the
*path* at the truck's projected position. That mixes a measurement with a plan.
The truck does not drive the path, it drives its own trajectory, and inside a
corner that trajectory is *wider* than the reference -- the controller lags the
turn-in. So the product describes a vehicle that does not exist: the speed of
the real one, the geometry of the planned one.

What the vehicle actually pulls is `v · ψ̇`, from the yaw rate, and that peaks
at **2.58 m/s²** on the harder half and 2.48 on the other. Cross-checked by
deriving the curvature of the driven trajectory from its own position trace:
the two agree to 1.4% with a correlation of 0.976, from entirely separate
telemetry.

| | s = 425 … 968 | s = 0 … 424 |
|---|---|---|
| planned peak | 2.54 m/s² | 2.51 m/s² |
| realized peak | **2.58 m/s²** | **2.48 m/s²** |
| frames over 2.5 | 0.9% | 0.0% |

So the profile is respected, within about 3%, and there is nothing here to fix.

The margin is thinner than it looks, though, and worth knowing about. Entering
a corner the truck is up to 14% above the planned speed, which is 30% in the
`v²` that matters, because the uncommandable deceleration band between 0.22 and
1.8 m/s² means it cannot shed small amounts. It stays inside the limit because
it simultaneously drives a wider arc than planned -- curvature 0.89 of the
path's through corners. **Two errors cancelling is not a design.** Anything that
makes the truck track the path more tightly, including the off-tracking
compensation that is still owed, will remove the second error and expose the
first.

The lesson is worth more than the number: a metric assembled from one measured
quantity and one planned quantity measures neither, and it reads as evidence
because half of it came from the vehicle. It survived a write-up and a commit.

## Two bugs the run found, both in what it reported

Neither touched the driving. Both made the report wrong, which is its own kind
of failure: the numbers above are the argument, and a report that flatters is
not worth having.

**The distance budget was measuring the wrong thing.** `SafetyMonitor` tracked
straight-line displacement from where the run began, not road covered. The
543 m run reported 311 m, so a 551 m budget was never close to firing. On a
closed loop it is worse than imprecise -- come round far enough and the figure
*falls*, and the one guard that knows the road runs out never fires at all.
This is the guard that exists because a stale distance figure once cost ten
points of chassis. It now accumulates step by step, and the closing run stopped
itself at exactly 425 m of a 425 m budget, which is what it was always supposed
to do.

**The trailer was reported 227 m off the path.** At the start of the second
run the trailer is behind s = 0, and on a closed loop behind the start is the
far end. An unseeded nearest-point search answers with the other side of the
loop, confidently. Same mistake as the 227 m in an earlier throwaway analysis,
which is the tell: it is not a slip, it is a missing concept. `TrailerTracker`
now owns it -- seeded from the tractor, silent until the trailer is on the
route -- and it is in `core` with tests rather than inline in a tool.

## Not covered

- **Speeds above 40 km/h**, and the lookahead settings at them. Everything here
  is simulation-tuned; the real run gives no reason to change it, but one route
  at one speed is not evidence that it is right.
- **The thin lateral-acceleration margin.** Explained, not designed away: the
  truck is fast into corners and wide through them, and the two cancel.
- **Off-tracking compensation.** Measured on two runs now; still not steered on.
- **Repeatability.** One lap, in two runs, one direction.
