# System identification — first session

**Date:** 2026-09-21
**Game:** ETS2 1.61.1.0, Linux native
**Vehicle:** Mercedes Actros 2014, 4 wheels, box trailer attached, **empty**
**Site:** flat straight road, ~790 m used across six runs
**Tools:** `tools/sysid.py` to drive and record, `tools/analyse_sysid.py` to fit

Everything below comes from recordings under `data/sysid/`, so it can be
re-derived without driving again. Re-run with:

```bash
python tools/analyse_sysid.py data/sysid
```

## Brake command to deceleration

The important result of the session, because it changes what Phase 6 can do.

| Command | Deceleration | g |
|---|---|---|
| 0.02 | 0.04 m/s² | 0.00 |
| 0.05 | 0.02 m/s² | 0.00 |
| **0.06** | **1.81 m/s²** | 0.18 |
| 0.08 | 2.24 m/s² | 0.23 |
| 0.10 | 2.84 m/s² | 0.29 |
| 0.12 | 3.30 m/s² | 0.34 |
| 0.14 | 3.70 m/s² | 0.38 |
| 0.15 | 4.06 m/s² | 0.41 |
| 0.25 | 5.27 m/s² | 0.54 |
| 0.40 | 6.83 m/s² | 0.70 |
| 0.60 | 8.89 m/s² | 0.91 |
| 0.85 | 9.71 m/s² | 0.99 |

Three things matter here.

**There is a dead zone below about 0.055.** Commands of 0.02 and 0.05 produce
no braking at all -- both runs were still rolling when their sixty-second
timeout expired. A controller that ramps the brake up from zero does nothing,
then suddenly decelerates at 1.8 m/s².

**The comfortable band is about 0.02 wide.** Comfortable deceleration for a
combination vehicle is roughly 1.5–2.5 m/s², which here means commands between
about 0.055 and 0.089. Everything an ACC does in normal traffic lives in two
percent of the available command range, immediately above a dead zone. Command
resolution is therefore a real constraint, not a rounding concern, and the
first ACC implementation should expect to spend its time near a discontinuity.

**Authority far exceeds comfort.** 0.85 gives 9.71 m/s², about 1 g. The gap
between what is available for emergency braking and what is acceptable in
normal driving is roughly fourfold, which is useful -- but it also means a
factor-of-four command error is the difference between smooth and violent.

Measured empty. A loaded trailer will differ, and the dead zone in particular
may move.

## Throttle command to acceleration, by gear

| Throttle | Gear 1 | Gears 3–5 | Gears 6–7 |
|---|---|---|---|
| 0.20 | 0.73 m/s² | 0.65 | — |
| 0.35 | 1.18 | 1.05 | 1.01 |
| 0.50 | 1.59 | 1.41 | 1.21 |
| 0.70 | 2.08 | 1.58 | 1.37 |
| 1.00 | 2.55 | 1.71 | 1.50 |

Monotone in throttle, falling with gear as torque multiplication drops. Note
how much less the response varies at the top of the range than at the bottom:
between 0.70 and 1.00 the gain is roughly a third of what it is between 0.20
and 0.35.

This is why the first drive's single figure of +0.70 m/s² was called not a map:
it was the mean across a pull that ran through to gear 5, so it described the
gearbox rather than the throttle.

The automatic gearbox is the awkward part. It shifts when it chooses, the
acceleration steps at every shift, and a longitudinal controller has no notice.
Phase 6 either models the shift schedule or treats it as a disturbance to
reject; the data above is what either approach is built on.

## Steering position to road wheel angle

Stationary, so no road was needed.

Right-hand targets shown; the left-hand half is symmetric to within 0.001,
which is itself worth knowing.

| Target | Applied | Wheel 0 | Wheel 1 | Mean |
|---|---|---|---|---|
| 0.05 | 0.0491 | 1.99° | 1.95° | 1.97° |
| 0.10 | 0.0992 | 4.05° | 3.89° | 3.97° |
| 0.20 | 0.1991 | 8.29° | 7.66° | 7.98° |
| 0.30 | 0.2992 | 12.72° | 11.30° | 12.01° |
| 0.50 | 0.4991 | 22.08° | 18.21° | 20.14° |
| 0.70 | 0.6975 | 31.98° | 24.68° | 28.33° |
| 0.90 | 0.8788 | 41.43° | 30.36° | 35.90° |
| 1.00 | 0.9488 | 45.14° | 32.50° | 38.82° |

**The position loop tracks to within 0.0025 up to ±0.7** -- 0.2992 for 0.30,
0.4991 for 0.50, 0.6975 for 0.70 -- and falls short at the extremes, reaching
0.879 for 0.90 and 0.949 for 1.00. That is not the loop failing but the hold
time: the steering rate drops near the end stops, and the 2.5 seconds allowed
per step is not enough to get there. The measurement is of the loop, not the
mechanism, and a longer hold would close it.

Because full lock was not reached, **38.8° is a lower bound on the maximum
wheel angle**, not the value.

**Ackermann geometry is clearly present.** The inner and outer wheels differ by
by up to 12.65° at the largest angle reached. A bicycle model's single wheel angle is therefore an
average of the two, not either one, and off-tracking calculations in Phase 7
need the real pair.

**The inner wheel's ratio is progressive** -- 0.112 rotations per unit near
centre to 0.132 at the largest angle reached -- but that is Ackermann geometry,
not the steering linkage. The *mean* of the pair, which is what a bicycle model
wants, is close to linear: 40.1 to 40.9 degrees per unit across the whole
range.

One trap in reading this table: the command axis for any derived map must be
the **applied** column, not the target. They agree to 0.0025 up to +/-0.7 and
diverge at the top, where 1.00 requested produced 0.9488. Indexing a steering
map by the request makes it appear to lose sensitivity at full lock when it
gains it -- a quiet 3 % angle error at the extremes, caught later by a test in
`core/control/lateral.py`.

## Engine braking, and the gap it does not fill

Measured because the service brake cannot decelerate gently and this truck has
no retarder. Each run coasts first and then engages, so the comparison is
against the same road at the same speed rather than a figure from another
afternoon.

| Run | Phase | Speed | Deceleration |
|---|---|---|---|
| 1 | coast | 45.5 → 41.4 km/h | 0.244 m/s² |
| 1 | engine brake | 41.3 → 20.7 km/h | 1.373 m/s² |
| 2 | coast | 46.1 → 42.4 km/h | 0.218 m/s² |
| 2 | engine brake | 42.4 → 0 km/h | 2.142 m/s² |

The two engine-brake figures disagree because both span gear changes, and a
downshift while engine braking is a large transient -- one 10 km/h band fits at
6.0 m/s². Restricting to stretches where the gear is constant, and trimming a
quarter-second from each end:

| Gear | Speed | RPM | Deceleration |
|---|---|---|---|
| 6 | 21.5–27.7 km/h | 1571 | 1.843 m/s² |
| 7 | 28.6–40.7 km/h | 1704 | 2.093 m/s² |

**Engine braking gives about 2.0 m/s² in steady gear**, which is inside the
comfortable band. It is binary: on or off, with no modulation, and this truck's
`motor_brake` channel is a boolean rather than a level.

### The deceleration palette

Putting everything measured so far in one place, because the shape of it is the
finding:

| Means | Deceleration |
|---|---|
| Lift-off (throttle released, in gear) | 0.22 m/s² |
| Service brake, command ≤ 0.05 | ~0 (dead zone) |
| **Engine brake** | **~1.97 m/s²** |
| **Service brake, command 0.06** | **1.81 m/s²** |
| Service brake, command 0.15 | 4.06 m/s² |
| Service brake, command 0.85 | 9.71 m/s² |

The engine brake was measured in the hope that it would sit *below* the service
brake's floor. It does not -- it lands within 0.2 m/s² of it. So it duplicates
the smallest available braking rather than extending the range downwards, and
**the gap between 0.22 and 1.8 m/s² remains uncommandable by either.**

That is an awkward range to be missing, because it is where an adaptive cruise
control holding a steady gap spends most of its time.

### Partial throttle does not fill it either

The obvious candidate was partial throttle: at any speed and gear there is a
throttle that exactly holds the speed, so commanding below it should decelerate
gently and continuously. Measured by holding 50 km/h in tenth gear and stepping
the throttle down. The road sloped away during the last two steps, so these are
corrected for gradient using the validated `-g·sin(pitch)` term:

| Throttle | Measured | Pitch | Gravity term | Corrected |
|---|---|---|---|---|
| 0.30 | +0.150 | −0.18° | +0.031 | +0.118 m/s² |
| 0.25 | +0.189 | −0.11° | +0.018 | +0.171 |
| 0.20 | +0.112 | −0.10° | +0.017 | +0.095 |
| 0.15 | +0.032 | −0.04° | +0.007 | +0.024 |
| 0.10 | +0.059 | −1.15° | +0.196 | **−0.137** |
| 0.05 | +0.566 | −4.68° | +0.800 | **−0.234** |

Equilibrium is around 0.13. Below it the truck does decelerate -- and reaches
**0.234 m/s² at most**, which is the lift-off figure. Partial throttle spans
zero to lift-off and no further.

That is not a limitation of the measurement; it is arithmetic. Throttle can
reduce tractive force to zero but cannot make it negative, so the greatest
deceleration obtainable by lifting off *is* the total resistance at that speed
-- 0.22 m/s² here. Partial throttle only interpolates between holding speed and
that. No amount of better data changes it.

**So the gap between 0.22 and 1.8 m/s² is genuinely uncommandable.** Three
mechanisms were available and all three land outside it: lift-off and partial
throttle below, the service brake and the engine brake above. There is nothing
in between, and no fourth mechanism on this vehicle.

What remains for Phase 6, then:

1. **Plan earlier.** Use the 0.22 m/s² over a longer distance instead of
   needing 0.5 over a shorter one. This is how a comfortable ACC behaves
   anyway; it fails only when a lead vehicle does something sudden, which is
   when 1.8 m/s² is the right answer regardless.
2. **Modulate in time, not amplitude.** Pulse between lift-off and the brake to
   synthesise an average. Cheap, and it puts a square wave through a trailer;
   a last resort, and the reason to get the planning right.

The design consequence is that the longitudinal controller cannot be a
straightforward regulator on gap error. Its output is effectively quantised --
coast, or brake at 1.8 and up -- so it has to be predictive about when to start
rather than proportional about how hard to push.

## Coordinate conventions, settled

Three conventions were open. Establishing them needed no extra driving -- the
recordings already had the evidence, which is the return on recording at all.

**Forward acceleration is `-linear_acceleration.z`.** The vehicle-frame z axis
runs backwards. Checked against the derivative of speed across three
independent manoeuvres:

| Manoeuvre | Slope of dv/dt against `linear_acceleration.z` | Correlation |
|---|---|---|
| Coasting | −0.98 | −0.996 |
| Accelerating | −0.99 | −0.986 |
| Braking | −0.80 | −0.944 |

The weaker braking figure is suspension pitch during deceleration, not a
different convention. Reading the channel directly gives the right magnitude
with the wrong sign -- a controller built on it would accelerate to brake.

**Forward direction is `(−sin(heading), −cos(heading))` in world `(x, z)`.**
Not `(sin, cos)`, which is exactly backwards. Checked against the actual
direction of travel over five runs:

| Run | Heading range | Mean alignment |
|---|---|---|
| throttle-sweep | 0–360° | 1.00000 |
| brake-sweep | 270–272° | 1.00000 |
| brake-sweep | 179–184° | 0.99997 |
| brake-sweep | 91–92° | 1.00000 |
| coastdown | 270° | 1.00000 |

All four quadrants, worst individual sample 0.999.

**Negative pitch is downhill.** Fitting lift-off acceleration against
`sin(pitch)` over 1282 samples gives

```
a = −8.33 · sin(pitch) − 0.244    R² = 0.920
```

The slope has the right sign and 85 % of g's magnitude. The constant term,
−0.244 m/s², is drag plus rolling resistance plus engine braking on the flat.

All three are implemented in `core/bridge/convert.py`, with the measurements
named in the docstrings so they can be repeated rather than trusted.

**The hitch angle sign is still open.** The magnitude tracks manoeuvres and the
wrap behaves, but which sign means which way the trailer is folded has not been
checked against a known manoeuvre. Phase 8 must establish it: a reversed sign
there is a controller that jackknifes on purpose.

## Coast-down: attempted, unusable

Ran one lift-off deceleration from 46 km/h. It cannot be fitted, for two
reasons that are worth recording so the next attempt is set up differently.

**The road was not flat.** Pitch ranged from −5.26° to +0.16°, contributing up
to 0.90 m/s² of gravitational acceleration against a mean deceleration of
0.035 m/s² -- roughly twenty-six times the quantity being measured. Correcting
for the gradient afterwards did not rescue it: neither sign convention produced
a physical fit, with R² of 0.16 and 0.06.

**The speed range was too narrow.** 46.2 down to 35.0 km/h in 22 seconds.
Separating a constant term from one growing with the square of speed needs
perhaps 40 km/h of range; this had 11.

`tools/analyse_sysid.py` now checks both conditions and reports the reason
instead of a coefficient. A usable coast-down needs a verified flat stretch --
pitch within a few tenths of a degree throughout -- and a road whose speed
limit allows starting well above 50 km/h.

That said, the failure produced the more useful finding: **at these speeds the
gradient term dwarfs the drag term.** A 5° grade is 0.86 m/s²; everything else
combined is about 0.24 m/s². Phase 6's feedforward should start with gradient,
and aerodynamic drag can wait for the road that can measure it.

## Retarder: not fitted

`retarder_step_count` reads 0 on this truck, so there is no retarder to
command. Given the service brake's dead zone, engine braking is the remaining
candidate for gentle deceleration and has not been tried -- the input device
does not currently register a `motorbrake` input.

## Not measured

- **A usable coast-down.** Attempted and rejected; see above. The
  aerodynamic and rolling-resistance split remains unknown.
- **Anything loaded.** Every number here is with an empty trailer.
- **Gradient effects on tractive effort.** The pitch channel is now
  validated against gravity, but how much acceleration a given throttle loses
  on a grade has not been measured.
- **Deceleration below 1.8 m/s².** Neither the service brake nor the engine
  brake reaches it. Partial throttle is the untested candidate; see above.
- **Engine braking below top gear**, and its behaviour through a downshift,
  which produced a 6 m/s² transient in one band.
- **Steering dynamics at speed.** The steering map is stationary. Yaw-rate
  response, and its speed dependence, is what the lateral controller actually
  needs and none of it has been measured.
- **Brake response time.** Deceleration was fitted over the whole application;
  the air-brake build-up delay was not separated out.

## What went wrong, and what it cost

**A manoeuvre drove the truck backwards for a full run.** Every "settle" phase
held `brake=1.0` to keep the truck still. `abackward` is not a brake at a
standstill -- it is a request to reverse -- so the settle phase selected
reverse and the throttle burst that followed accelerated backwards. Fifty-two
seconds of data, every gear negative. The finding had already been written down
in `longitudinal_first_drive.md`; writing it down was not enough to stop it
being designed around wrongly.

Fixed by releasing the controls at a standstill and letting the game's own hold
brake do the work, and by a `WrongDirection` guard that aborts rather than
recording a run nobody will notice is inverted.

**A run pushed against an obstacle at 70 % throttle for seventeen seconds** and
put 0.6 points of damage on the chassis and 0.45 on the cabin. The harness had
a speed ceiling, a run deadline and a direction guard, and not one of them
fires when the truck simply stops moving: zero speed is inside every limit.

Fixed with a stuck detector -- throttle applied, no movement, 2.5 seconds, stop
-- and a `--distance` budget, because the operator knows how much clear road
there is and the harness does not.

Both failures were in the experiment design rather than the stack, and both
were the kind that produce data instead of errors. That is the argument for the
guards: not that they prevent damage, though the second one would have, but
that they refuse to hand back a run that looks fine.

**Then the truck was crashed a second time, harder, with the guard in place.**
An engine-brake run was started with `--distance 500` after the game had been
restarted and the truck repositioned. The road actually ahead was about 120 m.
It hit something at 30.6 km/h, taking the chassis from 3.8 % to 13.5 % and the
cabin from 9.4 % to 17.2 %.

The stuck detector did fire -- on the *next* attempt, stopping it in 2.5
seconds rather than seventeen -- but it cannot prevent a collision, because the
truck is moving normally right up to the impact. Nothing in the harness can see
the road.

This one was not a design gap. The `--distance` budget existed precisely for
this and was filled in from a figure that had gone stale across a restart and a
reposition. So the flag is now **required** for any manoeuvre that drives, with
a message that says what it must describe: the road ahead *now*, not before the
last run. A default of `None` is what allowed a guess to be passed off as a
measurement.

The engine-brake measurement itself produced no data: the run ended during its
acceleration phase.

**Then the worst one: a guard prevented the stop it had just ordered.** The
throttle-taper run hit its distance budget, which raised
`DistanceBudgetExceeded`; the handler called `stop()`; and `stop()` immediately
raised the same exception again, because the budget was of course still
overspent. The stop never ran.

The recording ends with the truck at **54.8 km/h**, and the harness then exited
having commanded nothing. The plugin's watchdog released the controls 200 ms
later and the truck was left coasting until the operator took over manually.

Nothing was hit as a result -- the damage recorded around that time was the
operator's, by hand, and is not attributed here. But the outcome does not
depend on what happened next: a moving vehicle was handed back uncommanded by
the mechanism whose entire job was to bring it to rest. In both collisions
earlier in the session the vehicle was at least under control. Here it was
under none, at speed, because a safety check fired at the moment it was
supposed to stand aside.

The fix is `SafetyMonitor.suspend()`, used around the braking path: the budget
is already overspent and the truck is about to be stationary with the brake on,
so distance and stuck both trip on states a stop necessarily passes through.

The reason it was not caught earlier is more instructive than the bug. These
guards lived in `tools/sysid.py` and had no tests, because a tool is not
something one writes tests for. But stopping a moving vehicle is not experiment
plumbing. They are now `core/control/limits.py`, with an injectable clock and
twenty-six tests that care as much about when a limit must *not* fire as when
it must.
