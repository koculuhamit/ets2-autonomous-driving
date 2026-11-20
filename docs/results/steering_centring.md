# Why the truck ran wide: the game centres the steering

**Date:** 2026-09-24
**Data:** the drives of docs/results/adaptive_cruise.md (acc-01, -02, -03) and
two drives with the fix (steer-01, -02); Italian autostrada at night
**Code:** `core/control/steering.py` (`STEERING_CENTRING_BY_SPEED`,
`centring_at`, `SteeringPositionController.command`), `tools/follow.py`

## The problem

In bends the truck drifted to the outside, and more so the faster it went.
The error followed the curvature (correlation −0.39 at 60 km/h, −0.68 at
85 km/h). At 85 km/h it reached 1.51 m, and the supervisor saw the truck leave
its lane. A 2.5 m truck in a 4.5 m lane has 1.0 m each side. The simulation
under-read the error by two to three times; this had been an open question
since the first drives at 60 km/h (route_planning.md).

## What it was not

- **Understeer.** Measured yaw rate over the curvature the wheels ask for is
  the same at every speed from 30 to 86 km/h, to within a percent.
- **An actuator at its limit.** The steering command never came near ±1, and
  the steering moved at 0.009 per second or less (p99).
- **A dead zone in the game's input.** The controls file has none
  (`c_steer_dz 0`, linear curve). Taking the median ratio of steering rate to
  command in closed loop *suggested* one, 40–60 times too little response to
  small commands. It was the next effect seen the wrong way.

## What it was

In a bend at 85 km/h, the command asked the angle to go one way, and the
angle drifted the other, towards zero, halving in about four seconds. Fitting
the steering rate to both the command and the angle itself, 10 Hz, per speed:

    d(effective_steering)/dt = gain(v) x command - centring(v) x steering

| speed | gain /s | centring /s | R² | from |
|---|---|---|---|---|
| 11–29 km/h | 0.24–0.28 | 0.39 | 0.75–0.79 | both sets |
| 29–40 km/h | 0.26–0.27 | 0.71–0.80 | 0.76–0.94 | both sets |
| 40–50 km/h | 0.18–0.25 | 0.62–0.80 | 0.68–0.93 | both sets |
| 50–61 km/h | 0.24 | 1.15–1.16 | 0.86–0.96 | both sets |
| 61–72 km/h | 0.22 | 1.19 | 0.82 | old steering |
| 72–86 km/h | 0.18 | 1.19 | 0.84 | old steering |

"Both sets" means the old-steering drives and the new-steering drive
agree. The two drive the actuator with different commands.

**The game pulls the steering back to centre, with a time constant under a
second above 50 km/h.** The gain matches what doublets measured before
(input_path.md), because a short, strong pulse barely sees a slow pull. The
position controller was written for a pure integrator ("the game's steering
angle *is* the integrator"). With a leaky one it settles where its push and
the game's pull balance: about **63 % of the angle it wants**, in a steady
bend at 85 km/h.

## The fix

Given the speed, the controller now asks for the rate it wants **plus what
the pull will take away**, and divides by the measured gain for that speed:

    command = sign x (8 x error + centring(v) x angle) / gain(v)

Put into the simulated plant, the same term brought the simulation into
line: the old controller at 85 km/h gives 1.56 m in simulation against 1.33 m
driven. At 60 km/h the simulation still over-reads it, 1.39 against 0.57 m.

## In the game

| drive | steering | speed | sharpest bend | worst | > 1.0 m | in bends, on average |
|---|---|---|---|---|---|---|
| acc-02 | old | 85 km/h | 300 m | 1.33 m | 4.8 % | 0.35 m outside |
| acc-03 | old | 85 km/h | 187 m | **1.51 m** | 8.9 % | 0.57 m outside |
| steer-02 | **new** | 85 km/h | **138 m** | **0.88 m** | **0 %** | 0.38 m inside |
| steer-01 | new | 40–60 km/h | 79 m | 0.74 m | 0 % | 0.31 m inside |

On a route with sharper bends than either old drive, the new steering kept
the truck within its lane throughout at 85 km/h. It is now the default.
`follow.py --old-steering` keeps the old loop for comparison.

**It now leans the other way**, 0.3–0.4 m towards the inside of bends. That
is pure pursuit cutting corners: it aims at a point already round the bend,
and until now the old steering's shortfall hid it. The simulation shows it
too, but smaller: 0.09 m on steer-01's route against 0.31 m driven. It is
the next thing to take out of the lateral loop, and a smaller one than the
problem it replaces.

## Next: a shorter lookahead

Pure pursuit aims 1.8 s ahead, a value chosen when the steering answered
four times more slowly than it does now. In simulation (`follow.py
--simulate --lookahead`), shorter is better on every route tried. At 85 km/h
on steer-02's route:

| lookahead | worst | in bends, on average |
|---|---|---|
| 1.8 s | 0.69 m | 0.19 m inside |
| 1.5 s | 0.46 m | 0.11 m inside |
| 1.2 s | 0.28 m | 0.06 m inside |
| 1.0 s | 0.19 m | 0.04 m inside |

A steering command arriving 0.05, 0.10 or 0.15 s late (`--sim-delay`) changes
none of this: the steering loop is slow enough that such delays do not
unsettle it.

### Driven: 1.2 s

la-01, 2026-09-24: 4170 m of a planned route at up to 85 km/h, with bends
down to 11 m radius at junctions, `--lookahead 1.2`. Against steer-02, the
same steering with 1.8 s. The routes differ, so the comparison is by speed:

| in bends | 1.8 s (steer-02) | **1.2 s (la-01)** |
|---|---|---|
| 75–100 km/h, towards the inside | 0.52 m, worst 0.88 | **0.18 m, worst 0.23** |
| 50–75 km/h | 0.41 m, worst 0.82 | 0.25 m, worst 0.68 |
| 0–50 km/h | 0.17 m, worst 0.33 | 0.43 m, worst 0.97 |
| steering-rate reversals at speed | 3.6 /s | 2.6 /s |

At speed the lean is a third of what it was, and the steering is calmer, not
busier. The worst error of the drive, 0.97 m, came at 21 km/h in a 16 m
radius junction turn, and the trailer cut 2.23 m there. steer-02 had no
bends that tight, so the slow row compares different roads. Tight junction
turns are the next lateral problem, the trailer's more than the tractor's.

`follow.py` now aims 1.2 s ahead by default.

## Not established

- **Not an A/B on the same road.** The drives were on different stretches.
  The old loop was never driven on steer-02's route, and the new one never on
  acc-03's.
- **Below 18 km/h** no centring was measured. The table assumes it fades
  to nothing at rest.
- **One truck, one load, one game version.** Whether the pull depends on
  the truck, the load or a game setting (steering assistance, for instance)
  is untested.
