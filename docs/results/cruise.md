# First autonomous speed hold

**Date:** 2026-09-21
**Game:** ETS2 1.61.1.0, Linux native
**Vehicle:** Mercedes Actros 2014, empty box trailer
**Tool:** `tools/cruise.py --speed 30 --distance 280 --duration 30`

The first time the stack held a speed on its own. Longitudinal control only:
it regulates speed and holds the steering angle where it is. It does not
follow a lane.

## Result

| | |
|---|---|
| Target | 30.0 km/h |
| 95 % of target reached | 9.5 s |
| Settled within ±0.5 km/h | 9.8 s |
| Steady-state mean error | **−0.013 km/h** |
| Steady-state RMSE | **0.101 km/h** |
| Steady-state band | 29.51 – 30.17 km/h (0.67 wide) |
| Overshoot | +0.175 km/h |
| Distance | 209 m of a 280 m budget |

0.1 km/h of RMSE is 0.03 m/s. The gradient feedforward and the PI loop between
them hold the speed to well inside what the quantised actuator suggested was
possible, because on level ground the controller stays on the throttle side,
where the response is continuous.

The throttle it settled on is worth noting: it alternates between roughly 0.14
and zero, duty-cycling across the equilibrium the taper measurement put at
0.13. The measured calibration predicted the operating point correctly.

Braking was used in 4 % of commands, almost all of it in the final stop.

## Where it exceeded its own comfort limit

The controller requested at most 1.2 m/s² and the vehicle delivered **2.23**
during the launch, in first gear.

Not a controller fault and not a guard failure: the throttle map is a single
curve taken from cruising gears, and first gear is far stronger than it
describes. The controller asked for 1.2, the feedforward chose a throttle that
would give 1.2 in a high gear, and in first gear that same throttle gave
almost twice as much. The feedback loop corrected within a second.

Two ways to fix it, neither done: carry a per-gear throttle map, since the
`gear` channel is already in the block; or rate-limit the throttle command so
the launch is gentle regardless. The second is simpler and does not depend on
the map being right, which argues for it.

Everything after the launch was inside the limit: from 12 s onwards the
acceleration stayed between −0.25 and +0.14 m/s².

## Manual override works, and the stack fights it

The road was not perfectly straight, and the operator pressed the steer key
several times to avoid leaving it. That worked -- which confirms ADR-0002's
claim from the mix expressions in real conditions: the semantical device adds
to the game's control mixes rather than replacing them, so a human can steer
while the stack drives without disengaging anything.

But the steering loop holds the wheel at zero, so every correction was
actively undone. The recording shows eight excursions, none beyond 0.053 of
wheel angle and each pulled back within a fraction of a second; our command
peaked at 0.56 and never saturated, so the loop was winning comfortably.

Mild here, wrong in principle. Someone steering to avoid a collision should be
given the wheel, not argued with. The stack knows what it commanded, so a
disagreement between the command and the resulting angle is a usable detector
-- and detecting the driver's intervention, to disengage on it, is already
on the plan for Phase 2.

## Also found: commands must stop while the game is paused

The first attempt did not move at all. The map was open, so the game was
paused; it ignored the commands, the truck stayed still, and the stuck
detector fired after 2.5 seconds -- correct behaviour, wrong diagnosis
("something is in the way").

The plugin was reporting `paused` the whole time and nothing was reading it.
Worse than the misleading message: the controller kept winding up and holding
47 % throttle against a paused game, which the game would have applied the
instant the map closed.

`cruise.py` now refuses to start while paused, and during a run releases the
controls, resets the loop and waits. The same handling belongs in
`tools/sysid.py`, which does not have it yet.

## Not covered

- **Anything above 30 km/h**, and any gradient worth the name: the site was
  flat to within 0.2°.
- **Gains.** These are the simulation's starting values, unchanged. The run
  gives no reason to move them, but a flat road at one speed is not evidence
  that they are right.
- **Lane keeping.** The truck goes straight only because the road did, and not
  even quite then.
