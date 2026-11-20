# Input path — measurements

**Date:** 2026-09-20
**Game:** ETS2 1.61.1.0, Linux native, X11, 1920×1080 windowed, uncapped FPS
**Plugin:** SCS SDK 1.14, semantical input device, 12 inputs
**Decision this feeds:** [ADR-0002](../adr/0002-input-path.md)

Everything below was run with the truck stationary, engine on, parking brake
engaged.

## Does input arrive, and unaltered

Commanding one axis at a time and reading the raw `input_*` telemetry channels,
which report what the game received before applying any vehicle logic:

| Commanded | `input_steering` | `input_throttle` | `input_brake` | `input_clutch` |
|---|---|---|---|---|
| neutral | 0.000 | 0.000 | 0.000 | 0.000 |
| throttle 0.6 | 0.000 | **0.600** | 0.000 | 0.000 |
| brake 0.6 | 0.000 | 0.000 | **0.600** | 0.000 |
| clutch 0.6 | 0.000 | 0.000 | 0.000 | **0.600** |

Exact, and on the right axis. The plugin's index mapping is correct.

This table also corrected a wrong conclusion. Reading `effective_*` instead had
suggested the brake command was landing on the throttle, because `abackward` at
a standstill makes the game engage reverse — `effective_throttle` rises and
`effective_brake` falls. The raw channels separate what was *received* from
what the game decided to *do* with it, and only the first is a question about
our code.

## Latency

Command to first change in the corresponding `input_*` channel, 12 steps each:

| Axis | Median | Min | Max |
|---|---|---|---|
| steering | 27.9 ms | 10.9 | 34.1 |
| throttle | 25.2 ms | 14.5 | 37.8 |
| brake | 15.4 ms | 11.2 | 35.2 |
| clutch | 17.8 ms | 13.2 | 38.1 |

Publish interval over the same runs: 16.6 ms median. The 11–38 ms spread is
therefore a frame-boundary lottery — a command lands somewhere within a frame
and waits for the next one — not jitter in the input path.

Command to `effective_steering` and to the front wheel angle, 14 steps of ±0.5:
median **20.9 ms** for both, min 11.1, max 29.8. The wheels move on the same
frame the input is applied; there is no separate steering actuator delay to
model on top of the input path.

### A threshold that measured the wrong thing

The first attempt reported 130–640 ms with 10 of 12 steps timing out. The
threshold was set to 10 % of the commanded step, which works for a plant whose
output follows its input — and steering does not. It integrates, so the
threshold measured how long the wheel took to *travel* 10 % of full command: a
slew time wearing a latency's clothes. Alternating step signs also broke the
unsigned comparison, hence the timeouts.

Latency is the delay before a command has *any* effect, so the threshold
belongs at the noise floor. At 1e-4 the numbers above fell out immediately and
consistently. Recorded because the failure was quiet and plausible: 130 ms is
exactly the sort of figure one might accept and design around.

## Steering behaves as a rate command

Commanding a value and measuring the rate of change of `effective_steering`
over a fixed hold:

| Command | Δ per second | ratio |
|---|---|---|
| 0.05 | 0.0200 | 0.400 |
| 1.00 | 0.3866 | 0.387 |
| 0.05 | 0.0200 | 0.400 |
| 1.00 | 0.3733 | 0.373 |
| 0.25 | 0.0967 | 0.387 |

Proportional, so the analog value is being used — as a rate. Near centre the
rate is higher, around 0.75 per second at full command; it falls off towards
the end stops. With zero command the angle **holds** rather than returning to
centre: watched for six seconds at −0.687, unchanged.

The cause is in the game's own control mix. `mix steering` is
`dsteering - <joystick term> - semantical.steering`, and the game applies its
digital-steering behaviour to that mix whenever no analog steering axis is
connected. Our input enters at the same point as the keyboard, so it inherits
keyboard semantics.

The sign is inverted: the mix *subtracts* `semantical.steering`.

## Closing the loop on steering

A proportional loop on `effective_steering`, commanding rate:

| Target | Settling | Steady-state error | Overshoot |
|---|---|---|---|
| −0.20 | ~0.50 s | 0.0002 | none |
| +0.20 (0.40 swing) | ~0.75 s | 0.0003 | none |
| 0.00 | ~0.50 s | 0.0003 | none |

0.0003 is 0.03 % of full travel. The loop used a rate gain of 0.387 — half the
true value near centre — and still settled without overshoot, so the result
does not depend on that constant being identified accurately.

Full-lock traverse, −1.0 to 0.0 under saturated command: about 1.3 s.

## Watchdog

Engaged with `throttle=0.6`, then stopped publishing entirely, simulating a
crashed stack:

```
command flowing  : input_throttle = 0.600
publishing stopped
  -> RELEASED     t = 132 ms, input_throttle = 0.000
commands resumed : input_throttle = 0.600
```

It released, and recovered when commands came back.

**But 132 ms was not the configured 200 ms**, and the gap matters. The watchdog
counted input callbacks, which the game runs once per *rendered* frame — so its
duration tracked the frame rate. Twelve frames is 200 ms at 60 FPS, 50 ms at
240, 400 ms at 30. A safety timeout that shortens when the GPU is idle and
stretches exactly when the game is struggling is backwards.

The reasoning behind the frame count was that the two processes share no clock.
True for comparing timestamps, irrelevant here: measuring elapsed time since
the last observed change happens entirely inside the plugin. It now uses
`std::chrono::steady_clock` and a 200 ms threshold, independent of frame rate.

Re-measured after the change, five runs: median **226 ms**, range 213–240.
The release itself is at 200 ms; the remainder is one telemetry frame plus the
observer's own polling, and it no longer moves with the frame rate.

## Not yet established

- **`abackward` as a brake.** At a standstill the game reads it as reverse.
  Whether it brakes while moving forward, and how it interacts with the
  automatic gearbox, needs a moving test.
- **Buttons.** Registered, never exercised.
- **Behaviour at speed.** Every number here was taken stationary. Steering
  rate, in particular, may well depend on speed — the game models steering
  assistance, and Phase 3 should not assume these constants carry over.

## Steering rate while driving (2026-09-23)

The 0.75 per second above was measured with the truck standing still and the
command at full. That is not what a controller does: it sends small commands,
at road speed. Measured the way it is used -- steering doublets of known size
added while the controller holds the lane (`tools/follow.py --excite`), each
amplitude up for 0.3 s and down for 0.3 s, every 5 s, on an autostrada from
standstill to 60 km/h:

| Speed | rate per unit command |
|---|---|
| 8 km/h | 0.25 /s |
| 23–33 km/h | 0.24 /s |
| 43 km/h | 0.21 /s |
| 51–60 km/h | 0.19 /s |

At 50--60 km/h the gain is the same for every amplitude tried -- 0.194, 0.184,
0.195 and 0.187 for 0.05, 0.1, 0.2 and 0.3 -- so over the range a controller
uses the actuator is linear, and it slows with speed. That fits what the
section above found: the game treats the stack's steering like a keyboard's,
and keyboard steering slows down as the truck speeds up. A small command at a
standstill, sent while checking the input path the same day, gave about
0.4 /s; the 0.75 is the full-command, stationary figure.

The alignment took two tries. Taking the first large step in the command as
the start of the first doublet found the middle of it instead -- the step from
+A to −A is twice as big as the one from 0 to +A -- and every window was 0.3 s
late. Sliding the whole known doublet train over the command and taking the
best fit put it at 0 ms, with every edge where it should be.

`core/control/steering.py` has the table as `STEERING_RATE_BY_SPEED`. The
simulation now uses it as its plant. Whether the steering loop should use it
too is a separate question; see `docs/results/route_planning.md`.
