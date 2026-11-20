# ADR-0002: Input path into the game

- **Status:** Accepted
- **Date:** 2026-09-20
- **Measurements:** [`docs/results/input_path.md`](../results/input_path.md)

## Context

The stack must feed continuous steering, throttle and brake values into the
game. Two families of approach exist:

1. **SCS SDK Input API** — the plugin registers an input device with the game
   directly. Cross-platform by construction, no kernel modules, no drivers for
   the user to install. Documented limitation: devices cannot be added or
   removed after `scs_input_init`.
2. **OS-level virtual gamepad** — `/dev/uinput` via python-evdev on Linux,
   ViGEmBus plus `vgamepad` on Windows. Proven, but platform-specific on both
   sides, and it costs the user a driver install on Windows and a udev rule
   plus `input` group membership on Linux.

The decision was deferred until it could be measured rather than argued.

## What the spike found

Built the Input API path and ran it against ETS2 1.55.0.0 on Linux.

**It works, with no user configuration at all.** The device registers as
`SCS_INPUT_DEVICE_TYPE_semantical`, which binds inputs to the game's control
mixes *by name* — no binding UI, no `controls.sii` edit. The names were
confirmed against a freshly generated `controls.sii`, where a supported mix
carries a `semantical.<name>?0` term in its expression.

**Latency is one to two frames.** Command to a change in the corresponding
`input_*` telemetry channel, 12 samples per axis:

| Axis | Median | Min | Max |
|---|---|---|---|
| steering | 27.9 ms | 10.9 | 34.1 |
| throttle | 25.2 ms | 14.5 | 37.8 |
| brake | 15.4 ms | 11.2 | 35.2 |
| clutch | 17.8 ms | 13.2 | 38.1 |

The publish interval is 16.6 ms, so the spread is a frame-boundary lottery
rather than jitter in the path itself.

**Values arrive faithfully.** Every axis appears in its `input_*` channel
exactly as commanded — `brake=0.600` reads back `input_brake=0.600`.

**The device contributes additively**, so the wheel and keyboard keep working
while the stack drives. That is the manual override path, and getting it for
free matters more than it sounds: with a virtual joystick it would have to be
arranged deliberately.

**Steering is a rate command, not a position command.** This is the one
surprise and it is worth stating plainly. The game applies its digital-steering
behaviour to the `steering` mix whenever no analog steering *axis* is connected
(`j_steer_c`), and our semantical input enters that mix at the same point the
keyboard does. So the commanded value sets a steering *rate*, the applied angle
integrates it, and the wheel stays where it was left — it does not self-centre
at a standstill. The rate is proportional to the command, around 0.75 per
second at full deflection near centre.

Closing an inner loop on it solves the problem completely:

| Target | Settling | Steady-state error | Overshoot |
|---|---|---|---|
| −0.20 | ~0.50 s | 0.0002 | none |
| +0.20 (0.40 swing) | ~0.75 s | 0.0003 | none |
| 0.00 | ~0.50 s | 0.0003 | none |

0.0003 is 0.03 % of full travel. The loop settled cleanly even though its rate
gain was set to half the true value, so it is not sensitive to that constant
being wrong.

**`abackward` is "accelerate backward", not "brake".** The game decides what it
means from the vehicle's motion; at a standstill it engages reverse. Braking
behaviour while moving forward is still to be confirmed.

## Options considered

### Option A — SCS Input API, semantical device
- Zero user configuration; nothing to install on either platform.
- One `.so`/`.dll` covers both platforms, so `InputDevice` leaves the PAL
  entirely and Phase 12 loses an item.
- Manual override works without being arranged.
- Steering needs an inner rate loop, measured above as a non-problem.

### Option B — SCS Input API, generic device
- Would likely present as an analog axis and give direct position control.
- Costs the user a one-time binding in the game's control options, and makes
  the stack depend on that binding being correct.
- Not built: Option A's inner loop already delivers 0.03 % accuracy, so this
  would buy nothing for a real setup cost.

### Option C — OS-level virtual gamepad
- True analog axis, position control, no inner loop.
- Two platform-specific implementations to write and keep in parity, a
  ViGEmBus install on Windows, and a udev rule plus group membership on Linux.
- Everything it offers over Option A is already available for the price of a
  proportional loop.

## Decision

**Option A.** Input lives inside the plugin as a semantical device. Steering is
commanded as a rate, with a position loop closed on `effective_steering`.

`InputDevice` is therefore **removed from the PAL**: there is no OS-specific
input code on either platform. The `LinuxInputDevice` and `WindowsInputDevice`
skeletons come out, and the Phase 12 checklist loses the ViGEmBus work.

Option C stays available if a future need for true position control appears,
but nothing found in this spike argues for it.

## Consequences

- No driver or permission setup for the user, on either platform. This is the
  difference between a project someone can run and one they give up on.
- Steering has an integrator in the plant. Lateral controllers output a desired
  steering *position*, and a small inner loop converts it to a rate. That loop
  is part of the actuator model Phase 3 identifies, not an afterthought.
- Steering does not self-centre. Disengaging leaves the wheel where it was, so
  "disengaged" is not the same as "straight" — the handover story has to say so.
- The watchdog and the engage flag both live in the command block; see
  `plugin/include/ets2ad/command_layout.h`.
- Because the device binds by mix name, a game update that renames a mix breaks
  input silently. `docs/compat.md` records the names in use, and the spike is
  cheap to repeat.
