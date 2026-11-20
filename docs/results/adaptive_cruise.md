# Adaptive cruise

**Date:** 2026-09-24
**Code:** `core/control/following.py`, `core/control/acc.py`,
`core/perception/service.py`, `core/perception/tracking.py`,
`tools/follow.py --acc`, `tools/track_lead.py`
**Status:** simulated, run in the shadow of two hand drives, and driven three
times in the game; on the third it followed a slower vehicle for a minute
(below).

## What it does

Following is one more cap on the speed. The route's corner speeds and the
road's limit already cap it:

    target = min(route profile, road limit, following speed)
    wanted gap = 12 m + 2.0 s x own speed
    following speed = lead speed + (gap - wanted gap) / 4 s

The speed controller already handles the actuator, which cannot decelerate
between 0.22 and 1.81 m/s². When the gap closes faster than 2.5 m/s² can
stop, following asks for an **urgent** deceleration, up to 7 m/s², and the
caller commands it directly. It aims at the 12 m standstill gap while
there is room, and at 6 m once inside it. (Until the second stop test below
these were 8 and 4 m; the simulation tables in the next section are from
then.)

Perception runs in its own process: capture, the fine-tuned detector and
the tracker, at 30 Hz, 17 ms per frame. The control loop picks the lead
from the newest report, against the route's lane seen from the driver's
seat. **A report older than 0.5 s, or one taken with another window in
front of the game, stops the truck.** Driving on without sight of the road
is not an option it has.

## In simulation

`tests/unit/test_following.py` and `tests/unit/test_acc.py`. The truck uses
the measured actuator with a 0.3 s lag. Perception reports 0.1 s late at
20 Hz. The end-to-end cases run the real tracker on boxes drawn by the
camera model, with a pixel of noise.

| scenario | closest gap | hardest braking |
|---|---|---|
| catching up, 90 → 70 km/h, from 120 m | 46.8 m | 1.4 m/s² |
| lead brakes 4 m/s² to a stop, from 80 km/h | 6.9 m | 4.1 m/s² |
| lead brakes 7 m/s² to a stop, from 80 km/h | 6.7 m | 7.0 m/s² |
| car cuts in 12 m ahead, 70 vs 90 km/h | 6.6 m | 5.8 m/s² |
| stopped car first seen 90 m ahead, 80 km/h | 7.0 m | 3.8 m/s² |

**How far ahead a stopped car must be seen**, through the real tracker, five
seeds:

| truck speed | seen at | closest |
|---|---|---|
| 80 km/h | 80 m | 5.0 m |
| 80 km/h | 70 m | 2.3–2.9 m |
| 80 km/h | 60 m | **contact** |
| 50 km/h | 40 m | 4.6 m |
| 50 km/h | 30 m | 2.7 m |

At 80 km/h a stopped vehicle has to be seen by about 75 m. A car there is
15 px tall, where the detector finds about half of them in one frame. This
is the margin to watch when the truck first follows in the game. Waiting
less before a closing speed counts, or before braking hard, barely moves
it (70 m: 2.3 → 2.9 m). What limits it is 0.4 s of delay and 7 m/s² of
braking.

## In the shadow of hand drives

`tools/track_lead.py` runs the whole chain on a recording: detector,
tracker, lane from the map, lead, following decision. It uses the speed
the driver actually drove. The question it answers: **when would it have
braked hard where the driver did not?** Each such case is a phantom stop.

| | motorway, 5 min | city, 5 min |
|---|---|---|
| first version | **7**, driver braked in none | **6** where the driver did not |
| now | 1 (a single image, 120 m out) | 1 (ambiguous, below) |

Each fix came from looking at a phantom stop:

1. **Closing speed from box growth alone.** A car coming out from behind
   another widened its box from 49 to 91 px in 0.3 s, at a steady 22.7 m.
   Now growth and the ground-distance trend are each a median of pairwise
   slopes. They must agree on direction, and the smaller counts. A track
   gives no closing speed until it has 5 sightings over 0.25 s.
2. **Oncoming cars.** Cars coming the other way through a junction read as
   leads closing at 58–78 km/h. A lead coming towards the truck faster than
   4 m/s, or 30 % of the truck's speed, is not followed. The first threshold,
   2 m/s, declared a *stopped* car oncoming. It was ignored, and hit, in
   simulation.
3. **Crossing cars.** A car crossing a junction in front is not a lead. What
   counts is moving across *the lane*, not across the image. In a bend the
   whole scene sweeps sideways, and measuring against the camera dropped a
   third of the motorway leads. Crossing is only judged within 40 m: further
   out, distance error alone moves a bend's lane under the lead.
4. **Boxes the dashboard cuts.** A box whose foot is within 12 px of the
   bottom of the view has no ground distance. A crossing car's box doubled
   in width there. A track no longer measured for 0.3 s is not followed.
5. **Lag.** Distance was a one-second median, half a second old: 2.5 m too
   far for a car closing at 5 m/s. Lateral position was lagging too, 2 m
   behind the lane at 60 m in a bend. Distance is now carried forward to
   the present at the closing speed, and lateral is the median of the last
   three sightings.

The shadow tool does not apply the 0.1 s wait before hard braking that the
live cruise does. The motorway's remaining one-image request would not
reach the pedals.

On the motorway drive, with the truck above 54 km/h, the lead's own speed
has a median of 80 km/h. It is below 10 km/h in 2 of 647 images; the
first version had 7.7 %.

## In the game

Three drives on 2026-09-24, Italian autostrada at night, each on a route
planned from the map with no traffic lights, supervised, each to the end
of its route. Traffic there keeps to the 80 km/h limit or above. At 60 km/h,
then, nothing ahead is slower. The second and third drives therefore used
`--above-road-limit` at 85 km/h, a test-only flag: the first time steering
was driven past 60.

| | 1: 60 km/h | 2: 85 km/h | 3: 85 km/h |
|---|---|---|---|
| driven | 1992 m | 1988 m | 3103 m |
| perception report age, median / worst | 66 / 116 ms | 65 / 134 ms | 67 / 118 ms |
| a lead in … of control steps | 22 % | 23 % | 81 % |
| following capped the speed | never | never | 37 % of steps |
| urgent braking | none | none | none |
| worst lateral error, tractor | 0.57 m | 1.33 m | **1.51 m** |

On the first two drives every lead was faster, from 64 to 100 km/h, and
pulling away. On the third the truck closed on a lead and **followed it for
58 s**, slowing from 84 to 43 km/h as the lead slowed:

| while following lead 49 (108–166 s) | |
|---|---|
| gap over wanted gap, median | 1.09 |
| closest gap | 34.1 m at 43 km/h (wanted 32 m) |
| time gap beyond the 8 m floor then in use, steady following | median 2.55 s (10–90 %: 2.13–4.64) |
| brake applications | **18 in 58 s**, median 0.34 s each |
| acceleration, 0.5 s average, 1–99 % | −2.16 … +0.53 m/s² |

**Following works; it is not smooth.** The brake cannot decelerate at less
than 1.81 m/s², so small corrections come out as short brake pulses, one
every three seconds. Holding a gap is safe that way, but a passenger would
feel it. Following needs to meet small speed differences by coasting, and
brake only when the gap is really closing.

**Steering is not ready for 80–85 km/h.** The lateral error reached 1.33
and 1.51 m. A 2.5 m wide truck in a 4.5 m lane has 1.0 m each side, and the
supervisor saw the truck leave its lane at around 80 km/h. At 60 km/h the
worst was 0.43–0.57 m.

Distances here are the camera's own. Nothing in these drives measured the
gap to the lead independently. The camera was checked against a trailer to
about 1 % at 30–50 m (docs/results/distance_estimation.md).

The first attempt stopped where it stood. The perception child reports ready
once its model is loaded; the control loop started before the first frame
had been seen, and "no perception report yet" stopped the truck, as it
should. Starting now waits for a first real view of the road.

## Stopping behind a stopped vehicle

2026-09-24: the trailer dropped in the middle of a lane, the truck 87 m
behind it in the same lane, `follow.py --acc --speed 25 --distance 74`.
The distance budget was set as a backstop: were following to fail
completely, it would stop the truck about a metre short.

| | |
|---|---|
| trailer first detected | 81 m ahead |
| slowing | 25 → 5 km/h, from about 45 m out |
| then | **stuck at 5.2 km/h for 3 s**, "braking" with the brake flicking 0.06–0.10 |
| trailer lost from view | 10.5 m ahead: its wheels behind the dashboard |
| then | **the road taken for clear: throttle 0.37 towards the trailer** |
| stopped by | **the distance budget**, bumper about 5 m from the trailer |

Two faults, both fixed since:

1. **A close lead that drops out of view is remembered.** The camera cannot
   measure anything nearer than about 9.5 m. A lead last seen within 15 m is
   now held as standing where it was, and the truck stops for it. The memory
   ends only if a lead is seen again at most 20 m beyond it, which is the
   same vehicle moving off. It is not cleared by something further down the
   road, which the remembered one would be hiding. `test_acc.py` reproduces
   the game's failure with the near range hidden. It fails without the
   memory and passes with it.
2. **Stopping holds the brake.** Asked for (nearly) zero below 3 m/s, the
   speed controller now holds a 0.2 brake, instead of modulating against
   the engine's idle push. The simulation has no idle push, so only a unit
   test covers this; the game is the real test.

Not driven again yet.

### Second attempt

Same trailer, moved 117 m ahead, the same backstop budget, with the memory
and the brake hold in.

| | |
|---|---|
| trailer first detected | 100 m ahead (measured 89.6 m: this session's horizon reads far distances 10–20 % short) |
| slowing | 25 → 5 km/h by 16 m, then braking and **holding the brake** at 10 m, 2.4 km/h |
| then | measured distance **stuck at 10.0–10.5 m**, the edge of view; wanting 8 m, it crept on |
| stopped by | the budget, the eye 7.7 m from the trailer's rear wheels (true) |

Near the edge of view the measurement was right: 10.5 m measured, 9.9 m
true; 15.0 m, 15.2 m. The fault was the goal. **An 8 m standstill gap is
inside the range the camera cannot see**, so the truck can never measure
itself there. It is 12 m now, with 6 m as the last line of an urgent stop.
That also widened the stopping margins in simulation: a stopped car first
seen 70 m ahead at 80 km/h is now passed at 5.7 m, where before it was
2.3 m.

### Third attempt: it stops

Same trailer, 104.6 m ahead in a gentle bend, the 12 m standstill gap, a
backstop budget of 91 m that the truck never reached (it travelled 85 m).

| | |
|---|---|
| trailer first detected | 86.5 m ahead |
| stop | **by following alone**, from 25 km/h, hardest 3.1 m/s² (0.5 s average), no urgent request |
| measured gap at rest | 12.4–12.7 m |
| true gap at rest | eye to the trailer's rear wheels **15.4 m**; bumper to its rear face about 12.5 m |
| holding | 29 s on the brake, creeping 0.83 m in that time |

It stopped where it meant to, by its own measure, and about 3 m further back
than it meant to. On the previous attempt 15 m measured 15.2 m true. This
time the trailer stood in a bend, which may have changed which part of it
meets the road nearest; the horizon also moves between sessions. The error
fell on the safe side. Over a long wait, the brake alone lets the truck
creep; that wants the parking brake.

### Holding the stop: the parking brake

After the third attempt, probing the controls with the truck at rest on a
1° slope showed two things:

- **The parking-brake button works.** It had never been exercised. It is a
  toggle, and telemetry shows the state change.
- **A stopped truck is not held by the service brake.** A brake command
  applied at rest reaches the game (`input_brake` 0.30) but is not applied
  (`effective_brake` 0.00). With the parking brake released the truck rolled
  back at 1.35 km/h within a quarter of a second. A brake applied while
  still moving did hold through the stop (third attempt).

So a stop that lasts is now held on the parking brake (`core.control.parking`).
It is set after a second at rest, and the pedals are released while it holds.
Moving off, the throttle comes first and the parking brake is released 0.7 s
later. At the end of a drive the truck is braked to rest and the parking
brake set before the controls are let go. Every press is decided from the
telemetry's state, because pressing a toggle blind can as easily release it.

The fourth attempt, with this in:

| | |
|---|---|
| start | parked; released the parking brake itself and moved off |
| stop | 12.3 m measured, trailer first seen 86 m out |
| then | measured gap wandered to 13.1 m, target to 0.31 m/s, **just over the stop threshold**: the stop was dropped before the parking brake was set, the service brake let go, and **the truck rolled back** until the safety monitor ("commanded forward but moving at −1.8 km/h") stopped it |
| end of drive | parking brake set, truck still |

A stop now latches. It lasts until the target reaches 1.5 m/s, a lead that
has really moved off, rather than ending whenever the target wanders over
0.3 m/s. The supervisor had seen the same thing from the cab: "it stopped
twice, moving off again at 2 km/h in between".

### Fifth attempt: one stop, held

After a game update (`def.scs` and `version.scs` changed; the map
re-extracted from the updated archives came out lane for lane identical to
the cache, and telemetry reported the same versions). Trailer 107.9 m ahead
in a bend, backstop budget 95 m, not reached.

| | |
|---|---|
| came to rest | **once**; never moved off again |
| true gap at rest | eye to the trailer's rear wheels 15.4 m; bumper to its rear face about 12.6 m |
| parking brake | set 1.0 s after coming to rest |
| moved while stopped | **0.00 m** in about 30 s (0.83 m on the service brake before) |
| deceleration | hardest 2.6 m/s², 4 brake applications |
| end of drive | parking brake set, truck still |

### Sixth attempt: a cut-in, and a dead GPU

Meant to be the same stop at 45 km/h. An AI truck pulled in between the truck
and the trailer. The trailer dropped out as the lead 83 m out; 1.7 s later the
AI truck appeared as the lead **16.4 m ahead, with the truck at 38 km/h**.
Following slowed it to rest in about 5 s, the closest measured gap being
10.0 m, the edge of view. The parking brake was set. The drive ended when
another window came in front of the game, and it stopped, as it should. No
urgent request was needed; the AI truck was moving.

The perception child had not run on the GPU. The NVIDIA driver logged a
fatal UVM error at 16:19, just after the game update ("requiring os reboot
to recover"), and ONNX Runtime fell back to the CPU **without stopping**:
report age went from a median of 66 ms to 186 ms. The drive went ahead on
stopping margins worked out for 0.1 s. The child now reports where its model
runs, and `follow.py --acc` refuses a CPU model unless given `--allow-cpu`.

### Seventh attempt: from 40 km/h

After a reboot (CUDA back: the detector on the GPU, report age median
68 ms). Trailer 150.5 m ahead, set speed 45 km/h, backstop budget 135 m,
not reached (129 m).

| | |
|---|---|
| top speed | 39.8 km/h, 75 m from the trailer's wheels; the distance allowed no more |
| came to rest | once; never moved off again |
| true gap at rest | eye to the trailer's rear wheels 15.7 m; bumper to its rear face about 12.8 m |
| parking brake | set 1.0 s after coming to rest; 0.00 m of creep |
| deceleration | hardest 3.6 m/s², no urgent request, **8 brake applications** |

It stops where it did from 25 km/h. On the way down the target rose again
briefly, from 12.5 to 13.5 km/h, and fell: the approach is safe, but not yet
smooth. Brake pulses are the next thing to take out of following, here and
when behind a moving lead.

### Brake pulses: smoothing the lead's speed

While following in the game (acc-03, 58 s) the target speed wandered by
3.1 km/h (std), 9.8 km/h at the 99th percentile. Almost all of it came from
the lead's own speed, the truck's speed plus the closing estimate
(3.3 km/h std); the gap term added 0.9. Through the speed controller's gain
of 2 m/s² per m/s, 3 km/h is a request of 1.7 m/s², past the brake's 0.9
threshold: 18 brake applications a minute.

The lead's speed is now smoothed with a 1.5 s time constant while it stays
the same lead. The urgent check still reads the raw estimate, so a lead that
really brakes hard is not waited for. In closed loop, through the real
tracker, a lead slowing from 70 to 43 km/h:

| lead speed | brake applications a minute | closest gap |
|---|---|---|
| raw | 41 | 33.9 m |
| smoothed over 1.5 s | **8** | 33.4 m |

A first, simpler simulation drew the noise independently for every
report. It gave 80–330 a minute: correlated noise matters. What remains is
physics. The lead slowed at 0.45 m/s², inside the range the brake cannot
produce (0.22–1.81), so the truck alternates between coasting and braking.
Not yet driven.

On the same drive (la-01) following never had to act: every lead was
faster. It did produce two urgent requests. One, 3.9 m/s² for 0.3 s, came
from the smoothed estimate still carrying a bad first reading while the raw
one asked for nothing. That was a flaw in how the two were merged: a hard
stop now comes from the raw estimate alone. The other was a single control
step, 2.9 m/s² for 16 ms, for something 114 m ahead closing at 88 km/h.
For a stopped car that request would be right; it may have been a
short-lived box.

## Known limits

- **Brake pulses while following**: 18 in a minute. **Steering leaves the lane at 80–85 km/h.**
- **Stopping behind a stopped vehicle** works from 25 and 40 km/h, held on the parking brake; not tried faster.
- **After such a stop the truck waits** until the vehicle is seen moving off; if it never is, the driver takes over.
- **Nothing nearer than about 9.5 m has a distance.** There the vehicle's
  foot is behind the dashboard. The standstill gap is 12 m for that reason.
- **The one remaining city case** (image 2751): a car 14 m ahead, moving
  our way at about 22 km/h, just after crossing into the lane at a
  junction. Braking at 4.3 m/s² for it is arguable either way.
- **A crash cushion at a motorway split** is detected as a car. It stands
  at the lane's edge in a bend, 2.0–2.7 m from the lane's middle, and
  caused the motorway's hard brake before the other fixes. The detector
  needs to learn it; no rule can tell it from a stopped car.
- **Traffic lights and junctions** are not handled. Following is for a
  lane with a vehicle in it.
- **Nothing independent of the detector** brakes for what it misses.
