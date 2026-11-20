# Traffic lights: where from the map, what colour from the picture

**Date:** 2026-09-24
**Data:** city-01, a manual night drive of 5 minutes through Italian junctions
(15 light-governed lanes entered, 3 stops at red), game time 00:12–00:26
**Code:** `core/perception/lights.py`, `core/control/traffic_light.py`,
`core/map/sector.py` (`SemaphoreProfile`, `Signal`), `Route.stop_lines`,
`tools/follow.py`, `tools/map_extract` (`semaphore_profiles`, `cat`)

## Where: the map

A prefab lane a light governs names the light's semaphore, and starts at the
stop line: the governed lanes are 1.7 m long pieces that begin where the
traffic waits. So the stop line is the start of the first governed lane the
route enters (`Route.stop_lines`). The semaphore's own position in the prefab
descriptor is a locator on the ground 5–8 m from the line, not the lamp, and
is not used.

What the semaphore *is* comes from `/def/world/semaphore_profile*.sii`, which
the extractor now copies into the extraction (306 profiles). A profile gives
each semaphore id a type -- `traffic_light_major`, `barrier_manual`,
`barrier_distance` (level crossings), `barrier_gas` -- and many give a
**sleep window: 23:30 to 03:00, when the lights are dark**. The map cache
(format 3) stores, per lane, which of 93 signals stands at its start.

That explained what the picture alone could not: 8 of city-01's 15 lights
never showed a lamp. All 8 are `cr_1x1` or `cr_1x2`, which sleep, and the
drive was at 00:12–00:26. The `cr_2x1` lights, which do not sleep, were lit.

A lane can also name a semaphore its junction does not define; in city-01
that was a junction with no light on that approach. Such lanes are not
lights.

## What colour: the picture

`find_lamps` looks for small, bright (V ≥ 200), saturated (S ≥ 100), round
blobs whose surroundings are dark (median brightness around ≤ 0.6 of the
lamp's), not inside a detected vehicle, and high enough: at least 1.6 m up if
no nearer than the stop line. Far from the line that is the horizon; at the
line, the green lamp of the pole beside it is 100 px below it.

Each frame's reading is the colour with the most lamp area, so the light
ahead outweighs the next junction's. `LightFilter` confirms a colour over 3
frames, holds it 2 s after its lamps leave the view, and only believes the
two changes a lookalike can fake where a light could make them:

- **amber after green** only just above a green lamp just seen. Streets under
  sodium lamps are amber everywhere; without this, a row of lit windows
  170 px from a green lamp read as amber and stopped the truck at a green;
- **green after red** only once no red has been seen for 1 s -- these lights
  show amber alone for 2 s between them -- or in the red lamp's own head. A
  green sign beside a red light is not the light turning green.

## What to do: the decision

`TrafficLights.step` turns the next stop line and the colour into a target
speed, in the same `min()` as the corners, the road's limit and following:

| colour | decision |
|---|---|
| green | nothing; committed once too close to stop at 2.5 m/s² |
| amber | stop if ≤ 3 m/s², otherwise go through |
| red | stop, up to 5 m/s²; later than that, go through |
| unknown | approach as if red, at comfort only; a light never seen is waited for 45 s (a whole 42 s cycle), then passed |
| asleep | nothing |

It stops 13 m before the start of the governed lane (truck reference point);
see *In the game* for why not 8. A stop once
begun is finished, because the quantised brake lets the deceleration still
needed wander past the go-through thresholds on the way down.

## Checked against city-01, in the shadow

The whole chain run over the recording, with the driver's own path:

| lights | what happened |
|---|---|
| 8 asleep (`cr_1x1`, `cr_1x2`) | passed as dark lights, as the driver did |
| 3 where the driver stopped at red | red seen from 90–110 m; green seen when the driver went (the last one 0.5 s before he moved) |
| 1 red, then green on the approach | stopped for red, went on the green at 14 m, as the driver did |
| 2 with no lamp ever seen | stopped and waited; the driver went. No lamp from 150 m to the line, at a gate-house and a junction beside it |

The full cycle -- green, amber, red, green -- was read at one junction, with
the change seen within 0.1 s of the frames where it happened.

## In the game

Two drives on 2026-09-24, by day (16:45), in an Italian town, through two
`cr_2x1` lights, at up to 40 km/h (data/follow/lights-01, -02):

- **lights-01.** Green, amber, red read from 53 m; stopped for red. From the
  stop 8 m short of the lane the map says the light governs, no lamp was in
  view for 21 s. Then green was seen, but a car crossing the junction had
  been the lead for a moment and was remembered as standing 9.8 m ahead: it
  held the truck through the whole green. The light had committed to going
  through at rest, so when the memory cleared the truck moved off **on red**,
  6 m over the line, before stopping for traffic. Fixed: a green commits only
  a moving truck; only a lead followed for 1 s is remembered.
- **lights-02.** The first light passed on green. At the second, red was seen
  at 26 m and the truck stopped -- at the zebra, its light's lamps out of
  view: the pole's to the right, the overhead ones behind the cab roof. It
  went only when the supervisor turned the camera. Frames saved every 0.5 s
  showed the pole lamp leaving the view about 10 m back, and the overhead
  red lamps, plain to see, never found: by day they are framed by sky, and
  the median of their surroundings was sky. Fixed: stop 13 m short, and
  judge the surroundings by their darker quarter, which is the housing.

- **lights-03**, with both fixes, the same two lights. The first passed on
  green at 30 km/h. At the second, red seen from 35 m; stopped 13.1 m short,
  the pole's red lamp in view (1553, 433) the whole time; it went out and
  the green came on 50 px below it, in the same head, and the truck moved
  off by itself. At the first light, over the line, the state turned red:
  not our light -- the heads for the cross street, seen from the side, show
  red while ours shows green. Committed by then, it did not matter; stopped
  at a red, their green is the same trap the other way, and the rule that
  green after red must replace the red is what stands against it.

With both fixes the shadow of city-01 is unchanged but for one light: a
sleeping light where a red 64 m beyond it, now found, would have stopped the
truck. Red is no longer obeyed at a sleeping light.

## Inside a head (7 October 2026)

The lamps were found by colour, shape and surroundings alone, and each
lookalike that stopped the truck got a rule: a poster's red, a give-way
sign, a no-parking ring, tail lamps. Short of Linz a filling station's
logo, red under yellow among trees, read as red, and then the next
junction's light, 95 m on, held the truck 10 m from a line whose junction
has none. Now the detector finds signal heads (docs/results/vehicle_detection.md)
and a lamp counts only inside one (`find_lamps(heads=...)`), and only a
head near enough for the light whose line is near: a head's box is
`HEAD_HEIGHT_M` 1.5 m tall -- 112-140 px waiting 10 m from lines in Bern,
Milan and Linz -- so the next junction's, 13-14 px there, is 88 m off.

On the black box's pictures, the old reading against the new:

| case | old | heads |
|---|---|---|
| Linz: the station's logo | red and amber | nothing |
| Linz: waiting for the next junction's red | amber | nothing |
| Bern: a wait with no light at all | amber | nothing |
| red signs by a crossing, a red brick wall 130 m off, the satnav's red | red | nothing |
| real waits in Bern, near Linz, Milan by night | red, then green | the same |
| approaching six reds | first read at 32-42 m | the same distance |

Until a model trained on heads is loaded (`model.json` `signal_heads`), the
lamps are read as before.

## Not established

- **Only at night, only in Italy, only one drive.** Daylight lamps are less
  bright against their surroundings; nothing here was tested by day.
- **Not driven.** Everything above is the shadow of a manual drive.
- **The two unseen lights.** Why they never showed a lamp is not known; the
  45 s wait-and-go is a way out, and at a light that is there but hidden it
  can go on red.
- **Country variants.** Profiles like `cr_1x1.denmark` (never sleeps) are
  extracted but the prefab names the base profile; which one the game uses
  where is not known. A light the map calls asleep is ignored, camera and
  all: where a country's lights do not sleep, that is a red light run.
- **Barriers** (tolls, level crossings) are counted, not handled.
