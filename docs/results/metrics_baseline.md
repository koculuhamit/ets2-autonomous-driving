# Metrics baseline: the recorded missions, 2 to 4 October 2026

Every drive tools/mission.py recorded up to 4 October, measured by
`tools/report.py` (core.evaluation). This is the number to beat after each
change: run the same command over the new sessions and compare.

    python tools/report.py data/missions/* --markdown report.md

## Summary

| | |
|---|---|
| drives | 33 |
| distance | 4333 km (the game's), 4.6 h |
| interventions | 23 (11 taken over, 10 safety stops, 2 interrupted) |
| **interventions / 100 km** | **0.53** |
| hits | 6 |
| hard brakes (> 3 m/s² for 0.3 s) | 167 |
| damage | 38.9 % summed |

How to read it:

- Distances are the game's kilometres, from the truck's odometer, as the
  game's own GPS and dashboard give them; the speeds are the speedometer's.
  The game's map runs at a different scale from its clock, so 4333 km in
  4.6 h is not a mistake. Until 15 November 2026 the report summed the
  world's metres instead (258 km here), a sixteenth to an eighteenth of the
  game's on the open road and a third in towns, and gave 8.9 interventions
  per 100 km.
- A drive stopped by a hit is counted as a hit, with the damage it reported,
  even where the recording does not show it: the frame that showed the hit
  was not written until the fix of 15 November (3 of the 6 hits here).
- These sessions span the stack's development: the customs crash, the mirror
  lane change and the Milan–Bern problems are all in them, and each was fixed
  after its drive. The figure is where the stack *was*, not where it is.
- An intervention is a drive that ended taken over (`disengaged`,
  `interrupted`) or stopped for safety (`unsafe`). Some takeovers were the
  driver testing, not the truck failing; the outcome files do not say which.
- Lateral distance is to the route's lane middle; the tractor swings wide in
  tight turns on purpose. Further than 20 m is counted as *off the route*:
  the drive replanned and the recorded route is no longer the one it
  followed.
- Speed is resampled to 50 Hz before it is differentiated; the recording's
  timestamps are when each block was read, and raw differences read a
  full-frame speed change over 7 ms as 18 m/s².
- Memory growth is new (`health.npy`, from 6 October): none of these drives
  has it.

## Per drive

| drive | duration_s | distance_km | mean_speed_kmh | lateral_mean_m | lateral_max_m | out_of_lane_s | off_route_s | hard_brakes | max_deceleration_mps2 | max_jerk_mps3 | min_ttc_s | min_gap_m | steering_reversals_per_min | impacts | damage_pct | ended | memory_mb_per_h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-014612/drive-02 | 2171.80 | 457.32 | 59.14 | 0.85 | 19.99 | 155.90 | 1104.22 | 34 | 7.12 | 14.24 | 0.42 | 7.20 | 2.15 | 0 | 0.00 | interrupted | - |
| 20260930-014612/drive-03 | 245.62 | 42.68 | 50.95 | 0.12 | 0.59 | 0.00 | 0.00 | 1 | 3.83 | 8.28 | 0.51 | 11.46 | 3.42 | 0 | 0.00 | end_of_route | - |
| 20261001-192945/drive-01 | 41.60 | 1.93 | 9.15 | 0.28 | 0.52 | 0.00 | 0.00 | 0 | 2.54 | 4.96 | - | - | 4.33 | 0 | 0.00 | disengaged | - |
| 20261001-192945/drive-02 | 2522.49 | 882.71 | 70.61 | 0.30 | 19.98 | 21.42 | 1757.27 | 11 | 5.65 | 9.19 | 2.21 | 10.02 | 1.43 | 0 | 0.00 | disengaged | - |
| 20261001-192945/drive-03 | 1577.10 | 537.86 | 65.75 | 0.82 | 19.98 | 181.31 | 24.53 | 15 | 8.61 | 13.64 | 1.66 | 9.88 | 1.29 | 0 | 0.00 | disengaged | - |
| 20261001-214305/drive-01 | 1328.54 | 489.80 | 71.69 | 0.06 | 0.59 | 0.00 | 0.00 | 14 | 7.07 | 12.76 | 0.60 | 10.23 | 0.68 | 0 | 0.00 | end_of_route | - |
| 20261001-221736/drive-01 | 1961.76 | 694.55 | 72.44 | 0.70 | 19.97 | 153.95 | 3.78 | 13 | 5.15 | 7.95 | 0.51 | 9.58 | 0.76 | 0 | 0.06 | end_of_route | - |
| 20261001-231250/drive-01 | 1172.10 | 314.78 | 54.28 | 0.97 | 16.53 | 95.65 | 0.00 | 10 | 5.77 | 9.76 | 0.36 | 6.31 | 1.13 | 0 | 14.98 | end_of_route | - |
| 20261001-231250/drive-02 | 14.81 | 0.00 | 0.02 | 0.61 | 0.61 | 0.00 | 0.00 | 0 | 0.34 | 1.48 | - | - | 0.00 | 0 | 0.00 | disengaged | - |
| 20261001-231250/drive-03 | 8.34 | 0.00 | 0.00 | 0.61 | 0.61 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | - | - | 0.00 | 0 | 0.00 | unsafe | - |
| 20261001-231250/drive-04 | 622.48 | 105.34 | 43.02 | 0.40 | 19.99 | 8.70 | 251.55 | 5 | 4.70 | 8.15 | 1.94 | 13.85 | 2.60 | 0 | 0.00 | end_of_route | - |
| 20261001-231250/drive-05 | 852.41 | 149.55 | 40.57 | 0.33 | 19.93 | 16.96 | 14.74 | 6 | 4.95 | 7.42 | 0.97 | 9.49 | 3.17 | 0 | 0.00 | unsafe | - |
| 20261002-020235/drive-02 | 28.24 | 0.12 | 5.12 | 0.25 | 0.47 | 0.00 | 0.00 | 1 | 3.17 | 5.80 | 4.39 | 13.63 | 4.25 | 0 | 0.00 | unsafe | - |
| 20261002-020235/drive-03 | 72.82 | 0.64 | 10.68 | 0.96 | 19.96 | 4.65 | 14.39 | 1 | 3.25 | 8.76 | 6.89 | 13.48 | 4.94 | 0 | 0.00 | end_of_route | - |
| 20261002-020235/drive-04 | 86.23 | 0.85 | 11.87 | 0.46 | 1.60 | 5.14 | 0.00 | 2 | 3.84 | 7.45 | 8.63 | 16.45 | 3.48 | 0 | 0.00 | interrupted | - |
| 20261002-022859/drive-01 | 31.17 | 0.40 | 15.62 | 0.28 | 0.67 | 0.00 | 0.00 | 1 | 4.01 | 7.52 | 4.18 | 12.00 | 5.77 | 0 | 0.00 | end_of_route | - |
| 20261002-022859/drive-03 | 7.71 | 0.04 | 6.30 | 2.20 | 3.85 | 5.31 | 0.00 | 1 | 9.78 | 20.99 | - | - | 7.79 | 1 | 3.41 | unsafe | - |
| 20261002-022859/drive-05 | 529.12 | 83.16 | 40.21 | 0.11 | 0.67 | 0.00 | 0.00 | 10 | 10.27 | 20.62 | 2.58 | 9.61 | 2.04 | 1 | 1.11 | unsafe | - |
| 20261002-031018/drive-01 | 63.52 | 1.42 | 27.28 | 0.18 | 0.79 | 0.00 | 0.00 | 1 | 5.43 | 5.50 | 2.87 | 15.48 | 3.78 | 0 | 0.00 | end_of_route | - |
| 20261002-031018/drive-02 | 415.68 | 68.58 | 44.29 | 0.48 | 20.00 | 20.45 | 5.53 | 3 | 4.45 | 8.66 | 1.43 | 20.46 | 3.61 | 0 | 0.00 | disengaged | - |
| 20261003-201028/drive-01 | 445.20 | 82.18 | 45.08 | 0.15 | 1.07 | 0.00 | 0.00 | 6 | 7.58 | 14.69 | 2.86 | 9.66 | 2.29 | 0 | 0.00 | end_of_route | - |
| 20261004-202354/drive-01 | 636.12 | 136.50 | 44.94 | 0.17 | 6.32 | 5.69 | 0.00 | 11 | 6.34 | 13.35 | 2.48 | 7.97 | 2.45 | 1 | 18.70 | unsafe | - |
| 20261004-205330/drive-01 | 40.80 | 5.49 | 26.19 | 0.54 | 6.91 | 3.35 | 0.00 | 0 | 1.72 | 3.55 | 3.41 | 34.16 | 7.35 | 1 | 0.00 | unsafe | - |
| 20261004-205330/drive-03 | 411.64 | 131.61 | 61.39 | 0.07 | 0.67 | 0.00 | 0.00 | 3 | 5.59 | 6.62 | 4.92 | 10.46 | 1.31 | 1 | 0.20 | unsafe | - |
| 20261004-205330/drive-04 | 586.34 | 137.58 | 53.24 | 0.13 | 19.80 | 3.45 | 127.68 | 5 | 4.95 | 6.13 | 1.66 | 12.02 | 1.74 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-06 | 27.16 | 0.00 | 0.00 | 0.20 | 0.20 | 0.00 | 0.00 | 0 | 0.01 | 0.02 | - | - | 0.00 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-07 | 24.13 | 0.41 | 20.74 | 0.60 | 1.52 | 0.12 | 0.00 | 0 | 2.58 | 3.48 | - | - | 12.43 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-08 | 71.58 | 0.27 | 4.61 | 0.36 | 1.65 | 2.12 | 0.00 | 2 | 4.90 | 6.65 | 3.50 | 13.20 | 2.51 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-10 | 29.68 | 0.32 | 12.90 | 0.39 | 1.32 | 0.00 | 0.00 | 1 | 7.39 | 9.10 | - | - | 8.09 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-12 | 28.10 | 0.17 | 7.45 | 1.80 | 3.82 | 20.30 | 0.00 | 1 | 3.66 | 6.76 | - | - | 6.41 | 1 | 0.40 | unsafe | - |
| 20261004-205330/drive-13 | 314.99 | 5.79 | 22.21 | 0.17 | 1.75 | 1.04 | 0.00 | 7 | 8.24 | 15.73 | 1.58 | 6.79 | 3.43 | 0 | 0.03 | unsafe | - |
| 20261004-205330/drive-14 | 54.60 | 0.50 | 11.10 | 0.11 | 0.62 | 0.00 | 0.00 | 1 | 4.20 | 5.42 | 5.79 | 27.21 | 1.10 | 0 | 0.00 | disengaged | - |
| 20261004-205330/drive-15 | 41.34 | 0.50 | 14.79 | 0.01 | 0.03 | 0.00 | 0.00 | 1 | 4.12 | 5.31 | 6.20 | 17.72 | 0.00 | 0 | 0.00 | end_of_route | - |
