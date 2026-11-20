# ETS2 autopilot 2.1.2

Three fixes found after 2.1.0, three more trucks known, and the measurements
behind the stack published. 2.1.1 was a test build that was never released:
on Windows it did not drive at all.

Install: [Windows 11](install-windows.md) · [Linux](install-linux.md). The
plugin has not changed since 2.1.0. Coming from 2.0.0, run the launcher (or
`tools/setup.py`) once with the game closed.

## Fixed

- **Windows: the beacon did nothing but stop.** With the game in front,
  every drive stopped at once for "no view of the road: is the game in
  front?". When the perception process failed, nobody was told why. A grab
  that fails is now a view not had, retried, with its reason shown, and a
  perception process that fails says what failed and where. A window whose
  grab comes out at another size than the window's, as with display scaling,
  is taken at the size grabbed, and a minimised game window is said to be
  one.
- **Windows: the first start could time out.** Loading the detector the first
  time after installing may take minutes, while the virus scanner reads
  CUDA's libraries. It is now given 3 minutes, not 1. A beacon switched on
  before then is told to wait for "standby: ready".

On the Windows test the cause was the GPU: the game ran it near 100 % and
left the detector no time. Capped, the truck drove. The install guides now
say what to look at.
- **Rolling back at a red light.** Held on the brake at a standstill, the
  game's automatic gearbox put the truck into reverse within half a second,
  and it rolled backwards. The brake is now let go and the parking brake set
  the moment the gearbox shows reverse.
- **The route in the ROS 2 view.** It was drawn flat at the truck's own
  height, up to 8.5 m above or below the lanes on hills. Each point now sits
  at the road's height, and the route is sent from 500 m behind the truck to
  3 km ahead.

## New

- The Iveco Hi-Way's, the MAN TGX's and the Volvo FH16 2012's dashboards and
  windscreens, learned on the way: 18 trucks are known now.
- [Results](results/README.md): every drive recorded on Linux summed by
  period, the latency budget from the picture to the brake, and the dated
  lab notes and [design decisions](adr/) the stack was built from.
- The README explains the ROS 2 interface and why the controllers are pure
  pursuit and PI.

## Known limits

As in [2.1.0](release-notes-2.1.0.md).
