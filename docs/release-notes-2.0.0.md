# ETS2 autopilot 2.0.0

The first release driven on Windows 11. Unzip it, run the launcher, take a
job and switch the truck's beacon on with O (the game's beacon key), and the
truck drives the delivery; switch it off with O again to take the truck back. Before
release the developer drove deliveries with it until three in a row went
without a hit or a take-over: Bergen to Calais, Milan to Verona, Munich to
Nuremberg.

The release was then installed from its zip into an empty folder, the way a
new user would, on Windows 11 and on Ubuntu 22.04. The Linux test drive, a
delivery to Amsterdam, turned up the problems fixed in this release: lane
changes after a late reroute, the game restarted while the launcher waited,
and a car coming up fast in the mirror.

Install: [Windows 11](install-windows.md) · [Linux](install-linux.md)

Required: [this Steam Workshop mod](https://steamcommunity.com/sharedfiles/filedetails/?id=2131396358), enabled in the game's mod
manager. It takes the collision away from toll and yard gates; the truck
does not stop for their barriers and would hit them without it.

## What it does

- Follows the game's own GPS route, planned from the map read out of your
  game. Where the route on the game's screen leaves the map's route, for a
  place you marked or a road the map weighs differently, it follows the
  minimap's red line.
- With no job, drives to a place marked on the game's map and stops there.
- Keeps to its lane and changes lanes only where the route needs it: an
  exit, a fork, a lane that ends. Before and while it moves across, it looks
  in the HUD mirror on that side. It waits for a vehicle that is near or
  coming up fast, and turns back for one coming up fast while it is less
  than 1.5 m across.
- When a late reroute leaves the start of a lane change behind the truck,
  it still makes the change if the lane is clear and there is room. The
  change is then shorter than usual, but never under 4.8 s.
- Adaptive cruise to a stop and away, give-way at junctions, red lights
  (including one lost from view behind a lorry), and level crossroads taken
  at 60 km/h at most.
- Takes tight junction turns in its own lane, never over the other way's,
  going deeper into the junction first where that keeps the trailer off the
  kerb.
- Stops at the depot's gate. Ports and trains are the driver's: it stops
  where their own roads begin, before the passport booths and the quay.
- Never drives into a filling station; the driver fills up. Under 600 km of
  range the launcher's window says so, and after filling up, the beacon off
  and on sends the truck on.
- At borders and other places the map gets wrong, drives the line a driver
  taught it. [The seven border crossings](borders.md) of the base map and
  Scandinavia are taught.
- Says in the launcher's window what it does and why it stopped. Nothing
  pops up on the desktop and nothing sounds there.
- On Linux, if the game is closed and started again while the launcher
  waits, the launcher finds it again by itself.
- Hands over at once when the beacon goes off or a pedal is pressed. It
  stops and hands over on a hit, damage, standing still too long, the camera
  moved out of the cab, or its own route planner failing.

## Known limits

- The HUD mirrors are read for vehicles coming up behind by day; by night
  they show little. A car coming up very fast can still reach the truck as
  it moves across. On the Linux drive one did: it showed in the mirror only
  after the change had begun. The mirror rule added for it has been tried on
  recorded frames, not yet over many drives.
- On Linux the game must be full screen. A 1920x1080 window does not fit on
  a 1920x1080 screen beside the desktop's top bar, and no drive starts.
- Without an NVIDIA GPU, perception runs on the CPU, about three times
  slower. No delivery has been driven that way.
- Booths, posts and bollards are not on its map. In a tight turn the cab
  keeps within 0.6 m of its lane's middle on the outside, and the trailer
  may ride over the kerb on the inside. Drive out of a yard's gate by hand.
- Switched on at speed into a turn, it takes the turn wide, so switch it on
  where the road runs straight.
- Borders and junctions in DLCs the developer does not have are not taught.
  Teach them with `tools/record.py` and `tools/teach_crossing.py`.
- At a light it cannot see from the stop line, it waits up to 45 s. Where
  the map has a light the road does not, list it in `configs/no_lights.json`.
- Cap the game's frame rate at 60 FPS. Uncapped, the game at times left the
  detector no GPU for half a second, and the drive stopped because its view
  of the road was too old.
- It does not stop for toll or yard gates: the barrier mod above is
  required.
- The distances it reports are the game's kilometres. Along a route they are
  averaged, so in towns they run a little long.
- Trucks only: buses (the Coaches DLC) are not driven yet.
