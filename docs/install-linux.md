# Installing on Linux

Driven on Ubuntu 22.04 in an X11 session, with the native Linux build of Euro
Truck Simulator 2 (not Proton) and an NVIDIA GPU.

## What you need

- Euro Truck Simulator 2 from Steam, the native Linux build.
- An X11 session. Wayland does not work, because the stack reads the road off
  the screen and finds the game's window through X11. On Ubuntu with GDM, set
  `WaylandEnable=false` in `/etc/gdm3/custom.conf` and log in again.
- An NVIDIA GPU with its proprietary driver is best. Without one (an AMD or
  Intel GPU, or none), perception runs on the CPU, about three times slower,
  and no delivery has been driven that way yet.
- [This Steam Workshop mod](https://steamcommunity.com/sharedfiles/filedetails/?id=2131396358), subscribed to and enabled in the
  game's mod manager. It is required: it takes the collision away from toll
  and yard gates, and the truck drives through toll booths without stopping
  for their barriers. Without it the truck hits the closed barrier.
- About 4 GB of free disk, and `curl`:

```bash
sudo apt install curl
```

## Install

1. Download `ets2ad-<version>-linux-x64.zip` from the
   [releases](https://github.com/koculuhamit/ets2-autonomous-driving/releases) and unzip it:

   ```bash
   unzip ets2ad-<version>-linux-x64.zip -d ~/ets2ad
   ```

2. Close the game, then run the launcher:

   ```bash
   ~/ets2ad/ets2ad-<version>-linux-x64/ets2ad.sh
   ```

   The first time, it fetches [uv](https://docs.astral.sh/uv/) into
   `~/.local/bin`, and uv brings Python and the packages: the GPU build of
   ONNX Runtime if `nvidia-smi` is found, the CPU build if not. It then finds
   the game through Steam, copies the plugin into
   `<game>/bin/linux_x64/plugins`, and reads the map of your game and DLCs out
   of its archives. The map took about a minute on the test machine; it is
   read once, and again after a game update.

3. Start the game. It warns about SDK features (that is the plugin): press OK.
   It asks every time the game starts.
4. In the game's mod manager, enable the barrier mod (see "What you need")
   for your profile. The game keeps it enabled after that.

## In the game

- Display: full screen, 1920x1080. The stack reads the road off the screen,
  so the whole game window has to be on it. A 1920x1080 window does not fit
  on a 1920x1080 screen, because the desktop's top bar pushes it past the
  bottom edge, and then no drive starts ("reaches past the screen"). If you
  restart the game, check that it is still full screen.
- Frame rate: capped at 60 FPS (Options, Graphics). Uncapped, the game takes
  the GPU from the detector and the drive can stop for a late view.
- Camera: the interior camera (1), looking ahead, where the game puts it. If
  you move it, the dashboard no longer looks as measured, and the drive stops
  with "the game's camera is not in the cab as calibrated".
- Mirrors: turn on both of the HUD's mirror views, left and right (the game's
  HUD settings). Before and during a lane change the truck looks in the
  mirror on that side for a vehicle coming up, and waits or turns back for
  one coming up fast. By night it sees little there.
- Gearbox: automatic.
- Barriers: the truck does not stop for toll or yard gates, so the barrier
  mod from "What you need" must be enabled.

## Drive

Each time you want to drive, start the launcher first. It is a separate
program that drives the truck, and the beacon only tells it to go.

1. Run `ets2ad.sh`. After the first time it starts in a few seconds. Keep its
   terminal open; that is where it tells you what it is doing.
2. In the game, take a job, hitch the trailer and drive out of the yard onto
   the road. Go on to a straight stretch before handing over: switched on at
   speed into a turn, the truck takes the turn wide. Its lanes start at the
   yard's gate, and the gate's booth is not on its map.
3. Switch on the beacon (O). The truck drives the delivery and stops at the
   depot's gate. Ports and trains are yours: it stops where their own roads
   begin, before the passport booths and the quay. It never drives into a
   filling station and does not refuel. Under 600 km of range the launcher
   says so; fill up by hand and switch the beacon back on.
4. To take over, switch the beacon off or press the brake or the throttle.
   Switch the beacon off and on again to hand it back.

Without a job, set a place on the game's map as the navigation target. The
launcher then says "to where the GPS goes", and the truck drives there and
stops near it.

## If something is wrong

- "requires an X11 session": log in to an X11 session (see above).
- The truck does not move when the beacon is on: the launcher's terminal
  says why. "is the game in front?" means the game window was hidden or
  minimised. "reaches past the screen" means it does not fit on the screen;
  make the game full screen.
- The game says nothing about SDK features: the plugin is not installed. Run
  `ets2ad.sh` again with the game closed.
- A border, toll or junction it drives badly: drive it by hand once while
  `tools/record.py` records, then teach it with `tools/teach_crossing.py`
  (see its help). Taught places are in `configs/crossings.json`.
