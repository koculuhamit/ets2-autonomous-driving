# Installing on Windows 11

Driven on Windows 11 with Euro Truck Simulator 2 1.62, an NVIDIA RTX 2060
and 16 GB of RAM.

## What you need

- Euro Truck Simulator 2 from Steam, installed on an SSD. On a laptop's hard
  disk the game stalled for seconds at a time, and the drive stopped with it.
- An NVIDIA GPU is best. Without one, perception runs on the CPU, about three
  times slower, which is enough on a fast processor.
- About 4 GB of free disk: 2.5 GB for ONNX Runtime with CUDA, the rest for
  the map read out of your game.
- [This Steam Workshop mod](https://steamcommunity.com/sharedfiles/filedetails/?id=2131396358), subscribed to and enabled in the
  game's mod manager. It is required: it takes the collision away from toll
  and yard gates, and the truck drives through toll booths without stopping
  for their barriers. Without it the truck hits the closed barrier.
- An internet connection for the first start.

You install nothing else by hand. Python, its packages and the tools come
with the launcher.

## Install

1. Download `ets2ad-<version>-win-x64.zip` from the
   [releases](https://github.com/koculuhamit/ets2-autonomous-driving/releases) and unzip it
   somewhere with a short path, such as `C:\ets2ad`.
2. Close the game, then double-click `ets2ad.bat`. The first time, it fetches
   [uv](https://docs.astral.sh/uv/), and uv brings Python and the packages:
   the GPU build of ONNX Runtime if `nvidia-smi` is found, the CPU build if
   not. It then finds the game through Steam and copies the plugin into
   `<game>\bin\win_x64\plugins`; if Steam is under `Program Files`, Windows
   may ask for permission. Last, it reads the map of your game and DLCs out of
   its archives, which takes a minute or two on an SSD, once, and again after
   a game update.
3. Start the game. It warns about SDK features (that is the plugin): press OK.
   It asks every time the game starts.
4. In the game's mod manager, enable the barrier mod (see "What you need")
   for your profile. The game keeps it enabled after that.

## In the game

- Display: windowed or borderless full screen at 1920x1080, which is what it
  was driven at. The stack reads the road off the screen, so the game window
  has to be visible.
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

1. Run `ets2ad.bat`. After the first time it starts in a few seconds. Keep its
   window open; that is where it tells you what it is doing.
2. In the game, take a job, hitch the trailer and drive out of the yard onto
   the road. Go on to a straight stretch before handing over: switched on at
   speed into a turn, the truck takes the turn wide. Its lanes start at the
   yard's gate, and the gate's booth is not on its map.
3. Switch on the beacon with O, the game's default key for the beacon (or
   the key you bound to it in the game's controls); the same key switches it
   off. With the beacon on, the truck drives the delivery and stops at the
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

- The truck does not move when the beacon is on: the launcher's window says
  why. "is the game in front?" means the game window was hidden or minimised.
- It stops at once for "no view of the road", or perception does not start
  in time, with the game in front: look at the GPU's load (Task Manager,
  Performance). Near 100 %, the game leaves the detector no time. Cap the
  frame rate at 60 FPS, or lower the graphics settings, until it is well
  under.
- The game says nothing about SDK features: the plugin is not installed. Run
  `ets2ad.bat` again with the game closed.
- A border, toll or junction it drives badly: drive it by hand once while
  `tools/record.py` records, then teach it with `tools/teach_crossing.py`
  (see its help). Taught places are in `configs/crossings.json`.

## Uninstall

Close the game and double-click `uninstall.bat` in the folder you unzipped.
It asks about each of these in turn, and keeps whatever you do not answer
`y` to:

| | where | size |
|---|---|---|
| the plugin | `<game>\bin\win_x64\plugins\ets2ad_telemetry.dll` | small; the game stops warning about SDK features |
| uv's downloads of this project's packages | `%LOCALAPPDATA%\uv\cache` | about 2.5 GB with the GPU build |
| the Python uv fetched | `%APPDATA%\uv\python` | about 100 MB |
| uv | `%USERPROFILE%\.local\bin` | about 50 MB |
| this folder | where you unzipped it | its packages, the map, and the drives recorded in it |

Python and uv may be used by other projects of yours: say no to them if
so. On the same disk uv links its downloads into the folder's packages, so
the two take their space once, and it comes back when both are gone.
Python, uv and the folder are removed a few seconds after the last
question. Last, unsubscribe from the barrier mod in Steam if you no longer
want it.
