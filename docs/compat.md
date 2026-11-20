# Compatibility matrix

Every layer in this project is tied to a specific game build. The game does not
promise stability across versions: the telemetry struct gains fields, the `.scs`
archive format changes, and the map data is rebuilt wholesale in map DLC
updates. When something that worked stops working, this table is the first
place to look.

**Rule:** pin the game version. When an update lands, do not chase it on `main`
-- branch, retest, record the result here, then merge.

## Current target

| Component | Version | Notes |
|---|---|---|
| Euro Truck Simulator 2 | **1.62.3.3** | Steam buildid `26644872`, Linux native; 1.62 on Windows 11 |
| Steam app ID | 227300 | |
| Platform | Ubuntu 22.04.5 LTS, X11 | NVIDIA 595.99.02, RTX 2060 6 GB |
| Platform | Windows 11 | RTX 2060, 16 GB RAM |
| SCS Telemetry SDK | **1.14** | Vendored, MIT; `plugin/vendor/scs_sdk/` |
| Telemetry block layout | **v1** | 4080 bytes; `plugin/include/ets2ad/telemetry_layout.h` |
| Command block layout | **v1** | 96 bytes; `plugin/include/ets2ad/command_layout.h` |
| Recording format | **v1** | `core/recording/format.py` |
| Map extraction | TruckLib **0.5.1** | `tools/map_extract`; the map is read on each machine from its own game (`tools/setup.py`) |

## Control mixes the input device binds to

The plugin registers a *semantical* device, which binds by mix name. A game
update that renames a mix breaks input silently, so the names are recorded
here and re-checked against a fresh `controls.sii` after any update.

| Purpose | Mix | Notes |
|---|---|---|
| Steering | `steering` | **subtracted** in the mix, so the sign is inverted; a rate command: it sets how fast the wheel turns, not where it points |
| Throttle | `aforward` | position command |
| Brake | `abackward` | "accelerate backward": reverse at a standstill, brake when moving |
| Clutch | `clutch` | |
| Buttons | `parkingbrake`, `engine`, `gearup`, `geardown`, `lblinker`, `rblinker`, `horn`, `light`, `motorbrake`, `engbrakeup`, `engbrakedwn` | |
| Hazard lights | `flasher4way` | was registered as `hazard`, which is no mix: the lights never came on |
| Activate (Enter) | `activate` | bound, unused: it filled the tank at a pump before refuelling was left to the driver |
| Retarder | `retarderup`, `retarderdown` | a step per press; the level read back from `retarder_level` (core/control/retarder.py) |
| Trailer attach (T) | `attach` | bound, unused: it hitched the trailer before parking and hitching were left to the driver |

The version is the pack set's, from the game's own log:

```bash
grep -m1 "pack set version" ~/.local/share/"Euro Truck Simulator 2"/game.log.txt
```

The strings in the binary are not the version: read from them, this table
said 1.55.0.0 from the start, while the game was 1.61.1.0 all along.

## Installation layout (Linux)

```
<game root>/
  bin/linux_x64/
    eurotrucks2            # main binary
    libfmod.so.14          # audio
    libfmodstudio.so.14
    libsteam_api.so
    plugins/               # created by us; the game loads *.so from here
  *.scs                    # 107 archives on 1.62.3.3: base, def, core, map, DLC
```

`plugins/` does not exist in a fresh installation, and the game does not
create it: the launcher (`tools/setup.py`) does. It survives Steam updates, but
a "verify integrity of game files" pass may remove unknown files from it; the
launcher then installs the plugin again, and `plugin/` builds it from source.

## Keeping the version pinned

Steam auto-updates by default, which would silently swap the build under a
running experiment. Two settings, both in the Steam client:

1. **Library → Euro Truck Simulator 2 → Properties → Updates →**
   "Only update this game when I launch it".
2. **Properties → Betas →** select a specific branch if a future experiment
   needs one. `None` is the public branch, which every version here came from.

Neither is scriptable: Steam rewrites `appmanifest_227300.acf` from its own
state, so editing the file while the client runs achieves nothing.

## Update log

| Date | Game version | What broke | Resolution |
|---|---|---|---|
| 2026-09-20 | 1.61.1.0 | — | Initial target (buildid `25346574`; recorded here as 1.55.0.0 by mistake) |
| by 2026-11-14 | 1.62.3.3 | Nothing seen | Map read again by `tools/setup.py`; the 2.0.0 deliveries on Linux and Windows were driven on 1.62 |
