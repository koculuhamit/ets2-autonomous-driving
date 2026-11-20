# ETS2 autopilot 2.1.3

An uninstaller. Driving is as in [2.1.2](release-notes-2.1.2.md).

Install: [Windows 11](install-windows.md) · [Linux](install-linux.md).

## New

- **`uninstall.bat` / `uninstall.sh`**, beside the launcher. It asks about
  each thing the launcher put on the machine in turn, and keeps whatever is
  not answered `y` to:
  - the plugin in the game's folder (with the game closed),
  - uv's downloads of this project's packages, CUDA's among them, about
    2.5 GB with the GPU build,
  - the Python uv fetched for it,
  - uv itself,
  - this folder, with its packages, the map and the recorded drives.

  Deleting the folder alone left those downloads behind. On the same disk
  they share their space with the folder's packages, so it did not come
  back. The install guides now say what is where, and how much.
  `python tools/uninstall.py --dry-run` lists it all and takes nothing.

## Known limits

As in [2.1.0](release-notes-2.1.0.md).
