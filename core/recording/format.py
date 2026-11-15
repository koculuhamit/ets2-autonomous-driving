"""On-disk format for recorded sessions.

A recording is a directory, not a file:

    2026-09-20T16-30-00_highway/
      manifest.json     metadata, counts, layout versions
      telemetry.bin     flat array of fixed-size records
      commands.bin      flat array of fixed-size records
      frames.bin        flat array of fixed-size records, one per image
      images/           the captured game window, 000000.jpg, 000001.jpg, ...

The streams are flat arrays of fixed-size records and nothing else -- no
framing, no compression, no header. That buys three things worth more than the
disk space it costs:

* **`np.memmap` reads them directly.** A recording opens in constant time
  regardless of length, and a replay that only wants the last thirty seconds
  pays for the last thirty seconds.
* **Seeking is arithmetic.** Record *n* is at offset *n * itemsize*. Binary
  search by timestamp needs no index to maintain and no index to get wrong.
* **A truncated file is still a recording.** A process killed mid-write leaves
  a partial final record; whole records before it are unaffected, and the
  reader simply stops at the last complete one.

The records embed the shared memory blocks verbatim, so a recorded frame and a
live one are the same bytes and the same dtype. Replay code and live code are
therefore not two code paths that happen to agree.

Images are the one stream that cannot be fixed-size records: a frame of the
game window is megabytes raw and a few hundred kilobytes as a JPEG, so each
is its own file, named by its index in `frames.bin`. The record is written
after the file, so a record always has its image; a writer killed between
the two leaves an image no record points at, which is ignored. Recordings
without images -- all of them, before perception -- simply have no frames.

The manifest records the layout versions the blocks were written with. A
recording made by an older plugin is rejected with an explanation rather than
reinterpreted under the current layout.

The cost of storing blocks verbatim is size: 4088 bytes per frame at 60 Hz is
about 0.9 GB per hour, so Phase 11's long endurance runs will need either
decimation or compression. That is a later problem and a cheap one -- the
records are fixed-size and independent, so either can be added without
disturbing anything that reads them. Paying it now, in complexity, to save disk
on runs that do not exist yet would be the wrong trade.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from core.bridge.command import COMMAND_BLOCK_DTYPE, COMMAND_LAYOUT_VERSION
from core.bridge.layout import BLOCK_DTYPE, LAYOUT_VERSION

__all__ = [
    "COMMANDS_FILE",
    "COMMAND_RECORD_DTYPE",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "FRAMES_FILE",
    "FRAME_RECORD_DTYPE",
    "IMAGES_DIR",
    "MANIFEST_FILE",
    "TELEMETRY_FILE",
    "TELEMETRY_RECORD_DTYPE",
    "RecordingFormatError",
    "image_name",
]

FORMAT_NAME: Final = "ets2ad-recording"

#: Bumped when the directory layout or the record framing changes. The block
#: layouts have their own versions and are recorded separately.
FORMAT_VERSION: Final = 1

MANIFEST_FILE: Final = "manifest.json"
TELEMETRY_FILE: Final = "telemetry.bin"
COMMANDS_FILE: Final = "commands.bin"
FRAMES_FILE: Final = "frames.bin"
IMAGES_DIR: Final = "images"

#: One recorded telemetry frame.
#:
#: `t` is seconds since the recording started, on a monotonic clock. Absolute
#: time is in the manifest: `perf_counter` values are meaningless across
#: processes, so storing them raw would produce a timeline that only made sense
#: to the process that wrote it.
TELEMETRY_RECORD_DTYPE: Final = np.dtype([("t", "<f8"), ("block", BLOCK_DTYPE)])

#: One recorded command frame, same convention.
COMMAND_RECORD_DTYPE: Final = np.dtype([("t", "<f8"), ("block", COMMAND_BLOCK_DTYPE)])

#: One captured image of the game window, same convention. `sequence` is the
#: capture's own counter, so gaps show grabs that were not recorded.
FRAME_RECORD_DTYPE: Final = np.dtype(
    [("t", "<f8"), ("sequence", "<i8"), ("width", "<u2"), ("height", "<u2")]
)


def image_name(index: int) -> str:
    """The file, under `IMAGES_DIR`, of the image recorded as frame `index`."""
    return f"{index:06d}.jpg"


class RecordingFormatError(Exception):
    """The directory is not a recording this build can read."""


def manifest_template(started_at: str) -> dict[str, object]:
    """A manifest with everything the reader needs to validate itself."""
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "started_at": started_at,
        "duration": 0.0,
        "telemetry": {
            "file": TELEMETRY_FILE,
            "count": 0,
            "record_size": TELEMETRY_RECORD_DTYPE.itemsize,
            "layout_version": LAYOUT_VERSION,
        },
        "commands": {
            "file": COMMANDS_FILE,
            "count": 0,
            "record_size": COMMAND_RECORD_DTYPE.itemsize,
            "layout_version": COMMAND_LAYOUT_VERSION,
        },
        "frames": {
            "file": FRAMES_FILE,
            "images": IMAGES_DIR,
            "encoding": "jpeg",
            "count": 0,
            "record_size": FRAME_RECORD_DTYPE.itemsize,
        },
    }
