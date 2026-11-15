"""Recording and replaying sessions.

Recording is what makes everything above the bridge testable. Controllers are
tuned against recorded runs rather than against a game that has to be running,
CI exercises the real reader on real data with no GPU and no X server, and a
failure gets reproduced instead of described.

    from core.recording import Recorder, open_recording

    with Recorder(Path("data/runs/a2-east"), notes="highway, dry") as rec:
        rec.write_telemetry(reader.read())

    recording = open_recording("data/runs/a2-east")
    speed = recording.telemetry["block"]["truck"]["speed"]

To run the live stack against a recording, give `TelemetryReader` a backend
whose shared memory is a `ReplaySharedMemoryReader`. The same reader, the same
seqlock check, the same parsing -- offline tests exercise the production path
rather than a parallel one that agrees with it for now.
"""

from core.recording.format import (
    COMMAND_RECORD_DTYPE,
    FORMAT_VERSION,
    TELEMETRY_RECORD_DTYPE,
    RecordingFormatError,
)
from core.recording.reader import Recording, ReplaySharedMemoryReader, open_recording
from core.recording.replay import ReplayBackend, replay_frames
from core.recording.writer import Recorder

__all__ = [
    "COMMAND_RECORD_DTYPE",
    "FORMAT_VERSION",
    "TELEMETRY_RECORD_DTYPE",
    "Recorder",
    "Recording",
    "RecordingFormatError",
    "ReplayBackend",
    "ReplaySharedMemoryReader",
    "open_recording",
    "replay_frames",
]
