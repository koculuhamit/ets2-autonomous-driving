"""The messages the stack sends the ROS 2 bridge, and how they are framed.

The stack runs on Python 3.11 or later; ROS 2 Humble's rclpy is built for
the system's Python 3.10. So the two never share a process: the stack writes
these messages to a Unix socket (adapters/ros2/feed.py) and the bridge node
(ets2ad_ros.bridge) reads them and publishes topics. This module is the one
both sides import, and it keeps to the standard library and to Python 3.10.

A frame is

    MAGIC (4 bytes) | header length (u32 LE) | blob length (u32 LE)
    | header (UTF-8 JSON) | blob

The header is a JSON object with at least "kind" and "t" (seconds on the
drive's clock); the blob carries what does not belong in JSON: a JPEG.
Coordinates in the messages are the game's own (x east, y up, z south,
metres); ets2ad_ros.frames turns them into ROS's.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"E2R1"
"""Bumped with any change to the framing or to a message's meaning."""

_LENGTHS = struct.Struct("<II")
_PREFIX = len(MAGIC) + _LENGTHS.size

MAX_HEADER = 4 * 1024 * 1024
"""A route a thousand kilometres long fits; anything larger is a broken stream."""

MAX_BLOB = 16 * 1024 * 1024

# What a message describes. Each one stands alone: the bridge keeps no state
# between messages of a kind but the latest.
POSE = "pose"
"""The truck and its trailer, from telemetry: every turn of the stack's loop."""

ROUTE = "route"
"""The line the truck is to drive, as points: when it changes."""

LANES = "lanes"
"""The map's lanes round the truck, as polylines: every few seconds."""

OBJECTS = "objects"
"""What perception sees: vehicles ahead, the traffic light, the mirror."""

IMAGE = "image"
"""A view of the road, JPEG in the blob, with the camera it came from."""

STATUS = "status"
"""What the stack is doing, in words: its state and its last line."""

KINDS = (POSE, ROUTE, LANES, OBJECTS, IMAGE, STATUS)


class WireError(ValueError):
    """The stream is not this format, or not this version of it."""


@dataclass(frozen=True)
class Message:
    kind: str
    t: float
    fields: dict = field(default_factory=dict)
    blob: bytes = b""


def encode(message: Message) -> bytes:
    header = dict(message.fields)
    header["kind"] = message.kind
    header["t"] = float(message.t)
    text = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(text) > MAX_HEADER or len(message.blob) > MAX_BLOB:
        raise WireError(f"a {message.kind} message too large to send")
    return MAGIC + _LENGTHS.pack(len(text), len(message.blob)) + text + message.blob


class Decoder:
    """Turns a byte stream into messages, however it was split."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[Message]:
        self._buffer += data
        messages = []
        while len(self._buffer) >= _PREFIX:
            if bytes(self._buffer[: len(MAGIC)]) != MAGIC:
                raise WireError(
                    f"expected {MAGIC!r}, got {bytes(self._buffer[: len(MAGIC)])!r}: "
                    "not this stack, or another version of it"
                )
            header_length, blob_length = _LENGTHS.unpack_from(self._buffer, len(MAGIC))
            if header_length > MAX_HEADER or blob_length > MAX_BLOB:
                raise WireError(f"a frame of {header_length} + {blob_length} bytes")
            end = _PREFIX + header_length + blob_length
            if len(self._buffer) < end:
                break
            header = json.loads(bytes(self._buffer[_PREFIX : _PREFIX + header_length]))
            blob = bytes(self._buffer[_PREFIX + header_length : end])
            del self._buffer[:end]
            kind = header.pop("kind")
            t = float(header.pop("t"))
            messages.append(Message(kind, t, header, blob))
        return messages


def socket_path() -> Path:
    """Where the bridge listens: $ETS2AD_ROS_SOCKET, else a file in the
    user's own runtime directory, which no other user can enter."""
    named = os.environ.get("ETS2AD_ROS_SOCKET")
    if named:
        return Path(named)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        return Path(runtime) / "ets2ad-ros.sock"
    return Path.home() / ".cache" / "ets2ad" / "ros.sock"
