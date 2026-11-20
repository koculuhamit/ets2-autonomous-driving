"""What a drive says, kept beside its recording.

tools/follow.py tells the driver why it does what it does -- a lane change
held for a car in the mirror, a route the GPS gave and the drive refused --
on its console, and nowhere else: a drive looked at afterwards could only
be guessed at (a turn in Bern not taken, 15 November). `MessageLog` passes
what is written to a stream on, and writes each finished line, stamped
with the drive's clock, to a file too. The live line rewritten in place
with a carriage return a few times a second is the console's alone.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

MESSAGES_FILE = "messages.txt"


class MessageLog(io.TextIOBase):
    """A stream that also keeps its finished lines in `file`."""

    def __init__(
        self, stream: TextIO, file: TextIO, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.stream = stream
        self._file = file
        self._clock = clock
        self._pending = ""
        self.origin: float | None = None
        """Where the drive's clock starts, on `clock`; until set, lines are
        stamped from when the log was made."""
        self._made = clock()

    def write(self, text: str) -> int:
        self.stream.write(text)
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            if "\r" in line:
                continue  # the live line, rewritten in place
            if line.strip():
                at = self._clock() - (self.origin if self.origin is not None else self._made)
                self._file.write(f"{at:9.2f}  {line}\n")
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.stream, "encoding", "utf-8")


def keep_messages(folder: Path, streams: dict[str, TextIO]) -> tuple[TextIO, dict[str, MessageLog]]:
    """`MessageLog`s over each of `streams` (by name: stdout, stderr), all
    into `folder`/messages.txt; the file, to close, and the logs."""
    folder.mkdir(parents=True, exist_ok=True)
    file = (folder / MESSAGES_FILE).open("a", encoding="utf-8", buffering=1)
    return file, {name: MessageLog(stream, file) for name, stream in streams.items()}
