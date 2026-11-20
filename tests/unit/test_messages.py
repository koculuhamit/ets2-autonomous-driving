"""A drive's console lines, kept beside its recording."""

from __future__ import annotations

import io
import sys

from core.recording.messages import MESSAGES_FILE, MessageLog, keep_messages


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_finished_lines_are_kept_stamped_and_the_live_line_is_not() -> None:
    console, kept, clock = io.StringIO(), io.StringIO(), Clock()
    log = MessageLog(console, kept, clock)
    log.origin = 90.0
    log.write("\r 54.0/80.0 km/h  lat +0.10 m")
    log.write("\r 55.0/80.0 km/h  lat +0.12 m")
    clock.now = 102.5
    print("\n[GPS: the red line leaves the route 120 m on]", file=log)
    log.write("half a ")
    log.write("line\n")
    assert console.getvalue().count("km/h") == 2, "the console sees everything"
    assert kept.getvalue().splitlines() == [
        "    12.50  [GPS: the red line leaves the route 120 m on]",
        "    12.50  half a line",
    ]


def test_both_streams_go_to_one_file_in_the_drive_s_folder(tmp_path) -> None:
    file, logs = keep_messages(tmp_path, {"stdout": io.StringIO(), "stderr": io.StringIO()})
    logs["stdout"].write("out\n")
    logs["stderr"].write("err\n")
    file.close()
    lines = (tmp_path / MESSAGES_FILE).read_text().splitlines()
    assert [line.split()[-1] for line in lines] == ["out", "err"]


def test_the_console_s_streams_come_back(tmp_path) -> None:
    from tools.follow import stop_keeping

    before = sys.stdout, sys.stderr
    file, logs = keep_messages(tmp_path, {"stdout": sys.stdout, "stderr": sys.stderr})
    sys.stdout, sys.stderr = logs["stdout"], logs["stderr"]
    try:
        print("kept")
    finally:
        stop_keeping(file)
    assert (sys.stdout, sys.stderr) == before
    assert file.closed and "kept" in (tmp_path / MESSAGES_FILE).read_text()
