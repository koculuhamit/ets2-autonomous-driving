"""What the plugin does when the stack goes quiet: its own checks, built with
the plugin (plugin/tools/safety_test.cpp), run here so CI fails on them."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

BUILD_DIR = Path(__file__).resolve().parents[2] / "plugin" / "build"
NAMES = ("safety_test", "safety_test.exe")


def _find() -> Path | None:
    for candidate in sorted(BUILD_DIR.rglob("*")) if BUILD_DIR.is_dir() else ():
        if candidate.is_file() and candidate.name in NAMES:
            return candidate
    return None


def test_a_quiet_stack_has_the_truck_held_then_stopped_then_left_to_the_driver() -> None:
    executable = _find()
    if executable is None:
        pytest.skip("safety_test has not been built: cmake --build plugin/build")
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout
    assert "ok" in result.stdout
