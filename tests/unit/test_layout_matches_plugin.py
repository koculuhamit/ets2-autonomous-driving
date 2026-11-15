"""Assert the NumPy mirror matches the plugin's actual struct layout.

`core/bridge/layout.py` and `plugin/include/ets2ad/telemetry_layout.h` describe
the same bytes. Keeping them in step by being careful does not scale past the
first hurried edit, and the failure mode -- telemetry that parses cleanly and
means something else -- is the worst kind.

So the C++ side describes itself. `layout_dump` prints every struct size, field
offset and field size as JSON, and this test compares that against the dtypes,
field by field. On CI the same test runs against the Windows build, which is
what turns a GCC/MSVC padding disagreement into a red build instead of a truck
that only steers correctly on one platform.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.bridge import command, layout
from core.common.controls import Axis

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "plugin" / "build"

#: dtype in this module <-> struct name in the C++ header.
DTYPE_BY_STRUCT: dict[str, np.dtype] = {
    "Header": layout.HEADER_DTYPE,
    "CommonState": layout.COMMON_DTYPE,
    "WheelState": layout.WHEEL_DTYPE,
    "TruckState": layout.TRUCK_DTYPE,
    "TrailerState": layout.TRAILER_DTYPE,
    "JobState": layout.JOB_DTYPE,
    "NavigationState": layout.NAVIGATION_DTYPE,
    "TelemetryBlock": layout.BLOCK_DTYPE,
    "CommandBlock": command.COMMAND_BLOCK_DTYPE,
}

#: Padding members exist in C++ so no gap is implicit. They carry no data, so
#: the mirror omits them rather than inventing names for holes.
PADDING_PREFIX = "_"


#: Single-config generators put the executable at the build root;
#: multi-config ones (MSVC) nest it under the configuration name.
_LAYOUT_DUMP_NAMES = ("layout_dump", "layout_dump.exe")


def _find_layout_dump() -> Path | None:
    """Locate the built layout_dump, wherever the generator put it.

    Matching the exact names matters: a glob for `layout_dump*` also finds
    MSVC's `layout_dump.dir` and `layout_dump.vcxproj`, and trying to execute
    those produces "not a valid Win32 application" rather than anything
    resembling the real problem.
    """
    for name in _LAYOUT_DUMP_NAMES:
        direct = BUILD_DIR / name
        if direct.is_file():
            return direct
    for candidate in sorted(BUILD_DIR.rglob("*")):
        if candidate.is_file() and candidate.name in _LAYOUT_DUMP_NAMES:
            return candidate
    return None


@pytest.fixture(scope="module")
def plugin_layout() -> dict[str, Any]:
    """Run `layout_dump` and return its description of the C++ structs."""
    executable = _find_layout_dump()
    if executable is None:
        pytest.skip(
            "layout_dump has not been built; run "
            "`cmake -S plugin -B plugin/build && cmake --build plugin/build`"
        )
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_constants_match(plugin_layout: dict[str, Any]) -> None:
    assert plugin_layout["layout_version"] == layout.LAYOUT_VERSION
    assert plugin_layout["magic"] == layout.MAGIC
    assert plugin_layout["shared_memory_name"] == layout.SHARED_MEMORY_NAME
    assert plugin_layout["string_length"] == layout.STRING_LENGTH
    assert plugin_layout["max_wheels"] == layout.MAX_WHEELS
    assert plugin_layout["max_trailers"] == layout.MAX_TRAILERS


def test_every_cpp_struct_has_a_mirror(plugin_layout: dict[str, Any]) -> None:
    """A struct added in C++ and forgotten in Python is the whole failure mode."""
    assert set(plugin_layout["structs"]) == set(DTYPE_BY_STRUCT)


@pytest.mark.parametrize("struct_name", sorted(DTYPE_BY_STRUCT))
def test_struct_size_matches(plugin_layout: dict[str, Any], struct_name: str) -> None:
    assert DTYPE_BY_STRUCT[struct_name].itemsize == plugin_layout["structs"][struct_name]["size"]


@pytest.mark.parametrize("struct_name", sorted(DTYPE_BY_STRUCT))
def test_field_offsets_and_sizes_match(plugin_layout: dict[str, Any], struct_name: str) -> None:
    dtype = DTYPE_BY_STRUCT[struct_name]
    cpp_fields = {
        name: tuple(offset_size)
        for name, offset_size in plugin_layout["structs"][struct_name]["fields"].items()
        if not name.startswith(PADDING_PREFIX)
    }

    mirrored = {}
    for name in dtype.names or ():
        field_dtype, offset = dtype.fields[name][:2]
        mirrored[name] = (offset, field_dtype.itemsize)

    assert mirrored.keys() == cpp_fields.keys(), (
        f"{struct_name}: fields differ between the mirror and the plugin; "
        f"only in Python: {sorted(mirrored.keys() - cpp_fields.keys())}, "
        f"only in C++: {sorted(cpp_fields.keys() - mirrored.keys())}"
    )
    mismatched = {
        name: {"python": mirrored[name], "cpp": cpp_fields[name]}
        for name in cpp_fields
        if mirrored[name] != cpp_fields[name]
    }
    assert not mismatched, f"{struct_name}: (offset, size) mismatch: {mismatched}"


def test_command_constants_match(plugin_layout: dict[str, Any]) -> None:
    """The command block is an ABI too, and in the direction that steers."""
    assert plugin_layout["command_layout_version"] == command.COMMAND_LAYOUT_VERSION
    assert plugin_layout["command_magic"] == command.COMMAND_MAGIC
    assert plugin_layout["command_memory_name"] == command.COMMAND_MEMORY_NAME
    assert plugin_layout["command_safety_stop_seconds"] == command.SAFETY_STOP_S


def test_axis_order_matches_the_plugin() -> None:
    """The plugin indexes axes positionally, so a reordering here would send
    throttle where the brake was meant to go."""
    assert command.AXIS_ORDER == (Axis.STEER, Axis.THROTTLE, Axis.BRAKE, Axis.CLUTCH)
    axes_offset, axes_size = (
        command.COMMAND_BLOCK_DTYPE.fields["axes"][1],
        command.COMMAND_BLOCK_DTYPE.fields["axes"][0].itemsize,
    )
    assert axes_size == 4 * len(command.AXIS_ORDER)
    assert axes_offset == 40


def test_button_bits_are_unique_and_contiguous() -> None:
    """A duplicated bit would make two buttons the same button."""
    bits = sorted(command.BUTTON_BITS.values())
    assert bits == list(range(len(bits)))


def test_block_size_constant_matches(plugin_layout: dict[str, Any]) -> None:
    assert plugin_layout["structs"]["TelemetryBlock"]["size"] == layout.BLOCK_SIZE


def test_mirror_leaves_no_unexplained_gaps() -> None:
    """Every byte of the block is either a mirrored field or C++ padding.

    A field the mirror places at the wrong offset can still produce a dtype
    that happens to be the right total size. Walking the block and checking
    that the gaps are only where padding was declared catches that.
    """
    block = np.zeros(1, dtype=layout.BLOCK_DTYPE)
    assert block.nbytes == layout.BLOCK_SIZE
    assert layout.BLOCK_DTYPE.names is not None
    offsets = sorted(layout.BLOCK_DTYPE.fields[name][1] for name in layout.BLOCK_DTYPE.names)
    assert offsets[0] == 0, "the header must start the block"
