"""NumPy mirror of the plugin's shared memory layout.

This describes the same bytes as `plugin/include/ets2ad/telemetry_layout.h`.
Two descriptions of one binary format drift apart the moment somebody edits one
of them, and the symptom -- plausible but wrong telemetry -- is miserable to
debug. So the C++ side is the source of truth and this mirror is checked
against it: `plugin/tools/layout_dump` prints every offset and size, and
`tests/unit/test_layout_matches_plugin.py` asserts that the dtypes below agree,
field by field.

Offsets are stated explicitly rather than inferred. NumPy would happily pack
these structs its own way; saying where every field goes means a mismatch is a
failing test rather than silently shifted data.

Byte order is pinned little-endian. Both target platforms are x86-64, but a
wire format with implied endianness is a trap left for somebody else.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "BLOCK_DTYPE",
    "BLOCK_SIZE",
    "COMMON_DTYPE",
    "HEADER_DTYPE",
    "JOB_DTYPE",
    "LAYOUT_VERSION",
    "MAGIC",
    "MAX_TRAILERS",
    "MAX_WHEELS",
    "NAVIGATION_DTYPE",
    "ORIENTATION_DTYPE",
    "PLACEMENT_DTYPE",
    "SHARED_MEMORY_NAME",
    "STRING_LENGTH",
    "TRAILER_DTYPE",
    "TRUCK_DTYPE",
    "VECTOR3D_DTYPE",
    "VECTOR3F_DTYPE",
    "WHEEL_DTYPE",
    "GameId",
    "HeaderFlag",
]

#: "ET2A" little-endian. A block without this was never written by our plugin.
MAGIC: Final = 0x41325445

#: Bumped whenever any field moves. Readers refuse anything else.
LAYOUT_VERSION: Final = 1

SHARED_MEMORY_NAME: Final = "ets2ad_telemetry"
STRING_LENGTH: Final = 64
MAX_WHEELS: Final = 16
MAX_TRAILERS: Final = 3


class HeaderFlag:
    """Bits in the header's `flags` field."""

    TELEMETRY_ACTIVE: Final = 1 << 0
    PAUSED: Final = 1 << 1
    TIMESTAMPS_VALID: Final = 1 << 2


class GameId:
    """Values of the header's `game_id` field."""

    UNKNOWN: Final = 0
    ETS2: Final = 1
    ATS: Final = 2


def _struct(fields: list[tuple[str, object, int]], itemsize: int) -> np.dtype:
    """Build a dtype with explicit offsets and a fixed size.

    Args:
        fields: (name, format, offset) triples.
        itemsize: total size including any trailing padding.
    """
    return np.dtype(
        {
            "names": [name for name, _, _ in fields],
            "formats": [fmt for _, fmt, _ in fields],
            "offsets": [offset for _, _, offset in fields],
            "itemsize": itemsize,
        }
    )


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

VECTOR3F_DTYPE: Final = _struct([("x", "<f4", 0), ("y", "<f4", 4), ("z", "<f4", 8)], 12)
VECTOR3D_DTYPE: Final = _struct([("x", "<f8", 0), ("y", "<f8", 8), ("z", "<f8", 16)], 24)

#: Orientation in *rotations*, not radians: 0.0 to 1.0 is a full turn. Heading
#: grows counter-clockwise from north. `core.bridge.convert` turns this into a
#: conventional yaw, in exactly one place.
ORIENTATION_DTYPE: Final = _struct(
    [("heading", "<f4", 0), ("pitch", "<f4", 4), ("roll", "<f4", 8)], 12
)

PLACEMENT_DTYPE: Final = _struct(
    [("position", VECTOR3D_DTYPE, 0), ("orientation", ORIENTATION_DTYPE, 24)], 40
)

# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

HEADER_DTYPE: Final = _struct(
    [
        ("magic", "<u4", 0),
        ("layout_version", "<u4", 4),
        ("block_size", "<u4", 8),
        ("sequence", "<u4", 12),
        ("flags", "<u4", 16),
        ("game_id", "<u4", 20),
        ("game_version", "<u4", 24),
        ("sdk_version", "<u4", 28),
        ("write_count", "<u8", 32),
        ("write_time", "<f8", 40),
    ],
    80,
)

COMMON_DTYPE: Final = _struct(
    [("game_time", "<u4", 0), ("next_rest_stop", "<i4", 4), ("scale", "<f4", 8)], 16
)

WHEEL_DTYPE: Final = _struct(
    [
        ("steering", ("<f4", MAX_WHEELS), 0),
        ("rotation", ("<f4", MAX_WHEELS), 64),
        ("lift", ("<f4", MAX_WHEELS), 128),
        ("suspension", ("<f4", MAX_WHEELS), 192),
        ("position", (VECTOR3F_DTYPE, MAX_WHEELS), 256),
        ("radius", ("<f4", MAX_WHEELS), 448),
        ("on_ground", ("u1", MAX_WHEELS), 512),
        ("steerable", ("u1", MAX_WHEELS), 528),
        ("powered", ("u1", MAX_WHEELS), 544),
        ("liftable", ("u1", MAX_WHEELS), 560),
        ("count", "<u4", 576),
    ],
    592,
)

TRUCK_DTYPE: Final = _struct(
    [
        ("placement", PLACEMENT_DTYPE, 0),
        ("linear_velocity", VECTOR3F_DTYPE, 40),
        ("angular_velocity", VECTOR3F_DTYPE, 52),
        ("linear_acceleration", VECTOR3F_DTYPE, 64),
        ("angular_acceleration", VECTOR3F_DTYPE, 76),
        ("cabin_offset_position", VECTOR3F_DTYPE, 88),
        ("head_offset_position", VECTOR3F_DTYPE, 100),
        ("speed", "<f4", 112),
        ("engine_rpm", "<f4", 116),
        ("engine_rpm_max", "<f4", 120),
        ("input_steering", "<f4", 124),
        ("input_throttle", "<f4", 128),
        ("input_brake", "<f4", 132),
        ("input_clutch", "<f4", 136),
        ("effective_steering", "<f4", 140),
        ("effective_throttle", "<f4", 144),
        ("effective_brake", "<f4", 148),
        ("effective_clutch", "<f4", 152),
        ("cruise_control_speed", "<f4", 156),
        ("fuel", "<f4", 160),
        ("fuel_capacity", "<f4", 164),
        ("fuel_average_consumption", "<f4", 168),
        ("fuel_range", "<f4", 172),
        ("adblue", "<f4", 176),
        ("adblue_capacity", "<f4", 180),
        ("brake_air_pressure", "<f4", 184),
        ("brake_temperature", "<f4", 188),
        ("oil_pressure", "<f4", 192),
        ("oil_temperature", "<f4", 196),
        ("water_temperature", "<f4", 200),
        ("battery_voltage", "<f4", 204),
        ("wear_engine", "<f4", 208),
        ("wear_transmission", "<f4", 212),
        ("wear_cabin", "<f4", 216),
        ("wear_chassis", "<f4", 220),
        ("wear_wheels", "<f4", 224),
        ("odometer", "<f4", 228),
        ("hook_position", ("<f4", 3), 232),
        ("gear", "<i4", 244),
        ("gear_displayed", "<i4", 248),
        ("forward_gear_count", "<i4", 252),
        ("reverse_gear_count", "<i4", 256),
        ("retarder_level", "<i4", 260),
        ("retarder_step_count", "<u4", 264),
        ("engine_enabled", "u1", 268),
        ("electric_enabled", "u1", 269),
        ("parking_brake", "u1", 270),
        ("motor_brake", "u1", 271),
        ("indicator_left", "u1", 272),
        ("indicator_right", "u1", 273),
        ("light_parking", "u1", 274),
        ("light_low_beam", "u1", 275),
        ("light_high_beam", "u1", 276),
        ("light_brake", "u1", 277),
        ("light_reverse", "u1", 278),
        ("light_beacon", "u1", 279),
        ("wipers", "u1", 280),
        ("cruise_control_active", "u1", 281),
        ("hazard_warning", "u1", 282),
        ("wheels", WHEEL_DTYPE, 284),
        ("brand_id", f"S{STRING_LENGTH}", 876),
        ("model_id", f"S{STRING_LENGTH}", 940),
    ],
    1008,
)

TRAILER_DTYPE: Final = _struct(
    [
        ("placement", PLACEMENT_DTYPE, 0),
        ("linear_velocity", VECTOR3F_DTYPE, 40),
        ("angular_velocity", VECTOR3F_DTYPE, 52),
        ("linear_acceleration", VECTOR3F_DTYPE, 64),
        ("angular_acceleration", VECTOR3F_DTYPE, 76),
        ("hook_position", ("<f4", 3), 88),
        ("cargo_damage", "<f4", 100),
        ("wear_chassis", "<f4", 104),
        ("wear_wheels", "<f4", 108),
        ("axle_count", "<u4", 112),
        ("connected", "u1", 116),
        ("wheels", WHEEL_DTYPE, 120),
        ("id", f"S{STRING_LENGTH}", 712),
        ("cargo_accessory_id", f"S{STRING_LENGTH}", 776),
    ],
    848,
)

JOB_DTYPE: Final = _struct(
    [
        ("cargo_mass", "<f4", 0),
        ("cargo_unit_mass", "<f4", 4),
        ("cargo_unit_count", "<u4", 8),
        ("income", "<u4", 12),
        ("delivery_time", "<u4", 16),
        ("planned_distance", "<u4", 20),
        ("cargo_loaded", "u1", 24),
        ("special_job", "u1", 25),
        ("job_active", "u1", 26),
        ("cargo_id", f"S{STRING_LENGTH}", 32),
        ("cargo", f"S{STRING_LENGTH}", 96),
        ("source_city", f"S{STRING_LENGTH}", 160),
        ("source_company", f"S{STRING_LENGTH}", 224),
        ("destination_city", f"S{STRING_LENGTH}", 288),
        ("destination_company", f"S{STRING_LENGTH}", 352),
    ],
    416,
)

NAVIGATION_DTYPE: Final = _struct(
    [("distance", "<f4", 0), ("time", "<f4", 4), ("speed_limit", "<f4", 8)], 16
)

BLOCK_DTYPE: Final = _struct(
    [
        ("header", HEADER_DTYPE, 0),
        ("common", COMMON_DTYPE, 80),
        ("truck", TRUCK_DTYPE, 96),
        ("trailers", (TRAILER_DTYPE, MAX_TRAILERS), 1104),
        ("job", JOB_DTYPE, 3648),
        ("navigation", NAVIGATION_DTYPE, 4064),
    ],
    4080,
)

#: Size of the shared memory block, in bytes.
BLOCK_SIZE: Final = BLOCK_DTYPE.itemsize
