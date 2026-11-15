#!/usr/bin/env python3
"""Compare `layout_dump` output from two platforms.

The shared memory blocks are an ABI between the plugin and every reader, and
the plugin is built by GCC on Linux and MSVC on Windows. Nothing in the
language guarantees those two agree on where a field lands -- the layout header
is written so that they must, with explicit padding and static assertions, but
"written so that they must" is a claim, and this is the check.

    python scripts/compare_layouts.py linux.json windows.json

Exits non-zero on any disagreement, naming each field. A padding difference
caught here is a build failure; missed, it is telemetry that is correct on one
platform and plausible nonsense on the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Top-level keys that describe the format rather than the layout.
SCALAR_KEYS = (
    "layout_version",
    "magic",
    "shared_memory_name",
    "string_length",
    "max_wheels",
    "max_trailers",
    "command_layout_version",
    "command_magic",
    "command_memory_name",
    "command_watchdog_seconds",
)


def compare(
    left: dict[str, Any], right: dict[str, Any], left_name: str, right_name: str
) -> list[str]:
    problems: list[str] = []

    for key in SCALAR_KEYS:
        if left.get(key) != right.get(key):
            problems.append(f"{key}: {left_name}={left.get(key)!r} {right_name}={right.get(key)!r}")

    left_structs, right_structs = left.get("structs", {}), right.get("structs", {})
    only_left = sorted(left_structs.keys() - right_structs.keys())
    only_right = sorted(right_structs.keys() - left_structs.keys())
    if only_left:
        problems.append(f"structs only in {left_name}: {only_left}")
    if only_right:
        problems.append(f"structs only in {right_name}: {only_right}")

    for name in sorted(left_structs.keys() & right_structs.keys()):
        a, b = left_structs[name], right_structs[name]
        if a["size"] != b["size"]:
            problems.append(f"{name}: size {left_name}={a['size']} {right_name}={b['size']}")
        if a["align"] != b["align"]:
            problems.append(f"{name}: align {left_name}={a['align']} {right_name}={b['align']}")

        fields_a, fields_b = a["fields"], b["fields"]
        for field in sorted(fields_a.keys() | fields_b.keys()):
            if field not in fields_a:
                problems.append(f"{name}.{field}: missing from {left_name}")
            elif field not in fields_b:
                problems.append(f"{name}.{field}: missing from {right_name}")
            elif fields_a[field] != fields_b[field]:
                problems.append(
                    f"{name}.{field}: (offset, size) "
                    f"{left_name}={tuple(fields_a[field])} {right_name}={tuple(fields_b[field])}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))

    problems = compare(left, right, args.left.stem, args.right.stem)
    if problems:
        print(f"ABI MISMATCH between {args.left.stem} and {args.right.stem}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nThe layout header exists to make these identical. Fix the struct, "
            "do not relax the check.",
            file=sys.stderr,
        )
        return 1

    struct_count = len(left.get("structs", {}))
    field_count = sum(len(s["fields"]) for s in left.get("structs", {}).values())
    print(f"layouts identical: {struct_count} structs, {field_count} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
