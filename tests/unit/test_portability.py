"""Enforce the portability rules the project depends on.

`core/` must stay platform-independent: everything that differs between Linux
and Windows lives behind `pal/`, and ROS 2 is an adapter, not the backbone.
These rules are cheap to state and easy to violate by accident, so they are
checked mechanically rather than in review.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages that must never leak an OS-specific dependency.
PLATFORM_FREE_PACKAGES = ("core",)

#: Modules that only exist, or only behave sensibly, on one operating system.
OS_SPECIFIC_MODULES = frozenset(
    {
        # POSIX-only
        "fcntl",
        "termios",
        "tty",
        "pty",
        "grp",
        "pwd",
        "posix",
        "evdev",
        "Xlib",
        "uinput",
        # Windows-only
        "msvcrt",
        "winreg",
        "nt",
        "winsound",
        "win32api",
        "win32con",
        "win32gui",
        "win32process",
        "pywintypes",
        "vgamepad",
        "dxcam",
    }
)

#: ROS 2 belongs in adapters/, never in the core stack.
ROS_MODULES = frozenset({"rclpy", "rosbag2_py", "rclcpp"})


def _python_files(package: str) -> Iterator[Path]:
    yield from sorted((REPO_ROOT / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level name of every module imported by `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # `from . import x` has no module; relative imports are always local.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _violations(forbidden: frozenset[str]) -> list[str]:
    found = []
    for package in PLATFORM_FREE_PACKAGES:
        for path in _python_files(package):
            for module in sorted(_imported_roots(path) & forbidden):
                found.append(f"{path.relative_to(REPO_ROOT)}: imports {module!r}")
    return found


def test_core_has_no_os_specific_imports() -> None:
    violations = _violations(OS_SPECIFIC_MODULES)
    assert not violations, (
        "core/ must stay platform-independent; move this behind pal/:\n  " + "\n  ".join(violations)
    )


def test_core_does_not_import_ros() -> None:
    violations = _violations(ROS_MODULES)
    assert not violations, (
        "ROS 2 is an adapter, not the backbone; move this into adapters/ros2/:\n  "
        + "\n  ".join(violations)
    )


def test_pal_package_does_not_shadow_stdlib_platform() -> None:
    """ADR-0004: the PAL lives in pal/, because a top-level platform/ package
    would shadow the standard library's platform module for the whole process."""
    import platform as stdlib_platform

    assert Path(stdlib_platform.__file__).name == "platform.py"
    assert not (REPO_ROOT / "platform").exists()
    assert (REPO_ROOT / "pal" / "__init__.py").exists()
