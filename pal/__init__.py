"""Platform abstraction layer.

`get_backend()` is the only entry point the rest of the stack needs:

    from pal import get_backend

    backend = get_backend()
    with backend.input_device() as device:
        ...

The backend is chosen from `sys.platform`, overridable with the
`ETS2AD_PAL_BACKEND` environment variable -- which is how tests, CI and offline
replay all run the full stack with no game and no operating system support.
"""

from __future__ import annotations

import os
import sys

from pal.base import (
    GAME_WINDOW_TITLE,
    Backend,
    Frame,
    GamePaths,
    HotkeyAction,
    HotkeyListener,
    OverlayWindow,
    PalError,
    PalNotConnectedError,
    PalResource,
    PalUnsupportedError,
    ScreenCapture,
    SharedMemoryReader,
    SharedMemoryWriter,
    WindowGeometry,
    WindowNotFoundError,
    WindowNotInFrontError,
    WindowOffScreenError,
)

__all__ = [
    "BACKEND_ENV_VAR",
    "GAME_WINDOW_TITLE",
    "Backend",
    "Frame",
    "GamePaths",
    "HotkeyAction",
    "HotkeyListener",
    "OverlayWindow",
    "PalError",
    "PalNotConnectedError",
    "PalResource",
    "PalUnsupportedError",
    "ScreenCapture",
    "SharedMemoryReader",
    "SharedMemoryWriter",
    "WindowGeometry",
    "WindowNotFoundError",
    "WindowNotInFrontError",
    "WindowOffScreenError",
    "get_backend",
]

BACKEND_ENV_VAR = "ETS2AD_PAL_BACKEND"


def get_backend(name: str | None = None, **kwargs: object) -> Backend:
    """Return a backend instance.

    Args:
        name: "linux", "windows" or "fake". Defaults to `ETS2AD_PAL_BACKEND` if
            set, otherwise the running platform.
        **kwargs: passed to the backend constructor.

    Raises:
        PalUnsupportedError: unknown name, or a platform with no backend.

    Backends are imported lazily so that importing `pal` never pulls in another
    platform's dependencies -- `pal.windows` must remain importable on Linux,
    but its libraries need not be installed.
    """
    resolved = (name or os.environ.get(BACKEND_ENV_VAR) or _platform_backend_name()).lower()

    if resolved == "fake":
        from pal.fake import FakeBackend

        return FakeBackend(**kwargs)  # type: ignore[arg-type]
    if resolved == "linux":
        from pal.linux import LinuxBackend

        return LinuxBackend(**kwargs)  # type: ignore[arg-type]
    if resolved == "windows":
        from pal.windows import WindowsBackend

        return WindowsBackend(**kwargs)  # type: ignore[call-arg]

    raise PalUnsupportedError(
        f"unknown PAL backend {resolved!r}; expected one of: fake, linux, windows"
    )


def _platform_backend_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    raise PalUnsupportedError(
        f"no PAL backend for platform {sys.platform!r}; "
        f"set {BACKEND_ENV_VAR}=fake to run without one"
    )
