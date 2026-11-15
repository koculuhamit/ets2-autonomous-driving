"""Structural contract every PAL backend must satisfy.

These run against *all* backends, including the ones that are still skeletons.
They do not check that a capability works -- they check that it is declared,
wired up, and fails in the agreed way when it cannot work yet. That is what
stops Phase 12 from turning into an archaeology exercise.

The two rules with teeth:

* `neutral()` and `close()` never raise, on any backend, in any state. Both run
  on the panic path and inside `finally` blocks, where an exception would mask
  the failure that got us there.
* A capability that is merely unwritten raises `NotImplementedError`; one that
  the platform genuinely cannot provide raises `PalUnsupportedError`. Anything
  else -- `AttributeError`, `TypeError` -- means the abstraction has a hole.
"""

from __future__ import annotations

import contextlib
import inspect

import pytest

from pal.base import (
    Backend,
    GamePaths,
    HotkeyListener,
    OverlayWindow,
    PalError,
    PalResource,
    ScreenCapture,
    SharedMemoryReader,
    SharedMemoryWriter,
)
from pal.fake import FakeBackend
from pal.linux import LinuxBackend
from pal.windows import WindowsBackend

BACKEND_CLASSES = (FakeBackend, LinuxBackend, WindowsBackend)

#: (factory name, arguments, declared interface)
FACTORIES = (
    ("shared_memory", ("scs_telemetry", 32), SharedMemoryReader),
    ("shared_memory_writer", ("ets2ad_command", 32), SharedMemoryWriter),
    ("screen_capture", ("Euro Truck Simulator 2",), ScreenCapture),
    ("hotkeys", (), HotkeyListener),
    ("game_paths", (), GamePaths),
    ("overlay", (), OverlayWindow),
)


@pytest.fixture(params=BACKEND_CLASSES, ids=lambda cls: cls.name)
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param()


def test_backend_is_fully_concrete(backend: Backend) -> None:
    """No abstract method left unimplemented -- a missing stub is a design gap."""
    leftover = getattr(type(backend), "__abstractmethods__", frozenset())
    assert not leftover, f"{type(backend).__name__} does not implement: {sorted(leftover)}"


def test_backend_has_a_name(backend: Backend) -> None:
    assert backend.name and backend.name == backend.name.lower()
    assert backend.name in repr(backend)


@pytest.mark.parametrize(("factory", "args", "interface"), FACTORIES, ids=lambda x: str(x)[:20])
def test_factory_returns_declared_interface(
    backend: Backend, factory: str, args: tuple[object, ...], interface: type
) -> None:
    resource = getattr(backend, factory)(*args)
    assert isinstance(resource, interface)
    leftover = getattr(type(resource), "__abstractmethods__", frozenset())
    assert not leftover, f"{type(resource).__name__} does not implement: {sorted(leftover)}"


@pytest.mark.parametrize(("factory", "args", "interface"), FACTORIES, ids=lambda x: str(x)[:20])
def test_resources_support_the_context_manager_protocol(
    backend: Backend, factory: str, args: tuple[object, ...], interface: type
) -> None:
    """Lifetimes are the caller's to manage, so `with` has to work everywhere."""
    resource = getattr(backend, factory)(*args)
    if not isinstance(resource, PalResource):
        assert isinstance(resource, GamePaths)  # the one stateless interface
        return
    assert hasattr(resource, "__enter__") and hasattr(resource, "__exit__")


def test_close_is_idempotent_and_silent(backend: Backend) -> None:
    """`close()` runs in `finally` blocks and on error paths, where raising
    would mask the failure that got us there."""
    for factory, args, _ in FACTORIES:
        resource = getattr(backend, factory)(*args)
        if not isinstance(resource, PalResource):
            continue
        resource.close()
        resource.close()
        assert resource.is_open is False


@pytest.mark.parametrize(("factory", "args", "interface"), FACTORIES, ids=lambda x: str(x)[:20])
def test_unimplemented_capabilities_fail_in_the_agreed_way(
    backend: Backend, factory: str, args: tuple[object, ...], interface: type
) -> None:
    """Skeletons raise NotImplementedError or PalError -- never AttributeError."""
    resource = getattr(backend, factory)(*args)
    if not isinstance(resource, PalResource):
        return
    try:
        with contextlib.suppress(NotImplementedError, PalError):
            resource.open()
    finally:
        resource.close()


def test_every_interface_method_is_documented() -> None:
    """The PAL is the contract between core/ and three implementations; an
    undocumented method there costs more than the docstring it is missing."""
    interfaces = (
        SharedMemoryReader,
        ScreenCapture,
        HotkeyListener,
        GamePaths,
        OverlayWindow,
        Backend,
    )
    undocumented = [
        f"{cls.__name__}.{name}"
        for cls in interfaces
        for name, member in vars(cls).items()
        if not name.startswith("_")
        and (inspect.isfunction(member) or isinstance(member, property))
        and not (inspect.getdoc(member) or "").strip()
    ]
    assert not undocumented, f"undocumented PAL members: {undocumented}"
