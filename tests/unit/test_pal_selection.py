"""Tests for backend selection.

`get_backend()` is the single seam between `core/` and the operating system.
The environment override in particular is load-bearing: it is how CI, offline
replay and the whole test suite run the real stack with no game underneath it.
"""

from __future__ import annotations

import pytest

from pal import BACKEND_ENV_VAR, PalUnsupportedError, get_backend
from pal.fake import FakeBackend
from pal.linux import LinuxBackend
from pal.windows import WindowsBackend


@pytest.fixture(autouse=True)
def _clear_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a stray override in the developer's shell out of these tests."""
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fake", FakeBackend),
        ("linux", LinuxBackend),
        ("windows", WindowsBackend),
        ("LINUX", LinuxBackend),
    ],
)
def test_explicit_name_selects_the_backend(name: str, expected: type) -> None:
    assert isinstance(get_backend(name), expected)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("linux", LinuxBackend), ("linux2", LinuxBackend), ("win32", WindowsBackend)],
)
def test_platform_selects_the_backend(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: type
) -> None:
    monkeypatch.setattr("sys.platform", platform)
    assert isinstance(get_backend(), expected)


def test_environment_variable_overrides_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    assert isinstance(get_backend(), FakeBackend)


def test_explicit_name_beats_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    assert isinstance(get_backend("linux"), LinuxBackend)


def test_unknown_name_is_rejected_with_the_valid_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "nonsense")
    with pytest.raises(PalUnsupportedError, match="fake, linux, windows"):
        get_backend()


def test_unsupported_platform_points_at_the_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(PalUnsupportedError, match=BACKEND_ENV_VAR):
        get_backend()


def test_keyword_arguments_reach_the_backend(tmp_path: object) -> None:
    backend = get_backend("linux", steam_root=tmp_path)
    assert backend.game_paths()._explicit_root is tmp_path


def test_importing_pal_does_not_import_the_other_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy imports keep `pal.windows` from dragging Windows-only libraries
    into a Linux process, and vice versa."""
    import importlib
    import sys

    for module in ("pal", "pal.windows", "pal.linux", "pal.fake"):
        sys.modules.pop(module, None)
    importlib.import_module("pal")
    assert "pal.windows" not in sys.modules
    assert "pal.linux" not in sys.modules
