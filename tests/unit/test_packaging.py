"""A release that asks nothing of the player: the plugin built for release,
uv fetched by the launcher when it is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.install_plugin import LIBRARY_NAME, built_library
from scripts.package import LAUNCH_BAT, LAUNCH_SH


def test_a_single_config_build_is_at_the_top(tmp_path: Path) -> None:
    (tmp_path / LIBRARY_NAME).touch()
    assert built_library(tmp_path) == tmp_path / LIBRARY_NAME


def test_a_multi_config_build_is_taken_from_release(tmp_path: Path) -> None:
    for config in ("Debug", "Release"):
        (tmp_path / config).mkdir()
        (tmp_path / config / LIBRARY_NAME).touch()
    assert built_library(tmp_path) == tmp_path / "Release" / LIBRARY_NAME


def test_a_debug_build_alone_is_not_installed(tmp_path: Path) -> None:
    """A Debug DLL needs the debug runtime, which a player does not have."""
    (tmp_path / "Debug").mkdir()
    (tmp_path / "Debug" / LIBRARY_NAME).touch()
    with pytest.raises(SystemExit):
        built_library(tmp_path)


@pytest.mark.parametrize(
    ("launcher", "installer"),
    [(LAUNCH_SH, "astral.sh/uv/install.sh"), (LAUNCH_BAT, "astral.sh/uv/install.ps1")],
)
def test_the_launcher_fetches_uv_before_using_it(launcher: str, installer: str) -> None:
    assert launcher.index(installer) < launcher.index("uv sync")


def test_the_windows_launcher_looks_where_uv_installs_itself() -> None:
    assert "%USERPROFILE%\\.local\\bin;" in LAUNCH_BAT


def test_a_release_leaves_out_the_listed_paths_too(tmp_path: Path) -> None:
    from scripts.package import SKIPPED, left_out

    listing = tmp_path / "exclude.txt"
    listing.write_text("# kept out\nnotes.md\ndocs/drafts/\n", encoding="utf-8")
    assert left_out(listing) == (*SKIPPED, "notes.md", "docs/drafts/")
    assert left_out(tmp_path / "missing.txt") == SKIPPED, "a copy without the list"


@pytest.mark.parametrize("remover", ["UNINSTALL_SH", "UNINSTALL_BAT"])
def test_the_uninstaller_runs_on_the_folder_s_own_python_and_leaves_the_folder(
    remover: str,
) -> None:
    """Run through `uv run`, uv itself would be in use while it is removed;
    and a window still in the folder keeps the folder from being deleted."""
    import scripts.package as package

    text = getattr(package, remover)
    assert "tools/uninstall.py" in text.replace("\\", "/")
    assert ".venv" in text
    last = text.rstrip().splitlines()
    assert any(line.startswith(("cd /", 'cd /d "%TEMP%"')) for line in last[-3:])
