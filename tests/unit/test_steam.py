"""Tests for Steam library discovery.

The VDF parser is hand-written, so it gets tested against the shapes Valve
actually produces -- including the awkward ones: escaped quotes, platform
conditionals, comments, and a `libraryfolders.vdf` whose schema changed between
Steam versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pal._steam import ETS2_APP_ID, VdfParseError, app_install_dir, library_paths, parse_vdf


def test_parses_nested_blocks() -> None:
    parsed = parse_vdf(
        """
        "libraryfolders"
        {
            "0"
            {
                "path"      "/home/user/.steam"
                "apps"
                {
                    "227300"    "0"
                }
            }
        }
        """
    )
    assert parsed["libraryfolders"]["0"]["path"] == "/home/user/.steam"
    assert parsed["libraryfolders"]["0"]["apps"]["227300"] == "0"


def test_ignores_comments_and_platform_conditionals() -> None:
    parsed = parse_vdf(
        """
        // leading comment
        "root"
        {
            "linux_only"   "yes"   [$LINUX]
            "value"        "42"    // trailing comment
        }
        """
    )
    assert parsed["root"] == {"linux_only": "yes", "value": "42"}


def test_handles_escapes_and_windows_paths() -> None:
    parsed = parse_vdf(r'"root" { "path" "D:\\SteamLibrary" "quoted" "say \"hi\"" }')
    assert parsed["root"]["path"] == "D:\\SteamLibrary"
    assert parsed["root"]["quoted"] == 'say "hi"'


def test_duplicate_keys_keep_the_last_value() -> None:
    assert parse_vdf('"root" { "k" "first" "k" "second" }')["root"]["k"] == "second"


@pytest.mark.parametrize(
    "text",
    [
        '"root" { "dangling"',  # key with no value
        '"root" { "k" "v" } }',  # unbalanced closing brace
        '"root" {',  # unclosed block
        "{ }",  # block without a key
    ],
)
def test_malformed_input_raises(text: str) -> None:
    with pytest.raises(VdfParseError):
        parse_vdf(text)


def test_empty_input_is_an_empty_mapping() -> None:
    assert parse_vdf("") == {}
    assert parse_vdf("// only a comment\n") == {}


def _vdf_path(path: Path) -> str:
    """Render a path the way Steam writes it into a VDF file.

    Steam escapes backslashes, so a Windows library appears in the file as
    a doubled separator. Writing the raw path instead makes the parser eat the
    separators -- correctly, since a lone backslash starts an escape -- and the
    test then fails on Windows for a reason that has nothing to do with the
    parser.
    """
    return str(path).replace("\\", "\\\\")


def _make_library(root: Path, app_id: int, install_dir: str) -> None:
    steamapps = root / "steamapps"
    (steamapps / "common" / install_dir).mkdir(parents=True)
    (steamapps / f"appmanifest_{app_id}.acf").write_text(
        f'"AppState"\n{{\n\t"appid"\t\t"{app_id}"\n\t"installdir"\t\t"{install_dir}"\n}}\n',
        encoding="utf-8",
    )


def test_library_paths_defaults_to_the_root_when_manifest_is_missing(tmp_path: Path) -> None:
    (tmp_path / "steamapps").mkdir()
    assert library_paths(tmp_path) == [tmp_path]


def test_library_paths_includes_extra_libraries(tmp_path: Path) -> None:
    root, extra = tmp_path / "steam", tmp_path / "games"
    (root / "steamapps").mkdir(parents=True)
    (root / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n"0"\n{{\n"path" "{_vdf_path(root)}"\n}}\n'
        f'"1"\n{{\n"path" "{_vdf_path(extra)}"\n}}\n}}\n',
        encoding="utf-8",
    )
    assert library_paths(root) == [root, extra]


def test_library_paths_survives_a_corrupt_manifest(tmp_path: Path) -> None:
    """A broken libraryfolders.vdf must not take the whole stack down: the
    default library is still there and is where the game usually lives."""
    (tmp_path / "steamapps").mkdir()
    (tmp_path / "steamapps" / "libraryfolders.vdf").write_text('"broken', encoding="utf-8")
    assert library_paths(tmp_path) == [tmp_path]


def test_app_install_dir_finds_the_game(tmp_path: Path) -> None:
    (tmp_path / "steamapps").mkdir()
    _make_library(tmp_path, ETS2_APP_ID, "Euro Truck Simulator 2")
    found = app_install_dir(tmp_path)
    assert found == tmp_path / "steamapps" / "common" / "Euro Truck Simulator 2"


def test_app_install_dir_searches_secondary_libraries(tmp_path: Path) -> None:
    root, extra = tmp_path / "steam", tmp_path / "games"
    (root / "steamapps").mkdir(parents=True)
    (root / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n"0"\n{{\n"path" "{_vdf_path(root)}"\n}}\n'
        f'"1"\n{{\n"path" "{_vdf_path(extra)}"\n}}\n}}\n',
        encoding="utf-8",
    )
    _make_library(extra, ETS2_APP_ID, "Euro Truck Simulator 2")
    assert app_install_dir(root) == extra / "steamapps" / "common" / "Euro Truck Simulator 2"


def test_app_install_dir_returns_none_when_not_installed(tmp_path: Path) -> None:
    (tmp_path / "steamapps").mkdir()
    assert app_install_dir(tmp_path) is None


def test_manifest_without_a_directory_is_not_an_install(tmp_path: Path) -> None:
    """Steam leaves manifests behind after an uninstall; the directory is the
    thing that actually tells us whether files are on disk."""
    steamapps = tmp_path / "steamapps"
    steamapps.mkdir()
    (steamapps / f"appmanifest_{ETS2_APP_ID}.acf").write_text(
        '"AppState" { "installdir" "Euro Truck Simulator 2" }', encoding="utf-8"
    )
    assert app_install_dir(tmp_path) is None
