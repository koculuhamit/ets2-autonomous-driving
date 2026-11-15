"""First-run setup: finding the map extractor, wherever it is."""

from __future__ import annotations

from pathlib import Path

from tools.setup import extractor_command, plugin_name


def nothing(_name: str) -> None:
    return None


def test_a_named_extractor_comes_first(tmp_path: Path) -> None:
    command = extractor_command(tmp_path, {"ETS2AD_MAP_EXTRACT": "/opt/x/MapExtract"}, "linux")
    assert command == ["/opt/x/MapExtract"]


def test_a_release_s_own_extractor_is_used(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "MapExtract.exe").touch()
    command = extractor_command(tmp_path, {}, "win32", which=nothing)
    assert command == [str(tmp_path / "bin" / "MapExtract.exe")]


def test_a_built_extractor_is_run_by_dotnet(tmp_path: Path) -> None:
    built = tmp_path / "tools" / "map_extract" / "bin" / "Debug" / "net10.0"
    built.mkdir(parents=True)
    (built / "MapExtract.dll").touch()
    command = extractor_command(tmp_path, {}, "linux", which=lambda _: "/usr/bin/dotnet")
    assert command == ["/usr/bin/dotnet", str(built / "MapExtract.dll")]


def test_with_only_the_sdk_the_source_is_run(tmp_path: Path) -> None:
    command = extractor_command(tmp_path, {}, "linux", which=lambda _: "/usr/bin/dotnet")
    assert command is not None and command[1:3] == ["run", "--project"]


def test_the_plugin_is_named_for_the_platform() -> None:
    assert plugin_name("win32") == "ets2ad_telemetry.dll"
    assert plugin_name("linux") == "ets2ad_telemetry.so"
