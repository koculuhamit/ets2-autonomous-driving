"""Taking off a machine what the launcher put on it: asked, one by one."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import uninstall
from tools.uninstall import (
    Plan,
    ask,
    finishing_job,
    game_running,
    installed_plugin,
    main,
    project_packages,
    uv_binaries,
    uv_install_dir,
    venv_python,
)

LOCK = """version = 1
[[package]]
name = "ets2-ad"
version = "2.1.3"
[[package]]
name = "onnxruntime-gpu"
version = "1.30.0"
[[package]]
name = "numpy"
version = "2.5.3"
"""


def _install(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A release folder that has been set up, a game with its plugin in, and
    a folder of uv's."""
    folder = tmp_path / "ets2ad"
    (folder / ".venv").mkdir(parents=True)
    (folder / "uv.lock").write_text(LOCK, encoding="utf-8")
    (folder / "data" / "missions" / "20261120-1").mkdir(parents=True)
    (folder / "data" / "missions" / "20261120-1" / "telemetry.bin").write_bytes(b"x" * 2048)
    game = tmp_path / "game"
    plugin_in(game).parent.mkdir(parents=True)
    plugin_in(game).write_bytes(b"plugin")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in UV_NAMES:
        (bin_dir / name).write_bytes(b"")
    return folder, game, bin_dir


WINDOWS = sys.platform == "win32"
UV_NAMES = ("uv.exe", "uvx.exe") if WINDOWS else ("uv", "uvx")


def plugin_in(game: Path) -> Path:
    binary = "win_x64" if WINDOWS else "linux_x64"
    name = "ets2ad_telemetry.dll" if WINDOWS else "ets2ad_telemetry.so"
    return game / "bin" / binary / "plugins" / name


def test_only_yes_is_yes() -> None:
    assert ask("?", lambda _: "y") and ask("?", lambda _: " YES ") and ask("?", lambda _: "e")
    assert not ask("?", lambda _: "") and not ask("?", lambda _: "n")

    def closed(_):
        raise EOFError

    assert not ask("?", closed), "no one to answer keeps everything"


def test_the_downloads_are_this_project_s_packages_not_itself(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(LOCK, encoding="utf-8")
    assert project_packages(lock) == ["numpy", "onnxruntime-gpu"]
    assert project_packages(tmp_path / "none.lock") == []


def test_uv_is_looked_for_where_its_installer_puts_it(tmp_path: Path) -> None:
    assert uv_install_dir({"UV_INSTALL_DIR": "/opt/uv"}, tmp_path) == Path("/opt/uv")
    assert uv_install_dir({"XDG_BIN_HOME": "/x/bin"}, tmp_path) == Path("/x/bin")
    assert uv_install_dir({}, tmp_path) == tmp_path / ".local" / "bin"
    (tmp_path / "uv.exe").touch()
    (tmp_path / "uvx.exe").touch()
    assert uv_binaries(tmp_path, "win32") == [tmp_path / "uv.exe", tmp_path / "uvx.exe"]


def test_python_is_offered_only_if_uv_fetched_it(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    managed = tmp_path / "uv-python" / "cpython-3.12.15-linux" / "bin"
    managed.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {managed}\nversion_info = 3.12.15\n", encoding="utf-8"
    )

    def run(*_, **__):
        return SimpleNamespace(stdout=str(tmp_path / "uv-python") + "\n")

    assert venv_python(venv, Path("uv"), run) == ("3.12.15", str(managed))

    def elsewhere(*_, **__):
        return SimpleNamespace(stdout="/somewhere/else\n")

    assert venv_python(venv, Path("uv"), elsewhere)[0] is None, "the system's is not uv's"
    assert venv_python(venv, None, run)[0] is None


def test_the_plugin_is_found_in_the_game_s_folder(tmp_path: Path) -> None:
    _, game, _ = _install(tmp_path)
    plugin, _ = installed_plugin(game)
    assert plugin == plugin_in(game)
    assert installed_plugin(tmp_path / "no-game")[0] is None


def test_the_game_is_seen_running_by_its_process(tmp_path: Path) -> None:
    (tmp_path / "123").mkdir()
    (tmp_path / "123" / "comm").write_text("eurotrucks2\n", encoding="utf-8")
    assert game_running("linux", tmp_path)
    (tmp_path / "123" / "comm").write_text("bash\n", encoding="utf-8")
    assert not game_running("linux", tmp_path)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_python_goes_before_uv_and_uv_last(platform: str) -> None:
    plan = Plan(
        python="3.12.15",
        folder=Path("/home/a b/ets2ad"),
        uv_files=[Path("/home/a b/.local/bin/uv")],
        uv=Path("/home/a b/.local/bin/uv"),
    )
    lines = finishing_job(plan, platform)
    python = next(i for i, line in enumerate(lines) if "python uninstall 3.12.15" in line)
    folder = next(i for i, line in enumerate(lines) if "ets2ad" in line and "python" not in line)
    uv = next(i for i, line in enumerate(lines) if line.rstrip("\"'").endswith("bin/uv"))
    assert lines[0].startswith(("sleep", "timeout")), "waits for this window first"
    assert python < folder < uv
    assert "$0" in lines[-1] or "%~f0" in lines[-1], "and removes itself"
    if platform == "linux":
        assert "'/home/a b/ets2ad'" in lines[folder], "a path with a space quoted"


def _no_uv_on_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv("UV_INSTALL_DIR", str(bin_dir))
    monkeypatch.setattr(uninstall.shutil, "which", lambda name: None)
    monkeypatch.setattr(uninstall, "venv_python", lambda venv, uv: ("3.12.15", "/uv/python"))
    monkeypatch.setattr(uninstall, "game_running", lambda: False)


def test_a_dry_run_takes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    folder, game, bin_dir = _install(tmp_path)
    _no_uv_on_path(monkeypatch, bin_dir)
    started: list[Plan] = []
    monkeypatch.setattr(uninstall, "start_finishing_job", started.append)

    def nobody(_):
        raise AssertionError("a dry run asks nothing")

    assert main(["--dry-run", "--game", str(game), "--folder", str(folder)], nobody) == 0
    out = capsys.readouterr().out
    assert out.count("would ask") == 5
    assert "of it your recorded drives" in out
    assert plugin_in(game).is_file()
    assert started == []


def test_each_is_asked_and_only_what_is_said_yes_to_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, game, bin_dir = _install(tmp_path)
    _no_uv_on_path(monkeypatch, bin_dir)
    started: list[Plan] = []
    monkeypatch.setattr(uninstall, "start_finishing_job", started.append)
    cleaned: list[list[str]] = []

    def run(command, **_):
        if "clean" in command:
            cleaned.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uninstall.subprocess, "run", run)
    answers = iter(["y", "", "n", "", "y"])  # plugin, downloads, Python, uv, folder
    asked: list[str] = []

    def answer(question: str) -> str:
        asked.append(question)
        return next(answers)

    assert main(["--game", str(game), "--folder", str(folder)], answer) == 0
    assert len(asked) == 5, "one question each"
    assert not plugin_in(game).exists()
    assert cleaned == [], "the downloads were kept"
    [plan] = started
    assert plan.folder == folder.resolve() and plan.python is None and plan.uv_files == []


def test_yes_to_all_cleans_this_project_s_downloads_and_hands_on_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, game, bin_dir = _install(tmp_path)
    _no_uv_on_path(monkeypatch, bin_dir)
    started: list[Plan] = []
    monkeypatch.setattr(uninstall, "start_finishing_job", started.append)
    cleaned: list[list[str]] = []

    def run(command, **_):
        if "clean" in command:
            cleaned.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uninstall.subprocess, "run", run)
    assert main(["--game", str(game), "--folder", str(folder)], lambda _: "y") == 0
    uv = bin_dir / UV_NAMES[0]
    assert cleaned == [[str(uv), "cache", "clean", "numpy", "onnxruntime-gpu"]]
    [plan] = started
    assert plan.python == "3.12.15" and plan.uv == uv
    assert plan.uv_files == [bin_dir / name for name in UV_NAMES]


def test_a_running_game_keeps_its_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder, game, bin_dir = _install(tmp_path)
    _no_uv_on_path(monkeypatch, bin_dir)
    monkeypatch.setattr(uninstall, "game_running", lambda: True)
    monkeypatch.setattr(uninstall, "start_finishing_job", lambda plan: None)
    answers = iter(["y", "n", "n", "n", "n"])
    main(["--game", str(game), "--folder", str(folder)], lambda _: next(answers))
    assert plugin_in(game).is_file()
