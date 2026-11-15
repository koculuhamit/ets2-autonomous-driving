"""Steam library discovery, shared by the Linux and Windows backends.

Locating the game is the same problem on both platforms once the Steam root is
known: read `steamapps/libraryfolders.vdf`, walk the libraries, and look for
`appmanifest_<appid>.acf`. Only *finding the Steam root* is platform-specific,
so that part stays in the backends and everything below is shared.

The VDF ("KeyValues") parser here is deliberately minimal. It handles what
Valve actually writes into these two files -- quoted keys and values, nested
blocks, `//` comments and platform conditionals -- and nothing else. A full
KeyValues implementation would be a dependency and a liability; this is 60
lines we can test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "ETS2_APP_ID",
    "VdfParseError",
    "app_install_dir",
    "library_paths",
    "parse_vdf",
]

ETS2_APP_ID = 227300

_TOKEN = re.compile(
    r"""
    "(?P<quoted>(?:[^"\\]|\\.)*)"    # quoted string, backslash escapes allowed
  | (?P<brace>[{}])
  | //[^\n]*                          # line comment
  | \[[^\]]*\]                        # platform conditional, e.g. [$WINDOWS]
  | (?P<bare>[^\s{}"]+)               # bare token
  | \s+
    """,
    re.VERBOSE,
)

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}


class VdfParseError(ValueError):
    """The file is not valid KeyValues text."""


def _unescape(raw: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            index += 1
            out.append(_ESCAPES.get(raw[index], raw[index]))
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _tokenize(text: str) -> list[str | None]:
    """Yield tokens; `None` marks a brace, disambiguated by its own value."""
    tokens: list[str | None] = []
    for match in _TOKEN.finditer(text):
        if (quoted := match.group("quoted")) is not None:
            tokens.append(_unescape(quoted))
        elif (brace := match.group("brace")) is not None:
            tokens.append(brace)
        elif (bare := match.group("bare")) is not None:
            tokens.append(bare)
    return tokens


def parse_vdf(text: str) -> dict[str, Any]:
    """Parse KeyValues text into nested dictionaries.

    Duplicate keys keep the last value, matching how Steam itself resolves
    them. Values are always strings or dictionaries; no type inference, because
    the caller knows which fields it wants and how to read them.
    """
    tokens = _tokenize(text)
    root: dict[str, Any] = {}
    stack: list[dict[str, Any]] = [root]
    index = 0

    while index < len(tokens):
        key = tokens[index]
        index += 1

        if key == "}":
            if len(stack) == 1:
                raise VdfParseError("unbalanced '}'")
            stack.pop()
            continue
        if key == "{":
            raise VdfParseError("block without a key")
        if index >= len(tokens):
            raise VdfParseError(f"key {key!r} has no value")

        value = tokens[index]
        index += 1
        if value == "{":
            child: dict[str, Any] = {}
            stack[-1][str(key)] = child
            stack.append(child)
        elif value == "}":
            raise VdfParseError(f"key {key!r} has no value")
        else:
            stack[-1][str(key)] = value

    if len(stack) != 1:
        raise VdfParseError("unclosed block")
    return root


def library_paths(steam_root: Path) -> list[Path]:
    """Every Steam library folder, the install root first.

    A missing or unreadable `libraryfolders.vdf` is not an error: a fresh
    install has only the default library, which is the root itself.
    """
    paths = [steam_root]
    manifest = steam_root / "steamapps" / "libraryfolders.vdf"
    if not manifest.is_file():
        return paths

    try:
        data = parse_vdf(manifest.read_text(encoding="utf-8", errors="replace"))
    except VdfParseError:
        return paths

    folders = data.get("libraryfolders", {})
    if not isinstance(folders, dict):
        return paths

    for entry in folders.values():
        if isinstance(entry, dict) and isinstance(path := entry.get("path"), str):
            candidate = Path(path)
            if candidate not in paths:
                paths.append(candidate)
    return paths


def app_install_dir(steam_root: Path, app_id: int = ETS2_APP_ID) -> Path | None:
    """Return the app's installation directory, or None if it is not installed.

    "Installed" here means Steam has a manifest for the app *and* the directory
    exists. A download in progress satisfies both long before the game is
    playable, so callers that need a complete install must check for a specific
    file rather than trusting this.
    """
    for library in library_paths(steam_root):
        manifest = library / "steamapps" / f"appmanifest_{app_id}.acf"
        if not manifest.is_file():
            continue
        try:
            data = parse_vdf(manifest.read_text(encoding="utf-8", errors="replace"))
        except VdfParseError:
            continue
        app_state = data.get("AppState", {})
        if not isinstance(app_state, dict):
            continue
        install_dir = app_state.get("installdir")
        if not isinstance(install_dir, str):
            continue
        candidate = library / "steamapps" / "common" / install_dir
        if candidate.is_dir():
            return candidate
    return None
