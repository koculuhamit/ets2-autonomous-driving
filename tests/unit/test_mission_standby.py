"""The mission's word file to its standby, out of the shared /tmp."""

from __future__ import annotations

import stat
import sys

import pytest

from tools.mission import private_dir


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_the_word_file_lives_where_only_this_user_can_enter() -> None:
    """Under a guessable name in /tmp, another account could make the word
    file first and bring the standby down."""
    folder = private_dir()
    assert folder.is_dir()
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert folder.name.startswith("ets2ad-") and folder.name != "ets2ad-"
    assert private_dir() == folder, "one for the whole run"
