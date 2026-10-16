"""Tests for ftea.console: text a terminal can show, commands it can run."""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys

import pytest

from ftea import console

NAMES = [
    "plain", "my clip", "Bob's clip", "Café 東京 clip", "say \"hi\"",
    "D:\\Videos\\output\\clip", "ends in a backslash\\", "",
    "$HOME and %PATH%",
]


class TestShellArg:
    @pytest.mark.parametrize("text, expected", [
        ("plain", "plain"),
        ("clip_2.v1", "clip_2.v1"),
        ("my clip", '"my clip"'),
        ("Bob's clip", '"Bob\'s clip"'),
        ("D:\\Videos\\output\\clip", '"D:\\Videos\\output\\clip"'),
        ("output\\first", '"output\\first"'),
        ("C:\\", '"C:\\\\"'),
        ("", '""'),
    ])
    def test_should_use_double_quotes_on_windows(self, text, expected):
        assert console.shell_arg(text, windows=True) == expected

    @pytest.mark.parametrize("text, expected", [
        ("plain", "plain"),
        ("my clip", "'my clip'"),
        ("Bob's clip", "'Bob'\"'\"'s clip'"),
    ])
    def test_should_use_posix_quoting_elsewhere(self, text, expected):
        assert console.shell_arg(text, windows=False) == expected

    @pytest.mark.parametrize("text", NAMES)
    def test_should_survive_a_posix_shell(self, text):
        assert shlex.split(console.shell_arg(text, windows=False)) == [text]

    @pytest.mark.skipif(
        os.name != "nt", reason="needs the Windows command-line parser",
    )
    @pytest.mark.parametrize("text", NAMES)
    def test_should_survive_the_windows_command_line(self, text):
        """CreateProcess hands the line to the program, which splits it by
        the same rules every Windows program (ftea.exe included) uses."""
        line = (
            subprocess.list2cmdline([sys.executable, "-c",
                                     "import sys; print(ascii(sys.argv[1:]))"])
            + " " + console.shell_arg(text, windows=True)
        )
        result = subprocess.run(
            line, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == ascii([text])

    def test_should_follow_the_platform_by_default(self, monkeypatch):
        monkeypatch.setattr(console, "_on_windows", lambda: True)
        assert console.shell_arg("my clip") == '"my clip"'
        monkeypatch.setattr(console, "_on_windows", lambda: False)
        assert console.shell_arg("my clip") == "'my clip'"


class TestUnprintableText:
    def test_should_write_unprintable_characters_as_escapes(
        self, monkeypatch,
    ):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        console.tolerate_unprintable_text()
        print("клип")
        stream.flush()
        assert raw.getvalue().decode("cp1252").strip() == (
            "\\u043a\\u043b\\u0438\\u043f"
        )

    def test_should_leave_a_stream_without_reconfigure_alone(
        self, monkeypatch,
    ):
        monkeypatch.setattr(sys, "stdout", object())
        console.tolerate_unprintable_text()  # no AttributeError
