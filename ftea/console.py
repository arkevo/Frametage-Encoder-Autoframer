"""What ftea prints for a terminal: text it can show, commands it can run.

Standard library only.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Optional

# What a printed command can carry unquoted in cmd.exe, PowerShell and
# Git Bash alike. A backslash is not here: Git Bash would drop it.
_PLAIN = re.compile(r"[A-Za-z0-9_./:-]+")


def _on_windows() -> bool:
    return os.name == "nt"


def shell_arg(text: str, *, windows: Optional[bool] = None) -> str:
    """`text` as one argument of a command printed for the user to run.

    POSIX shell quoting, except on Windows, where cmd.exe and PowerShell
    take single quotes literally. There it is double quotes by the rules
    every Windows program splits its command line with
    (subprocess.list2cmdline), which Git Bash also reads correctly for a
    file name. PowerShell and cmd.exe still expand a $ or a %...% inside
    double quotes; file names rarely hold one.
    """
    if not (_on_windows() if windows is None else windows):
        return shlex.quote(text)
    if _PLAIN.fullmatch(text):
        return text
    quoted = subprocess.list2cmdline([text])
    if quoted.startswith('"'):
        return quoted
    # list2cmdline adds quotes only for a space, a tab or an empty
    # string; backslashes before the closing quote added here must be
    # doubled, or they would escape it.
    trailing = len(quoted) - len(quoted.rstrip("\\"))
    return '"' + quoted + "\\" * trailing + '"'


def tolerate_unprintable_text() -> None:
    """Never let text the console cannot show end a command.

    On Windows, output sent to a pipe or a file (Git Bash, or
    `ftea run > log.txt`) is encoded in the ANSI code page, and printing
    a clip name or a folder outside it raises UnicodeEncodeError. Such
    characters are written as backslash escapes instead, as Python
    already does on stderr.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
