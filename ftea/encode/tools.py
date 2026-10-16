"""Run the external programs the encoder depends on.

Every process the encoder starts (ffmpeg, ffprobe and the reframe
stage's interpreter) goes through run_tool, for three reasons:

* Output is decoded as UTF-8 and undecodable bytes are replaced. On
  Windows, Python's default is the system code page, which cannot decode
  a file name such as "Café 東京.mp4" when ffmpeg echoes it back; the
  UnicodeDecodeError would fail the clip for no real reason.
* A failure carries the program's own last error lines, so a clip report
  says why ffmpeg failed instead of only "exit status 1".
* Inside clip_deadline, every call obeys the clip's time limit: a tool
  still running when the time is up is stopped, and none starts after
  it. A hung ffmpeg then fails its clip instead of stalling the batch.

A program that starts programs of its own, as the reframe stage starts
ffmpeg, is run with kill_tree=True, so stopping it stops those too.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

STDERR_TAIL_LINES = 3
STDERR_TAIL_CHARS = 600

Arg = Union[str, Path]

# The deadline of the clip being encoded, as a time.monotonic() value, or
# None outside clip_deadline. Module-level rather than thread-local, so
# the VMAF scoring threads obey the clip's deadline too. Clips are
# encoded one at a time, so one deadline is enough.
_clip_deadline: Optional[float] = None


class ToolError(RuntimeError):
    """An external program could not start, failed, or ran too long."""


class ToolTimeout(ToolError):
    """An external program did not finish within its time limit."""


# Seconds allowed for taskkill, and for collecting the last output of a
# stopped program.
_STOP_WAIT_SEC = 30


def run_process_tree(
    args: Sequence[str], *, timeout: Optional[float] = None,
    capture_output: bool = False, **kwargs,
) -> subprocess.CompletedProcess:
    """subprocess.run that also stops the program's own child processes.

    subprocess.run stops only the program it started, when the time runs
    out or the wait is interrupted. A program that runs ffmpeg itself
    would leave that ffmpeg running, still writing into the folder the
    next attempt empties, which Windows then refuses. Here the whole
    tree is stopped: on Windows with taskkill /T, on Linux by looking up
    the program's descendants in /proc. The program stays in this
    process's group, so Ctrl+C and a closed terminal still reach all of
    it, as they did before. Raises subprocess.TimeoutExpired as
    subprocess.run does.
    """
    if capture_output:
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with subprocess.Popen(args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            _stop_tree(process)
            raise
    return subprocess.CompletedProcess(
        process.args, process.returncode, stdout, stderr,
    )


def _stop_tree(process: subprocess.Popen) -> None:
    """Stop a program and everything it started, then reap it."""
    if os.name == "nt":
        taskkill = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32",
            "taskkill.exe",
        )
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [taskkill, "/F", "/T", "/PID", str(process.pid)],
                capture_output=True, timeout=_STOP_WAIT_SEC,
            )
    else:
        # Stopped first, so it cannot start another program between the
        # list being read and the kill.
        with contextlib.suppress(OSError):
            os.kill(process.pid, signal.SIGSTOP)
        for pid in _descendants(process.pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
    process.kill()
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        process.communicate(timeout=_STOP_WAIT_SEC)


def _descendants(pid: int, proc: Path = Path("/proc")) -> list[int]:
    """Every process below `pid`, read from /proc; empty without it."""
    children: dict[int, list[int]] = {}
    try:
        entries = [entry for entry in os.listdir(proc) if entry.isdigit()]
    except OSError:
        return []
    for entry in entries:
        try:
            stat = (proc / entry / "stat").read_bytes()
            # pid (name) state ppid ...: the name may hold spaces and ")".
            parent = int(stat.rsplit(b")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found: list[int] = []
    waiting = [pid]
    while waiting:
        for child in children.get(waiting.pop(), []):
            found.append(child)
            waiting.append(child)
    return found


def stderr_tail(text: Optional[str]) -> str:
    """The last few non-blank lines of a program's error output."""
    lines = [line.strip() for line in (text or "").splitlines()]
    tail = " | ".join([line for line in lines if line][-STDERR_TAIL_LINES:])
    return tail[-STDERR_TAIL_CHARS:] if tail else "no error output"


@contextlib.contextmanager
def clip_deadline(seconds: float) -> Iterator[None]:
    """Give every run_tool call inside the block one shared time limit.

    Each call's timeout is capped at the time left, and once the time is
    up no new tool starts. Leaving the block, however it is left, lifts
    the limit.
    """
    global _clip_deadline
    _clip_deadline = time.monotonic() + seconds
    try:
        yield
    finally:
        _clip_deadline = None


def _capped_timeout(
    name: str, timeout: Optional[float],
) -> tuple[Optional[float], bool]:
    """The timeout to run with, and whether the clip deadline set it."""
    if _clip_deadline is None:
        return timeout, False
    left = _clip_deadline - time.monotonic()
    if left <= 0:
        raise ToolTimeout(
            f"clip timed out: its time limit ran out before {name} "
            "could start"
        )
    capped = left if timeout is None else min(timeout, left)
    return capped, capped == left


def run_tool(
    cmd: Sequence[Arg],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    kill_tree: bool = False,
) -> subprocess.CompletedProcess:
    """Run one program and capture its output as text.

    Raises ToolTimeout when it runs longer than `timeout` seconds, or past
    the clip's deadline, and ToolError when it cannot start or (with
    check=True) exits non-zero. With kill_tree, stopping it stops the
    programs it started too (see run_process_tree).
    """
    args = [str(part) for part in cmd]
    name = Path(args[0]).name
    if cwd is not None and not Path(cwd).is_dir():
        raise ToolError(
            f"cannot run {name}: working folder {cwd} does not exist"
        )
    run_timeout, by_deadline = _capped_timeout(name, timeout)
    runner = run_process_tree if kill_tree else subprocess.run
    try:
        result = runner(
            args,
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=run_timeout,
        )
    except FileNotFoundError:
        raise ToolError(
            f"{name} could not be started: it is not installed or not "
            "on PATH"
        ) from None
    except subprocess.TimeoutExpired:
        if by_deadline:
            raise ToolTimeout(
                f"clip timed out: its time limit ran out while {name} "
                "was running"
            ) from None
        raise ToolTimeout(f"{name} did not finish within {timeout}s") from None
    except OSError as exc:
        raise ToolError(f"{name} could not be started: {exc}") from None
    if check and result.returncode != 0:
        raise ToolError(
            f"{name} failed (exit {result.returncode}): "
            f"{stderr_tail(result.stderr)}"
        )
    return result
