"""Tests for ftea.encode.tools, the runner for every external program."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ftea.encode import tools
from ftea.encode.main import _pid_alive
from ftea.encode.tools import (
    ToolError,
    ToolTimeout,
    clip_deadline,
    run_process_tree,
    run_tool,
    stderr_tail,
)


def _py(code: str) -> list[str]:
    """A command that runs `code` in this interpreter."""
    return [sys.executable, "-c", code]


class TestRunTool:
    def test_should_return_decoded_output_when_the_tool_succeeds(self):
        result = run_tool(_py("print('ok')"))
        assert result.returncode == 0
        assert result.stdout.strip() == "ok"

    def test_should_decode_non_ascii_text_as_utf8(self):
        code = (
            "import sys; sys.stdout.buffer.write("
            "'Caf\\u00e9 \\u6771\\u4eac'.encode('utf-8'))"
        )
        assert run_tool(_py(code)).stdout == "Café 東京"

    def test_should_replace_bytes_that_are_not_utf8(self):
        # 0x9d is undefined in the Windows code page and invalid UTF-8.
        code = "import sys; sys.stderr.buffer.write(b'bad \\x9d\\xff end')"
        result = run_tool(_py(code), check=False)
        assert result.stderr.startswith("bad ")
        assert result.stderr.endswith(" end")
        assert "�" in result.stderr

    def test_should_raise_with_the_last_error_lines_when_the_tool_fails(
        self,
    ):
        code = (
            "import sys; sys.stderr.write('l1\\nl2\\nl3\\nl4\\n'); "
            "sys.exit(3)"
        )
        with pytest.raises(ToolError) as info:
            run_tool(_py(code))
        message = str(info.value)
        assert "exit 3" in message
        assert "l2 | l3 | l4" in message
        assert "l1" not in message

    def test_should_return_the_result_when_check_is_off(self):
        result = run_tool(_py("import sys; sys.exit(2)"), check=False)
        assert result.returncode == 2

    def test_should_name_a_program_that_is_not_installed(self):
        with pytest.raises(ToolError, match="not installed or not on PATH"):
            run_tool(["ftea-no-such-program-0f3c", "-version"])

    def test_should_raise_a_timeout_when_the_tool_runs_too_long(self):
        with pytest.raises(ToolTimeout, match="did not finish within 0.5s"):
            run_tool(_py("import time; time.sleep(10)"), timeout=0.5)

    def test_should_run_in_the_given_folder(self, tmp_path):
        result = run_tool(_py("import os; print(os.getcwd())"), cwd=tmp_path)
        assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()

    def test_should_refuse_a_working_folder_that_does_not_exist(
        self, tmp_path,
    ):
        with pytest.raises(ToolError, match="does not exist"):
            run_tool(_py("pass"), cwd=tmp_path / "absent")

    def test_should_pass_the_given_environment(self):
        env = {**os.environ, "FTEA_TOOL_TEST": "yes"}
        code = "import os; print(os.environ['FTEA_TOOL_TEST'])"
        assert run_tool(_py(code), env=env).stdout.strip() == "yes"

    def test_should_accept_path_arguments(self, tmp_path):
        script = tmp_path / "hello.py"
        script.write_text("print('hi')\n", encoding="utf-8")
        assert run_tool([Path(sys.executable), script]).stdout.strip() == "hi"


class TestStderrTail:
    def test_should_say_so_when_there_is_no_error_output(self):
        assert stderr_tail("") == "no error output"
        assert stderr_tail(None) == "no error output"

    def test_should_cap_very_long_output(self):
        assert len(stderr_tail("x" * 5000)) == 600


class TestClipDeadline:
    def test_should_stop_a_tool_still_running_at_the_deadline(self):
        started = time.monotonic()
        with clip_deadline(0.5):
            with pytest.raises(ToolTimeout, match="timed out"):
                run_tool(_py("import time; time.sleep(10)"), timeout=60)
        assert time.monotonic() - started < 5

    def test_should_start_no_tool_once_the_deadline_has_passed(
        self, tmp_path,
    ):
        marker = tmp_path / "ran"
        with clip_deadline(0.05):
            time.sleep(0.2)
            with pytest.raises(ToolTimeout, match="timed out"):
                run_tool(_py(f"open({str(marker)!r}, 'w').close()"))
        assert not marker.exists()

    def test_should_lift_the_deadline_when_the_block_ends(self):
        with clip_deadline(0.05):
            time.sleep(0.2)
        assert run_tool(_py("print('ok')")).stdout.strip() == "ok"


def _parent_of_a_sleeper(pid_file: Path) -> str:
    """A program that starts a child of its own, records the child's
    process id, and then waits, as the reframe stage does with ffmpeg.

    The child keeps none of the parent's pipes, like the stage's ffmpeg,
    whose output the stage captures; otherwise subprocess.run would wait
    for it and the test could not tell a stopped child from a finished
    one.
    """
    return textwrap.dedent(f"""
        import subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open({str(pid_file)!r}, "w") as handle:
            handle.write(str(child.pid))
        time.sleep(60)
    """)


def _gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def _stop(pid: int) -> None:
    """Tidy up after a failed test, so no sleeper outlives it."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)], capture_output=True,
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


class TestProcessTree:
    """The reframe stage starts ffmpeg itself. When its time runs out, an
    ffmpeg left running keeps writing into the clip's work folder, and
    on Windows the retry then cannot empty that folder."""

    def test_should_stop_the_programs_children_when_its_time_runs_out(
        self, tmp_path,
    ):
        pid_file = tmp_path / "child.pid"

        with pytest.raises(ToolTimeout):
            run_tool(
                _py(_parent_of_a_sleeper(pid_file)), timeout=3,
                kill_tree=True,
            )

        child = int(pid_file.read_text(encoding="utf-8"))
        try:
            assert _gone(child), "the program's own child was left running"
        finally:
            _stop(child)

    def test_should_behave_like_subprocess_run_otherwise(self):
        done = run_process_tree(
            _py("import sys; print('out'); sys.exit(3)"),
            stdout=subprocess.PIPE, encoding="utf-8",
        )

        assert (done.returncode, done.stdout.strip()) == (3, "out")
        assert done.stderr is None

    @pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
    def test_should_keep_the_program_in_this_process_group(self):
        """Ctrl+C and a closed terminal signal the foreground process
        group; a program moved out of it would outlive this one."""
        done = run_process_tree(
            _py("import os; print(os.getpgrp())"),
            stdout=subprocess.PIPE, encoding="utf-8",
        )

        assert int(done.stdout) == os.getpgrp()

    def test_should_find_every_descendant_in_proc(self, tmp_path):
        table = {
            10: b"10 (python) S 1 10",
            11: b"11 (ff) mpeg)) S 10 10",
            12: b"12 (x) R 11 10",
            13: b"13 (other) S 1 13",
            14: b"14 (gone",
        }
        for pid, stat in table.items():
            (tmp_path / str(pid)).mkdir()
            (tmp_path / str(pid) / "stat").write_bytes(stat)
        (tmp_path / "self").mkdir()

        assert tools._descendants(10, proc=tmp_path) == [11, 12]
        assert tools._descendants(13, proc=tmp_path) == []
        assert tools._descendants(10, proc=tmp_path / "absent") == []

    def test_should_raise_timeout_expired_like_subprocess_run(self):
        with pytest.raises(subprocess.TimeoutExpired):
            run_process_tree(
                _py("import time; time.sleep(10)"), timeout=0.5,
                capture_output=True,
            )
