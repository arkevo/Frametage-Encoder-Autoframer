"""Tests for the `ftea` command line."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

import pytest

import ftea
import ftea.encode.main as encode_main
from ftea import cli, console, paths, preview, vertical
from ftea.cli import main
from ftea.encode.config import PipelineConfig
from ftea.encode.main import PipelineError, RunResult
from ftea.vertical import VerticalError, VerticalResult

REPO = Path(__file__).resolve().parents[2]
VERSION_LINE = f"ftea {ftea.__version__}"


class TestMain:
    def test_should_print_version_when_given_version_flag(self, capsys):
        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == VERSION_LINE

    def test_should_print_version_when_given_short_flag(self, capsys):
        assert main(["-V"]) == 0
        assert capsys.readouterr().out.strip() == VERSION_LINE

    def test_should_print_usage_when_asked_for_help(self, capsys):
        assert main(["--help"]) == 0
        out = capsys.readouterr().out
        assert "usage: ftea" in out
        for command in ("run", "preview", "reframe"):
            assert command in out

    def test_should_exit_2_and_name_commands_for_an_unknown_command(
        self, capsys,
    ):
        assert main(["bogus"]) == 2
        err = capsys.readouterr().err
        for command in ("run", "preview", "reframe"):
            assert command in err

    def test_should_exit_2_when_given_no_arguments(self, capsys):
        assert main([]) == 2
        assert "usage: ftea" in capsys.readouterr().err

    def test_should_derive_the_usage_line_from_the_commands(self):
        assert cli.USAGE == (
            "usage: ftea {" + ",".join(cli.COMMANDS) + "} ... | ftea --version"
        )

    @pytest.mark.parametrize("command", cli.COMMANDS)
    def test_should_give_every_command_its_own_help(self, command, capsys):
        assert main([command, "--help"]) == 0
        assert f"usage: ftea {command}" in capsys.readouterr().out


class TestEntryPoints:
    def test_should_run_as_python_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "ftea", "--version"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == VERSION_LINE

    def test_should_install_console_script(self):
        name = "ftea.exe" if os.name == "nt" else "ftea"
        script = Path(sysconfig.get_path("scripts")) / name
        if not script.exists():
            pytest.skip(f"ftea is not installed here ({script})")
        result = subprocess.run(
            [str(script), "--version"], capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == VERSION_LINE

    def test_should_report_one_version_everywhere(self):
        try:
            installed = metadata.version("ftea")
        except metadata.PackageNotFoundError:
            pytest.skip("ftea is not installed in this environment")
        assert installed == ftea.__version__

    def test_should_load_no_command_until_one_runs(self):
        """--version stays instant, and `ftea run` never loads the
        preview server or the reframe stage."""
        code = (
            "import sys, ftea.cli\n"
            "watched = ('ftea.preview', 'ftea.vertical', 'ftea.encode', "
            "'ftea.reframe', 'http.server')\n"
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith(watched))))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""


@pytest.fixture
def run_calls(monkeypatch):
    calls = []
    outcome = {"result": RunResult(), "raise": None}

    def fake(input_path=None, *, reframe=None, force=False):
        calls.append((input_path, reframe, force))
        if outcome["raise"] is not None:
            raise outcome["raise"]
        return outcome["result"]

    monkeypatch.setattr(encode_main, "run", fake)
    return calls, outcome


class TestRun:
    def test_should_use_the_input_folder_by_default(self, run_calls):
        calls, _ = run_calls
        assert main(["run"]) == 0
        assert calls == [(None, None, False)]

    def test_should_pass_the_input_and_the_flags_on(self, run_calls):
        calls, _ = run_calls
        assert main(["run", "clips", "--no-reframe", "--force"]) == 0
        assert main(["run", "--reframe", "one.mp4"]) == 0
        assert calls == [
            (Path("clips"), False, True), (Path("one.mp4"), True, False),
        ]

    def test_should_refuse_both_reframe_flags(self, run_calls, capsys):
        calls, _ = run_calls
        assert main(["run", "--reframe", "--no-reframe"]) == 2
        assert calls == []
        assert "not allowed with" in capsys.readouterr().err

    def test_should_return_the_runs_exit_code(self, run_calls):
        _, outcome = run_calls
        outcome["result"] = RunResult(failed=("a",))
        assert main(["run"]) == 1

    @pytest.mark.parametrize("windows, quoted", [
        (True, '"my clip"'), (False, "'my clip'"),
    ])
    def test_should_say_how_to_watch_each_new_clip(
        self, run_calls, tmp_path, capsys, monkeypatch, windows, quoted,
    ):
        """The command is quoted for the shell of the platform: cmd.exe
        and PowerShell take single quotes literally."""
        monkeypatch.setattr(console, "_on_windows", lambda: windows)
        _, outcome = run_calls
        outcome["result"] = RunResult(
            outputs=(tmp_path / "plain", tmp_path / "my clip"),
        )
        assert main(["run"]) == 0
        out = capsys.readouterr().out
        assert "Watch it on this computer or a phone:" in out
        assert "  ftea preview plain\n" in out
        assert f"  ftea preview {quoted}\n" in out

    def test_should_list_at_most_five_clips(
        self, run_calls, tmp_path, capsys,
    ):
        _, outcome = run_calls
        outcome["result"] = RunResult(
            outputs=tuple(tmp_path / f"c{i}" for i in range(7)),
        )
        main(["run"])
        out = capsys.readouterr().out
        assert out.count("ftea preview c") == 5
        assert "...and 2 more in" in out

    @pytest.mark.parametrize("error", [
        PipelineError("ffmpeg is not on PATH"),
        paths.DataRootError("FTEA_HOME is set to 'x'"),
        paths.RepoRootError("ftea is not running from a cloned repository"),
    ])
    def test_should_report_a_setup_problem_without_a_traceback(
        self, run_calls, capsys, error,
    ):
        _, outcome = run_calls
        outcome["raise"] = error
        assert main(["run"]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: ")
        assert "Traceback" not in err

    def test_should_exit_130_when_interrupted(self, run_calls, capsys):
        _, outcome = run_calls
        outcome["raise"] = KeyboardInterrupt()
        assert main(["run"]) == 130
        assert "Interrupted." in capsys.readouterr().err


@pytest.fixture
def serve_calls(monkeypatch):
    calls = []

    def fake(clip_dir, *, port, local_only):
        calls.append((clip_dir, port, local_only))
        return 0

    monkeypatch.setattr(preview, "serve", fake)
    return calls


def _clip(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    return folder


class TestPreview:
    def test_should_leave_the_port_to_the_server_when_none_is_chosen(
        self, serve_calls, tmp_path,
    ):
        """None: the server tries its default port, then the next ones."""
        clip = _clip(tmp_path / "clip")
        assert main(["preview", str(clip)]) == 0
        assert serve_calls == [(clip.resolve(), None, False)]

    def test_should_keep_the_port_and_scope_the_user_chose(
        self, serve_calls, tmp_path,
    ):
        clip = _clip(tmp_path / "clip")
        assert main(
            ["preview", str(clip), "--port", "9000", "--local-only"],
        ) == 0
        assert serve_calls == [(clip.resolve(), 9000, True)]

    def test_should_preview_the_only_clip_in_the_output_folder(
        self, serve_calls,
    ):
        clip = _clip(paths.output_dir() / "only")
        assert main(["preview"]) == 0
        assert serve_calls[0][0] == clip.resolve()

    @pytest.mark.parametrize("port", [
        "70000", "-1", "eighty", "8_765", " 80 ", "+80",
        "\u0668\u0660",  # 80 in Arabic-Indic digits
    ])
    def test_should_refuse_a_port_that_does_not_exist(
        self, serve_calls, capsys, port,
    ):
        assert main(["preview", "x", "--port", port]) == 2
        assert "is not a port number" in capsys.readouterr().err
        assert serve_calls == []

    def test_should_explain_a_folder_with_nothing_to_play(
        self, serve_calls, tmp_path, capsys,
    ):
        (tmp_path / "empty").mkdir()
        assert main(["preview", str(tmp_path / "empty")]) == 1
        assert "not a finished clip folder" in capsys.readouterr().err

    def test_should_take_port_0_as_any_free_port(self, serve_calls, tmp_path):
        clip = _clip(tmp_path / "clip")
        assert main(["preview", str(clip), "--port", "0"]) == 0
        assert serve_calls[0][1] == 0

    def test_should_say_in_help_what_port_0_means(self, capsys):
        assert main(["preview", "--help"]) == 0
        out = " ".join(capsys.readouterr().out.split())  # undo wrapping
        assert "0 picks any free port" in out

    def test_should_preview_a_clip_path_when_the_data_folder_is_missing(
        self, serve_calls, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "typo"))
        clip = _clip(tmp_path / "clip")
        assert main(["preview", str(clip)]) == 0
        assert serve_calls[0][0] == clip.resolve()

    @pytest.mark.parametrize("error", [
        paths.MissingDataRootError("FTEA_HOME is set to 'typo'"),
        paths.RepoRootError("ftea is not running from a cloned repository"),
    ])
    def test_should_name_the_setup_problem_when_a_named_clip_is_not_found(
        self, serve_calls, monkeypatch, tmp_path, capsys, error,
    ):
        def broken():
            raise error

        monkeypatch.setattr(paths, "output_dir", broken)
        monkeypatch.chdir(tmp_path)
        assert main(["preview", "myclip"]) == 1
        err = capsys.readouterr().err
        assert "myclip was not found" in err
        assert str(error) in err
        assert serve_calls == []

    def test_should_report_a_missing_data_folder_when_no_clip_is_named(
        self, serve_calls, monkeypatch, tmp_path, capsys,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "typo"))
        assert main(["preview"]) == 1
        assert paths.ENV_HOME in capsys.readouterr().err

    def test_should_quote_the_real_default_port_in_help(self, capsys):
        assert main(["preview", "--help"]) == 0
        out = " ".join(capsys.readouterr().out.split())  # undo wrapping
        assert f"default: {preview.DEFAULT_PORT}" in out

    def test_should_exit_with_what_the_server_returns(
        self, monkeypatch, tmp_path,
    ):
        """serve() returns 0 after Ctrl+C, and ftea preview passes it on:
        130 is for ftea run and ftea reframe only."""
        clip = _clip(tmp_path / "clip")
        monkeypatch.setattr(preview, "serve", lambda clip_dir, **kwargs: 7)
        assert main(["preview", str(clip)]) == 7


@pytest.fixture
def vertical_calls(monkeypatch, tmp_path):
    calls = []
    outcome = {"result": None, "raise": None}

    def fake(video, *, out_dir=None, debug_video=False):
        calls.append((video, out_dir, debug_video))
        if outcome["raise"] is not None:
            raise outcome["raise"]
        return outcome["result"]

    out = tmp_path / "v-vertical"
    outcome["result"] = VerticalResult(
        "reframed", out, vertical=out / "vertical.mp4",
        crops=out / "crops.json",
    )
    monkeypatch.setattr(vertical, "make_vertical", fake)
    return calls, outcome, out


def _cp1252_console(monkeypatch):
    """What Windows gives a program whose output goes to a pipe or a
    file (Git Bash, `ftea preview > log.txt`): the ANSI code page."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    return stream, raw


class TestUnprintableNames:
    def test_should_preview_a_clip_whose_name_the_console_cannot_show(
        self, monkeypatch, tmp_path,
    ):
        def interrupted(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(preview.PreviewServer, "serve_forever", interrupted)
        clip = _clip(paths.output_dir() / "\u043a\u043b\u0438\u043f")
        stream, raw = _cp1252_console(monkeypatch)
        status = main(["preview", clip.name, "--port", "0", "--local-only"])
        stream.flush()
        assert status == 0
        text = raw.getvalue().decode("cp1252")
        assert "\\u043a\\u043b\\u0438\\u043f" in text
        assert text.splitlines()[-1] == "Stopped."

    def test_should_report_a_vertical_video_whose_name_the_console_cannot_show(
        self, vertical_calls, monkeypatch,
    ):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "reframed", out, vertical=out / "\u6771\u4eac.mp4",
        )
        stream, raw = _cp1252_console(monkeypatch)
        status = main(["reframe", "\u6771\u4eac.mp4"])
        stream.flush()
        assert status == 0
        assert "\\u6771\\u4eac.mp4" in raw.getvalue().decode("cp1252")


class TestReframe:
    def test_should_pass_the_video_and_options_on(self, vertical_calls):
        calls, _, _ = vertical_calls
        assert main(
            ["reframe", "v.mp4", "--debug-video", "--out", "results"],
        ) == 0
        assert main(["reframe", "w.mp4"]) == 0
        assert calls == [
            (Path("v.mp4"), Path("results"), True),
            (Path("w.mp4"), None, False),
        ]

    def test_should_name_the_files_it_wrote(self, vertical_calls, capsys):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "reframed", out, vertical=out / "vertical.mp4",
            crops=out / "crops.json", debug=out / "debug.mp4",
        )
        assert main(["reframe", "v.mp4", "--debug-video"]) == 0
        text = capsys.readouterr().out
        assert f"Vertical video: {out / 'vertical.mp4'}" in text
        assert f"Debug video:    {out / 'debug.mp4'}" in text

    def test_should_fail_when_the_debug_video_it_asked_for_failed(
        self, vertical_calls, capsys,
    ):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "reframed", out, vertical=out / "vertical.mp4",
            debug_error="RuntimeError: no NVENC session",
        )
        assert main(["reframe", "v.mp4", "--debug-video"]) == 1
        captured = capsys.readouterr()
        assert "Vertical video:" in captured.out
        assert "The debug video failed: RuntimeError" in captured.err
        assert "The vertical video is fine." in captured.err

    def test_should_say_there_was_nothing_to_reframe(
        self, vertical_calls, capsys,
    ):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "skipped", out, reason="already_vertical",
        )
        assert main(["reframe", "tall.mp4"]) == 0
        text = capsys.readouterr().out
        assert "Nothing to reframe: tall.mp4" in text
        assert "already_vertical" in text

    def test_should_point_at_the_partial_files_after_a_failure(
        self, vertical_calls, capsys,
    ):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "failed", out, error="no frames could be read",
        )
        assert main(["reframe", "v.mp4"]) == 1
        err = capsys.readouterr().err
        assert "Reframe failed: no frames could be read" in err
        assert str(out / vertical.PARTIAL_DIR) in err

    def test_should_report_a_stage_problem_without_a_traceback(
        self, vertical_calls, capsys,
    ):
        _, outcome, _ = vertical_calls
        outcome["raise"] = VerticalError("v.mp4 is not a video file")
        assert main(["reframe", "v.mp4"]) == 1
        assert capsys.readouterr().err.startswith(
            "Error: v.mp4 is not a video file",
        )

    def test_should_report_a_file_system_error_without_a_traceback(
        self, vertical_calls, capsys,
    ):
        _, outcome, _ = vertical_calls
        outcome["raise"] = PermissionError(13, "Access is denied", "results")
        assert main(["reframe", "v.mp4", "--out", "results"]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: ")
        assert "Access is denied" in err

    def test_should_say_when_no_crop_track_was_written(
        self, vertical_calls, capsys,
    ):
        _, outcome, out = vertical_calls
        outcome["result"] = VerticalResult(
            "reframed", out, vertical=out / "vertical.mp4",
        )
        assert main(["reframe", "v.mp4"]) == 0
        text = capsys.readouterr().out
        assert "Crop track:" not in text
        assert "No crop track was written" in text

    def test_should_say_in_help_that_a_video_left_alone_exits_0(self, capsys):
        assert main(["reframe", "--help"]) == 0
        out = " ".join(capsys.readouterr().out.split())  # undo wrapping
        assert "not landscape" in out
        assert "exits 0" in out

    def test_should_refuse_an_out_folder_inside_a_file(
        self, tmp_path, monkeypatch, capsys,
    ):
        """The real make_vertical, up to the point it would start the
        stage: the --out problem is reported before any work."""
        video = tmp_path / "v.mp4"
        video.write_bytes(b"video")
        exe = tmp_path / "reframe-python"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            encode_main, "load_run_config",
            lambda *args, **kwargs: PipelineConfig(reframe_python=str(exe)),
        )
        afile = tmp_path / "afile"
        afile.write_text("not a folder", encoding="utf-8")
        assert main(
            ["reframe", str(video), "--out", str(afile / "sub")],
        ) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: ")
        assert "Traceback" not in err
