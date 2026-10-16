"""Tests for ftea.vertical, which runs the reframe stage for `ftea reframe`."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ftea import paths, vertical
from ftea.encode.config import PipelineConfig
from ftea.encode.tools import run_process_tree
from ftea.tests import fake_stage

NAME = "It's a clip"


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "videos" / f"{NAME}.mp4"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"video")
    return path


def _config(tmp_path: Path) -> PipelineConfig:
    exe = tmp_path / "reframe-python"
    exe.write_text("", encoding="utf-8")
    return PipelineConfig(reframe_python=str(exe), reframe_timeout=60)


def _in_process(calls=None):
    """Run the stand-in stage in this process instead of a subprocess."""

    def run(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = fake_stage.main(list(cmd[3:]))
        return subprocess.CompletedProcess(cmd, code, buffer.getvalue(), None)

    return run


def _returning(stdout: str, returncode: int):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout, None)

    return run


def _held_path(path, *args, **kwargs):
    """What Windows raises for a file another program has open."""
    raise PermissionError(13, "The file is in use", str(path))


def _earlier_results_folder(tmp_path: Path) -> Path:
    """This video's default results folder, marked as its own, the way
    an earlier run leaves it."""
    out = paths.output_dir() / f"{NAME}-vertical"
    out.mkdir(parents=True)
    (out / vertical.MARKER_NAME).write_text(
        json.dumps({"schema": 1, "source": str(_video(tmp_path).resolve())}),
        encoding="utf-8",
    )
    return out


def _make(tmp_path, **options):
    options.setdefault("config", _config(tmp_path))
    options.setdefault("runner", _in_process())
    return vertical.make_vertical(_video(tmp_path), **options)


class TestResults:
    def test_should_put_the_vertical_video_in_its_own_folder(self, tmp_path):
        result = _make(tmp_path)
        out = paths.output_dir() / f"{NAME}-vertical"
        assert result.status == "reframed"
        assert result.out_dir == out.resolve()
        assert result.vertical == out.resolve() / "vertical.mp4"
        assert result.vertical.read_bytes() == b"vertical"
        assert result.crops == out.resolve() / "crops.json"
        assert not (out / ".partial").exists()
        assert result.exit_code == 0

    def test_should_ask_for_the_debug_video_beside_the_result(
        self, tmp_path,
    ):
        calls = []
        result = _make(
            tmp_path, debug_video=True, runner=_in_process(calls),
        )
        target = result.out_dir / "debug.mp4"
        cmd = calls[0][0]
        assert cmd[cmd.index("--debug-video") + 1] == str(target)
        assert result.debug == target
        assert result.exit_code == 0

    def test_should_fail_the_command_but_keep_the_video_when_the_debug_render_fails(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("FTEA_FAKE_STAGE", "debug-error")
        result = _make(tmp_path, debug_video=True)
        assert result.vertical.is_file()
        assert result.debug is None
        assert result.debug_error == "RuntimeError: no NVENC session"
        assert result.exit_code == 1

    def test_should_remove_an_old_debug_video_first(
        self, tmp_path, monkeypatch,
    ):
        out = _earlier_results_folder(tmp_path)
        (out / "debug.mp4").write_bytes(b"old")
        (out / "debug.sendcmd").write_text("old", encoding="utf-8")
        monkeypatch.setenv("FTEA_FAKE_STAGE", "debug-error")
        _make(tmp_path, debug_video=True)
        assert not (out / "debug.mp4").exists()
        assert not (out / "debug.sendcmd").exists()

    def test_should_leave_nothing_behind_for_a_vertical_video(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("FTEA_FAKE_STAGE", "skipped")
        result = _make(tmp_path)
        assert result.status == "skipped"
        assert result.reason == "already_vertical"
        assert not result.out_dir.exists()
        assert result.exit_code == 0

    def test_should_keep_the_partial_files_when_the_stage_fails(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("FTEA_FAKE_STAGE", "failed")
        result = _make(tmp_path)
        assert result.status == "failed"
        assert result.error == "no frames could be read"
        assert (result.out_dir / ".partial").is_dir()
        assert result.exit_code == 1

    def test_should_read_a_relative_out_folder_from_the_working_folder(
        self, tmp_path, monkeypatch,
    ):
        (tmp_path / "here").mkdir()
        monkeypatch.chdir(tmp_path / "here")
        result = _make(tmp_path, out_dir=Path("results"))
        assert result.out_dir == (tmp_path / "here" / "results").resolve()

    def test_should_refuse_a_folder_holding_other_files(self, tmp_path):
        out = tmp_path / "shorts"
        out.mkdir()
        (out / "notes.txt").write_text("mine", encoding="utf-8")
        calls = []
        with pytest.raises(vertical.VerticalError, match="--out"):
            _make(tmp_path, out_dir=out, runner=_in_process(calls))
        assert calls == []
        assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"

    def test_should_refuse_a_folder_holding_another_videos_results(
        self, tmp_path,
    ):
        out = tmp_path / "shorts"
        out.mkdir()
        other = tmp_path / "other.mp4"
        (out / "vertical.mp4").write_bytes(b"other")
        (out / vertical.MARKER_NAME).write_text(
            json.dumps({"schema": 1, "source": str(other.resolve())}),
            encoding="utf-8",
        )
        with pytest.raises(vertical.VerticalError, match="other.mp4"):
            _make(tmp_path, out_dir=out)
        assert (out / "vertical.mp4").read_bytes() == b"other"

    def test_should_record_which_video_the_results_belong_to(self, tmp_path):
        result = _make(tmp_path)
        marker = json.loads(
            (result.out_dir / vertical.MARKER_NAME).read_text(encoding="utf-8"),
        )
        assert marker["source"] == str(_video(tmp_path).resolve())

    def test_should_refuse_the_results_of_a_video_of_the_same_name_elsewhere(
        self, tmp_path,
    ):
        """Both default to output/<name>-vertical; only one may own it."""
        first = _make(tmp_path)
        elsewhere = tmp_path / "elsewhere" / f"{NAME}.mp4"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"another video")
        calls = []
        with pytest.raises(vertical.VerticalError) as info:
            vertical.make_vertical(
                elsewhere, config=_config(tmp_path),
                runner=_in_process(calls),
            )
        assert str(_video(tmp_path).resolve()) in str(info.value)
        assert "--out" in str(info.value)
        assert calls == []
        assert first.vertical.read_bytes() == b"vertical"
        assert first.crops.is_file()

    @pytest.mark.parametrize("names", [
        ("vertical.mp4",), ("vertical.mov", "crops.json"), (".partial",),
    ])
    def test_should_refuse_results_that_do_not_name_their_video(
        self, tmp_path, names,
    ):
        out = tmp_path / "shorts"
        out.mkdir()
        for name in names:
            if name.startswith("."):
                (out / name).mkdir()
            else:
                (out / name).write_bytes(b"unknown")
        calls = []
        with pytest.raises(vertical.VerticalError, match="--out"):
            _make(tmp_path, out_dir=out, runner=_in_process(calls))
        assert calls == []
        assert sorted(p.name for p in out.iterdir()) == sorted(names)

    def test_should_run_again_after_a_failed_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FTEA_FAKE_STAGE", "failed")
        _make(tmp_path)
        monkeypatch.setenv("FTEA_FAKE_STAGE", "reframed")
        assert _make(tmp_path).status == "reframed"

    def test_should_keep_an_empty_folder_the_user_chose_when_nothing_is_reframed(
        self, tmp_path, monkeypatch,
    ):
        out = tmp_path / "mine"
        out.mkdir()
        monkeypatch.setenv("FTEA_FAKE_STAGE", "skipped")
        assert _make(tmp_path, out_dir=out).status == "skipped"
        assert out.is_dir()
        assert list(out.iterdir()) == []

    def test_should_keep_earlier_results_when_nothing_is_reframed(
        self, tmp_path, monkeypatch,
    ):
        first = _make(tmp_path)
        monkeypatch.setenv("FTEA_FAKE_STAGE", "skipped")
        assert _make(tmp_path).status == "skipped"
        assert first.vertical.read_bytes() == b"vertical"
        assert (first.out_dir / vertical.MARKER_NAME).is_file()

    def test_should_drop_an_earlier_crop_track_the_new_run_did_not_write(
        self, tmp_path, monkeypatch,
    ):
        first = _make(tmp_path)
        assert first.crops.is_file()
        monkeypatch.setenv("FTEA_FAKE_STAGE", "no-crops")
        second = _make(tmp_path)
        assert second.crops is None
        assert not (second.out_dir / "crops.json").exists()
        assert second.vertical.read_bytes() == b"vertical"

    def test_should_drop_an_earlier_vertical_video_of_another_type(
        self, tmp_path,
    ):
        first = _make(tmp_path)
        stale = first.out_dir / "vertical.mkv"
        stale.write_bytes(b"older container")
        second = _make(tmp_path)
        assert not stale.exists()
        assert second.vertical.read_bytes() == b"vertical"

    def test_should_replace_this_videos_own_earlier_results(self, tmp_path):
        first = _make(tmp_path, debug_video=True)
        second = _make(tmp_path, debug_video=True)
        assert second.out_dir == first.out_dir
        assert second.vertical.read_bytes() == b"vertical"
        assert second.debug == second.out_dir / "debug.mp4"


class TestLaunch:
    def test_should_launch_the_stage_by_the_path_contract(
        self, tmp_path, monkeypatch,
    ):
        """Relative input on purpose: every path argument must be absolute."""
        video = _video(tmp_path)
        monkeypatch.chdir(video.parent)
        calls = []
        vertical.make_vertical(
            Path(video.name), config=_config(tmp_path),
            runner=_in_process(calls),
        )
        cmd, kwargs = calls[0]
        assert cmd[:3] == [
            str(tmp_path / "reframe-python"), "-m", paths.REFRAME_MODULE,
        ]
        assert cmd[cmd.index("--source") + 1] == str(video.resolve())
        assert Path(cmd[cmd.index("--work-dir") + 1]).is_absolute()
        assert "--config" not in cmd
        assert kwargs["cwd"] == str(paths.source_root())
        assert kwargs["env"]["PYTHONPATH"] == str(paths.checkout_root())
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        assert kwargs["stdout"] == subprocess.PIPE
        assert "stderr" not in kwargs  # progress goes to the terminal
        assert kwargs["timeout"] == 60

    def test_should_launch_a_real_interpreter_from_the_checkout(
        self, tmp_path, monkeypatch, capfd,
    ):
        monkeypatch.setattr(paths, "REFRAME_MODULE", "ftea.tests.fake_stage")
        monkeypatch.chdir(tmp_path)
        config = PipelineConfig(
            reframe_python=sys.executable, reframe_timeout=120,
        )
        result = vertical.make_vertical(_video(tmp_path), config=config)
        assert result.status == "reframed"
        assert result.vertical.read_bytes() == b"vertical"
        assert Path(result.report["cwd"]).resolve() == paths.source_root()
        assert result.report["pythonpath"] == str(paths.checkout_root())
        assert "fake stage running" in capfd.readouterr().err


class TestErrors:
    def test_should_refuse_a_missing_video(self, tmp_path):
        with pytest.raises(vertical.VerticalError, match="not a video file"):
            vertical.make_vertical(
                tmp_path / "missing.mp4", config=_config(tmp_path),
            )

    def test_should_explain_how_to_install_without_an_interpreter(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            paths, "reframe_venv_dir", lambda: tmp_path / "no-venv",
        )
        with pytest.raises(vertical.VerticalError, match="install.sh"):
            vertical.make_vertical(_video(tmp_path), config=PipelineConfig())

    def test_should_refuse_output_that_is_not_a_result(self, tmp_path):
        with pytest.raises(vertical.VerticalError, match="printed no result"):
            _make(tmp_path, runner=_returning("Traceback (most recent", 1))

    def test_should_refuse_an_exit_code_that_contradicts_the_status(
        self, tmp_path,
    ):
        payload = json.dumps({"schema": 1, "status": "reframed"})
        with pytest.raises(vertical.VerticalError, match="exited 2"):
            _make(tmp_path, runner=_returning(payload, 2))

    def test_should_name_the_setting_when_the_stage_runs_too_long(
        self, tmp_path,
    ):
        limits = []

        def hang(cmd, **kwargs):
            limits.append(kwargs["timeout"])
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with pytest.raises(
            vertical.VerticalError, match="REFRAME_TIMEOUT",
        ) as info:
            _make(tmp_path, runner=hang, debug_video=True)
        assert limits == [120]  # twice the configured 60 s
        work = (paths.output_dir() / f"{NAME}-vertical").resolve() / ".partial"
        assert f"partial files are in {work}" in str(info.value)
        assert "may already be" not in str(info.value)

    def test_should_stop_the_programs_the_stage_started_by_default(self):
        """The stage runs ffmpeg itself; subprocess.run would leave it
        running past the time limit."""
        default = inspect.signature(vertical.make_vertical).parameters[
            "runner"
        ].default
        assert default is run_process_tree

    def test_should_not_offer_an_option_ftea_reframe_does_not_have(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            paths, "reframe_venv_dir", lambda: tmp_path / ".venv-reframe",
        )
        with pytest.raises(vertical.VerticalError) as info:
            _make(tmp_path, config=PipelineConfig())
        assert "install.sh" in str(info.value)
        assert "--no-reframe" not in str(info.value)

    def test_should_ignore_encoder_settings_it_does_not_use(self, tmp_path):
        """A bad VMAF or bitrate value must not block a reframe-only run."""
        exe = tmp_path / "reframe-python"
        exe.write_text("", encoding="utf-8")
        paths.config_path().write_text(
            "VMAF_TARGET_1080=abc\nAV1_MAXRATE_360=fast\n"
            f"REFRAME_PYTHON={exe.as_posix()}\nREFRAME_TIMEOUT=45\n",
            encoding="utf-8",
        )
        calls = []

        result = _make(tmp_path, config=None, runner=_in_process(calls))

        assert result.status == "reframed"
        assert calls[0][1]["timeout"] == 45

    def test_should_say_when_the_interpreter_cannot_start(self, tmp_path):
        def refuse(cmd, **kwargs):
            raise OSError("not a program")

        with pytest.raises(vertical.VerticalError, match="could not be started"):
            _make(tmp_path, runner=refuse)

    def test_should_name_old_partial_files_that_are_held(
        self, tmp_path, monkeypatch,
    ):
        out = _earlier_results_folder(tmp_path)
        (out / ".partial").mkdir()
        monkeypatch.setattr(vertical.shutil, "rmtree", _held_path)
        calls = []
        with pytest.raises(vertical.VerticalError, match="close it in the"):
            _make(tmp_path, runner=_in_process(calls))
        assert calls == []

    def test_should_name_an_old_debug_video_that_is_open_elsewhere(
        self, tmp_path, monkeypatch,
    ):
        out = _earlier_results_folder(tmp_path)
        (out / "debug.mp4").write_bytes(b"old")
        monkeypatch.setattr(Path, "unlink", _held_path)
        calls = []
        with pytest.raises(vertical.VerticalError, match="debug.mp4 is in use"):
            _make(tmp_path, debug_video=True, runner=_in_process(calls))
        assert calls == []

    def test_should_name_an_earlier_result_that_is_open_elsewhere(
        self, tmp_path, monkeypatch,
    ):
        earlier = _make(tmp_path)
        monkeypatch.setattr(Path, "rename", _held_path)
        calls = []
        with pytest.raises(vertical.VerticalError, match="is in use"):
            _make(tmp_path, runner=_in_process(calls))
        assert calls == []
        assert earlier.vertical.read_bytes() == b"vertical"

    def test_should_name_the_partial_folder_when_the_result_cannot_be_moved(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(vertical.os, "replace", _held_path)
        with pytest.raises(vertical.VerticalError) as info:
            _make(tmp_path)
        work = (paths.output_dir() / f"{NAME}-vertical").resolve() / ".partial"
        assert str(work) in str(info.value)
        assert (work / "vertical.mp4").is_file()

    def test_should_say_where_each_file_is_when_the_crop_track_cannot_be_moved(
        self, tmp_path, monkeypatch,
    ):
        real = vertical.os.replace

        def crops_held(source, target):
            if Path(source).name == "crops.json":
                raise PermissionError(13, "The file is in use", str(target))
            real(source, target)

        monkeypatch.setattr(vertical.os, "replace", crops_held)
        with pytest.raises(vertical.VerticalError) as info:
            _make(tmp_path)
        out = (paths.output_dir() / f"{NAME}-vertical").resolve()
        message = str(info.value)
        assert f"vertical video is in {out / 'vertical.mp4'}" in message
        assert str(out / ".partial" / "crops.json") in message
        assert (out / "vertical.mp4").is_file()
        assert (out / ".partial" / "crops.json").is_file()

    def test_should_refuse_an_out_folder_inside_a_file(self, tmp_path):
        afile = tmp_path / "afile"
        afile.write_text("not a folder", encoding="utf-8")
        calls = []
        with pytest.raises(vertical.VerticalError, match="cannot be used"):
            _make(
                tmp_path, out_dir=afile / "sub", runner=_in_process(calls),
            )
        assert calls == []

    def test_should_refuse_a_results_folder_it_cannot_read(
        self, tmp_path, monkeypatch,
    ):
        out = tmp_path / "shorts"
        out.mkdir()

        def unreadable(self):
            raise PermissionError(13, "Access is denied", str(self))

        monkeypatch.setattr(Path, "iterdir", unreadable)
        with pytest.raises(vertical.VerticalError, match="cannot be used"):
            _make(tmp_path, out_dir=out)


class TestPayload:
    """The stage's JSON result is a published contract; a payload that
    breaks it is reported, never a traceback."""

    def test_should_expect_the_stages_schema(self):
        from ftea.reframe.main import SCHEMA_VERSION

        assert vertical.STAGE_SCHEMA == SCHEMA_VERSION

    @pytest.mark.parametrize("payload, code, field", [
        ({"schema": 2, "status": "reframed"}, 0, "schema"),
        ({"status": "reframed"}, 0, "schema"),
        ({"schema": 1, "status": ["reframed"]}, 0, "status"),
        ({"schema": 1, "status": "done"}, 0, "status"),
        ({"schema": 1, "status": "reframed", "output": 5}, 0, "output"),
        ({"schema": 1, "status": "reframed", "output": None}, 0, "output"),
        ({"schema": 1, "status": "reframed", "output": ""}, 0, "output"),
        ({"schema": 1, "status": "reframed", "output": "v.mp4",
          "crops": ["c"]}, 0, "crops"),
        ({"schema": 1, "status": "reframed", "output": "v.mp4",
          "debug": {}}, 0, "debug"),
        ({"schema": 1, "status": "failed", "error": 7}, 1, "error"),
        ({"schema": 1, "status": "skipped", "report": "x"}, 2, "report"),
    ])
    def test_should_refuse_a_payload_that_breaks_the_contract(
        self, tmp_path, payload, code, field,
    ):
        with pytest.raises(vertical.VerticalError, match=field):
            _make(tmp_path, runner=_returning(json.dumps(payload), code))


class TestResultShape:
    def test_should_refuse_an_unknown_status(self, tmp_path):
        with pytest.raises(ValueError, match="status"):
            vertical.VerticalResult("done", tmp_path)

    def test_should_refuse_a_reframed_result_without_its_video(self, tmp_path):
        with pytest.raises(ValueError, match="vertical"):
            vertical.VerticalResult("reframed", tmp_path)
