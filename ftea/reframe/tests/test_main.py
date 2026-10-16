"""Tests for ftea.reframe.main — the reframe stage entry point."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch


from ftea import paths
from ftea.reframe.debug_video import DebugResult, DebugVideoError
from ftea.reframe.errors import ReframeError
from ftea.reframe.main import (
    EXIT_FAILED,
    EXIT_REFRAMED,
    EXIT_SKIPPED,
    SCHEMA_VERSION,
    main,
)
from ftea.reframe.probe import ProbeError


def _payload(width, height, rotation=None):
    """An ffprobe JSON payload shaped like the real thing."""
    stream = {"codec_type": "video", "width": width, "height": height,
              "r_frame_rate": "25/1", "duration": "10.000000"}
    if rotation is not None:
        stream["side_data_list"] = [
            {"side_data_type": "Display Matrix", "rotation": rotation},
        ]
    return {"streams": [stream], "format": {"duration": "10.000000"}}


def _argv(tmp_path, source="clip.mp4"):
    return [
        "--source", str(tmp_path / source),
        "--work-dir", str(tmp_path / "work"),
        "--config", str(tmp_path / "config.env"),
    ]


class TestSkipPath:
    def test_should_exit_two_and_report_skip_when_already_vertical(
        self, tmp_path, capsys,
    ):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920)):
            code = main(_argv(tmp_path))

        captured = capsys.readouterr()
        assert code == EXIT_SKIPPED == 2
        result = json.loads(captured.out)
        assert result["schema"] == SCHEMA_VERSION
        assert result["status"] == "skipped"
        assert result["output"] is None
        assert result["crops"] is None
        assert result["error"] is None
        assert result["report"]["applied"] is False
        assert result["report"]["reason"] == "already_vertical"
        assert "skipped: already vertical" in captured.err

    def test_should_exit_two_when_square(self, tmp_path, capsys):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1080)):
            code = main(_argv(tmp_path))

        assert code == EXIT_SKIPPED
        assert json.loads(capsys.readouterr().out)["report"]["reason"] \
            == "square"

    def test_should_not_call_run_stages_when_skipping(self, tmp_path):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920)), \
             patch("ftea.reframe.main.run_stages") as stages:
            main(_argv(tmp_path))

        assert stages.call_count == 0

    def test_should_write_nothing_to_disk_when_skipping(self, tmp_path):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920)):
            main(_argv(tmp_path))

        assert not (tmp_path / "work").exists()


class TestReframePath:
    def test_should_exit_zero_and_report_paths_when_reframed(
        self, tmp_path, capsys,
    ):
        def fake_stages(source, meta, out_path, crops_path, settings):
            out_path.write_bytes(b"video")
            crops_path.write_text("[]")

        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=fake_stages):
            code = main(_argv(tmp_path))

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_REFRAMED == 0
        assert result["status"] == "reframed"
        assert result["report"]["applied"] is True
        assert result["report"]["reason"] == "landscape"
        assert result["report"]["elapsed_sec"] >= 0.0
        assert Path(result["output"]).exists()
        assert Path(result["crops"]).exists()

    def test_should_name_the_output_from_the_container_setting(
        self, tmp_path,
    ):
        (tmp_path / "config.env").write_text("REFRAME_CONTAINER=mkv\n")

        def fake_stages(source, meta, out_path, crops_path, settings):
            out_path.write_bytes(b"v")
            crops_path.write_text("[]")

        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=fake_stages):
            main(_argv(tmp_path))

        assert (tmp_path / "work" / "vertical.mkv").exists()

    def test_should_pass_display_geometry_to_run_stages(self, tmp_path):
        seen = {}

        def fake_stages(source, meta, out_path, crops_path, settings):
            seen["display"] = (meta.display_width, meta.display_height)
            seen["stride"] = settings.stride
            out_path.write_bytes(b"v")
            crops_path.write_text("[]")

        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920, rotation=90)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=fake_stages):
            main(_argv(tmp_path))

        assert seen["display"] == (1920, 1080)
        assert seen["stride"] == 3


class TestAtomicOutput:
    def test_should_fail_when_run_stages_leaves_no_output(
        self, tmp_path, capsys,
    ):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages", return_value=None):
            code = main(_argv(tmp_path))

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED == 1
        assert result["status"] == "failed"
        assert "no output" in result["error"]

    def test_should_remove_a_stale_partial_before_running(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir(parents=True)
        stale = work / "vertical.partial.mp4"
        stale.write_bytes(b"junk from a killed run")

        def fake_stages(source, meta, out_path, crops_path, settings):
            assert not stale.exists()
            out_path.write_bytes(b"v")
            crops_path.write_text("[]")

        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=fake_stages):
            assert main(_argv(tmp_path)) == EXIT_REFRAMED


class TestStdoutHygiene:
    def test_should_still_emit_parseable_json_when_a_stage_prints(
        self, tmp_path, capsys,
    ):
        """torch prints banners, and run_stages' own progress print()
        calls must not land where the payload does."""
        def fake_stages(source, meta, out_path, crops_path, settings):
            print("noise")
            out_path.write_bytes(b"v")
            crops_path.write_text("[]")

        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=fake_stages):
            code = main(_argv(tmp_path))

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert code == EXIT_REFRAMED
        assert result["status"] == "reframed"
        assert "noise" in captured.err


class TestFailurePaths:
    def test_should_exit_one_and_report_when_run_stages_raises(
        self, tmp_path, capsys,
    ):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=ReframeError("detector exploded")):
            code = main(_argv(tmp_path))

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED
        assert result["status"] == "failed"
        assert "detector exploded" in result["error"]
        assert result["output"] is None
        assert result["report"]["reason"] == "landscape"
        assert result["report"]["outcome"] == "failed"

    def test_should_exit_one_and_report_when_probe_fails(
        self, tmp_path, capsys,
    ):
        with patch("ftea.reframe.probe._run_ffprobe",
                   side_effect=ProbeError("ffprobe failed: no such file")):
            code = main(_argv(tmp_path))

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED
        assert result["status"] == "failed"
        assert result["report"] is None
        assert "no such file" in result["error"]

    def test_should_exit_one_when_settings_are_invalid(
        self, tmp_path, capsys,
    ):
        (tmp_path / "config.env").write_text("REFRAME_STRIDE=0\n")
        code = main(_argv(tmp_path))

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED
        assert "REFRAME_STRIDE" in result["error"]

    def test_should_emit_valid_json_even_on_failure(self, tmp_path, capsys):
        """The orchestrator parses stdout unconditionally — it must be JSON."""
        with patch("ftea.reframe.probe._run_ffprobe",
                   side_effect=ProbeError("boom")):
            main(_argv(tmp_path))

        json.loads(capsys.readouterr().out)


class TestArgumentParsing:
    def test_should_exit_one_when_source_missing(self, tmp_path, capsys):
        code = main(["--work-dir", str(tmp_path / "work")])

        assert code == EXIT_FAILED
        assert "--source" in capsys.readouterr().err

    def test_should_exit_one_when_work_dir_missing(self, tmp_path, capsys):
        code = main(["--source", str(tmp_path / "clip.mp4")])

        assert code == EXIT_FAILED
        assert "--work-dir" in capsys.readouterr().err

    def test_should_default_config_to_the_data_root_config_env(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Omitting --config reads <data root>/config.env, never the cwd's."""
        paths.config_path().write_text("REFRAME_STRIDE=0\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "config.env").write_text("REFRAME_BATCH_SIZE=0\n")
        monkeypatch.chdir(elsewhere)

        code = main([
            "--source", str(tmp_path / "clip.mp4"),
            "--work-dir", str(tmp_path / "work"),
        ])

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED
        assert "REFRAME_STRIDE" in result["error"]

class TestConfigFile:
    def test_should_say_so_when_the_config_file_is_missing(
        self, tmp_path, capsys,
    ):
        """The encoder always passes --config, even when no config.env
        exists, so a missing file is not an error. It is not silent
        either: a mistyped path must show up on stderr."""
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920)):
            code = main(_argv(tmp_path))

        err = capsys.readouterr().err
        assert code == EXIT_SKIPPED
        assert f"no config file at {tmp_path / 'config.env'}" in err
        assert "built-in defaults" in err

    def test_should_not_warn_when_the_config_file_exists(
        self, tmp_path, capsys,
    ):
        (tmp_path / "config.env").write_text("REFRAME_STRIDE=2\n")
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1080, 1920)):
            main(_argv(tmp_path))

        assert "no config file" not in capsys.readouterr().err


class TestRunStagesSeam:
    def test_should_be_implemented_by_render(self):
        """The seam is filled in.

        The behaviour of the filled-in body is covered by
        test_render.py::TestRunStages; all this asserts is that the seam
        is no longer a placeholder and still has its published signature.
        """
        import inspect

        from ftea.reframe.pipeline import run_stages

        assert "NotImplementedError" not in inspect.getsource(run_stages)
        assert list(
            inspect.signature(run_stages).parameters
        ) == ["source", "meta", "out_path", "crops_path", "settings"]


def _fake_stages(source, meta, out_path, crops_path, settings):
    out_path.write_bytes(b"video")
    crops_path.write_text("{}", encoding="ascii")


def _debug_result(path):
    return DebugResult(
        out_path=path, script=path.with_name("debug.sendcmd"),
        width=1686, height=720, frames=96, commands=1, elapsed_s=0.1,
        codec="h264_nvenc", command=("ffmpeg",),
    )


class TestDebugVideo:
    """--debug-video <out.mp4> renders the side-by-side video after a
    successful reframe, and reports it without ever failing the clip."""

    def _run(self, tmp_path, *, render=None, spec=None, extra=(),
             landscape=True):
        argv = _argv(tmp_path) + list(extra)
        size = (1920, 1080) if landscape else (1080, 1920)
        patches = [
            patch("ftea.reframe.probe._run_ffprobe",
                  return_value=_payload(*size)),
            patch("ftea.reframe.main.run_stages", side_effect=_fake_stages),
        ]
        if spec is not None:
            patches.append(
                patch("ftea.reframe.main.spec_from_crops", side_effect=spec))
        if render is not None:
            patches.append(patch(
                "ftea.reframe.main.render_debug_video", side_effect=render))
        for item in patches:
            item.start()
        try:
            return main(argv)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_should_report_the_debug_video_when_asked(
        self, tmp_path, capsys,
    ):
        target = tmp_path / "debug.mp4"
        seen = {}

        def fake_spec(crops, source, vertical, out_path, **kw):
            seen["spec"] = (source, vertical, out_path)
            return "SPEC"

        def fake_render(spec, *, ffmpeg=None, require_inputs=False):
            seen["render"] = (spec, ffmpeg, require_inputs)
            return _debug_result(target)

        code = self._run(tmp_path, spec=fake_spec, render=fake_render,
                         extra=["--debug-video", str(target)])

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_REFRAMED
        assert result["status"] == "reframed"
        assert result["debug"] == str(target)
        assert result["debug_error"] is None
        assert seen["spec"] == (tmp_path / "clip.mp4",
                                Path(result["output"]), target)
        assert seen["render"] == ("SPEC", None, True)

    def test_should_keep_the_reframe_when_the_debug_render_fails(
        self, tmp_path, capsys,
    ):
        def failing_render(spec, *, ffmpeg=None, require_inputs=False):
            raise DebugVideoError("ffmpeg failed: no NVENC session")

        code = self._run(tmp_path, spec=lambda *a, **k: "SPEC",
                         render=failing_render,
                         extra=["--debug-video", str(tmp_path / "d.mp4")])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert code == EXIT_REFRAMED
        assert result["status"] == "reframed"
        assert Path(result["output"]).exists()
        assert result["debug"] is None
        assert "no NVENC session" in result["debug_error"]
        assert "debug video FAILED" in captured.err

    def test_should_report_an_unreadable_crop_track_without_failing(
        self, tmp_path, capsys,
    ):
        code = self._run(tmp_path,
                         extra=["--debug-video", str(tmp_path / "d.mp4")])

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_REFRAMED
        assert result["debug"] is None
        assert "no 'track' object" in result["debug_error"]

    def test_should_reject_debug_video_without_a_path(
        self, tmp_path, capsys,
    ):
        code = main(_argv(tmp_path) + ["--debug-video"])

        captured = capsys.readouterr()
        assert code == EXIT_FAILED
        assert "--debug-video requires a value" in captured.err
        assert json.loads(captured.out)["status"] == "failed"

    def test_should_not_render_a_debug_video_when_skipping(
        self, tmp_path, capsys,
    ):
        def must_not_render(*args, **kwargs):
            raise AssertionError("a skipped clip has nothing to debug")

        code = self._run(tmp_path, render=must_not_render, landscape=False,
                         extra=["--debug-video", str(tmp_path / "d.mp4")])

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_SKIPPED
        assert result["debug"] is None
        assert result["debug_error"] is None

    def test_should_report_no_debug_video_when_not_asked(
        self, tmp_path, capsys,
    ):
        def must_not_render(*args, **kwargs):
            raise AssertionError("nobody asked for a debug video")

        code = self._run(tmp_path, render=must_not_render)

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_REFRAMED
        assert result["debug"] is None
        assert result["debug_error"] is None

    def test_should_render_with_the_configured_ffmpeg(
        self, tmp_path, capsys,
    ):
        (tmp_path / "config.env").write_text(
            "REFRAME_FFMPEG=/opt/ffmpeg/bin/ffmpeg\n")
        seen = {}

        def fake_render(spec, *, ffmpeg=None, require_inputs=False):
            seen["ffmpeg"] = ffmpeg
            return _debug_result(tmp_path / "d.mp4")

        self._run(tmp_path, spec=lambda *a, **k: "SPEC", render=fake_render,
                  extra=["--debug-video", str(tmp_path / "d.mp4")])

        assert json.loads(capsys.readouterr().out)["debug"] is not None
        assert seen["ffmpeg"] == "/opt/ffmpeg/bin/ffmpeg"

    def test_should_keep_the_reframe_when_the_debug_render_raises_anything(
        self, tmp_path, capsys,
    ):
        def broken_render(spec, *, ffmpeg=None, require_inputs=False):
            raise RuntimeError("a bug in the debug renderer")

        code = self._run(tmp_path, spec=lambda *a, **k: "SPEC",
                         render=broken_render,
                         extra=["--debug-video", str(tmp_path / "d.mp4")])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert code == EXIT_REFRAMED
        assert result["status"] == "reframed"
        assert result["debug"] is None
        assert result["debug_error"] == (
            "RuntimeError: a bug in the debug renderer")
        assert "Traceback" in captured.err

    def test_should_reject_a_relative_debug_video_path(
        self, tmp_path, capsys,
    ):
        code = main(_argv(tmp_path) + ["--debug-video", "debug.mp4"])

        captured = capsys.readouterr()
        assert code == EXIT_FAILED
        assert "--debug-video must be an absolute path" in captured.err
        assert json.loads(captured.out)["status"] == "failed"

    def test_should_remove_the_sendcmd_sidecar_after_a_good_render(
        self, tmp_path, capsys,
    ):
        target = tmp_path / "d.mp4"
        sidecar = target.with_name("debug.sendcmd")

        def fake_render(spec, *, ffmpeg=None, require_inputs=False):
            sidecar.write_text("0.0 drawbox x 0;\n", encoding="ascii")
            return _debug_result(target)

        code = self._run(tmp_path, spec=lambda *a, **k: "SPEC",
                         render=fake_render,
                         extra=["--debug-video", str(target)])

        assert code == EXIT_REFRAMED
        assert json.loads(capsys.readouterr().out)["debug"] == str(target)
        assert not sidecar.exists()


class TestUnexpectedFailure:
    def test_should_keep_the_one_json_object_when_a_bug_escapes(
        self, tmp_path, capsys,
    ):
        """A bug or an OSError is not a ReframeError, but the encoder
        still reads exactly one JSON object from stdout."""
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=RuntimeError("bug")):
            code = main(_argv(tmp_path))

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert code == EXIT_FAILED
        assert result["status"] == "failed"
        assert result["error"] == "RuntimeError: bug"
        assert result["report"]["outcome"] == "failed"
        assert "Traceback" in captured.err


class TestPostRunChecks:
    def _run(self, tmp_path, fake):
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages", side_effect=fake):
            return main(_argv(tmp_path))

    def test_should_fail_when_no_crop_track_was_written(
        self, tmp_path, capsys,
    ):
        def video_only(source, meta, out_path, crops_path, settings):
            out_path.write_bytes(b"v")

        code = self._run(tmp_path, video_only)

        result = json.loads(capsys.readouterr().out)
        assert code == EXIT_FAILED
        assert "no crop track" in result["error"]
        assert result["output"] is None

    def test_should_fail_when_the_output_is_empty(self, tmp_path, capsys):
        def empty_video(source, meta, out_path, crops_path, settings):
            out_path.write_bytes(b"")
            crops_path.write_text("{}")

        code = self._run(tmp_path, empty_video)

        assert code == EXIT_FAILED
        assert "no output" in json.loads(capsys.readouterr().out)["error"]

    def test_should_remove_every_stale_artifact_before_running(
        self, tmp_path,
    ):
        """A leftover crops.json would otherwise satisfy the post-run
        check, and a leftover crops.failed.json would sit beside a good
        result as if this run had failed."""
        work = tmp_path / "work"
        work.mkdir()
        stale = [work / name for name in (
            "vertical.mp4", "crops.json", "vertical.partial.mp4",
            "crops.partial.json", "crops.failed.json")]
        for path in stale:
            path.write_bytes(b"old")
        present = {}

        def fake(source, meta, out_path, crops_path, settings):
            present.update({path.name: path.exists() for path in stale})
            out_path.write_bytes(b"v")
            crops_path.write_text("{}")

        assert self._run(tmp_path, fake) == EXIT_REFRAMED
        assert present == {path.name: False for path in stale}


class TestDebugSidecar:
    def _run(self, tmp_path, render, extra=()):
        argv = _argv(tmp_path) + list(extra)
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=_payload(1920, 1080)), \
             patch("ftea.reframe.main.run_stages",
                   side_effect=_fake_stages), \
             patch("ftea.reframe.main.spec_from_crops",
                   side_effect=lambda *a, **k: "SPEC"), \
             patch("ftea.reframe.main.render_debug_video",
                   side_effect=render):
            return main(argv)

    def test_should_keep_the_sidecar_after_a_failed_debug_render(
        self, tmp_path, capsys,
    ):
        target = tmp_path / "d.mp4"
        sidecar = target.with_name("debug.sendcmd")

        def failing(spec, *, ffmpeg=None, require_inputs=False):
            sidecar.write_text("0.0 drawbox x 0;\n", encoding="ascii")
            raise DebugVideoError("ffmpeg failed: boom")

        code = self._run(tmp_path, failing,
                         ["--debug-video", str(target)])

        assert code == EXIT_REFRAMED
        assert json.loads(capsys.readouterr().out)["debug_error"]
        assert sidecar.exists()

    def test_should_still_report_the_video_when_the_sidecar_will_not_go(
        self, tmp_path, capsys,
    ):
        target = tmp_path / "d.mp4"

        class StuckScript:
            name = "debug.sendcmd"

            def unlink(self, missing_ok=False):
                raise PermissionError("in use by another process")

        def render(spec, *, ffmpeg=None, require_inputs=False):
            return dataclasses.replace(_debug_result(target),
                                       script=StuckScript())

        code = self._run(tmp_path, render, ["--debug-video", str(target)])

        captured = capsys.readouterr()
        assert code == EXIT_REFRAMED
        assert json.loads(captured.out)["debug"] == str(target)
        assert "could not remove debug.sendcmd" in captured.err


class TestArgumentRejection:
    def test_should_reject_an_unknown_argument(self, tmp_path, capsys):
        code = main(_argv(tmp_path) + ["--bogus", "x"])

        captured = capsys.readouterr()
        assert code == EXIT_FAILED
        assert "unknown argument '--bogus'" in captured.err
        assert json.loads(captured.out)["status"] == "failed"

    def test_should_reject_a_debug_video_that_is_not_mp4(
        self, tmp_path, capsys,
    ):
        code = main(_argv(tmp_path)
                    + ["--debug-video", str(tmp_path / "d.mkv")])

        captured = capsys.readouterr()
        assert code == EXIT_FAILED
        assert "--debug-video must be an absolute path to an .mp4" in \
            captured.err
