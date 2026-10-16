"""Tests for ftea.reframe.probe — the auto-reframe input gate."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import (
    EXIT_ERROR,
    EXIT_REFRAME,
    EXIT_SKIP,
    FFPROBE_TIMEOUT_SEC,
    OUTCOME_FAILED,
    OUTCOME_GATE_ONLY,
    OUTCOME_REFRAMED,
    OUTCOME_SKIPPED,
    REASON_ALREADY_VERTICAL,
    REASON_LANDSCAPE,
    REASON_SQUARE,
    GateDecision,
    ProbeError,
    SourceMeta,
    _run_ffprobe,
    decide,
    is_quarter_turn,
    main,
    normalize_rotation,
    probe_source,
    reframe_report_entry,
)


def _meta(width=1920, height=1080, rotation=0, fps=25.0, duration_s=10.0,
          has_audio=False):
    """Build a SourceMeta without touching ffprobe."""
    return SourceMeta(
        path=Path("clip.mp4"),
        width=width, height=height, rotation=rotation,
        fps=fps, duration_s=duration_s, has_audio=has_audio,
    )


class TestNormalizeRotation:
    def test_should_return_zero_when_none(self):
        assert normalize_rotation(None) == 0

    def test_should_pass_through_exact_quarter_turns(self):
        assert normalize_rotation(0) == 0
        assert normalize_rotation(90) == 90
        assert normalize_rotation(180) == 180
        assert normalize_rotation(270) == 270

    def test_should_wrap_negative_ninety_to_270(self):
        assert normalize_rotation(-90) == 270

    def test_should_wrap_negative_180_to_180(self):
        assert normalize_rotation(-180) == 180

    def test_should_wrap_360_to_zero(self):
        assert normalize_rotation(360) == 0

    def test_should_wrap_450_to_ninety(self):
        assert normalize_rotation(450) == 90

    def test_should_accept_string_and_float_forms(self):
        assert normalize_rotation("90") == 90
        assert normalize_rotation(-90.0) == 270

    def test_should_round_to_nearest_quarter_turn(self):
        assert normalize_rotation(89) == 90
        # 45 is an exact .5 tie; Python rounds half to even -> 0.
        assert normalize_rotation(45) == 0

    def test_should_return_zero_when_unparseable(self):
        assert normalize_rotation("sideways") == 0
        assert normalize_rotation("") == 0


class TestIsQuarterTurn:
    def test_should_be_true_for_90_and_270(self):
        assert is_quarter_turn(90) is True
        assert is_quarter_turn(270) is True

    def test_should_be_false_for_0_and_180(self):
        assert is_quarter_turn(0) is False
        assert is_quarter_turn(180) is False


class TestSourceMetaDisplayDimensions:
    def test_should_keep_dimensions_when_not_rotated(self):
        meta = _meta(width=1920, height=1080, rotation=0)
        assert (meta.display_width, meta.display_height) == (1920, 1080)

    def test_should_keep_dimensions_when_rotated_180(self):
        meta = _meta(width=1920, height=1080, rotation=180)
        assert (meta.display_width, meta.display_height) == (1920, 1080)

    def test_should_swap_dimensions_when_rotated_90(self):
        meta = _meta(width=1920, height=1080, rotation=90)
        assert (meta.display_width, meta.display_height) == (1080, 1920)

    def test_should_swap_dimensions_when_rotated_270(self):
        meta = _meta(width=1920, height=1080, rotation=270)
        assert (meta.display_width, meta.display_height) == (1080, 1920)

    def test_should_leave_coded_dimensions_untouched_when_rotated(self):
        meta = _meta(width=1920, height=1080, rotation=90)
        assert (meta.width, meta.height) == (1920, 1080)


class TestSourceMetaOrientation:
    def test_should_be_landscape_when_display_wider_than_tall(self):
        assert _meta(1920, 1080, rotation=0).orientation == "landscape"

    def test_should_be_portrait_when_display_taller_than_wide(self):
        assert _meta(1080, 1920, rotation=0).orientation == "portrait"

    def test_should_be_square_when_display_dimensions_equal(self):
        assert _meta(1080, 1080, rotation=0).orientation == "square"

    def test_should_be_portrait_when_landscape_file_rotated_90(self):
        """A phone-shot portrait clip: coded 1920x1080, 90 display matrix."""
        assert _meta(1920, 1080, rotation=90).orientation == "portrait"

    def test_should_be_landscape_when_portrait_file_rotated_270(self):
        assert _meta(1080, 1920, rotation=270).orientation == "landscape"

    def test_should_be_square_regardless_of_rotation(self):
        assert _meta(1080, 1080, rotation=90).orientation == "square"


class TestSourceMetaIsFrozen:
    def test_should_reject_mutation(self):
        meta = _meta()
        with pytest.raises(Exception):
            meta.width = 1


class TestSourceMetaHasAudio:
    """has_audio is populated by probe_source from the real
    ffprobe stream list; here it is only the SourceMeta field itself."""

    def test_should_default_to_false_in_the_meta_helper(self):
        assert _meta().has_audio is False


# --- ffprobe JSON fakes (captured from real ffprobe 8.0.1 output) ---

def _stream_json(width, height, r_frame_rate="25/1", duration="10.000000",
                 rotation=None, rotate_tag=None, avg_frame_rate=None,
                 format_duration="10.000000", disposition=None,
                 audio=False):
    """Build an ffprobe JSON payload shaped exactly like the real thing."""
    stream: dict = {"width": width, "height": height,
                    "r_frame_rate": r_frame_rate, "codec_type": "video"}
    if avg_frame_rate is not None:
        stream["avg_frame_rate"] = avg_frame_rate
    if duration is not None:
        stream["duration"] = duration
    if rotation is not None:
        stream["side_data_list"] = [
            {"side_data_type": "Display Matrix", "rotation": rotation},
        ]
    if rotate_tag is not None:
        stream["tags"] = {"rotate": rotate_tag}
    if disposition == "attached_pic":
        stream["disposition"] = {"attached_pic": 1}
    streams = [stream]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    payload: dict = {"programs": [], "streams": streams}
    if format_duration is not None:
        payload["format"] = {"duration": format_duration}
    return payload


def _fake_runner(payload):
    """A fake ffprobe runner: returns a captured payload, spawns nothing."""
    def run(source):
        return payload
    return run


class TestProbeSourceHappyPath:
    def test_should_read_landscape_geometry(self):
        payload = _stream_json(1920, 1080, r_frame_rate="30000/1001",
                               duration="12.512000",
                               format_duration="12.512000")
        meta = probe_source(Path("clip.mp4"), runner=_fake_runner(payload))

        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.rotation == 0
        assert meta.fps == 29.97
        assert meta.duration_s == 12.512
        assert meta.path == Path("clip.mp4")

    def test_should_read_rotation_from_display_matrix(self):
        payload = _stream_json(1920, 1080, rotation=-90)
        meta = probe_source(Path("phone.mp4"), runner=_fake_runner(payload))

        assert meta.rotation == 270
        assert (meta.display_width, meta.display_height) == (1080, 1920)
        assert meta.orientation == "portrait"

    def test_should_fall_back_to_legacy_rotate_tag(self):
        payload = _stream_json(1920, 1080, rotate_tag="90")
        meta = probe_source(Path("old.mov"), runner=_fake_runner(payload))

        assert meta.rotation == 90

    def test_should_prefer_display_matrix_over_rotate_tag(self):
        payload = _stream_json(1920, 1080, rotation=180, rotate_tag="90")
        meta = probe_source(Path("both.mp4"), runner=_fake_runner(payload))

        assert meta.rotation == 180

    def test_should_default_rotation_to_zero_when_absent(self):
        payload = _stream_json(1920, 1080)
        assert "side_data_list" not in payload["streams"][0]
        meta = probe_source(Path("plain.mp4"), runner=_fake_runner(payload))

        assert meta.rotation == 0


class TestProbeSourceFallbacks:
    def test_should_use_format_duration_when_stream_duration_absent(self):
        """Real MKV behaviour: the stream object carries no duration key."""
        payload = _stream_json(1920, 1080, duration=None,
                               format_duration="1.001000")
        assert "duration" not in payload["streams"][0]
        meta = probe_source(Path("clip.mkv"), runner=_fake_runner(payload))

        assert meta.duration_s == 1.001

    def test_should_use_format_duration_when_stream_duration_is_na(self):
        payload = _stream_json(1920, 1080, duration="N/A",
                               format_duration="7.500000")
        meta = probe_source(Path("clip.mkv"), runner=_fake_runner(payload))

        assert meta.duration_s == 7.5

    def test_should_use_avg_frame_rate_when_r_frame_rate_degenerate(self):
        payload = _stream_json(1920, 1080, r_frame_rate="0/0",
                               avg_frame_rate="24000/1001")
        meta = probe_source(Path("odd.mp4"), runner=_fake_runner(payload))

        assert meta.fps == 23.976


class TestProbeSourceHasAudio:
    def test_should_report_has_audio_when_an_audio_stream_is_present(self):
        payload = _stream_json(1920, 1080, audio=True)
        meta = probe_source(Path("clip.mp4"), runner=_fake_runner(payload))

        assert meta.has_audio is True

    def test_should_report_no_audio_when_video_only(self):
        payload = _stream_json(1920, 1080, audio=False)
        meta = probe_source(Path("clip.mp4"), runner=_fake_runner(payload))

        assert meta.has_audio is False


class TestProbeSourceErrors:
    def test_should_raise_when_no_video_stream(self):
        """An audio-only file has a streams list with no video entry."""
        payload = {"programs": [],
                   "streams": [{"codec_type": "audio", "codec_name": "aac"}],
                   "format": {"duration": "1.000000"}}
        with pytest.raises(ProbeError, match="no video stream"):
            probe_source(Path("audio.m4a"), runner=_fake_runner(payload))

    def test_should_raise_when_stream_is_an_attached_picture(self):
        payload = _stream_json(200, 100, r_frame_rate="90000/1",
                               disposition="attached_pic")
        with pytest.raises(ProbeError, match="attached picture"):
            probe_source(Path("cover.m4a"), runner=_fake_runner(payload))

    def test_should_raise_when_frame_rate_is_implausible(self):
        payload = _stream_json(1920, 1080, r_frame_rate="90000/1")
        with pytest.raises(ProbeError, match="implausible"):
            probe_source(Path("odd.mp4"), runner=_fake_runner(payload))

    def test_should_raise_when_dimensions_missing(self):
        payload = {"streams": [{"codec_type": "video",
                                "r_frame_rate": "25/1", "duration": "1.0"}],
                   "format": {"duration": "1.0"}}
        with pytest.raises(ProbeError, match="dimensions"):
            probe_source(Path("weird.mp4"), runner=_fake_runner(payload))

    def test_should_raise_when_dimensions_not_positive(self):
        payload = _stream_json(0, 1080)
        with pytest.raises(ProbeError, match="dimensions"):
            probe_source(Path("zero.mp4"), runner=_fake_runner(payload))

    def test_should_raise_when_frame_rate_undeterminable(self):
        payload = _stream_json(1920, 1080, r_frame_rate="0/0")
        with pytest.raises(ProbeError, match="frame rate"):
            probe_source(Path("norate.mp4"), runner=_fake_runner(payload))

    def test_should_raise_when_duration_undeterminable(self):
        payload = _stream_json(1920, 1080, duration=None,
                               format_duration=None)
        with pytest.raises(ProbeError, match="duration"):
            probe_source(Path("nodur.mp4"), runner=_fake_runner(payload))


class TestRunFfprobeSubprocessBoundary:
    """The one place with no seam left to inject: patch the boundary."""

    def test_should_build_a_single_full_json_ffprobe_command(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = '{"streams": []}'
            mock_run.return_value.stderr = ""
            _run_ffprobe(Path("clip.mp4"))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffprobe"
        assert cmd[-1] == "clip.mp4"
        assert "-show_streams" in cmd
        assert "-show_format" in cmd
        assert "-of" in cmd and "json" in cmd
        # -show_entries section names vary by ffprobe build and hard-fail
        # when unsupported; the full-JSON form has no such dependency.
        assert "-show_entries" not in cmd
        assert mock_run.call_count == 1
        assert mock_run.call_args.kwargs["timeout"] == FFPROBE_TIMEOUT_SEC
        assert mock_run.call_args.kwargs["errors"] == "replace"

    def test_should_raise_with_stderr_when_ffprobe_exits_nonzero(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = "{\n\n}"
            mock_run.return_value.stderr = (
                "nope.mp4: No such file or directory"
            )
            with pytest.raises(ProbeError,
                               match="No such file or directory"):
                _run_ffprobe(Path("nope.mp4"))

    def test_should_raise_when_ffprobe_emits_invalid_json(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "not json at all"
            mock_run.return_value.stderr = ""
            with pytest.raises(ProbeError, match="invalid JSON"):
                _run_ffprobe(Path("clip.mp4"))

    def test_should_raise_probe_error_when_ffprobe_is_missing(self):
        with patch("subprocess.run",
                   side_effect=FileNotFoundError("ffprobe")):
            with pytest.raises(ProbeError, match="could not run ffprobe"):
                _run_ffprobe(Path("clip.mp4"))

    def test_should_raise_probe_error_when_ffprobe_hangs(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=60),
        ):
            with pytest.raises(ProbeError, match="timed out"):
                _run_ffprobe(Path("clip.mp4"))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)
class TestProbeSourceAgainstRealFfprobe:
    """Confirms the parser against the real ffprobe binary, not a fake."""

    @pytest.fixture(autouse=True)
    def _clips(self, tmp_path):
        base = tmp_path / "base.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "testsrc=size=640x360:rate=25:duration=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base)],
            check=True,
        )
        rot = tmp_path / "rot90.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-display_rotation", "90",
             "-i", str(base), "-c", "copy", str(rot)],
            check=True,
        )
        self.base = base
        self.rot = rot

    def test_should_read_coded_dimensions_and_zero_rotation_from_a_real_file(
        self,
    ):
        meta = probe_source(self.base)

        assert (meta.width, meta.height) == (640, 360)
        assert meta.rotation == 0
        assert meta.orientation == "landscape"

    def test_should_read_ninety_degree_rotation_from_a_real_file(self):
        meta = probe_source(self.rot)

        assert (meta.width, meta.height) == (640, 360)
        assert meta.rotation == 90
        assert (meta.display_width, meta.display_height) == (360, 640)
        assert meta.orientation == "portrait"


class TestDecideGate:
    def test_should_reframe_when_landscape(self):
        decision = decide(_meta(1920, 1080))

        assert decision.should_reframe is True
        assert decision.reason == REASON_LANDSCAPE

    def test_should_skip_when_already_vertical(self):
        decision = decide(_meta(1080, 1920))

        assert decision.should_reframe is False
        assert decision.reason == REASON_ALREADY_VERTICAL

    def test_should_skip_when_square(self):
        """Square is its own reason here, not "landscape", so a square
        clip passes through untouched. encode.scene.detect_orientation
        is two-way and reads a square frame as portrait."""
        decision = decide(_meta(1080, 1080))

        assert decision.should_reframe is False
        assert decision.reason == REASON_SQUARE

    def test_should_skip_when_landscape_file_is_rotated_to_vertical(self):
        """Phone-shot portrait: coded 1920x1080 + 90 display matrix."""
        decision = decide(_meta(1920, 1080, rotation=90))

        assert decision.should_reframe is False
        assert decision.reason == REASON_ALREADY_VERTICAL

    def test_should_reframe_when_vertical_file_is_rotated_to_landscape(self):
        decision = decide(_meta(1080, 1920, rotation=270))

        assert decision.should_reframe is True
        assert decision.reason == REASON_LANDSCAPE

    def test_should_skip_when_square_and_rotated(self):
        decision = decide(_meta(1080, 1080, rotation=90))

        assert decision.should_reframe is False
        assert decision.reason == REASON_SQUARE

    def test_should_reframe_when_barely_wider_than_tall(self):
        """No tolerance band: the gate is an exact comparison."""
        decision = decide(_meta(1081, 1080))

        assert decision.should_reframe is True

    def test_should_reframe_four_by_three(self):
        decision = decide(_meta(1440, 1080))

        assert decision.should_reframe is True
        assert decision.reason == REASON_LANDSCAPE

    def test_should_carry_the_probed_meta_through(self):
        meta = _meta(1920, 1080, fps=23.976, duration_s=42.0)
        decision = decide(meta)

        assert decision.meta is meta
        assert decision.meta.fps == 23.976
        assert decision.meta.duration_s == 42.0


class TestDecideFromPath:
    def test_should_probe_when_given_a_path(self):
        payload = _stream_json(1920, 1080)
        decision = decide(Path("clip.mp4"), runner=_fake_runner(payload))

        assert decision.should_reframe is True
        assert decision.meta.path == Path("clip.mp4")

    def test_should_propagate_probe_errors(self):
        payload = {"programs": [], "streams": [], "format": {}}
        with pytest.raises(ProbeError):
            decide(Path("audio.m4a"), runner=_fake_runner(payload))


class TestGateDecisionLogLine:
    def test_should_say_already_vertical_when_skipping_portrait(self):
        line = decide(_meta(1080, 1920)).log_line

        assert "skipped: already vertical" in line
        assert "1080x1920" in line

    def test_should_say_square_when_skipping_square(self):
        line = decide(_meta(1080, 1080)).log_line

        assert "skipped: square" in line
        assert "1080x1080" in line

    def test_should_report_display_dimensions_when_reframing(self):
        line = decide(_meta(1920, 1080)).log_line

        assert "1920x1080" in line
        assert "skipped" not in line

    def test_should_name_rotation_when_present(self):
        line = decide(_meta(1920, 1080, rotation=90)).log_line

        assert "rot 90" in line
        # The displayed geometry, not the coded geometry.
        assert "1080x1920" in line


class TestReframeReportEntry:
    def test_should_describe_a_skipped_vertical_clip(self):
        decision = decide(_meta(1080, 1920, fps=30.0, duration_s=8.5))
        entry = reframe_report_entry(decision, applied=False)

        assert entry["applied"] is False
        assert entry["outcome"] == OUTCOME_SKIPPED
        assert entry["reason"] == REASON_ALREADY_VERTICAL
        assert entry["elapsed_sec"] == 0.0
        assert entry["source"] == {
            "width": 1080, "height": 1920, "rotation": 0,
            "display_width": 1080, "display_height": 1920,
            "orientation": "portrait", "fps": 30.0, "duration_s": 8.5,
        }

    def test_should_describe_a_skipped_square_clip(self):
        entry = reframe_report_entry(decide(_meta(1080, 1080)), applied=False)

        assert entry["applied"] is False
        assert entry["outcome"] == OUTCOME_SKIPPED
        assert entry["reason"] == REASON_SQUARE
        assert entry["source"]["orientation"] == "square"

    def test_should_describe_a_reframed_clip(self):
        decision = decide(_meta(1920, 1080))
        entry = reframe_report_entry(decision, applied=True, elapsed_sec=12.3)

        assert entry["applied"] is True
        assert entry["outcome"] == OUTCOME_REFRAMED
        assert entry["reason"] == REASON_LANDSCAPE
        assert entry["elapsed_sec"] == 12.3

    def test_should_derive_gate_only_when_landscape_and_not_applied(self):
        """A landscape decision recorded before the stage has run at all —
        ftea/reframe/main.py builds this entry before calling
        `run_stages`."""
        decision = decide(_meta(1920, 1080))
        entry = reframe_report_entry(decision, applied=False)

        assert entry["applied"] is False
        assert entry["outcome"] == OUTCOME_GATE_ONLY
        assert entry["reason"] == REASON_LANDSCAPE

    def test_should_accept_explicit_failed_outcome(self):
        """main.py replaces the gate_only entry with outcome=FAILED when the
        stage raises."""
        decision = decide(_meta(1920, 1080))
        entry = reframe_report_entry(
            decision, applied=False, outcome=OUTCOME_FAILED,
        )

        assert entry["outcome"] == OUTCOME_FAILED
        assert entry["applied"] is False

    def test_should_reject_reframed_outcome_when_not_applied(self):
        decision = decide(_meta(1920, 1080))
        with pytest.raises(ValueError, match="outcome"):
            reframe_report_entry(
                decision, applied=False, outcome=OUTCOME_REFRAMED,
            )

    @pytest.mark.parametrize("applied,outcome", [
        (True, OUTCOME_FAILED),
        (True, OUTCOME_SKIPPED),
        (True, OUTCOME_GATE_ONLY),
    ])
    def test_should_reject_an_outcome_that_contradicts_applied(
        self, applied, outcome,
    ):
        """applied is True exactly when the outcome is "reframed"."""
        decision = decide(_meta(1920, 1080))
        with pytest.raises(ValueError, match="outcome"):
            reframe_report_entry(decision, applied=applied, outcome=outcome)

    @pytest.mark.parametrize("size,outcome", [
        ((1920, 1080), OUTCOME_SKIPPED),
        ((1080, 1920), OUTCOME_FAILED),
        ((1080, 1920), OUTCOME_GATE_ONLY),
    ])
    def test_should_reject_an_outcome_that_contradicts_the_gate(
        self, size, outcome,
    ):
        """"skipped" exactly when the gate said not to reframe."""
        decision = decide(_meta(*size))
        with pytest.raises(ValueError, match="gate decision"):
            reframe_report_entry(decision, applied=False, outcome=outcome)

    def test_should_reject_an_unknown_outcome(self):
        decision = decide(_meta(1920, 1080))
        with pytest.raises(ValueError, match="bogus"):
            reframe_report_entry(decision, applied=False, outcome="bogus")

    def test_should_record_rotation_and_display_geometry(self):
        decision = decide(_meta(1920, 1080, rotation=270))
        entry = reframe_report_entry(decision, applied=False)

        assert entry["source"]["rotation"] == 270
        assert entry["source"]["width"] == 1920
        assert entry["source"]["display_width"] == 1080
        assert entry["source"]["display_height"] == 1920

    def test_should_round_elapsed_to_one_decimal(self):
        decision = decide(_meta(1920, 1080))
        entry = reframe_report_entry(decision, applied=True,
                                     elapsed_sec=12.3456)

        assert entry["elapsed_sec"] == 12.3

    def test_should_reject_applied_true_for_a_skipped_clip(self):
        decision = decide(_meta(1080, 1920))
        with pytest.raises(ValueError, match="applied"):
            reframe_report_entry(decision, applied=True)

    def test_should_be_json_serializable(self):
        entry = reframe_report_entry(decide(_meta(1920, 1080)), applied=True)

        assert json.loads(json.dumps(entry)) == entry

    def test_should_not_leak_the_source_path(self):
        """The report already carries `source`; no second copy, no absolute
        Windows path leaking into a git-tracked report file."""
        entry = reframe_report_entry(decide(_meta(1920, 1080)), applied=True)

        assert "path" not in entry
        assert "path" not in entry["source"]


class TestCli:
    def test_should_exit_zero_and_print_json_when_reframing(self, capsys):
        payload = _stream_json(1920, 1080)
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=payload):
            code = main(["clip.mp4"])

        captured = capsys.readouterr()
        assert code == EXIT_REFRAME == 0
        entry = json.loads(captured.out)
        assert entry["reason"] == REASON_LANDSCAPE
        assert entry["applied"] is False
        assert "reframing" in captured.err

    def test_should_exit_two_when_already_vertical(self, capsys):
        payload = _stream_json(1080, 1920)
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=payload):
            code = main(["clip.mp4"])

        captured = capsys.readouterr()
        assert code == EXIT_SKIP == 2
        assert json.loads(captured.out)["reason"] == REASON_ALREADY_VERTICAL
        assert "skipped: already vertical" in captured.err

    def test_should_exit_two_when_square(self, capsys):
        payload = _stream_json(1080, 1080)
        with patch("ftea.reframe.probe._run_ffprobe",
                   return_value=payload):
            code = main(["clip.mp4"])

        assert code == EXIT_SKIP
        assert "skipped: square" in capsys.readouterr().err

    def test_should_exit_one_without_a_traceback_when_probe_fails(
        self, capsys,
    ):
        with patch("ftea.reframe.probe._run_ffprobe",
                   side_effect=ProbeError("ffprobe failed for x: boom")):
            code = main(["missing.mp4"])

        captured = capsys.readouterr()
        assert code == EXIT_ERROR == 1
        assert "probe failed" in captured.err
        assert "Traceback" not in captured.err

    def test_should_exit_one_when_given_no_arguments(self, capsys):
        assert main([]) == EXIT_ERROR
        assert "Usage:" in capsys.readouterr().err

    def test_should_exit_one_when_given_extra_arguments(self, capsys):
        assert main(["a.mp4", "b.mp4"]) == EXIT_ERROR
        assert "Usage:" in capsys.readouterr().err


class TestGateDecisionInvariants:
    def test_should_refuse_a_reframe_decision_for_a_square_clip(self):
        with pytest.raises(ValueError, match="square"):
            GateDecision(True, REASON_SQUARE, _meta(1080, 1080))

    def test_should_refuse_a_skip_decision_for_a_landscape_clip(self):
        with pytest.raises(ValueError, match="landscape"):
            GateDecision(False, REASON_LANDSCAPE, _meta(1920, 1080))

    def test_should_refuse_an_unknown_reason(self):
        with pytest.raises(ValueError, match="sideways"):
            GateDecision(False, "sideways", _meta(1080, 1920))


class TestProbeErrorLineage:
    def test_should_be_catchable_as_a_reframe_error(self):
        """errors.py's contract: every reframe module raises a
        ReframeError, so one except clause tells a domain failure from a
        crash."""
        assert issubclass(ProbeError, ReframeError)
