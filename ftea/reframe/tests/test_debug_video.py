"""Tests for ftea.reframe.debug_video -- the side-by-side harness."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ftea.reframe import debug_video
from ftea.reframe.debug_video import (
    DEBUG_SENDCMD_NAME,
    DEFAULT_BOX_PX,
    FALLBACK_CODEC,
    STILL_FRACTIONS,
    STILL_QUALITY,
    BinaryNotFoundError,
    DebugVideoError,
    box_thickness,
    build_command,
    even_up,
    extract_stills,
    filtergraph,
    flatten_x,
    main,
    render_debug_video,
    sendcmd_lines,
    spec_from_crops,
    x_changes,
)


def _crops(x_by_shot=((100, 100, 300, 300),), **kw):
    flat = [v for shot in x_by_shot for v in shot]
    shots = []
    start = 0
    for values in x_by_shot:
        shots.append({
            "index": len(shots), "start_frame": start,
            "end_frame": start + len(values),
            "has_subject": True, "engagements": 1, "switch_frames": [],
            "x": list(values),
            "subject_x": [v + 304 for v in values],
            "observed": [True] * len(values),
            "inside": [True] * len(values),
        })
        start += len(values)
    doc = {
        "schema": 1,
        "source": {
            "name": "clip.mp4", "width": 1920, "height": 1080,
            "rotation": 0, "display_width": 1920, "display_height": 1080,
            "orientation": "landscape", "fps": 24.0, "duration_s": 10.0,
            "has_audio": True,
        },
        "output": {"name": "vertical.mp4", "width": 608, "height": 1080,
                   "frames": len(flat), "codec": "hevc_nvenc",
                   "container": "mp4", "cq": 10, "preset": "p7"},
        "aspect": {"w": 9, "h": 16},
        "track": {
            "crop_w": 608, "crop_h": 1080, "y": 0,
            "frame_width": 1920, "frame_height": 1080,
            "fps": 24.0, "stride": 3, "total_frames": len(flat),
            "dead_zone": 0.3, "dead_zone_release": 0.25,
            "smoothing_hz": 1.5, "max_pan_frac_per_sec": 0.2,
            "shots": shots,
        },
    }
    doc.update(kw)
    return doc


def _spec(tmp_path, crops=None):
    return spec_from_crops(
        crops or _crops(), tmp_path / "clip.mp4", tmp_path / "vertical.mp4",
        tmp_path / "debug.mp4",
    )


class TestEvenUp:
    @pytest.mark.parametrize(
        "value,want", [(0.0, 0), (1.0, 2), (2.0, 2), (405.33, 406),
                       (406.0, 406), (1279.1, 1280)],
    )
    def test_should_round_to_even(self, value, want):
        assert even_up(value) == want

    @pytest.mark.parametrize(
        "value,want", [(1706.666, 1706), (1694.117, 1694), (405.0, 406)],
    )
    def test_should_match_ffmpeg_scale_minus_two(self, value, want):
        # MEASURED 2026-09-13 against ffmpeg 8.1 with scale=-2:720:
        # 2560x1080 -> 1706 and 1920x816 -> 1694 (so NOT round-up), while
        # 810x1440 -> 406 from an exact 405.0 (so NOT banker's rounding
        # either). The rule ffmpeg actually uses is round-HALF-UP to the
        # nearest even, which is camera.even_round. Rounding up instead
        # planned a 2114px-wide debug video from an ultrawide source
        # while ffmpeg produced 2112.
        assert even_up(value) == want

    def test_should_be_the_camera_rounding_rule(self):
        from ftea.reframe.camera import even_round

        for value in (0.0, 405.0, 405.33, 1694.117, 1706.666, 1279.1):
            assert even_up(value) == even_round(value)


class TestFlattenX:
    def test_should_concatenate_shots_in_order(self):
        crops = _crops(((0, 2), (4, 6), (8, 10)))
        assert flatten_x(crops) == (0, 2, 4, 6, 8, 10)

    def test_should_raise_when_the_length_disagrees_with_total_frames(self):
        crops = _crops(((0, 2),))
        crops["track"]["total_frames"] = 99
        with pytest.raises(DebugVideoError, match="total_frames"):
            flatten_x(crops)

    def test_should_raise_when_there_are_no_shots(self):
        crops = _crops()
        crops["track"]["shots"] = []
        with pytest.raises(DebugVideoError, match="no shots"):
            flatten_x(crops)

    def test_should_raise_when_the_track_key_is_absent(self):
        with pytest.raises(DebugVideoError, match="track"):
            flatten_x({"schema": 1})


class TestXChanges:
    def test_should_emit_the_first_frame_always(self):
        assert x_changes([4, 4, 4])[0] == (0, 4)

    def test_should_emit_one_entry_per_change(self):
        assert x_changes([0, 0, 2, 2, 4]) == [(0, 0), (2, 2), (4, 4)]

    def test_should_reconstruct_the_series_exactly(self):
        series = [0, 0, 2, 2, 2, 8, 8, 4]
        rebuilt, current = [], None
        changes = dict(x_changes(series))
        for i in range(len(series)):
            current = changes.get(i, current)
            rebuilt.append(current)
        assert rebuilt == series

    def test_should_raise_on_an_empty_series(self):
        with pytest.raises(DebugVideoError, match="empty"):
            x_changes([])


class TestBoxThickness:
    @pytest.mark.parametrize(
        "display_h,height,want",
        [(2160, 720, 12), (1080, 720, 6), (720, 720, 4), (816, 720, 5)],
    )
    def test_should_scale_so_the_box_is_four_px_after_the_scale(
        self, display_h, height, want,
    ):
        assert box_thickness(display_h, height) == want

    def test_should_never_go_below_two(self):
        assert box_thickness(100, 720, target_px=DEFAULT_BOX_PX) == 2


class TestSpecFromCrops:
    def test_should_take_the_crop_rectangle_verbatim(self, tmp_path):
        spec = _spec(tmp_path)
        assert (spec.crop_w, spec.crop_h, spec.y) == (608, 1080, 0)

    def test_should_take_the_per_frame_x_from_the_track(self, tmp_path):
        assert _spec(tmp_path).x == (100, 100, 300, 300)

    def test_should_compute_both_pane_widths_even(self, tmp_path):
        spec = _spec(tmp_path)
        assert spec.left_w == 1280
        assert spec.right_w == 406
        assert spec.out_w == 1686
        assert spec.left_w % 2 == 0 and spec.right_w % 2 == 0

    @pytest.mark.parametrize(
        "dw,dh,crop_w,left_w,right_w",
        [
            (1280, 720, 406, 1280, 406),
            (3840, 2160, 1216, 1280, 406),
            (2560, 1080, 608, 1706, 406),
            (1920, 816, 460, 1694, 406),
            (2560, 1440, 810, 1280, 406),
        ],
    )
    def test_should_size_panes_for_every_tier(
        self, tmp_path, dw, dh, crop_w, left_w, right_w,
    ):
        crops = _crops()
        crops["source"]["display_width"] = dw
        crops["source"]["display_height"] = dh
        crops["track"].update(
            crop_w=crop_w, crop_h=dh, frame_width=dw, frame_height=dh)
        spec = _spec(tmp_path, crops)
        assert (spec.left_w, spec.right_w) == (left_w, right_w)

    def test_should_raise_when_fps_is_not_positive(self, tmp_path):
        crops = _crops()
        crops["track"]["fps"] = 0.0
        crops["source"]["fps"] = 0.0
        with pytest.raises(DebugVideoError, match="fps"):
            _spec(tmp_path, crops)

    def test_should_raise_when_the_crop_is_wider_than_the_frame(
        self, tmp_path,
    ):
        crops = _crops()
        crops["track"]["crop_w"] = 5000
        with pytest.raises(DebugVideoError, match="crop_w"):
            _spec(tmp_path, crops)


class TestSendcmdLines:
    def test_should_start_with_an_ascii_comment(self, tmp_path):
        assert sendcmd_lines(_spec(tmp_path))[0].startswith("#")

    def test_should_emit_one_line_per_change(self, tmp_path):
        lines = sendcmd_lines(_spec(tmp_path))
        assert len(lines) == 3          # header + two changes

    def test_should_clamp_the_first_command_to_zero(self, tmp_path):
        assert sendcmd_lines(_spec(tmp_path))[1].startswith("0.000000 ")

    def test_should_use_the_half_frame_early_convention(self, tmp_path):
        # A command fires on the first frame whose PTS >= T, so
        # T = (frame - 0.5)/fps is frame-exact. Frame 2 at 24 fps ->
        # (2 - 0.5)/24 = 0.062500.
        assert sendcmd_lines(_spec(tmp_path))[2].startswith("0.062500 ")

    def test_should_target_drawbox_x(self, tmp_path):
        assert sendcmd_lines(_spec(tmp_path))[2].endswith("drawbox x 300;")

    def test_should_be_pure_ascii(self, tmp_path):
        for line in sendcmd_lines(_spec(tmp_path)):
            line.encode("ascii")


class TestFiltergraph:
    def test_should_reference_the_script_by_bare_name(self, tmp_path):
        graph = filtergraph(_spec(tmp_path))
        assert f"sendcmd=f={DEBUG_SENDCMD_NAME}" in graph
        # The Windows lavfi path trap: no drive letter may appear.
        assert ":\\" not in graph and ":/" not in graph

    def test_should_reset_pts_on_both_inputs(self, tmp_path):
        assert filtergraph(_spec(tmp_path)).count("setpts=PTS-STARTPTS") == 2

    def test_should_draw_the_box_before_scaling(self, tmp_path):
        graph = filtergraph(_spec(tmp_path))
        assert graph.index("drawbox") < graph.index("scale=-2:")

    def test_should_stack_with_shortest(self, tmp_path):
        assert "hstack=inputs=2:shortest=1" in filtergraph(_spec(tmp_path))

    def test_should_carry_no_drawtext(self, tmp_path):
        # drawtext needs fontfile=<path> INSIDE the graph, which is
        # exactly the trap the encoder's probe works around.
        assert "drawtext" not in filtergraph(_spec(tmp_path))


class TestBuildCommand:
    def test_should_take_two_inputs_in_source_then_vertical_order(
        self, tmp_path,
    ):
        cmd = build_command(_spec(tmp_path))
        first = cmd.index("-i")
        second = cmd.index("-i", first + 1)
        assert cmd[first + 1].endswith("clip.mp4")
        assert cmd[second + 1].endswith("vertical.mp4")

    def test_should_drop_audio(self, tmp_path):
        assert "-an" in build_command(_spec(tmp_path))

    def test_should_pass_the_frame_timing_through(self, tmp_path):
        cmd = build_command(_spec(tmp_path))
        assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"

    def test_should_use_the_fallback_codec_when_asked(self, tmp_path):
        cmd = build_command(_spec(tmp_path), codec="libx264")
        assert "libx264" in cmd and "h264_nvenc" not in cmd

    def test_should_map_only_the_stacked_stream(self, tmp_path):
        cmd = build_command(_spec(tmp_path))
        assert cmd[cmd.index("-map") + 1] == "[v]"


class _Runner:
    """A real fake: records argv, writes the output, returns canned JSON."""

    def __init__(self, width=1686, height=720, frames=4, fail_on=None):
        self.calls = []
        self.width, self.height, self.frames = width, height, frames
        self.fail_on = fail_on

    def __call__(self, argv, *, cwd=None):
        self.calls.append((list(argv), cwd))
        if self.fail_on and self.fail_on in " ".join(str(a) for a in argv):
            raise DebugVideoError("ffmpeg failed: boom")
        if "ffprobe" in str(argv[0]):
            return json.dumps({"streams": [{
                "width": self.width, "height": self.height,
                "nb_read_packets": str(self.frames)}]})
        target = Path(argv[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"video")
        return ""


class TestRenderDebugVideo:
    def test_should_write_the_script_beside_the_output(self, tmp_path):
        runner = _Runner()
        result = render_debug_video(_spec(tmp_path), runner=runner)
        assert result.script.name == DEBUG_SENDCMD_NAME
        assert result.script.parent == tmp_path
        assert result.script.is_file()

    def test_should_run_ffmpeg_with_cwd_at_the_output_dir(self, tmp_path):
        # The sendcmd script is referenced by bare NAME, so ffmpeg must
        # be started in the directory holding it, as render.py does.
        runner = _Runner()
        render_debug_video(_spec(tmp_path), runner=runner)
        ffmpeg_calls = [
            c for c in runner.calls if "ffprobe" not in str(c[0][0])]
        assert ffmpeg_calls[0][1] == tmp_path

    def test_should_report_the_probed_geometry(self, tmp_path):
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        assert (result.width, result.height) == (1686, 720)

    def test_should_report_the_command_count(self, tmp_path):
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        assert result.commands == 2

    def test_should_keep_the_exact_argv(self, tmp_path):
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        # Windows resolves this to ...\\ffmpeg.EXE, so compare the stem
        # case-insensitively rather than the whole string.
        assert Path(result.command[0]).stem.lower() == "ffmpeg"
        assert "-filter_complex" in result.command

    def test_should_raise_when_the_geometry_is_not_what_was_planned(
        self, tmp_path,
    ):
        with pytest.raises(DebugVideoError, match="1686x720"):
            render_debug_video(_spec(tmp_path), runner=_Runner(width=1280))

    def test_should_raise_when_the_output_is_empty(self, tmp_path):
        class Empty(_Runner):
            def __call__(self, argv, *, cwd=None):
                if "ffprobe" in str(argv[0]):
                    return super().__call__(argv, cwd=cwd)
                Path(argv[-1]).write_bytes(b"")
                return ""

        with pytest.raises(DebugVideoError, match="empty"):
            render_debug_video(_spec(tmp_path), runner=Empty())

    def test_should_raise_when_the_source_is_absent(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(DebugVideoError, match="clip.mp4"):
            render_debug_video(spec, runner=_Runner(), require_inputs=True)

    def test_should_fall_back_to_libx264_when_nvenc_fails(self, tmp_path):
        runner = _Runner(fail_on="h264_nvenc")
        result = render_debug_video(_spec(tmp_path), runner=runner)
        assert result.codec == FALLBACK_CODEC
        assert any("libx264" in " ".join(str(a) for a in c[0])
                   for c in runner.calls)


class TestFallbackReporting:
    """The NVENC failure is never thrown away: it is logged before the
    libx264 retry and named again if the retry fails too."""

    def test_should_log_the_nvenc_failure_before_retrying(
        self, tmp_path, capsys,
    ):
        render_debug_video(_spec(tmp_path),
                           runner=_Runner(fail_on="h264_nvenc"))

        err = capsys.readouterr().err
        assert "h264_nvenc failed" in err
        assert "boom" in err
        assert "libx264" in err

    def test_should_name_both_failures_when_the_fallback_fails_too(
        self, tmp_path,
    ):
        class BothFail(_Runner):
            def __call__(self, argv, *, cwd=None):
                self.calls.append((list(argv), cwd))
                codec = argv[argv.index("-c:v") + 1]
                raise DebugVideoError(f"ffmpeg failed: {codec} broke")

        with pytest.raises(DebugVideoError) as excinfo:
            render_debug_video(_spec(tmp_path), runner=BothFail())

        message = str(excinfo.value)
        assert "libx264 broke" in message
        assert "h264_nvenc broke" in message

    def test_should_not_retry_when_ffmpeg_itself_is_missing(self, tmp_path):
        """A missing binary is not a codec problem; a retry would only
        report it as a libx264 failure."""
        runner = _Runner()

        def missing(argv, *, cwd=None):
            runner.calls.append((list(argv), cwd))
            raise BinaryNotFoundError(f"{argv[0]} not found on PATH")

        with pytest.raises(BinaryNotFoundError):
            render_debug_video(_spec(tmp_path), runner=missing)

        assert len(runner.calls) == 1

    def test_should_report_a_missing_binary_as_its_own_error(self):
        with pytest.raises(BinaryNotFoundError):
            debug_video._run(["ffmpeg-that-does-not-exist-anywhere"])

    def test_should_keep_a_missing_binary_a_debug_video_error(self):
        assert issubclass(BinaryNotFoundError, DebugVideoError)


class TestFfprobeBesideFfmpeg:
    """The debug video's ffprobe comes from beside its ffmpeg, matched
    the way render.ffprobe_for matches it."""

    @pytest.mark.parametrize("ffmpeg,expected", [
        (r"C:\tools\FFmpeg.exe", r"C:\tools\ffprobe.exe"),
        (r"C:\tools\ffmpeg.EXE", r"C:\tools\ffprobe.EXE"),
        ("/opt/ff/bin/ffmpeg", "/opt/ff/bin/ffprobe"),
        ("ffmpeg", "ffprobe"),
    ])
    def test_should_find_the_ffprobe_beside_any_ffmpeg(self, ffmpeg,
                                                       expected):
        assert debug_video._ffprobe_for(ffmpeg) == expected

    def test_should_fall_back_to_path_for_a_binary_not_named_ffmpeg(self):
        """The debug video never fails a clip, so an odd name falls back
        to ffprobe from PATH instead of raising."""
        assert debug_video._ffprobe_for("/usr/bin/avconv") == "ffprobe"


class TestExtractStills:
    def test_should_write_one_still_per_fraction(self, tmp_path):
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        runner = _Runner()
        stills = extract_stills(result, tmp_path / "stills", runner=runner)
        assert len(stills) == len(STILL_FRACTIONS)
        assert all(p.suffix == ".webp" for p in stills)

    def test_should_write_stills_as_quality_bounded_webp(self, tmp_path):
        """Evidence stills go into git history, so they must stay small.

        PNG was measured at 14.1 MB for one nine-still bundle against
        2.8 MB as WebP, with no loss of legibility that matters here.
        """
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        runner = _Runner()
        stills = extract_stills(result, tmp_path / "stills", runner=runner)

        assert all(p.suffix == ".webp" for p in stills)
        for argv, _ in runner.calls:
            assert "-quality" in argv
            index = argv.index("-quality")
            assert argv[index + 1] == str(STILL_QUALITY)

    def test_should_seek_inside_the_clip(self, tmp_path):
        result = render_debug_video(_spec(tmp_path), runner=_Runner())
        runner = _Runner()
        extract_stills(result, tmp_path / "stills", runner=runner)
        for argv, _ in runner.calls:
            index = argv.index("-ss")
            assert 0.0 <= float(argv[index + 1]) <= result.frames


class TestCli:
    def _crops_file(self, tmp_path):
        path = tmp_path / "crops.json"
        path.write_text(json.dumps(_crops()), encoding="ascii")
        (tmp_path / "clip.mp4").write_bytes(b"x")
        (tmp_path / "vertical.mp4").write_bytes(b"x")
        return path

    def test_should_exit_two_when_the_crops_file_is_missing(self, tmp_path):
        assert main(["--crops", str(tmp_path / "nope.json"),
                     "--source", str(tmp_path / "clip.mp4"),
                     "--out", str(tmp_path / "d.mp4")]) == 2

    def test_should_exit_one_with_no_arguments(self):
        assert main([]) == 1

    def test_should_need_the_source_video(self, tmp_path, capsys):
        """crops.json records only the source's file name, and neither
        `ftea run` nor `ftea reframe` keeps the source beside it."""
        crops = self._crops_file(tmp_path)

        code = main(["--crops", str(crops),
                     "--out", str(tmp_path / "debug.mp4")])

        assert code == 1
        assert "--source" in capsys.readouterr().err

    def test_should_print_one_json_object_on_stdout(self, tmp_path, capsys,
                                                   monkeypatch):
        monkeypatch.setattr(
            "ftea.reframe.debug_video._run", _Runner())
        crops = self._crops_file(tmp_path)
        code = main(["--crops", str(crops),
                     "--out", str(tmp_path / "debug.mp4"),
                     "--source", str(tmp_path / "clip.mp4"),
                     "--vertical", str(tmp_path / "vertical.mp4")])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["width"] == 1686
        assert payload["frames"] == 4


@pytest.mark.gpu
class TestRealFfmpegRender:
    """One real render. Skips when ffmpeg is absent; needs no GPU (it
    falls back to libx264), but carries the marker so `-m gpu` selects
    the slow suite on a GPU machine."""

    def test_should_render_a_moving_box_when_ffmpeg_is_real(self, tmp_path):
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not on PATH")
        source = tmp_path / "clip.mp4"
        vertical = tmp_path / "vertical.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i",
             "testsrc2=size=1920x1080:rate=24:duration=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-vf", "crop=608:1080:100:0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vertical)],
            check=True, capture_output=True,
        )
        crops = _crops(tuple([tuple([100] * 12 + [300] * 12)]))
        spec = spec_from_crops(
            crops, source, vertical, tmp_path / "debug.mp4")
        result = render_debug_video(spec, codec=FALLBACK_CODEC)
        assert (result.width, result.height) == (1686, 720)
        assert abs(result.frames - 24) <= 2


# A schema-2 crops.json (per-shot treatment and title box), built on the
# schema-1 document render.py writes. Kept here because this file is the
# only consumer.
def _track(x, *, crop_w=608, crop_h=1080, fps=24.0, switch_frames=(),
           engagements=1, frame_width=1920, frame_height=1080, stride=3):
    return {
        "crop_w": crop_w, "crop_h": crop_h, "y": 0,
        "frame_width": frame_width, "frame_height": frame_height,
        "fps": fps, "stride": stride, "total_frames": len(x),
        "dead_zone": 0.3, "dead_zone_release": 0.25,
        "smoothing_hz": 1.5, "max_pan_frac_per_sec": 0.2,
        "dead_zone_fraction": 0.987179, "observed_samples": 702,
        "inside_samples": 693, "clamped_samples": 4, "switch_samples": 5,
        "shots": [{
            "index": 0, "start_frame": 0, "end_frame": len(x),
            "has_subject": True, "engagements": engagements,
            "switch_frames": list(switch_frames),
            "x": list(x), "subject_x": [v + crop_w // 2 for v in x],
            "observed": [True] * len(x), "inside": [True] * len(x),
        }],
    }


def _doc(x=(0, 0, 0, 0), *, width=1920, height=1080, **kw):
    track = _track(x, **kw)
    return {
        "schema": 1, "generator": "ftea/reframe/render.py",
        "source": {
            "name": "clip.mp4", "width": width, "height": height,
            "rotation": 0, "display_width": width, "display_height": height,
            "orientation": "landscape", "fps": track["fps"],
            "duration_s": 10.0, "has_audio": True,
        },
        "output": {
            "name": "vertical.mp4", "codec": "hevc_nvenc", "container": "mp4",
            "cq": 10, "preset": "p7", "width": track["crop_w"],
            "height": track["crop_h"], "frames": len(x),
        },
        "aspect": {"w": 9, "h": 16},
        "shots": [{"index": 0, "start_frame": 0, "end_frame": len(x),
                   "frame_count": len(x), "start_sec": 0.0,
                   "end_sec": round(len(x) / track["fps"], 3)}],
        "track": track,
    }


def crops_with(treatments, *, canvas=True, frame_width=1920,
               crop_w=608, title_span=(462.0, 1458.0), xs=None):
    """A schema-2 crops.json with one 120-frame shot per treatment."""
    doc = _doc((656,) * 120)
    doc["schema"] = 2
    if canvas:
        doc["canvas"] = {"width": 1920, "height": 3410, "pad_x": 0,
                         "pad_y": 1165}
    shots = []
    for index, treatment in enumerate(treatments):
        shot = {
            "index": index, "start_frame": index * 120,
            "end_frame": (index + 1) * 120, "has_subject": True,
            "engagements": 0, "switch_frames": [],
            "x": list(xs or [656] * 120),
            "subject_x": [960] * 120,
            "observed": [True] * 120, "inside": [True] * 120,
            "treatment": treatment,
        }
        if treatment == "letterbox":
            shot["title"] = {
                "x1": title_span[0], "y1": 222.0,
                "x2": title_span[1], "y2": 841.0,
                "centre_x": (title_span[0] + title_span[1]) / 2.0,
                "width": title_span[1] - title_span[0],
                "line_count": 5, "frames_seen": 5, "frames_sampled": 5,
            }
        shots.append(shot)
    doc["track"]["shots"] = shots
    doc["track"]["total_frames"] = 120 * len(treatments)
    doc["track"]["crop_w"] = crop_w
    doc["track"]["frame_width"] = frame_width
    return doc


class TestTreatmentAwareOverlay:
    """spec_from_crops takes (crops, source, vertical, out_path)."""

    def _spec(self, doc, tmp_path):
        return spec_from_crops(doc, tmp_path / "src.mp4",
                               tmp_path / "vertical.mp4",
                               tmp_path / "debug.mp4")

    def test_should_draw_the_whole_frame_on_a_letterbox_shot(
        self, tmp_path,
    ):
        spec = self._spec(crops_with(["letterbox"]), tmp_path)

        assert spec.rects[0] == (0, 0, 1920, 1080)

    def test_should_draw_the_crop_window_on_a_normal_shot(
        self, tmp_path,
    ):
        spec = self._spec(crops_with(["crop"], canvas=False), tmp_path)

        assert spec.rects[0] == (656, 0, 608, 1080)

    def test_should_command_all_four_components_at_a_cut(self, tmp_path):
        spec = self._spec(crops_with(["letterbox", "crop"]), tmp_path)

        at_cut = [line for line in sendcmd_lines(spec) if " drawbox w " in line]

        assert at_cut

    def test_should_command_only_x_on_a_clip_with_no_title(
        self, tmp_path,
    ):
        """The common path stays exactly as it was: one `drawbox x`
        per change, no w/h/y noise."""
        spec = self._spec(crops_with(["crop", "crop"], canvas=False),
                          tmp_path)

        lines = [line for line in sendcmd_lines(spec)
                 if not line.startswith("#")]

        assert all(" drawbox x " in line for line in lines)
        assert not any(" drawbox w " in line for line in lines)

    def test_should_read_the_treatment_and_never_recompute_it(
        self, tmp_path,
    ):
        """This tool re-plans nothing. If
        crops.json and the rendered file disagree, the debug video must
        SHOW the disagreement."""
        doc = crops_with(["letterbox"])
        doc["track"]["shots"][0]["treatment"] = "crop"

        spec = self._spec(doc, tmp_path)

        assert spec.rects[0] == (656, 0, 608, 1080)
