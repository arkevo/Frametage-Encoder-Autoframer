"""Tests for ftea.reframe.shots -- per-shot frame boundaries."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import EXIT_ERROR, ProbeError, SourceMeta
from ftea.reframe.shots import (
    DEFAULT_MIN_SHOT_SEC,
    DEFAULT_SCENE_THRESHOLD,
    Shot,
    ShotsError,
    TailProbe,
    _scene_timestamps_checked,
    _validate_tiling,
    assert_decodable,
    assert_frame_count_consistent,
    boundaries_to_shots,
    cut_frames,
    detect_shots,
    drop_short_shots,
    main,
    seconds_to_frame,
    shots_to_json,
    tail_decode_probe,
    total_frame_count,
)


def _meta(fps=25.0, duration_s=6.0, path="clip.mp4"):
    """Build a SourceMeta without touching ffprobe."""
    return SourceMeta(
        path=Path(path), width=1920, height=1080, rotation=0,
        fps=fps, duration_s=duration_s, has_audio=True,
    )


class TestSecondsToFrame:
    def test_should_convert_exact_boundaries(self):
        assert seconds_to_frame(0.0, 25.0) == 0
        assert seconds_to_frame(2.0, 25.0) == 50
        assert seconds_to_frame(6.0, 25.0) == 150

    def test_should_round_rather_than_truncate(self):
        assert seconds_to_frame(1.9999999, 25.0) == 50

    def test_should_handle_ntsc_frame_rates(self):
        assert seconds_to_frame(1.001, 29.97) == 30

    def test_should_raise_when_fps_is_not_positive(self):
        with pytest.raises(ShotsError, match="fps"):
            seconds_to_frame(1.0, 0.0)


class TestShotsErrorReframeErrorLineage:
    def test_should_be_catchable_as_a_reframe_error(self):
        with pytest.raises(ReframeError):
            raise ShotsError("boom")


class TestTotalFrameCount:
    def test_should_derive_frames_from_duration_and_fps(self):
        assert total_frame_count(_meta(25.0, 6.0)) == 150

    def test_should_match_a_real_ntsc_clip(self):
        assert total_frame_count(_meta(29.97, 1.001)) == 30

    def test_should_be_zero_for_a_sub_frame_duration(self):
        assert total_frame_count(_meta(25.0, 0.01)) == 0


class TestCutFrames:
    def test_should_always_include_frame_zero(self):
        assert cut_frames([], 25.0, 150) == [0]

    def test_should_convert_seconds_to_frames(self):
        assert cut_frames([0.0, 2.0, 4.0], 25.0, 150) == [0, 50, 100]

    def test_should_deduplicate_repeated_boundaries(self):
        assert cut_frames([0.0, 2.0, 2.0], 25.0, 150) == [0, 50]

    def test_should_sort_out_of_order_input(self):
        assert cut_frames([4.0, 0.0, 2.0], 25.0, 150) == [0, 50, 100]

    def test_should_drop_a_cut_at_the_end_of_the_clip(self):
        """A boundary at total_frames would create an empty last shot."""
        assert cut_frames([0.0, 6.0], 25.0, 150) == [0]

    def test_should_drop_a_cut_past_the_end_of_the_clip(self):
        assert cut_frames([0.0, 9.0], 25.0, 150) == [0]

    def test_should_collapse_sub_frame_cuts_into_frame_zero(self):
        assert cut_frames([0.0, 0.01], 25.0, 150) == [0]


class TestShotProperties:
    def test_frame_count_should_be_the_half_open_length(self):
        assert Shot(0, 50, 100, 25.0).frame_count == 50

    def test_should_convert_frames_back_to_seconds(self):
        shot = Shot(1, 50, 100, 25.0)

        assert shot.start_sec == 2.0
        assert shot.end_sec == 4.0
        assert shot.duration_s == 2.0

    def test_contains_should_exclude_the_end_frame(self):
        shot = Shot(0, 50, 100, 25.0)

        assert shot.contains(50) is True
        assert shot.contains(99) is True
        assert shot.contains(100) is False
        assert shot.contains(49) is False

    def test_should_be_frozen(self):
        with pytest.raises(Exception):
            Shot(0, 0, 10, 25.0).start_frame = 5


class TestDropShortShots:
    def test_should_keep_boundaries_that_are_far_enough_apart(self):
        assert drop_short_shots([0, 50, 100], 150, 10) == [0, 50, 100]

    def test_should_drop_a_boundary_that_creates_a_short_shot(self):
        """A 2-frame flash at frame 50 must not become its own shot."""
        assert drop_short_shots([0, 50, 52], 150, 10) == [0, 50]

    def test_should_drop_a_short_first_shot(self):
        assert drop_short_shots([0, 3, 100], 150, 10) == [0, 100]

    def test_should_drop_a_short_final_shot(self):
        """encode.scene.merge_short_scenes leaves this one in place."""
        assert drop_short_shots([0, 50, 145], 150, 10) == [0, 50]

    def test_should_collapse_to_one_shot_when_the_clip_is_short(self):
        assert drop_short_shots([0, 5, 8], 12, 20) == [0]

    def test_should_never_add_a_boundary(self):
        """The split_long_scenes trap: no artificial mid-take cuts."""
        result = drop_short_shots([0], 100000, 10)
        assert result == [0]

    def test_should_be_a_no_op_when_min_frames_is_trivial(self):
        assert drop_short_shots([0, 1, 2], 3, 1) == [0, 1, 2]


class TestBoundariesToShots:
    def test_should_produce_half_open_contiguous_ranges(self):
        shots = boundaries_to_shots([0, 50, 100], 150, 25.0)

        assert [(s.start_frame, s.end_frame) for s in shots] == [
            (0, 50), (50, 100), (100, 150),
        ]

    def test_should_number_shots_from_zero(self):
        shots = boundaries_to_shots([0, 50, 100], 150, 25.0)

        assert [s.index for s in shots] == [0, 1, 2]

    def test_should_produce_one_shot_when_there_are_no_cuts(self):
        shots = boundaries_to_shots([0], 150, 25.0)

        assert len(shots) == 1
        assert (shots[0].start_frame, shots[0].end_frame) == (0, 150)


class TestShotSecondsRoundTrip:
    def test_end_sec_of_one_shot_should_equal_start_sec_of_the_next(self):
        shots = boundaries_to_shots([0, 50, 100], 150, 29.97)

        assert shots[0].end_sec == shots[1].start_sec

    def test_should_keep_start_plus_duration_equal_to_end_at_ntsc(self):
        """start_sec + duration_s must equal end_sec exactly at a
        non-integer frame rate, where three independent roundings of
        the same interval could otherwise disagree.

        Compared at the 3-dp precision the three properties are
        published at: binary float addition of two rounded decimals
        (1.668 + 1.669) lands at 3.3369999999999997, so exact `==` here
        would be asserting IEEE arithmetic, not the module's rounding.
        """
        shots = boundaries_to_shots([0, 50, 100], 150, 29.97)

        for shot in shots:
            total = round(shot.start_sec + shot.duration_s, 3)
            assert total == shot.end_sec


def _fake_integrity(frames=2, last_pts=None):
    """A fake decode-integrity probe returning a TailProbe."""
    def run(meta):
        pts = last_pts
        if pts is None:
            pts = meta.duration_s - (1.0 / meta.fps if frames else 0.0)
        return TailProbe(frames=frames, last_pts=pts)
    return run


class TestAssertDecodable:
    def test_should_pass_when_frames_decode(self):
        assert_decodable(_meta(), integrity_fn=_fake_integrity(3))

    def test_should_return_the_tail_probe(self):
        probe = assert_decodable(
            _meta(), integrity_fn=_fake_integrity(3, last_pts=5.92),
        )

        assert probe == TailProbe(frames=3, last_pts=5.92)

    def test_should_raise_when_no_frames_decode(self):
        with pytest.raises(ShotsError, match="decoded no frames"):
            assert_decodable(_meta(), integrity_fn=_fake_integrity(0))


class TestAssertFrameCountConsistent:
    def test_should_pass_on_an_exact_match(self):
        meta = _meta(25.0, 6.0)
        probe = TailProbe(frames=2, last_pts=5.96)

        assert_frame_count_consistent(meta, probe)

    def test_should_pass_on_a_legitimate_one_frame_difference(self):
        meta = _meta(25.0, 6.0)
        probe = TailProbe(frames=2, last_pts=5.92)

        assert_frame_count_consistent(meta, probe)

    def test_should_raise_on_a_vfr_shaped_mismatch(self):
        """r_frame_rate is the nominal MAX rate for a VFR source, so
        duration x fps can overstate the real decoded frame count."""
        meta = _meta(30.0, 6.0)
        probe = TailProbe(frames=2, last_pts=3.97)

        with pytest.raises(ShotsError, match="frame count mismatch"):
            assert_frame_count_consistent(meta, probe)

    def test_should_honour_a_custom_tolerance(self):
        meta = _meta(25.0, 6.0)
        probe = TailProbe(frames=2, last_pts=5.80)

        with pytest.raises(ShotsError):
            assert_frame_count_consistent(meta, probe, tolerance_frames=0)


class TestTailDecodeProbeSubprocessBoundary:
    """The one place with no seam left to inject: patch the boundary."""

    def test_should_build_a_bounded_tail_decode_command_with_copyts(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = "pts_time:5.96\n"
            tail_decode_probe(_meta(25.0, 6.0))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-copyts" in cmd
        assert "-sseof" in cmd and "-1" in cmd
        assert "-frames:v" in cmd
        assert cmd[-2:] == ["null", "-"]
        assert mock_run.call_args[1]["check"] is True
        assert mock_run.call_count == 1

    def test_should_decode_the_whole_tail_window_not_just_one_frame(self):
        """The frame cap must span the tail window, not stop at its
        first frame.

        `-frames:v 1` stops at the FIRST frame after the -sseof seek, so
        last_pts would report the START of the tail window instead of
        the clip's final frame -- measured on Windows ffmpeg 8.1 as
        3.04s on a clip whose real last frame is 3.96s, which makes
        assert_frame_count_consistent reject every healthy clip. The cap
        stays bounded (one window plus slack), never unlimited.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = ""
            tail_decode_probe(_meta(25.0, 6.0))

        cmd = mock_run.call_args[0][0]
        cap = int(cmd[cmd.index("-frames:v") + 1])
        assert cap >= 25
        assert cap <= 30

    def test_should_count_frames_and_report_the_last_pts(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = (
                "[Parsed_showinfo_0] n:0 pts_time:4.96    \n"
                "[Parsed_showinfo_0] n:1 pts_time:5.96 \n"
                "frame=2 fps=0.0\n"
            )
            probe = tail_decode_probe(_meta(25.0, 6.0))

        assert probe == TailProbe(frames=2, last_pts=5.96)

    def test_should_return_zero_frames_when_nothing_decoded(self):
        """The exit-0 trap: a truncated faststart mp4 decodes nothing."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = "Output file is empty\n"
            probe = tail_decode_probe(_meta(25.0, 6.0))

        assert probe == TailProbe(frames=0, last_pts=0.0)

    def test_should_raise_with_ffmpeg_stderr_when_it_exits_nonzero(self):
        error = subprocess.CalledProcessError(183, ["ffmpeg"])
        error.stderr = (
            "[in#0] Error opening input: Invalid data found when "
            "processing input\n"
            "Error opening input files: Invalid data found when "
            "processing input\n"
        )
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(ShotsError, match="Invalid data found"):
                tail_decode_probe(_meta(25.0, 6.0))

    def test_should_raise_when_ffmpeg_is_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ShotsError, match="ffmpeg not found"):
                tail_decode_probe(_meta(25.0, 6.0))


def _fake_scene(timestamps):
    """A fake scene detector: returns canned seconds, spawns nothing."""
    def run(source, threshold):
        return list(timestamps)
    return run


class TestDetectShots:
    def test_should_return_one_shot_when_no_cuts_detected(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0]),
            integrity_fn=_fake_integrity(),
        )

        assert len(shots) == 1
        assert (shots[0].start_frame, shots[0].end_frame) == (0, 150)

    def test_should_split_at_every_detected_cut(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 2.0, 4.0]),
            integrity_fn=_fake_integrity(),
        )

        assert [(s.start_frame, s.end_frame) for s in shots] == [
            (0, 50), (50, 100), (100, 150),
        ]

    def test_should_tile_the_whole_source_with_no_gaps(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 1.3, 2.7, 5.1]),
            integrity_fn=_fake_integrity(),
        )

        assert shots[0].start_frame == 0
        assert shots[-1].end_frame == 150
        assert sum(s.frame_count for s in shots) == 150
        for previous, current in zip(shots, shots[1:]):
            assert previous.end_frame == current.start_frame

    def test_should_append_the_end_bound_scene_py_omits(self):
        """detect_scene_timestamps never appends the total duration."""
        shots = detect_shots(
            _meta(25.0, 6.0), scene_fn=_fake_scene([0.0, 4.0]),
            integrity_fn=_fake_integrity(),
        )

        assert shots[-1].end_frame == 150

    def test_should_not_add_boundaries_at_max_scene_sec(self):
        """split_long_scenes would cut a 40s take into 4; this must not."""
        shots = detect_shots(
            _meta(25.0, 40.0), scene_fn=_fake_scene([0.0]),
            integrity_fn=_fake_integrity(),
        )

        assert len(shots) == 1
        assert shots[0].frame_count == 1000

    def test_should_merge_a_sub_threshold_flash_shot(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 2.0, 2.08]),
            integrity_fn=_fake_integrity(),
        )

        assert [(s.start_frame, s.end_frame) for s in shots] == [
            (0, 50), (50, 150),
        ]

    def test_should_honour_a_custom_min_shot_sec(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 2.0, 2.08]),
            min_shot_sec=0.0, integrity_fn=_fake_integrity(),
        )

        assert len(shots) == 3

    def test_should_pass_the_threshold_through_to_the_detector(self):
        seen = {}

        def spy(source, threshold):
            seen["threshold"] = threshold
            return [0.0]

        detect_shots(_meta(), threshold=0.55, scene_fn=spy,
                     integrity_fn=_fake_integrity())

        assert seen["threshold"] == 0.55

    def test_should_use_the_encoder_default_threshold(self):
        seen = {}

        def spy(source, threshold):
            seen["threshold"] = threshold
            return [0.0]

        detect_shots(_meta(), scene_fn=spy, integrity_fn=_fake_integrity())

        assert seen["threshold"] == DEFAULT_SCENE_THRESHOLD == 0.3

    def test_should_probe_the_source_path_from_the_meta(self):
        seen = {}

        def spy(source, threshold):
            seen["source"] = source
            return [0.0]

        detect_shots(_meta(path="a/b/clip.mp4"), scene_fn=spy,
                     integrity_fn=_fake_integrity())

        assert seen["source"] == Path("a/b/clip.mp4")


class TestDetectShotsFailsLoudly:
    def test_should_raise_when_ffmpeg_decodes_nothing(self):
        """The headline trap: a silent ffmpeg failure yields [0.0], which
        is indistinguishable from a legitimate single-shot clip."""
        with pytest.raises(ShotsError, match="truncated"):
            detect_shots(
                _meta(), scene_fn=_fake_scene([0.0]),
                integrity_fn=_fake_integrity(frames=0),
            )

    def test_should_check_decodability_before_scene_detection(self):
        """Corrupt input must cost one GOP, not a full decode pass."""
        called = []

        def spy(source, threshold):
            called.append(source)
            return [0.0]

        with pytest.raises(ShotsError):
            detect_shots(_meta(), scene_fn=spy,
                         integrity_fn=_fake_integrity(frames=0))

        assert called == []

    def test_should_propagate_an_integrity_probe_failure(self):
        def boom(source):
            raise ShotsError("ffmpeg cannot decode clip.mp4: Invalid data")

        with pytest.raises(ShotsError, match="Invalid data"):
            detect_shots(_meta(), scene_fn=_fake_scene([0.0]),
                         integrity_fn=boom)

    def test_should_raise_when_the_source_has_no_frames(self):
        with pytest.raises(ShotsError, match="no frames"):
            detect_shots(
                _meta(25.0, 0.01), scene_fn=_fake_scene([0.0]),
                integrity_fn=_fake_integrity(),
            )

    def test_should_not_run_the_integrity_probe_for_an_empty_source(self):
        called = []

        with pytest.raises(ShotsError):
            detect_shots(
                _meta(25.0, 0.0),
                scene_fn=_fake_scene([0.0]),
                integrity_fn=lambda source: called.append(source) or 2,
            )

        assert called == []


class TestSceneTimestampsChecked:
    """The scene-detection RUN is checked too, not just the file."""

    def test_should_build_the_same_command_as_gpu_scene(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = ""
            _scene_timestamps_checked(Path("clip.mp4"), 0.3)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert cmd[-1] == "-" and cmd[-2] == "null"
        assert "-vsync" in cmd and "vfr" in cmd
        assert any("gt(scene,0.3)" in part for part in cmd)
        assert mock_run.call_args[1]["check"] is True

    def test_should_raise_when_the_scene_pass_exits_nonzero(self):
        error = subprocess.CalledProcessError(183, ["ffmpeg"])
        error.stderr = "Invalid data found when processing input\n"
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(
                ShotsError, match="scene detection failed",
            ):
                _scene_timestamps_checked(Path("bad.mp4"), 0.3)


class TestSceneTimestampParsing:
    def test_should_read_every_cut_from_showinfo_lines(self):
        """The parser is a copy, not an import, of the encoder's, so its
        reading of ffmpeg's stderr needs its own test."""
        stderr = (
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':\n"
            "[Parsed_showinfo_1 @ 0x1] config in time_base: 1/24000\n"
            "[Parsed_showinfo_1 @ 0x1] n:   0 pts: 100900 "
            "pts_time:4.20417 duration:1001 checksum:ABC\n"
            "frame=  2 fps=0.0 q=-0.0 size=N/A time=00:00:09.00\n"
            "[Parsed_showinfo_1 @ 0x1] n:   1 pts: 216216 "
            "pts_time:9.009   duration:1001 checksum:DEF\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = stderr
            got = _scene_timestamps_checked(Path("clip.mp4"), 0.3)

        assert got == [0.0, 4.20417, 9.009]

    def test_should_report_a_single_take_as_one_start(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stderr = "no cuts here\n"
            assert _scene_timestamps_checked(Path("c.mp4"), 0.3) == [0.0]

    def test_should_raise_when_ffmpeg_is_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ShotsError, match="ffmpeg not found"):
                _scene_timestamps_checked(Path("clip.mp4"), 0.3)


class TestValidateTiling:
    """The last check before detection, subject and camera rely on the
    shots covering every frame exactly once."""

    def _shot(self, index, start, end):
        return Shot(index=index, start_frame=start, end_frame=end,
                    fps=24.0)

    def test_should_accept_shots_that_tile_the_clip(self):
        _validate_tiling([self._shot(0, 0, 10), self._shot(1, 10, 24)], 24)

    @pytest.mark.parametrize("spans,total,needle", [
        ((), 24, "produced no shots"),
        (((2, 24),), 24, "starts at 2, not 0"),
        (((0, 20),), 24, "ends at 20, not 24"),
        (((0, 10), (12, 24)), 24, "gap between shot 0 and 1"),
        (((0, 10), (10, 10), (10, 24)), 24, "shot 1 is empty"),
    ])
    def test_should_refuse_shots_that_do_not_tile(self, spans, total,
                                                  needle):
        shots = [self._shot(index, start, end)
                 for index, (start, end) in enumerate(spans)]

        with pytest.raises(ShotsError, match=needle):
            _validate_tiling(shots, total)


class TestDetectShotsUsesTheRealDetectorByDefault:
    def test_should_use_the_real_scene_detector_by_default(self):
        meta = _meta()
        with patch(
            "ftea.reframe.shots._scene_timestamps_checked",
            return_value=[0.0],
        ) as mock_detector:
            detect_shots(meta, integrity_fn=_fake_integrity())

        mock_detector.assert_called_once_with(
            meta.path, DEFAULT_SCENE_THRESHOLD,
        )


class TestDetectShotsOutOfRangeCuts:
    """A large out-of-range cut set signals a wrong duration."""

    def test_should_tolerate_a_single_cut_at_the_clip_end(self):
        shots = detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 6.0]),
            integrity_fn=_fake_integrity(),
        )

        assert len(shots) == 1

    def test_should_raise_when_many_cuts_fall_past_the_end(self):
        with pytest.raises(ShotsError, match="detected cuts"):
            detect_shots(
                _meta(), scene_fn=_fake_scene([0.0, 6.0, 7.0, 8.0]),
                integrity_fn=_fake_integrity(),
            )


class TestDetectShotsMergeReporting:
    """The short-shot merge announces itself on stderr."""

    def test_should_report_when_short_shots_are_merged(self, capsys):
        detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 2.0, 2.08]),
            integrity_fn=_fake_integrity(),
        )

        assert "merged 1 short shot" in capsys.readouterr().err

    def test_should_be_silent_when_nothing_merges(self, capsys):
        detect_shots(
            _meta(), scene_fn=_fake_scene([0.0, 2.0, 4.0]),
            integrity_fn=_fake_integrity(),
        )

        assert capsys.readouterr().err == ""


class TestShotsToJson:
    def test_should_serialize_every_shot(self):
        shots = boundaries_to_shots([0, 50], 150, 25.0)
        payload = shots_to_json(shots)

        assert payload == [
            {"index": 0, "start_frame": 0, "end_frame": 50,
             "frame_count": 50, "start_sec": 0.0, "end_sec": 2.0},
            {"index": 1, "start_frame": 50, "end_frame": 150,
             "frame_count": 100, "start_sec": 2.0, "end_sec": 6.0},
        ]

    def test_should_be_json_serializable(self):
        payload = shots_to_json(boundaries_to_shots([0], 150, 25.0))

        assert json.loads(json.dumps(payload)) == payload


class TestCli:
    def test_should_print_shots_and_exit_zero(self, capsys):
        meta = _meta()
        with patch("ftea.reframe.shots.probe_source",
                   return_value=meta), \
             patch("ftea.reframe.shots._scene_timestamps_checked",
                   return_value=[0.0, 2.0]), \
             patch("ftea.reframe.shots.tail_decode_probe",
                   return_value=TailProbe(frames=2, last_pts=5.96)):
            code = main(["clip.mp4"])

        captured = capsys.readouterr()
        assert code == 0
        assert len(json.loads(captured.out)) == 2
        assert "2 shot(s)" in captured.err

    def test_should_exit_with_the_error_code_when_probe_fails(self, capsys):
        with patch("ftea.reframe.shots.probe_source",
                   side_effect=ProbeError("no video stream in x")):
            code = main(["x.m4a"])

        captured = capsys.readouterr()
        assert code == EXIT_ERROR
        assert "shot detection failed" in captured.err
        assert "Traceback" not in captured.err

    def test_should_exit_with_the_error_code_when_detection_fails(
        self, capsys,
    ):
        with patch("ftea.reframe.shots.probe_source",
                   return_value=_meta()), \
             patch("ftea.reframe.shots.tail_decode_probe",
                   return_value=TailProbe(frames=0, last_pts=0.0)):
            code = main(["broken.mp4"])

        assert code == EXIT_ERROR
        assert "decoded no frames" in capsys.readouterr().err

    def test_should_fail_when_given_no_arguments(self, capsys):
        assert main([]) == EXIT_ERROR
        assert "Usage:" in capsys.readouterr().err

    def test_should_fail_when_given_extra_arguments(self, capsys):
        assert main(["a.mp4", "b.mp4"]) == EXIT_ERROR
        assert "Usage:" in capsys.readouterr().err

    def test_error_code_should_be_the_shared_reframe_failure_code(self):
        """The CLIs share probe's exit table, so they cannot drift."""
        assert EXIT_ERROR != 0


class TestDefaults:
    def test_min_shot_default_should_be_below_any_real_take(self):
        assert DEFAULT_MIN_SHOT_SEC == 0.4


class TestSceneCommandMatchesTheEncoder:
    """shots.py runs the encoder's scene-detection command without
    importing the encoder, so the stage never loads the encoder package.
    This keeps the two command lines identical."""

    def test_should_run_the_same_scene_command_as_the_encoder(
        self, monkeypatch, tmp_path,
    ):
        scene = pytest.importorskip("ftea.encode.scene")
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append([str(part) for part in cmd])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        source = tmp_path / "clip.mp4"

        scene.detect_scene_timestamps(source, 0.3)
        _scene_timestamps_checked(source, 0.3)

        assert len(seen) == 2
        assert seen[0] == seen[1]
