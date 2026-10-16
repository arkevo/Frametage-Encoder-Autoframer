"""Unit tests for ftea/reframe/camera.py."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from ftea.reframe.camera import (
    DEFAULT_ASPECT_H,
    DEFAULT_ASPECT_W,
    DEFAULT_DEAD_ZONE,
    DEFAULT_DEAD_ZONE_RELEASE,
    DEFAULT_MAX_PAN_FRAC_PER_SEC,
    DEFAULT_SMOOTHING_HZ,
    SC2_THRESHOLD,
    TREATMENT_CROP,
    TREATMENT_LETTERBOX,
    CameraError,
    Canvas,
    CropTrack,
    CropWindow,
    DeadZoneReport,
    ShotTrajectory,
    camera_summary,
    canvas_for,
    crop_changes,
    crop_rects,
    crop_track_to_json,
    crop_window,
    dead_zone_half,
    dead_zone_report,
    even_floor,
    even_round,
    expand_targets,
    plan_camera,
    plan_shot,
    rect_changes,
    round_up_even,
    smooth_step,
    title_treatment,
    title_x,
)
from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import SourceMeta


def meta_for(width, height, *, fps=24.0, duration_s=5.0, rotation=0):
    """A real SourceMeta -- probe.py's frozen dataclass, not a fake."""
    return SourceMeta(
        path=Path("clip.mp4"), width=width, height=height,
        rotation=rotation, fps=fps, duration_s=duration_s, has_audio=True,
    )


def sample_at(frame, cx, *, observed=True, kind="person", score=0.5,
              track_id=0, cy=0.5):
    """One real SubjectSample. Unobserved samples are zero-area."""
    from ftea.reframe.subject import SubjectSample

    half = 0.05 if observed else 0.0
    return SubjectSample(
        frame=frame, cx=cx, cy=cy,
        x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half,
        observed=observed, kind=kind if observed else "held",
        score=score if observed else 0.0,
        track_id=track_id if observed else -1,
    )


def shot_subject(points, *, start=0, stride=2, fps=24.0, shot_index=0,
                 frame_count=None, has_subject=True, switch_frames=(),
                 observed=None):
    """A real ShotSubject built from a list of normalised cx values."""
    from ftea.reframe.subject import ShotSubject

    flags = [True] * len(points) if observed is None else list(observed)
    frames = tuple(start + index * stride for index in range(len(points)))
    span = frame_count if frame_count is not None else len(points) * stride
    samples = tuple(
        sample_at(frame, point, observed=flag)
        for frame, point, flag in zip(frames, points, flags)
    )
    return ShotSubject(
        shot_index=shot_index, start_frame=start, end_frame=start + span,
        fps=fps, stride=stride, sampled_frames=frames, samples=samples,
        has_subject=has_subject, track_count=1 if has_subject else 0,
        switch_frames=tuple(switch_frames),
    )


def centre_shot(frame_count, *, start=0, stride=2, fps=24.0, shot_index=0):
    """A has_subject=False shot: every sample is centre and unobserved."""
    count = (frame_count + stride - 1) // stride
    return shot_subject(
        [0.5] * count, start=start, stride=stride, fps=fps,
        shot_index=shot_index, frame_count=frame_count,
        has_subject=False, observed=[False] * count,
    )


def walk(count, first, last):
    """A linear ramp of normalised cx values."""
    if count == 1:
        return [first]
    return [first + (last - first) * (i / (count - 1))
            for i in range(count)]


# name, width, height, crop_w, crop_h, max_x, centre_x
TIERS = [
    ("720p", 1280, 720, 406, 720, 874, 438),
    ("1080p", 1920, 1080, 608, 1080, 1312, 656),
    ("1440p", 2560, 1440, 810, 1440, 1750, 876),
    ("4K", 3840, 2160, 1216, 2160, 2624, 1312),
    ("ultrawide", 2560, 1080, 608, 1080, 1952, 976),
    ("letterboxed", 1920, 816, 460, 816, 1460, 730),
]


class TestEvenRounding:
    def test_should_round_up_to_the_next_even_when_odd(self):
        assert round_up_even(405) == 406
        assert round_up_even(403) == 404
        assert round_up_even(1215) == 1216
        assert round_up_even(459) == 460

    def test_should_leave_it_alone_when_already_even(self):
        assert round_up_even(404) == 404
        assert round_up_even(810.0) == 810

    def test_should_round_a_fraction_up_when_not_an_integer(self):
        assert round_up_even(607.5) == 608
        assert round_up_even(0.1) == 2

    def test_should_floor_to_even_when_flooring(self):
        assert even_floor(721) == 720
        assert even_floor(720) == 720
        assert even_floor(1.9) == 0
        assert even_floor(3.0) == 2

    def test_should_round_half_up_and_never_use_bankers_rounding(self):
        assert even_round(437) == 438
        assert even_round(875) == 876
        assert even_round(656) == 656
        assert even_round(404) == 404
        assert even_round(403) == 404
        assert even_round(1311.5) == 1312
        assert even_round(0.9) == 0
        assert even_round(1.0) == 2

    def test_should_be_even_for_every_input_when_rounding(self):
        for value in (0, 0.4, 1, 1.5, 7, 405, 1215.7):
            assert round_up_even(value) % 2 == 0
            assert even_floor(value) % 2 == 0
            assert even_round(value) % 2 == 0


class TestCropWindow:
    @pytest.mark.parametrize(
        "name,width,height,crop_w,crop_h,max_x,centre_x", TIERS,
    )
    def test_should_compute_the_crop_window_for_every_tier(
        self, name, width, height, crop_w, crop_h, max_x, centre_x,
    ):
        window = crop_window(meta_for(width, height))
        assert window.frame_width == width
        assert window.frame_height == height
        assert window.crop_w == crop_w
        assert window.crop_h == crop_h
        assert window.y == 0
        assert window.max_x == max_x
        assert window.centre_x == centre_x

    def test_should_produce_only_even_dimensions_when_any_tier(self):
        for _n, width, height, _cw, _ch, _mx, _cx in TIERS:
            window = crop_window(meta_for(width, height))
            assert window.crop_w % 2 == 0
            assert window.crop_h % 2 == 0
            assert window.max_x % 2 == 0
            assert window.centre_x % 2 == 0

    def test_should_floor_odd_source_dimensions_to_even(self):
        window = crop_window(meta_for(1281, 721))
        assert (window.frame_width, window.frame_height) == (1280, 720)
        assert window.crop_w == 406
        window = crop_window(meta_for(1920, 817))
        assert (window.frame_width, window.frame_height) == (1920, 816)
        assert window.crop_w == 460

    def test_should_allow_a_crop_exactly_as_wide_as_the_frame(self):
        window = crop_window(meta_for(1920, 1080), aspect_w=16, aspect_h=9)
        assert window.crop_w == 1920
        assert window.max_x == 0
        window = crop_window(meta_for(900, 1600))
        assert window.crop_w == 900
        assert window.max_x == 0

    def test_should_refuse_an_aspect_wider_than_the_source(self):
        """16:9 out of a 4:3 source would need a crop wider than the
        frame. Clamping it rendered a 4:3 file with no warning."""
        with pytest.raises(CameraError) as excinfo:
            crop_window(meta_for(1440, 1080), aspect_w=16, aspect_h=9)

        message = str(excinfo.value)
        assert "REFRAME_ASPECT_W" in message
        assert "1920" in message
        assert "1440" in message

    def test_should_use_display_geometry_on_a_rotated_source(self):
        window = crop_window(meta_for(1920, 1080, rotation=90))
        assert (window.frame_width, window.frame_height) == (1080, 1920)
        assert window.crop_w == 1080          # 1920*9/16 = 1080, clamped

    def test_should_expose_the_normalised_half_crop(self):
        window = crop_window(meta_for(1920, 1080))
        assert window.half_crop_norm == pytest.approx(608 / 3840.0)

    def test_should_raise_when_the_frame_has_no_area(self):
        with pytest.raises(CameraError) as excinfo:
            crop_window(meta_for(0, 1080))
        assert "0" in str(excinfo.value)
        with pytest.raises(CameraError):
            crop_window(meta_for(1920, 1))

    def test_should_raise_on_a_non_positive_aspect(self):
        with pytest.raises(CameraError) as excinfo:
            crop_window(meta_for(1920, 1080), aspect_w=0)
        assert "REFRAME_ASPECT_W" in str(excinfo.value)
        with pytest.raises(CameraError) as excinfo:
            crop_window(meta_for(1920, 1080), aspect_h=-1)
        assert "REFRAME_ASPECT_H" in str(excinfo.value)


class TestCameraError:
    def test_should_be_a_reframe_error_when_raised(self):
        assert issubclass(CameraError, ReframeError)


class TestDefaults:
    def test_should_publish_the_pinned_defaults(self):
        assert DEFAULT_ASPECT_W == 9
        assert DEFAULT_ASPECT_H == 16
        assert DEFAULT_DEAD_ZONE == 0.9
        assert DEFAULT_SMOOTHING_HZ == 0.6
        assert DEFAULT_MAX_PAN_FRAC_PER_SEC == 0.20
        assert DEFAULT_DEAD_ZONE_RELEASE == 0.6
        assert SC2_THRESHOLD == 0.95


class TestSmoothStep:
    def test_should_not_move_when_already_at_the_target(self):
        position, velocity = smooth_step(0.5, 0.0, 0.5, 1 / 24.0, 1.5)
        assert position == pytest.approx(0.5)
        assert velocity == pytest.approx(0.0)

    def test_should_move_toward_the_target_when_below_it(self):
        position, velocity = smooth_step(0.0, 0.0, 1.0, 1 / 24.0, 1.5)
        assert 0.0 < position < 1.0
        assert velocity > 0.0

    def test_should_move_toward_the_target_when_above_it(self):
        position, velocity = smooth_step(1.0, 0.0, 0.0, 1 / 24.0, 1.5)
        assert 0.0 < position < 1.0
        assert velocity < 0.0

    @pytest.mark.parametrize(
        "fps,frames_to_95", [(23.976, 13), (24.0, 13), (25.0, 14),
                             (29.97, 16), (30.0, 16), (60.0, 32)],
    )
    def test_should_settle_a_unit_step_in_about_half_a_second(
        self, fps, frames_to_95,
    ):
        position, velocity, reached = 0.0, 0.0, None
        for index in range(int(fps * 3)):
            position, velocity = smooth_step(
                position, velocity, 1.0, 1 / fps, 1.5)
            if reached is None and position >= 0.95:
                reached = index
        assert reached == frames_to_95

    @pytest.mark.parametrize("hz", [0.5, 1.0, 1.5, 3.0, 10.0])
    def test_should_never_overshoot_when_critically_damped(self, hz):
        position, velocity, history = 0.0, 0.0, []
        for _ in range(240):
            position, velocity = smooth_step(
                position, velocity, 1.0, 1 / 24.0, hz)
            history.append(position)
        assert max(history) <= 1.0
        assert all(b >= a for a, b in zip(history, history[1:]))

    def test_should_stay_stable_when_the_step_is_large(self):
        """No dt*omega stability bound: implicit Euler, not explicit."""
        position, velocity = 0.0, 0.0
        for _ in range(50):
            position, velocity = smooth_step(
                position, velocity, 1.0, 1 / 5.0, 50.0)
        assert position == pytest.approx(1.0, abs=1e-6)
        assert abs(velocity) < 1e-3

    def test_should_raise_on_a_non_positive_smoothing_hz(self):
        with pytest.raises(CameraError) as excinfo:
            smooth_step(0.0, 0.0, 1.0, 1 / 24.0, 0.0)
        assert "REFRAME_SMOOTHING_HZ" in str(excinfo.value)

    def test_should_raise_on_a_non_positive_dt(self):
        with pytest.raises(CameraError):
            smooth_step(0.0, 0.0, 1.0, 0.0, 1.5)


class TestDeadZoneHalf:
    def test_should_be_half_the_dead_zone_share_of_the_crop(self):
        window = crop_window(meta_for(1920, 1080))
        assert dead_zone_half(window, 0.3) == pytest.approx(0.0475)
        assert dead_zone_half(window, 0.3) * 1920 == pytest.approx(91.2)

    def test_should_scale_with_the_crop_not_the_frame(self):
        wide = crop_window(meta_for(2560, 1080))     # crop 608 of 2560
        tall = crop_window(meta_for(1920, 1080))     # crop 608 of 1920
        assert dead_zone_half(wide, 0.3) < dead_zone_half(tall, 0.3)
        assert dead_zone_half(wide, 0.3) * 2560 == pytest.approx(91.2)

    def test_should_raise_on_a_dead_zone_outside_the_unit_interval(self):
        window = crop_window(meta_for(1920, 1080))
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(CameraError) as excinfo:
                dead_zone_half(window, bad)
            assert "REFRAME_DEAD_ZONE" in str(excinfo.value)


class TestExpandTargets:
    def test_should_emit_one_entry_per_frame_not_per_sample(self):
        subject = shot_subject([0.3, 0.4, 0.5], stride=2)
        targets, observed = expand_targets(subject)
        assert len(targets) == 6
        assert len(observed) == 6

    def test_should_hold_each_sample_until_the_next_one(self):
        subject = shot_subject([0.3, 0.7], stride=3)
        targets, _observed = expand_targets(subject)
        assert targets == (0.3, 0.3, 0.3, 0.7, 0.7, 0.7)

    def test_should_hold_the_last_sample_to_the_end_of_the_shot(self):
        subject = shot_subject([0.3, 0.7], stride=2, frame_count=7)
        targets, _observed = expand_targets(subject)
        assert targets == (0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7)

    def test_should_backfill_before_the_first_sample(self):
        base = shot_subject([0.8], start=0, stride=2, frame_count=4)
        subject = dataclasses.replace(
            base, sampled_frames=(2,), samples=(sample_at(2, 0.8),))
        targets, observed = expand_targets(subject)
        assert targets == (0.8, 0.8, 0.8, 0.8)
        assert observed == (False, False, True, False)

    def test_should_mark_only_sampled_observed_frames_as_observed(self):
        subject = shot_subject([0.3, 0.4, 0.5], stride=2,
                               observed=[True, False, True])
        _targets, observed = expand_targets(subject)
        assert observed == (True, False, False, False, True, False)

    def test_should_return_the_centre_for_a_shot_with_no_subject(self):
        targets, observed = expand_targets(centre_shot(6))
        assert targets == (0.5,) * 6
        assert observed == (False,) * 6

    def test_should_handle_a_single_frame_shot(self):
        subject = shot_subject([0.42], stride=1, frame_count=1)
        targets, observed = expand_targets(subject)
        assert targets == (0.42,)
        assert observed == (True,)

    def test_should_not_mutate_its_argument(self):
        subject = shot_subject(walk(4, 0.2, 0.8))
        before = (subject.sampled_frames, subject.samples)
        expand_targets(subject)
        assert (subject.sampled_frames, subject.samples) == before


class TestShotSubjectValidation:
    def test_should_raise_when_samples_and_sampled_frames_disagree(self):
        broken = dataclasses.replace(
            shot_subject([0.3, 0.4]), sampled_frames=(0,))
        with pytest.raises(CameraError) as excinfo:
            expand_targets(broken)
        assert "sampled_frames" in str(excinfo.value)

    def test_should_raise_when_sampled_frames_go_backwards(self):
        broken = dataclasses.replace(
            shot_subject([0.3, 0.4]), sampled_frames=(2, 0),
            samples=(sample_at(2, 0.3), sample_at(0, 0.4)))
        with pytest.raises(CameraError) as excinfo:
            expand_targets(broken)
        assert "increasing" in str(excinfo.value)

    def test_should_raise_when_a_sample_falls_outside_its_shot(self):
        broken = dataclasses.replace(
            shot_subject([0.3], stride=2, frame_count=2),
            sampled_frames=(9,), samples=(sample_at(9, 0.3),))
        with pytest.raises(CameraError) as excinfo:
            expand_targets(broken)
        assert "9" in str(excinfo.value)

    def test_should_raise_when_a_sample_frame_does_not_match(self):
        broken = dataclasses.replace(
            shot_subject([0.3, 0.4]),
            samples=(sample_at(0, 0.3), sample_at(3, 0.4)))
        with pytest.raises(CameraError):
            expand_targets(broken)

    def test_should_raise_on_a_normalised_coordinate_out_of_range(self):
        with pytest.raises(CameraError) as excinfo:
            expand_targets(shot_subject([1.4]))
        assert "1.4" in str(excinfo.value)

    def test_should_raise_when_there_are_no_samples(self):
        broken = dataclasses.replace(
            shot_subject([0.3]), sampled_frames=(), samples=())
        with pytest.raises(CameraError):
            expand_targets(broken)

    def test_should_raise_on_an_empty_frame_range(self):
        with pytest.raises(CameraError):
            expand_targets(shot_subject([0.3], frame_count=0))

    def test_should_raise_on_a_non_positive_fps(self):
        with pytest.raises(CameraError) as excinfo:
            expand_targets(shot_subject([0.3], fps=0.0))
        assert "fps" in str(excinfo.value)

    def test_should_raise_on_a_stride_below_one(self):
        broken = dataclasses.replace(shot_subject([0.3]), stride=0)
        with pytest.raises(CameraError):
            expand_targets(broken)


def _reversals(values):
    """Count sign changes in the per-frame delta -- a stutter proxy."""
    deltas = [b - a for a, b in zip(values, values[1:])]
    signs = [1 if delta > 0 else -1 for delta in deltas if delta != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


class TestPlanShotShape:
    def test_should_emit_one_entry_per_frame_in_every_tuple(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(60, 0.2, 0.8)), window)
        assert shot.frame_count == 120
        assert len(shot.x) == 120
        assert len(shot.subject_x) == 120
        assert len(shot.observed) == 120
        assert len(shot.inside) == 120
        assert len(shot.clamped) == 120

    def test_should_copy_the_shot_identity_from_the_subject(self):
        window = crop_window(meta_for(1920, 1080))
        subject = shot_subject(walk(10, 0.3, 0.6), start=40, shot_index=3,
                               switch_frames=(52,))
        shot = plan_shot(subject, window)
        assert shot.shot_index == 3
        assert shot.start_frame == 40
        assert shot.end_frame == 60
        assert shot.fps == 24.0
        assert shot.has_subject is True
        assert shot.switch_frames == (52,)

    def test_should_produce_only_even_x_inside_the_pan_range(self):
        for _n, width, height, _cw, _ch, max_x, _cx in TIERS:
            window = crop_window(meta_for(width, height))
            shot = plan_shot(shot_subject(walk(60, 0.1, 0.9)), window)
            assert all(value % 2 == 0 for value in shot.x)
            assert all(0 <= value <= max_x for value in shot.x)

    def test_should_resolve_an_absolute_frame_with_x_at(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(10, 0.3, 0.6), start=40), window)
        assert shot.x_at(40) == shot.x[0]
        assert shot.x_at(59) == shot.x[-1]
        with pytest.raises(CameraError):
            shot.x_at(60)


class TestDeadZoneHolds:
    def test_should_never_move_for_a_static_centred_subject(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.5] * 60), window)
        assert set(shot.x) == {656}
        assert shot.engagements == 0
        assert all(shot.inside)

    def test_should_never_move_for_a_static_off_centre_subject(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.7] * 60), window)
        assert set(shot.x) == {1040}
        assert shot.engagements == 0

    def test_should_not_move_when_the_subject_jitters_inside_the_zone(self):
        window = crop_window(meta_for(1920, 1080))
        points = [0.5 + 0.02 * math.sin(2 * math.pi * i / 6.0)
                  for i in range(60)]
        shot = plan_shot(shot_subject(points), window)
        assert len(set(shot.x)) == 1
        assert shot.engagements == 0

    def test_should_hold_the_centre_for_a_shot_with_no_subject(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(centre_shot(120), window)
        assert set(shot.x) == {window.centre_x}
        assert shot.has_subject is False
        assert not any(shot.observed)
        assert shot.engagements == 0


class TestCameraMoves:
    def test_should_follow_a_walking_subject_monotonically(self):
        """The hold-heavy defaults trade zone-tightness
        for stillness: on a constant-velocity walk the camera
        lags behind and the subject spends 5 of 60 observed
        samples just outside the dead zone, where the old tight
        camera held all 60 inside. What must NOT happen is
        reversal -- the pan stays monotone, so the lag reads as
        a slow follow rather than as hunting.
        """
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(60, 0.2, 0.8)), window)
        assert shot.x[0] == 80
        assert shot.x[-1] == 1022
        assert all(b >= a for a, b in zip(shot.x, shot.x[1:]))
        assert shot.engagements == 3
        assert sum(shot.inside[frame] for frame in range(0, 120, 2)) == 55

    def test_should_re_engage_when_a_fast_walk_starts_off_frame(self):
        """Starts pinned at the left edge.

        Under the hold-heavy defaults the wide dead zone
        absorbs the whole walk in ONE engagement, where the old
        tight camera latched twice.
        """
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(60, 0.1, 0.9)), window)
        assert shot.x[0] == 0
        assert shot.x[-1] == 1240
        assert shot.engagements == 1
        assert sum(shot.inside[frame] for frame in range(0, 120, 2)) == 58

    def test_should_pan_not_snap_on_a_mid_shot_subject_switch(self):
        window = crop_window(meta_for(1920, 1080))
        points = [0.25] * 24 + [0.75] * 36
        shot = plan_shot(shot_subject(points, switch_frames=(48,)), window)
        assert shot.x[0] == 176
        assert shot.x[-1] == 1018
        assert max(abs(b - a) for a, b in zip(shot.x, shot.x[1:])) <= 16
        assert sum(shot.inside[frame] for frame in range(0, 120, 2)) == 39

    def test_should_never_exceed_the_pan_speed_cap(self):
        for _n, width, height, _cw, _ch, _mx, _cx in TIERS:
            window = crop_window(meta_for(width, height))
            shot = plan_shot(shot_subject(walk(60, 0.02, 0.98)), window)
            cap_px = 0.20 * window.frame_width / 24.0
            worst = max(abs(b - a) for a, b in zip(shot.x, shot.x[1:]))
            assert worst <= cap_px + 2      # +2: even-quantization step

    def test_should_honour_a_slower_pan_cap_when_configured(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(60, 0.1, 0.9)), window,
                         max_pan_frac_per_sec=0.02)
        worst = max(abs(b - a) for a, b in zip(shot.x, shot.x[1:]))
        assert worst <= 0.02 * 1920 / 24.0 + 2
        assert shot.x[-1] < 400          # far behind the subject


class TestEdgeClamping:
    def test_should_clamp_at_the_left_edge_and_stay_there(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.02] * 60), window)
        assert set(shot.x) == {0}
        assert all(shot.inside)          # clamped counts as inside
        assert all(shot.clamped)

    def test_should_clamp_at_the_right_edge_and_stay_there(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.98] * 60), window)
        assert set(shot.x) == {1312}
        assert all(shot.inside)
        assert all(shot.clamped)

    def test_should_leave_the_edge_at_once_when_the_subject_returns(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.02] * 30 + [0.5] * 30), window)
        assert shot.x[59] == 0
        assert shot.x[61] > 0            # velocity was killed, not stored

    def test_should_pin_x_to_zero_when_there_is_no_pan_range(self):
        window = crop_window(meta_for(1920, 1080), aspect_w=16, aspect_h=9)
        assert window.max_x == 0
        shot = plan_shot(shot_subject(walk(60, 0.1, 0.9)), window)
        assert set(shot.x) == {0}


class TestHardReset:
    def test_should_snap_to_the_first_subject_point_at_the_shot_start(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject([0.8] * 30), window)
        assert shot.x[0] == 1232
        assert shot.x[1] == 1232
        assert shot.inside[0] is True

    def test_should_reset_hard_at_every_shot_boundary(self):
        window = crop_window(meta_for(1920, 1080))
        first = plan_shot(
            shot_subject([0.8] * 30, start=0, frame_count=60), window)
        second = plan_shot(
            shot_subject([0.2] * 30, start=60, shot_index=1,
                         frame_count=60), window)
        assert first.x[-1] == 1232
        assert second.x[0] == 80

    def test_should_backfill_before_the_first_sample(self):
        window = crop_window(meta_for(1920, 1080))
        base = shot_subject([0.8] * 10, stride=2, frame_count=40)
        subject = dataclasses.replace(
            base,
            sampled_frames=tuple(20 + i * 2 for i in range(10)),
            samples=tuple(sample_at(20 + i * 2, 0.8) for i in range(10)),
        )
        shot = plan_shot(subject, window)
        assert len(set(shot.x)) == 1     # already in place on frame 0

    def test_should_handle_a_single_frame_shot(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(
            shot_subject([0.42], stride=1, frame_count=1), window)
        assert shot.frame_count == 1
        assert len(shot.x) == 1

    def test_should_not_carry_velocity_across_a_reset(self):
        window = crop_window(meta_for(1920, 1080))
        second = plan_shot(
            shot_subject([0.2] * 30, start=60, shot_index=1,
                         frame_count=60), window)
        assert len(set(second.x)) == 1   # static subject, static camera


class TestParameterValidation:
    def test_should_raise_on_a_dead_zone_outside_its_range(self):
        window = crop_window(meta_for(1920, 1080))
        with pytest.raises(CameraError) as excinfo:
            plan_shot(shot_subject([0.5] * 4), window, dead_zone=0.0)
        assert "REFRAME_DEAD_ZONE" in str(excinfo.value)

    def test_should_raise_on_a_release_outside_its_range(self):
        window = crop_window(meta_for(1920, 1080))
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(CameraError) as excinfo:
                plan_shot(shot_subject([0.5] * 4), window,
                          dead_zone_release=bad)
            assert "REFRAME_DEAD_ZONE_RELEASE" in str(excinfo.value)

    def test_should_raise_on_a_non_positive_smoothing_hz(self):
        window = crop_window(meta_for(1920, 1080))
        with pytest.raises(CameraError) as excinfo:
            plan_shot(shot_subject([0.5] * 4), window, smoothing_hz=0.0)
        assert "REFRAME_SMOOTHING_HZ" in str(excinfo.value)

    def test_should_raise_on_a_pan_cap_outside_its_range(self):
        window = crop_window(meta_for(1920, 1080))
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(CameraError) as excinfo:
                plan_shot(shot_subject([0.5] * 4), window,
                          max_pan_frac_per_sec=bad)
            assert "REFRAME_MAX_PAN_FRAC_PER_SEC" in str(excinfo.value)

    def test_should_accept_a_dead_zone_near_one(self):
        window = crop_window(meta_for(1920, 1080))
        subject = shot_subject(walk(60, 0.2, 0.8))
        wide = plan_shot(subject, window, dead_zone=0.999)
        usual = plan_shot(subject, window)
        assert max(wide.x) - min(wide.x) < max(usual.x) - min(usual.x)
        assert all(0 <= value <= window.max_x for value in wide.x)

    def test_should_accept_a_full_width_pan_cap(self):
        window = crop_window(meta_for(1920, 1080))
        shot = plan_shot(shot_subject(walk(60, 0.2, 0.8)), window,
                         max_pan_frac_per_sec=1.0)
        assert all(0 <= value <= window.max_x for value in shot.x)


class TestReleaseFraction:
    def test_should_stop_copying_jitter_when_release_is_the_default(self):
        """Release 0.0 chatters after a pan, 0.25 does not."""
        window = crop_window(meta_for(1920, 1080))
        points = []
        for index in range(120):
            frame = index * 2
            if frame < 60:
                points.append(0.35)
            elif frame < 120:
                points.append(0.35 + 0.30 * ((frame - 60) / 60.0))
            else:
                points.append(0.65 + 0.018 * math.sin(
                    2 * math.pi * (frame - 120) / 8.0))
        subject = shot_subject(points, frame_count=240)
        loose = plan_shot(subject, window, dead_zone_release=0.0)
        tight = plan_shot(subject, window, dead_zone_release=0.25)
        assert _reversals(loose.x) > 10
        assert _reversals(tight.x) == 0
        assert tight.engagements == 1
        assert loose.x[-1] == 944
        assert tight.x[-1] == 884


class TestCoastStability:
    def test_should_not_re_engage_at_any_smoothing_hz(self):
        """An engage-then-go-quiet shot, swept across smoothing_hz.

        The release branch does not zero velocity, so the camera eases
        out instead of stopping dead. The
        residual decay factor is 1 / (1 + dt * omega) ** 2 per frame --
        0.52 at 1.5 Hz but 0.78 at 0.5 Hz -- so a slow filter coasts
        further and could in principle carry the subject back outside
        the dead zone and re-engage in a limit cycle. It does not.
        """
        window = crop_window(meta_for(1920, 1080))
        points = []
        for index in range(120):
            frame = index * 2
            if frame < 60:
                points.append(0.35)
            elif frame < 120:
                points.append(0.35 + 0.30 * ((frame - 60) / 60.0))
            else:
                points.append(0.65 + 0.018 * math.sin(
                    2 * math.pi * (frame - 120) / 8.0))
        subject = shot_subject(points, frame_count=240)
        # With release 0.6 the camera stops while the
        # subject is still off-centre, so the ramp can latch a
        # SECOND time at 1.0 Hz and above. That is not the
        # failure this test guards against: a limit cycle means the
        # camera REVERSES, and it never does. Assert the
        # no-oscillation property directly rather than via an
        # engagement count that only held for a tight camera.
        for smoothing_hz in (0.5, 1.0, 1.5, 3.0):
            shot = plan_shot(subject, window, smoothing_hz=smoothing_hz)
            assert shot.engagements <= 2
            assert _reversals(shot.x) == 0


def two_shot_clip():
    """Shot 0: static centre, 120 frames. Shot 1: a walk, 120 frames."""
    return [
        shot_subject([0.5] * 60, start=0, shot_index=0),
        shot_subject(walk(60, 0.2, 0.8), start=120, shot_index=1),
    ]


class TestPlanCamera:
    def test_should_return_one_trajectory_per_shot_in_order(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        assert len(track.shots) == 2
        assert [s.shot_index for s in track.shots] == [0, 1]
        assert track.shots[0].start_frame == 0
        assert track.shots[1].start_frame == 120
        assert track.total_frames == 240
        assert track.fps == 24.0

    def test_should_share_one_crop_window_across_the_clip(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        assert track.window.crop_w == 608
        assert track.window.crop_h == 1080
        assert track.window.y == 0

    def test_should_record_the_motion_parameters_it_used(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080),
                            dead_zone=0.4, dead_zone_release=0.1,
                            smoothing_hz=2.0, max_pan_frac_per_sec=0.1)
        assert track.dead_zone == 0.4
        assert track.dead_zone_release == 0.1
        assert track.smoothing_hz == 2.0
        assert track.max_pan_frac_per_sec == 0.1

    def test_should_flatten_to_one_x_per_frame_of_the_clip(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        flat = track.flatten()
        assert len(flat) == 240
        assert flat[:120] == track.shots[0].x
        assert flat[120:] == track.shots[1].x

    def test_should_resolve_any_absolute_frame_with_x_at(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        assert track.x_at(0) == 656
        assert track.x_at(119) == 656
        assert track.x_at(120) == 80
        assert track.x_at(239) == 1022
        with pytest.raises(CameraError):
            track.x_at(240)
        with pytest.raises(CameraError):
            track.x_at(-1)

    def test_should_not_mutate_its_arguments(self):
        subjects = two_shot_clip()
        before = [(s.sampled_frames, s.samples) for s in subjects]
        plan_camera(subjects, meta_for(1920, 1080))
        assert [(s.sampled_frames, s.samples) for s in subjects] == before

    def test_should_be_deterministic_when_called_twice(self):
        subjects = two_shot_clip()
        first = plan_camera(subjects, meta_for(1920, 1080))
        second = plan_camera(subjects, meta_for(1920, 1080))
        assert first == second

    def test_should_raise_when_the_subject_list_is_empty(self):
        with pytest.raises(CameraError) as excinfo:
            plan_camera([], meta_for(1920, 1080))
        assert "empty" in str(excinfo.value)

    def test_should_raise_when_the_first_shot_does_not_start_at_zero(self):
        subjects = [shot_subject([0.5] * 10, start=4)]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080))
        assert "start at frame 0" in str(excinfo.value)

    def test_should_raise_on_a_non_contiguous_shot_tiling(self):
        subjects = [
            shot_subject([0.5] * 10, start=0, shot_index=0),
            shot_subject([0.5] * 10, start=30, shot_index=1),
        ]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080))
        assert "30" in str(excinfo.value)

    def test_should_raise_on_overlapping_shots(self):
        first = dataclasses.replace(
            shot_subject([0.5] * 10, start=0, shot_index=0), end_frame=25)
        second = shot_subject([0.5] * 10, start=20, shot_index=1)
        with pytest.raises(CameraError) as excinfo:
            plan_camera([first, second], meta_for(1920, 1080))
        assert "no gap and no overlap" in str(excinfo.value)

    def test_should_raise_when_shot_indices_are_out_of_order(self):
        subjects = [
            shot_subject([0.5] * 10, start=0, shot_index=1),
            shot_subject([0.5] * 10, start=20, shot_index=0),
        ]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080))
        assert "shot_index" in str(excinfo.value)

    def test_should_raise_when_shots_disagree_about_fps(self):
        subjects = [
            shot_subject([0.5] * 10, start=0, shot_index=0, fps=24.0),
            shot_subject([0.5] * 10, start=20, shot_index=1, fps=25.0),
        ]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080))
        assert "25.0" in str(excinfo.value)

    def test_should_raise_when_shots_disagree_about_stride(self):
        """Stride is cross-checked exactly like fps."""
        subjects = [
            shot_subject([0.5] * 10, start=0, shot_index=0, stride=2),
            shot_subject([0.5] * 10, start=20, shot_index=1, stride=3),
        ]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080))
        assert "stride" in str(excinfo.value)

    def test_should_record_the_stride_it_ran_at(self):
        """CropTrack carries the stride that produced the shots."""
        subjects = [shot_subject([0.5] * 10, stride=3)]
        track = plan_camera(subjects, meta_for(1920, 1080))
        assert track.stride == 3

    def test_should_raise_when_the_source_fps_disagrees_with_shots(self):
        """meta.fps is cross-checked against the shots, not just

        the shots against each other.
        """
        subjects = [shot_subject([0.5] * 10, start=0, shot_index=0,
                                 fps=25.0)]
        with pytest.raises(CameraError) as excinfo:
            plan_camera(subjects, meta_for(1920, 1080, fps=24.0))
        message = str(excinfo.value)
        assert "25.0" in message and "24.0" in message

    @pytest.mark.parametrize(
        "name,width,height,final_x,implied",
        [("720p", 1280, 720, 682, 0.691), ("1080p", 1920, 1080, 1022, 0.691),
         ("1440p", 2560, 1440, 1364, 0.691), ("4K", 3840, 2160, 2046, 0.691),
         ("ultrawide", 2560, 1080, 1584, 0.7375),
         ("letterboxed", 1920, 816, 1186, 0.7375)],
    )
    def test_should_produce_an_identical_trajectory_at_every_tier(
        self, name, width, height, final_x, implied,
    ):
        """One code path, no tier branch.

        The four 16:9 tiers still land on ONE normalised position
        (0.691 within 0.001) across 720p..4K, which is what
        resolution-independence means and is what this test exists to
        prove -- there is still no tier branch in the code.

        The two non-16:9 sources land elsewhere (0.7375)
        instead of joining them. The dead zone is a fraction of
        CROP width expressed in frame-normalised units, so it
        scales with crop_w/frame_width -- 0.317 on 16:9 but 0.238
        on 2560x1080. The old tight camera converged close enough
        to the subject that the zone width barely affected where
        it came to rest, which made every aspect agree by
        coincidence. A hold-heavy camera rests AT the zone edge,
        so the difference is now visible. Aspect-independence was
        never a stated criterion and the framing is sound either
        way (the subject rests 53% of the way to the crop edge on
        ultrawide vs 69% on 16:9).
        """
        subjects = [shot_subject(walk(60, 0.2, 0.8))]
        track = plan_camera(subjects, meta_for(width, height))
        shot = track.shots[0]
        assert shot.x[-1] == final_x
        actual = ((shot.x[-1] + track.window.crop_w / 2.0)
                  / track.window.frame_width)
        assert actual == pytest.approx(implied, abs=0.001)


class TestFrameRateIndependence:
    @pytest.mark.parametrize(
        "fps,frames,final_x,observed_inside",
        [(23.976, 120, 1022, 55), (24.0, 120, 1022, 55),
         (25.0, 125, 996, 58), (29.97, 150, 1036, 70),
         (30.0, 150, 1036, 70), (60.0, 300, 1032, 138)],
    )
    def test_should_pan_the_same_5_seconds_at_every_frame_rate(
        self, fps, frames, final_x, observed_inside,
    ):
        sampled_count = (frames + 1) // 2
        subjects = [shot_subject(walk(sampled_count, 0.2, 0.8), stride=2,
                                 fps=fps, frame_count=frames)]
        # meta.fps must agree with the shots, so it tracks fps here.
        track = plan_camera(subjects, meta_for(1920, 1080, fps=fps))
        shot = track.shots[0]
        assert shot.x[-1] == final_x
        assert sum(shot.inside[f] for f in range(0, frames, 2)) == (
            observed_inside)


class TestCropChanges:
    def test_should_always_start_at_frame_zero(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        changes = crop_changes(track)
        assert changes[0] == (0, track.flatten()[0])

    def test_should_collapse_a_static_clip_to_one_entry(self):
        subjects = [shot_subject([0.5] * 60)]
        track = plan_camera(subjects, meta_for(1920, 1080))
        assert crop_changes(track) == [(0, 656)]

    def test_should_replay_exactly_to_the_flat_trajectory(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        flat = track.flatten()
        replay, current = [], None
        pending = dict(crop_changes(track))
        for frame in range(track.total_frames):
            current = pending.get(frame, current)
            replay.append(current)
        assert tuple(replay) == flat

    def test_should_record_the_cut_as_a_change(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        assert any(frame == 120 for frame, _x in crop_changes(track))


class TestDeadZoneReport:
    def test_should_count_every_observed_sample_of_the_clip(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == 120
        # 5 samples sit just outside the zone while the
        # slow camera catches up on shot 1's walk. Still clears
        # the 0.95 threshold, so the dead-zone metric passes.
        assert report.inside_samples == 115
        assert report.clamped_samples == 0
        assert report.switch_samples == 0
        assert report.fraction == pytest.approx(0.958333, abs=1e-6)
        assert report.passes is True
        assert report.measurable is True

    def test_should_attribute_an_unreachable_subject_to_clamping(self):
        subjects = [shot_subject([0.02] * 60)]
        track = plan_camera(subjects, meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == 60
        assert report.inside_samples == 60
        assert report.clamped_samples == 60
        assert report.fraction == 1.0

    def test_should_attribute_a_mid_shot_switch_transient(self):
        subjects = [shot_subject([0.25] * 24 + [0.75] * 36,
                                 switch_frames=(48,))]
        track = plan_camera(subjects, meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == 60
        assert report.inside_samples == 39
        # The window is bounded to the recovery point:
        # the switch is at frame 48 and `inside` is True again at 90
        # (the wider zone recovers sooner than the old 102), so
        # the window [48, 90] holds 22 observed (even) samples -- not
        # the 36 that counting to the shot end would give.
        assert report.switch_samples == 22
        assert report.fraction == 0.65
        assert report.passes is False

    def test_should_exclude_unobserved_samples_from_the_report(self):
        flags = [True, False] * 30
        subjects = [shot_subject([0.5] * 60, observed=flags)]
        track = plan_camera(subjects, meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == 30
        assert report.inside_samples == 30

    def test_should_report_a_full_fraction_when_nothing_was_observed(self):
        track = plan_camera([centre_shot(120)], meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == 0
        assert report.fraction == 1.0
        assert report.passes is True
        assert report.measurable is False   # passed, but nothing measured

    @pytest.mark.parametrize(
        "stride,samples,fraction",
        [(1, 120, 0.941667), (2, 60, 0.916667), (3, 40, 0.85),
         (4, 30, 0.9)],
    )
    def test_should_measure_a_stale_zero_order_hold_at_each_stride(
        self, stride, samples, fraction,
    ):
        """The zero-order hold goes stale at a coarse stride.

        The fraction is not MONOTONE in stride --
        stride 4 (0.900) scores above stride 3 (0.850). With a
        hold-heavy camera the subject tracks along the zone edge,
        so whether a given stride's samples land inside or
        outside is an aliasing accident of where they fall, not a
        smooth function of coarseness. The staleness is still
        real (every stride is below stride 1); only the ordering
        was an artefact of the old tight camera, so this pins the
        measured value per stride instead of an ordering.
        """
        count = (120 + stride - 1) // stride
        subjects = [shot_subject(walk(count, 0.2, 0.8), stride=stride,
                                 frame_count=120)]
        track = plan_camera(subjects, meta_for(1920, 1080))
        report = dead_zone_report(track)
        assert report.observed_samples == samples
        assert report.fraction == pytest.approx(fraction, abs=1e-6)

    def test_should_bound_the_switch_window_to_the_recovery_point(self):
        """Only the recovery transient counts,
        not every observed sample from the switch to the shot end."""
        window = crop_window(meta_for(1920, 1080))
        shot = ShotTrajectory(
            shot_index=0, start_frame=0, end_frame=10, fps=24.0,
            x=(0,) * 10, subject_x=(0,) * 10,
            observed=(True,) * 10,
            inside=(True, True, False, False, True, True, True, False,
                    True, True),
            clamped=(False,) * 10,
            has_subject=True, switch_frames=(2,), engagements=1,
        )
        track = CropTrack(
            window=window, fps=24.0, stride=2, total_frames=10,
            dead_zone=0.3, dead_zone_release=0.25, smoothing_hz=1.5,
            max_pan_frac_per_sec=0.20, shots=(shot,),
        )
        report = dead_zone_report(track)
        # switch at frame 2; inside is False at 2, 3 and True again at
        # 4 -- the window is [2, 4] inclusive, 3 samples, not the 8
        # remaining observed samples to the shot end.
        assert report.switch_samples == 3

    def test_should_not_double_count_overlapping_switch_windows(self):
        window = crop_window(meta_for(1920, 1080))
        shot = ShotTrajectory(
            shot_index=0, start_frame=0, end_frame=8, fps=24.0,
            x=(0,) * 8, subject_x=(0,) * 8,
            observed=(True,) * 8,
            inside=(True, True, False, False, False, True, True, True),
            clamped=(False,) * 8,
            has_subject=True, switch_frames=(2, 4), engagements=2,
        )
        track = CropTrack(
            window=window, fps=24.0, stride=2, total_frames=8,
            dead_zone=0.3, dead_zone_release=0.25, smoothing_hz=1.5,
            max_pan_frac_per_sec=0.20, shots=(shot,),
        )
        report = dead_zone_report(track)
        # switch at 2 recovers at 5 -> window {2,3,4,5}; switch at 4
        # recovers at 5 too -> window {4,5}; unioned, not 4 + 2.
        assert report.switch_samples == 4

    def test_should_not_count_switch_samples_in_a_title_shot(self):
        """switch_samples explains misses among the observed samples,
        and a title shot's samples are not among them."""
        window = crop_window(meta_for(1920, 1080))
        shot = ShotTrajectory(
            shot_index=0, start_frame=0, end_frame=10, fps=24.0,
            x=(0,) * 10, subject_x=(0,) * 10,
            observed=(True,) * 10,
            inside=(True, True) + (False,) * 8,
            clamped=(False,) * 10,
            has_subject=True, switch_frames=(2,), engagements=0,
            treatment=TREATMENT_CROP, title=title_card(width=500.0),
        )
        track = CropTrack(
            window=window, fps=24.0, stride=2, total_frames=10,
            dead_zone=0.3, dead_zone_release=0.25, smoothing_hz=1.5,
            max_pan_frac_per_sec=0.20, shots=(shot,),
        )

        report = dead_zone_report(track)

        assert report.title_samples == 10
        assert report.observed_samples == 0
        assert report.switch_samples == 0

    def test_should_be_a_plain_dataclass_of_four_counts(self):
        report = DeadZoneReport(
            observed_samples=10, inside_samples=9, clamped_samples=1,
            switch_samples=0)
        assert report.fraction == 0.9
        assert report.passes is False
        assert report.measurable is True


class TestCameraSummary:
    def test_should_name_the_crop_the_shots_and_the_dead_zone(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        line = camera_summary(track)
        assert "608x1080" in line
        assert "2 shot(s)" in line
        assert "240 frame(s)" in line
        assert "95.8%" in line

    def test_should_be_pure_ascii_for_a_cp1252_stdout(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        camera_summary(track).encode("ascii")   # raises if it is not


class TestCropTrackToJson:
    def test_should_publish_the_window_and_the_motion_parameters(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        payload = crop_track_to_json(track)
        assert payload["crop_w"] == 608
        assert payload["crop_h"] == 1080
        assert payload["y"] == 0
        assert payload["frame_width"] == 1920
        assert payload["frame_height"] == 1080
        assert payload["fps"] == 24.0
        assert payload["total_frames"] == 240
        assert payload["dead_zone"] == 0.9
        assert payload["dead_zone_release"] == 0.6
        assert payload["smoothing_hz"] == 0.6
        assert payload["max_pan_frac_per_sec"] == 0.20
        assert payload["dead_zone_fraction"] == pytest.approx(
            0.958333, abs=1e-6)
        assert payload["observed_samples"] == 120
        assert payload["sc2_measurable"] is True

    def test_should_publish_the_stride_it_ran_at(self):
        """The dead-zone metric is stride-sensitive, so crops.json must
        say which stride produced it, not config.env at read time."""
        track = plan_camera([shot_subject([0.5] * 20, stride=3)],
                            meta_for(1920, 1080))
        assert crop_track_to_json(track)["stride"] == 3

    def test_should_publish_one_entry_per_frame_per_shot(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        payload = crop_track_to_json(track)
        assert len(payload["shots"]) == 2
        first = payload["shots"][0]
        assert first["index"] == 0
        assert first["start_frame"] == 0
        assert first["end_frame"] == 120
        assert first["has_subject"] is True
        assert first["switch_frames"] == []
        for key in ("x", "subject_x", "observed", "inside"):
            assert len(first[key]) == 120

    def test_should_round_trip_through_the_json_module(self):
        track = plan_camera(two_shot_clip(), meta_for(1920, 1080))
        payload = crop_track_to_json(track)
        restored = json.loads(json.dumps(payload))
        assert restored["shots"][1]["x"] == list(track.shots[1].x)


def subject_walking(shot_index, start, end, fps):
    """A walking ShotSubject spanning [start, end).

    A NAMING shim over shot_subject/walk above, not a second subject
    factory: every sample is built by the existing helpers, so there is
    only ever one implementation of what a ShotSubject looks like.
    """
    return shot_subject(
        walk((end - start) // 2, 0.2, 0.8), start=start, stride=2,
        fps=fps, shot_index=shot_index, frame_count=end - start,
    )


def title_card(centre_x=960.0, width=996.0, *, lines=5, seen=5,
               sampled=5, top=222.0, bottom=841.0):
    """A real TitleCard -- text.py's frozen dataclass, not a fake."""
    from ftea.reframe.text import TitleCard

    return TitleCard(
        x1=centre_x - width / 2.0, y1=top,
        x2=centre_x + width / 2.0, y2=bottom,
        centre_x=centre_x, line_count=lines,
        frames_seen=seen, frames_sampled=sampled,
    )


class TestCanvasFor:
    """The canvas formula, at every output tier.

    The canvas carries the OUTPUT aspect, not the source's, because the
    letterbox state crops the WHOLE canvas and a static scale then fits
    it to crop_w x crop_h. Any other shape stretches the picture.
    """

    @pytest.mark.parametrize("width,height,canvas_h,pad_y", [
        (1280, 720, 2270, 775),
        (1920, 1080, 3410, 1165),
        (2560, 1440, 4552, 1556),
        (3840, 2160, 6822, 2331),
        (2560, 1080, 4548, 1734),
        (1920, 816, 3406, 1295),
    ])
    def test_should_compute_the_canvas_for_every_tier(self, width,
                                                      height, canvas_h,
                                                      pad_y):
        canvas = canvas_for(crop_window(meta_for(width, height)))

        assert canvas.width == width
        assert canvas.height == canvas_h
        assert canvas.pad_x == 0
        assert canvas.pad_y == pad_y

    def test_should_reproduce_the_verified_v5_geometry(self):
        """A test render of exactly pad=1920:3410:0:1165 gave a
        constant 608x1080 (and a second check confirmed it)."""
        canvas = canvas_for(crop_window(meta_for(1920, 1080)))

        assert (canvas.width, canvas.height, canvas.pad_x,
                canvas.pad_y) == (1920, 3410, 0, 1165)

    def test_should_use_even_round_and_not_round_up_even(self):
        """It was measured that ffmpeg's scale=-2 rounds
        half-up to even -- 2560x1080 -> 1706, not 1708. round_up_even
        would give 3412 here instead of 3410, and the canvas and any
        downstream scale would then disagree by construction."""
        canvas = canvas_for(crop_window(meta_for(1920, 1080)))

        assert canvas.height == even_round(1920 * 1080 / 608)
        assert canvas.height != round_up_even(1920 * 1080 / 608)

    def test_should_give_a_canvas_taller_than_the_frame(self):
        canvas = canvas_for(crop_window(meta_for(1920, 1080)))

        assert canvas.height > 1080
        assert canvas.pad_y * 2 + 1080 <= canvas.height

    def test_should_carry_the_output_aspect_within_a_pixel(self):
        window = crop_window(meta_for(1920, 1080))
        canvas = canvas_for(window)

        assert abs(canvas.width / canvas.height
                   - window.crop_w / window.crop_h) < 0.001

    def test_should_refuse_a_source_whose_crop_spans_the_frame(self):
        """Unreachable from plan_camera -- title_treatment never
        letterboxes a crop that already spans the frame -- but a
        letterbox of a full-width crop is meaningless, so it raises
        rather than dividing by a degenerate ratio."""
        window = CropWindow(frame_width=1080, frame_height=1920,
                            crop_w=1080, crop_h=1920, y=0)

        with pytest.raises(CameraError) as excinfo:
            canvas_for(window)

        assert "spans the frame" in str(excinfo.value)


class TestTitleGeometry:
    def test_should_centre_the_crop_on_the_text_centroid(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_x(title_card(centre_x=960.0), window) == 656

    def test_should_quantise_to_an_even_pixel(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_x(title_card(centre_x=961.0), window) % 2 == 0

    def test_should_clamp_a_centroid_near_the_left_edge(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_x(title_card(centre_x=100.0), window) == 0

    def test_should_clamp_a_centroid_near_the_right_edge(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_x(title_card(centre_x=1900.0),
                       window) == window.max_x

    def test_should_letterbox_the_measured_credit_block(self):
        """996px of text, 608px of crop, 19.2px of margin each side."""
        window = crop_window(meta_for(1920, 1080))

        assert title_treatment(title_card(width=996.0), window,
                               0.01) == TREATMENT_LETTERBOX

    def test_should_simply_centre_a_narrow_title(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_treatment(title_card(width=500.0), window,
                               0.01) == TREATMENT_CROP

    def test_should_not_letterbox_a_crop_that_spans_the_frame(self):
        """With the crop already the full frame width there is nothing
        a letterbox could add, and canvas_for would refuse it."""
        window = crop_window(meta_for(1920, 1080), aspect_w=16, aspect_h=9)

        assert title_treatment(title_card(width=1900.0), window,
                               0.01) == TREATMENT_CROP

    def test_should_plan_a_title_shot_when_the_crop_spans_the_frame(self):
        subject = subject_walking(0, 0, 120, 24.0)

        track = plan_camera([subject], meta_for(1920, 1080),
                            aspect_w=16, aspect_h=9,
                            titles=[title_card(width=1900.0)])

        assert track.canvas is None
        assert track.shots[0].treatment == TREATMENT_CROP

    @pytest.mark.parametrize("width,want", [
        (569.0, TREATMENT_CROP),        # 569 + 38.4 = 607.4 <= 608
        (570.0, TREATMENT_LETTERBOX),   # 570 + 38.4 = 608.4 >  608
    ])
    def test_should_switch_treatment_at_the_margin_boundary(self, width,
                                                            want):
        window = crop_window(meta_for(1920, 1080))

        assert title_treatment(title_card(width=width), window,
                               0.01) == want

    def test_should_never_letterbox_at_a_zero_margin_when_it_fits(self):
        window = crop_window(meta_for(1920, 1080))

        assert title_treatment(title_card(width=608.0), window,
                               0.0) == TREATMENT_CROP


class TestPlanShotWithATitle:
    def test_should_hold_a_constant_x_across_the_whole_shot(self):
        """A title shot bypasses the dead zone, the damping and the pan
        cap entirely: there is nothing to follow and nothing to smooth."""
        window = crop_window(meta_for(1920, 1080))
        subject = subject_walking(0, 0, 120, 24.0)

        shot = plan_shot(subject, window, title=title_card(),
                         edge_margin=0.01)

        assert set(shot.x) == {656}

    def test_should_record_the_treatment_and_the_card(self):
        window = crop_window(meta_for(1920, 1080))
        card = title_card()

        shot = plan_shot(subject_walking(0, 0, 120, 24.0), window,
                         title=card, edge_margin=0.01)

        assert shot.treatment == TREATMENT_LETTERBOX
        assert shot.title is card

    def test_should_still_record_the_subject_track_for_audit(self):
        """The centroid governs only the "crop" treatment -- a
        letterbox shot crops the whole canvas, so its x is irrelevant to
        the rendered result. It is planned and recorded anyway: that is
        what makes the decision auditable, and what a later "was
        letterboxing actually needed?" review reads."""
        window = crop_window(meta_for(1920, 1080))
        subject = subject_walking(0, 0, 120, 24.0)

        shot = plan_shot(subject, window, title=title_card(),
                         edge_margin=0.01)

        assert len(shot.subject_x) == shot.frame_count
        assert len(set(shot.subject_x)) > 1
        assert any(shot.observed)

    def test_should_not_smooth_across_the_cut_into_a_title_shot(self):
        window = crop_window(meta_for(1920, 1080))

        shot = plan_shot(subject_walking(1, 120, 240, 24.0), window,
                         title=title_card(), edge_margin=0.01)

        assert shot.x[0] == 656


class TestPlanCameraWithTitles:
    def _subjects(self):
        return [
            subject_walking(0, 0, 120, 24.0),
            subject_walking(1, 120, 240, 24.0),
        ]

    def test_should_set_the_canvas_when_any_shot_letterboxes(self):
        track = plan_camera(self._subjects(),
                            meta_for(1920, 1080, duration_s=10.0),
                            titles=[title_card(), None])

        assert track.canvas == Canvas(width=1920, height=3410, pad_x=0,
                                      pad_y=1165)
        assert track.shots[0].treatment == TREATMENT_LETTERBOX
        assert track.shots[1].treatment == TREATMENT_CROP

    def test_should_leave_the_canvas_unset_for_a_narrow_title(self):
        """A centred narrow title needs no bars, so the clip keeps
        the graph used without title detection."""
        track = plan_camera(self._subjects(),
                            meta_for(1920, 1080, duration_s=10.0),
                            titles=[title_card(width=400.0), None])

        assert track.canvas is None
        assert track.shots[0].treatment == TREATMENT_CROP
        assert set(track.shots[0].x) == {656}

    def test_should_reject_a_titles_list_of_the_wrong_length(self):
        with pytest.raises(CameraError) as excinfo:
            plan_camera(self._subjects(),
                        meta_for(1920, 1080, duration_s=10.0),
                        titles=[title_card()])

        assert "1 title" in str(excinfo.value)
        assert "2 shot" in str(excinfo.value)


class TestCropRects:
    def _track(self):
        return plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card(), None])

    def test_should_crop_the_whole_canvas_on_a_letterbox_shot(self):
        rects = crop_rects(self._track())

        assert rects[0] == (1920, 3410, 0, 0)
        assert rects[119] == (1920, 3410, 0, 0)

    def test_should_offset_a_normal_shot_by_the_pad(self):
        """The real image sits in the middle of the canvas, so a normal
        crop reads at y = pad_y, not at y = 0."""
        rects = crop_rects(self._track())

        assert rects[120][0:2] == (608, 1080)
        assert rects[120][3] == 1165

    def test_should_raise_without_a_canvas(self):
        track = plan_camera(
            [subject_walking(0, 0, 120, 24.0)],
            meta_for(1920, 1080, duration_s=5.0))

        with pytest.raises(CameraError) as excinfo:
            crop_rects(track)

        assert "needs a canvas" in str(excinfo.value)
        assert "no letterbox shot" in str(excinfo.value)

    def test_should_emit_one_change_per_moving_component(self):
        changes = rect_changes(self._track())

        assert changes[0] == (0, {"w": 1920, "h": 3410, "x": 0, "y": 0})
        at_cut = [c for c in changes if c[0] == 120]
        assert at_cut and set(at_cut[0][1]) == {"w", "h", "x", "y"}

    def test_should_suppress_unchanged_components(self):
        """Inside the second shot only x moves, so w/h/y must not be
        re-emitted -- sendcmd's parse cost is super-linear."""
        changes = rect_changes(self._track())
        after_cut = [c for c in changes if c[0] > 120]

        assert after_cut
        assert all(set(delta) == {"x"} for _frame, delta in after_cut)

    def test_should_replay_back_to_the_rect_series(self):
        """Lossless, exactly like crop_changes: replaying the deltas
        must reconstruct crop_rects() entry for entry."""
        track = self._track()
        rects = crop_rects(track)
        current = {}
        rebuilt = []
        changes = dict(rect_changes(track))
        for frame in range(len(rects)):
            current.update(changes.get(frame, {}))
            rebuilt.append((current["w"], current["h"], current["x"],
                            current["y"]))

        assert tuple(rebuilt) == rects


class TestDeadZoneReportExcludesTitles:
    def test_should_count_title_frames_separately(self):
        """The dead-zone metric measures PEOPLE-following, and a
        title shot deliberately stops following people. Excluding a
        named, counted category is not tuning: the per-frame `inside`
        and `clamped` arrays are still computed honestly and still
        written to crops.json, and only the aggregate excludes them."""
        track = plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card(), None])

        report = dead_zone_report(track)
        plain = dead_zone_report(plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0)))

        assert report.title_samples > 0
        assert report.observed_samples < plain.observed_samples
        assert report.title_samples + report.observed_samples == \
            plain.observed_samples

    def test_should_apply_to_a_narrow_title_too(self):
        """A centred narrow title still frames text rather than people,
        so the dead-zone metric is equally meaningless there."""
        track = plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card(width=400.0), None])

        assert dead_zone_report(track).title_samples > 0

    def test_should_be_unmeasurable_when_every_shot_is_a_title(self):
        """A clip that is ONE shot and letterboxes end to end leaves
        the dead-zone metric nothing to measure -- and `measurable`
        already exists to say exactly that."""
        track = plan_camera(
            [subject_walking(0, 0, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card()])

        report = dead_zone_report(track)

        assert report.observed_samples == 0
        assert report.measurable is False
        assert report.passes is True

    def test_should_still_write_the_per_frame_arrays(self):
        track = plan_camera(
            [subject_walking(0, 0, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card()])

        doc = crop_track_to_json(track)

        assert len(doc["shots"][0]["inside"]) == 240
        assert len(doc["shots"][0]["observed"]) == 240
        assert any(doc["shots"][0]["observed"])


class TestCropTrackToJsonWithTitles:
    def test_should_publish_the_treatment_on_every_shot(self):
        doc = crop_track_to_json(plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card(), None]))

        assert [s["treatment"] for s in doc["shots"]] == \
            ["letterbox", "crop"]

    def test_should_publish_the_card_only_where_there_is_one(self):
        doc = crop_track_to_json(plan_camera(
            [subject_walking(0, 0, 120, 24.0),
             subject_walking(1, 120, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card(), None]))

        assert doc["shots"][0]["title"]["line_count"] == 5
        assert doc["shots"][0]["title"]["centre_x"] == 960.0
        assert "title" not in doc["shots"][1]

    def test_should_publish_the_title_sample_count(self):
        doc = crop_track_to_json(plan_camera(
            [subject_walking(0, 0, 240, 24.0)],
            meta_for(1920, 1080, duration_s=10.0),
            titles=[title_card()]))

        assert doc["title_samples"] > 0


class TestTitlesNoneIsTodaysBehaviour:
    """The regression contract. `titles=None` changes nothing.

    Captured as a value rather than as a shape: the flattened
    trajectory, the change list and the whole serialised document are
    compared, so a one-pixel drift anywhere in the filter fails here.
    """

    def _subjects(self):
        return two_shot_clip()

    def test_should_flatten_identically_with_titles_omitted(self):
        meta = meta_for(1920, 1080, duration_s=10.0)

        without = plan_camera(self._subjects(), meta)
        explicit = plan_camera(self._subjects(), meta, titles=None)

        assert explicit.flatten() == without.flatten()

    def test_should_serialise_identically_with_titles_omitted(self):
        meta = meta_for(1920, 1080, duration_s=10.0)

        without = crop_track_to_json(plan_camera(self._subjects(), meta))
        explicit = crop_track_to_json(
            plan_camera(self._subjects(), meta, titles=None))

        assert json.dumps(explicit, sort_keys=True) == \
            json.dumps(without, sort_keys=True)

    def test_should_leave_the_canvas_unset_when_nothing_letterboxes(self):
        track = plan_camera(self._subjects(), meta_for(1920, 1080,
                                                       duration_s=10.0))

        assert track.canvas is None
        assert all(shot.treatment == TREATMENT_CROP
                   for shot in track.shots)
        assert all(shot.title is None for shot in track.shots)


def _trajectory(frames=4, **overrides):
    fields = dict(
        shot_index=0, start_frame=0, end_frame=frames, fps=24.0,
        x=(0,) * frames, subject_x=(0,) * frames,
        observed=(True,) * frames, inside=(True,) * frames,
        clamped=(False,) * frames, has_subject=True, switch_frames=(),
        engagements=0,
    )
    fields.update(overrides)
    return ShotTrajectory(**fields)


def _track_of(shot, canvas=None):
    return CropTrack(
        window=crop_window(meta_for(1920, 1080)), fps=24.0, stride=2,
        total_frames=shot.end_frame, dead_zone=0.3, dead_zone_release=0.25,
        smoothing_hz=1.5, max_pan_frac_per_sec=0.20, shots=(shot,),
        canvas=canvas,
    )


class TestTreatmentInvariants:
    """treatment, title and canvas say the same thing three ways; the
    types refuse a combination where they disagree."""

    def test_should_refuse_a_letterbox_shot_without_a_title(self):
        with pytest.raises(CameraError, match="title"):
            _trajectory(treatment=TREATMENT_LETTERBOX, title=None)

    def test_should_refuse_an_unknown_treatment(self):
        with pytest.raises(CameraError, match="zoom"):
            _trajectory(treatment="zoom")

    def test_should_refuse_per_frame_values_of_the_wrong_length(self):
        with pytest.raises(CameraError, match="inside"):
            _trajectory(inside=(True,) * 3)

    def test_should_refuse_a_letterbox_track_without_a_canvas(self):
        shot = _trajectory(treatment=TREATMENT_LETTERBOX,
                           title=title_card())

        with pytest.raises(CameraError, match="canvas"):
            _track_of(shot, canvas=None)

    def test_should_refuse_a_canvas_with_no_letterbox_shot(self):
        canvas = canvas_for(crop_window(meta_for(1920, 1080)))

        with pytest.raises(CameraError, match="canvas"):
            _track_of(_trajectory(), canvas=canvas)

    def test_should_accept_a_letterbox_track_with_its_canvas(self):
        shot = _trajectory(treatment=TREATMENT_LETTERBOX,
                           title=title_card())
        canvas = canvas_for(crop_window(meta_for(1920, 1080)))

        assert _track_of(shot, canvas=canvas).canvas == canvas
