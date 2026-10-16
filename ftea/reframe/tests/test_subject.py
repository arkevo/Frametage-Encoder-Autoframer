"""Tests for ftea.reframe.subject -- per-shot subject selection."""

from pathlib import Path

import pytest

from ftea.reframe.detect import Detection, DetectionSet
from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import SourceMeta
from ftea.reframe.shots import Shot
from ftea.reframe.subject import (
    DEFAULT_MIN_CONFIDENCE_FACE,
    DEFAULT_MIN_CONFIDENCE_PERSON,
    KIND_BOTH,
    KIND_CENTER,
    KIND_FACE,
    KIND_HELD,
    KIND_PERSON,
    NO_TRACK,
    NormBox,
    Observation,
    SubjectError,
    Track,
    associate_faces,
    build_tracks,
    contains_ratio,
    follow,
    gap_samples,
    iou,
    link_affinity,
    normalize_box,
    observation_score,
    observations_for_frame,
    select_subjects,
    subject_summary,
    subject_to_json,
    track_scores,
)

TIERS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "ultrawide": (2560, 1080),
    "letterboxed": (1920, 816),
}


def _meta(width=1920, height=1080, fps=24.0, rotation=0, duration_s=10.0):
    return SourceMeta(
        path=Path("clip.mp4"), width=width, height=height,
        rotation=rotation, fps=fps, duration_s=duration_s, has_audio=False,
    )


def _person(frame, cx, cy, w, h, score=0.98, size=(1920, 1080)):
    width, height = size
    return Detection(
        frame=frame, cls="person",
        x1=(cx - w / 2) * width, y1=(cy - h / 2) * height,
        x2=(cx + w / 2) * width, y2=(cy + h / 2) * height,
        score=score,
    )


def _face(frame, cx, cy, w, h, score=0.60, size=(1920, 1080)):
    width, height = size
    return Detection(
        frame=frame, cls="face",
        x1=(cx - w / 2) * width, y1=(cy - h / 2) * height,
        x2=(cx + w / 2) * width, y2=(cy + h / 2) * height,
        score=score,
    )


def _box(cx, cy, w, h):
    return NormBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class TestSubjectErrorLineage:
    def test_should_be_catchable_as_a_reframe_error(self):
        with pytest.raises(ReframeError):
            raise SubjectError("boom")


class TestNormalizeBox:
    def test_should_map_pixels_onto_the_unit_square(self):
        det = Detection(0, "person", 480.0, 270.0, 960.0, 810.0, 0.9)

        box = normalize_box(det, 1920, 1080)

        assert (box.x1, box.y1, box.x2, box.y2) == (0.25, 0.25, 0.5, 0.75)

    def test_should_give_identical_values_at_4k_for_the_same_framing(self):
        hd = normalize_box(
            Detection(0, "person", 480.0, 270.0, 960.0, 810.0, 0.9),
            1920, 1080,
        )
        uhd = normalize_box(
            Detection(0, "person", 960.0, 540.0, 1920.0, 1620.0, 0.9),
            3840, 2160,
        )

        assert hd == uhd

    def test_should_clamp_a_box_that_extends_past_the_frame(self):
        det = Detection(0, "person", -40.0, -10.0, 2200.0, 1200.0, 0.9)

        box = normalize_box(det, 1920, 1080)

        assert (box.x1, box.y1, box.x2, box.y2) == (0.0, 0.0, 1.0, 1.0)

    def test_should_normalise_a_box_given_backwards(self):
        det = Detection(0, "person", 960.0, 810.0, 480.0, 270.0, 0.9)

        box = normalize_box(det, 1920, 1080)

        assert (box.x1, box.y1, box.x2, box.y2) == (0.25, 0.25, 0.5, 0.75)

    def test_should_raise_when_the_frame_has_no_area(self):
        with pytest.raises(SubjectError, match="frame dimensions"):
            normalize_box(Detection(0, "person", 0, 0, 1, 1, 0.9), 0, 1080)


class TestIou:
    def test_should_be_one_for_identical_boxes(self):
        box = _box(0.5, 0.5, 0.2, 0.4)

        assert iou(box, box) == pytest.approx(1.0)

    def test_should_be_zero_for_disjoint_boxes(self):
        assert iou(_box(0.2, 0.5, 0.1, 0.2), _box(0.8, 0.5, 0.1, 0.2)) == 0.0

    def test_should_be_invariant_under_independent_axis_scaling(self):
        a = NormBox(0.0, 0.0, 0.4, 0.6)
        b = NormBox(0.2, 0.3, 0.6, 0.9)
        wide_a = NormBox(0.0, 0.0, 0.8, 0.3)
        wide_b = NormBox(0.4, 0.15, 1.2, 0.45)

        assert iou(a, b) == pytest.approx(iou(wide_a, wide_b))

    def test_should_be_zero_for_a_zero_area_box(self):
        assert iou(NormBox(0.5, 0.5, 0.5, 0.5),
                   _box(0.5, 0.5, 0.2, 0.2)) == 0.0


class TestContainsRatio:
    def test_should_be_one_when_fully_inside(self):
        small = _box(0.5, 0.3, 0.04, 0.06)
        big = _box(0.5, 0.5, 0.2, 0.8)

        assert contains_ratio(small, big) == pytest.approx(1.0)

    def test_should_be_zero_when_outside(self):
        assert contains_ratio(_box(0.1, 0.1, 0.02, 0.02),
                              _box(0.8, 0.8, 0.2, 0.2)) == 0.0


class TestExtent:
    def test_should_be_the_geometric_mean_of_the_sides(self):
        assert _box(0.5, 0.5, 0.16, 0.64).extent == pytest.approx(0.32)

    def test_should_be_resolution_invariant(self):
        boxes = {
            name: normalize_box(
                _person(0, 0.5, 0.5, 0.16, 0.64, size=(w, h)), w, h
            )
            for name, (w, h) in TIERS.items()
        }
        extents = {name: box.extent for name, box in boxes.items()}

        assert len(set(round(v, 9) for v in extents.values())) == 1


def _obs(frame, person=None, face=None, pscore=0.98, fscore=0.60):
    return Observation(
        frame=frame, person=person, face=face,
        person_score=pscore if person is not None else 0.0,
        face_score=fscore if face is not None else 0.0,
    )


class TestAssociateFaces:
    def test_should_attach_a_face_whose_centre_is_inside_a_person(self):
        persons = [(_box(0.5, 0.5, 0.2, 0.8), 0.98)]
        faces = [(_box(0.5, 0.25, 0.05, 0.09), 0.6)]

        kept, orphans = associate_faces(persons, faces)

        assert kept == {0: 0}
        assert orphans == []

    def test_should_return_a_face_with_no_person_as_an_orphan(self):
        faces = [(_box(0.5, 0.5, 0.2, 0.36), 0.55)]

        kept, orphans = associate_faces([], faces)

        assert kept == {}
        assert orphans == [0]

    def test_should_attach_a_face_to_the_smallest_containing_person(self):
        persons = [
            (_box(0.5, 0.5, 0.9, 0.9), 0.90),
            (_box(0.5, 0.4, 0.2, 0.5), 0.95),
        ]
        faces = [(_box(0.5, 0.25, 0.05, 0.09), 0.6)]

        kept, _orphans = associate_faces(persons, faces)

        assert kept == {0: 1}

    def test_should_orphan_the_smaller_of_two_faces_in_one_person(self):
        persons = [(_box(0.5, 0.5, 0.6, 0.9), 0.98)]
        faces = [
            (_box(0.40, 0.25, 0.04, 0.07), 0.6),
            (_box(0.60, 0.25, 0.08, 0.14), 0.5),
        ]

        kept, orphans = associate_faces(persons, faces)

        assert kept == {1: 0}
        assert orphans == [0]


class TestObservationsForFrame:
    def test_should_pair_a_person_with_its_face(self):
        dets = [
            _person(0, 0.5, 0.5, 0.2, 0.8),
            _face(0, 0.5, 0.25, 0.05, 0.09),
        ]

        observations = observations_for_frame(0, dets, 1920, 1080)

        assert len(observations) == 1
        assert observations[0].kind == KIND_BOTH

    def test_should_emit_an_orphan_face_as_its_own_candidate(self):
        dets = [
            _person(0, 0.2, 0.5, 0.2, 0.8),
            _face(0, 0.8, 0.3, 0.06, 0.1),
        ]

        observations = observations_for_frame(0, dets, 1920, 1080)

        kinds = sorted(o.kind for o in observations)
        assert kinds == [KIND_FACE, KIND_PERSON]

    def test_should_emit_nothing_when_the_frame_had_no_detections(self):
        assert observations_for_frame(0, [], 1920, 1080) == []


class TestConfidenceFloorFiltering:
    def test_should_drop_a_person_below_its_confidence_floor(self):
        dets = [_person(0, 0.5, 0.5, 0.15, 0.6, score=0.5)]

        observations = observations_for_frame(
            0, dets, 1920, 1080, min_confidence_person=0.8
        )

        assert observations == []

    def test_should_drop_a_face_below_its_confidence_floor(self):
        dets = [_face(0, 0.5, 0.5, 0.06, 0.1, score=0.2)]

        observations = observations_for_frame(
            0, dets, 1920, 1080, min_confidence_face=0.35
        )

        assert observations == []

    def test_should_orphan_a_face_whose_person_is_below_floor(self):
        dets = [
            _person(0, 0.5, 0.5, 0.2, 0.8, score=0.5),
            _face(0, 0.5, 0.25, 0.05, 0.09, score=0.6),
        ]

        observations = observations_for_frame(
            0, dets, 1920, 1080, min_confidence_person=0.8
        )

        assert len(observations) == 1
        assert observations[0].kind == KIND_FACE


class TestObservationPointContinuity:
    def test_should_bound_the_point_shift_when_a_face_appears_mid_track(self):
        close_up_body = _box(0.60, 0.55, 0.30, 0.90)
        close_up_face = _box(0.55, 0.20, 0.08, 0.14)

        body_only = Observation(
            frame=0, person=close_up_body, face=None,
            person_score=0.95, face_score=0.0,
        )
        with_face = Observation(
            frame=1, person=close_up_body, face=close_up_face,
            person_score=0.95, face_score=0.60,
        )

        delta = abs(with_face.point[0] - body_only.point[0])

        assert delta < 0.10


class TestLinkAffinity:
    def test_should_use_person_iou_when_both_have_person_boxes(self):
        a = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8))
        b = _obs(2, person=_box(0.5, 0.5, 0.2, 0.8))

        assert link_affinity(a, b) == pytest.approx(1.0)

    def test_should_use_face_iou_when_neither_has_a_person_box(self):
        a = _obs(0, face=_box(0.5, 0.3, 0.1, 0.18))
        b = _obs(2, face=_box(0.5, 0.3, 0.1, 0.18))

        assert link_affinity(a, b) == pytest.approx(1.0)

    def test_should_bridge_a_person_only_sample_to_a_face_inside_it(self):
        person_only = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8))
        face_only = _obs(2, face=_box(0.5, 0.25, 0.05, 0.09))

        assert link_affinity(person_only, face_only) == pytest.approx(0.5)

    def test_should_be_zero_when_the_face_sits_outside_the_person(self):
        person_only = _obs(0, person=_box(0.2, 0.5, 0.2, 0.8))
        face_only = _obs(2, face=_box(0.85, 0.25, 0.05, 0.09))

        assert link_affinity(person_only, face_only) == 0.0

    def test_should_prefer_a_same_kind_match_over_a_cross_kind_one(self):
        previous = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8))
        same_kind = _obs(2, person=_box(0.52, 0.5, 0.2, 0.8))
        cross_kind = _obs(2, face=_box(0.5, 0.25, 0.05, 0.09))

        assert link_affinity(previous, same_kind) > link_affinity(
            previous, cross_kind
        )


class TestGapSamples:
    def test_should_convert_seconds_with_fps_and_stride(self):
        assert gap_samples(0.25, 24.0, 2) == 3
        assert gap_samples(0.25, 24.0, 3) == 2

    def test_should_be_stride_independent_in_wall_time(self):
        at_two = gap_samples(0.25, 24.0, 2) * 2 / 24.0
        at_three = gap_samples(0.25, 24.0, 3) * 3 / 24.0

        assert abs(at_two - at_three) < 0.05

    def test_should_raise_on_a_non_positive_fps(self):
        with pytest.raises(SubjectError, match="fps"):
            gap_samples(0.25, 0.0, 2)

    def test_should_raise_on_a_stride_below_one(self):
        with pytest.raises(SubjectError, match="stride"):
            gap_samples(0.25, 24.0, 0)


class TestBuildTracks:
    def test_should_group_a_steady_person_into_one_track(self):
        sampled = (0, 2, 4, 6)
        by_frame = {
            f: [_obs(f, person=_box(0.5, 0.5, 0.2, 0.8))] for f in sampled
        }

        tracks = build_tracks(sampled, by_frame)

        assert len(tracks) == 1
        assert len(tracks[0].observations) == 4

    def test_should_start_a_new_track_when_the_box_jumps(self):
        by_frame = {
            0: [_obs(0, person=_box(0.2, 0.5, 0.1, 0.4))],
            2: [_obs(2, person=_box(0.8, 0.5, 0.1, 0.4))],
        }

        tracks = build_tracks((0, 2), by_frame)

        assert len(tracks) == 2

    def test_should_keep_two_people_in_separate_tracks(self):
        sampled = (0, 2, 4)
        by_frame = {
            f: [
                _obs(f, person=_box(0.3, 0.5, 0.14, 0.6)),
                _obs(f, person=_box(0.7, 0.5, 0.14, 0.6)),
            ]
            for f in sampled
        }

        tracks = build_tracks(sampled, by_frame)

        assert len(tracks) == 2
        assert all(len(t.observations) == 3 for t in tracks)

    def test_should_bridge_a_one_sample_detection_gap(self):
        by_frame = {
            0: [_obs(0, person=_box(0.5, 0.5, 0.2, 0.8))],
            2: [],
            4: [_obs(4, person=_box(0.5, 0.5, 0.2, 0.8))],
        }

        tracks = build_tracks((0, 2, 4), by_frame, max_gap=1)

        assert len(tracks) == 1

    def test_should_close_a_track_once_the_gap_budget_expires(self):
        by_frame = {
            0: [_obs(0, person=_box(0.5, 0.5, 0.2, 0.8))],
            2: [],
            4: [],
            6: [_obs(6, person=_box(0.5, 0.5, 0.2, 0.8))],
        }

        tracks = build_tracks((0, 2, 4, 6), by_frame, max_gap=1)

        assert len(tracks) == 2

    def test_should_track_a_face_only_subject(self):
        sampled = (0, 2, 4)
        by_frame = {
            f: [_obs(f, face=_box(0.5, 0.45, 0.18, 0.32))] for f in sampled
        }

        tracks = build_tracks(sampled, by_frame)

        assert len(tracks) == 1
        assert all(o.kind == KIND_FACE for o in tracks[0].observations)


class TestObservationScore:
    def test_should_rank_a_bigger_face_above_a_bigger_body(self):
        big_face = _obs(
            0, person=_box(0.5, 0.5, 0.10, 0.50),
            face=_box(0.5, 0.30, 0.10, 0.18),
        )
        big_body = _obs(
            0, person=_box(0.5, 0.5, 0.14, 0.70),
            face=_box(0.5, 0.22, 0.03, 0.05),
        )

        assert observation_score(big_face) > observation_score(big_body)

    def test_should_score_a_face_only_observation(self):
        orphan = _obs(0, face=_box(0.5, 0.45, 0.18, 0.32))

        assert observation_score(orphan) > 0.0

    def test_should_rank_a_closeup_orphan_face_above_a_distant_body(self):
        closeup = _obs(0, face=_box(0.5, 0.45, 0.22, 0.40), fscore=0.62)
        distant = _obs(0, person=_box(0.5, 0.5, 0.06, 0.28), pscore=0.95)

        assert observation_score(closeup) > observation_score(distant)

    def test_should_prefer_the_more_central_subject_when_sizes_match(self):
        centre = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8))
        edge = _obs(0, person=_box(0.05, 0.5, 0.2, 0.8))

        assert observation_score(centre) > observation_score(edge)

    def test_should_normalise_confidence_against_the_person_floor(self):
        strong = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8), pscore=1.0)
        marginal = _obs(
            0, person=_box(0.5, 0.5, 0.2, 0.8),
            pscore=DEFAULT_MIN_CONFIDENCE_PERSON,
        )

        assert observation_score(strong) > observation_score(marginal)

    def test_should_not_zero_a_score_on_low_confidence_alone(self):
        marginal = _obs(
            0, face=_box(0.5, 0.45, 0.18, 0.32),
            fscore=DEFAULT_MIN_CONFIDENCE_FACE,
        )

        assert observation_score(marginal) > 0.0

    def test_should_let_a_detected_face_rescue_a_marginal_person(self):
        with_face = _obs(
            0, person=_box(0.5, 0.5, 0.2, 0.8),
            face=_box(0.5, 0.25, 0.001, 0.001),
            pscore=0.81, fscore=0.74,
        )
        without = _obs(0, person=_box(0.5, 0.5, 0.2, 0.8), pscore=0.81)

        assert observation_score(with_face) > observation_score(without)

    def test_should_be_identical_across_every_resolution_tier(self):
        scores = []
        for width, height in TIERS.values():
            dets = [
                _person(0, 0.5, 0.5, 0.16, 0.64, size=(width, height)),
                _face(0, 0.5, 0.26, 0.05, 0.09, size=(width, height)),
            ]
            observations = observations_for_frame(0, dets, width, height)
            scores.append(round(observation_score(observations[0]), 9))

        assert len(set(scores)) == 1

    def test_should_raise_when_face_weight_is_below_one(self):
        with pytest.raises(SubjectError, match="face_weight"):
            observation_score(
                _obs(0, person=_box(0.5, 0.5, 0.2, 0.8)), face_weight=0.5
            )


class TestTrackScores:
    def test_should_penalise_a_track_present_in_few_samples(self):
        box = _box(0.5, 0.5, 0.2, 0.8)
        brief = Track(0, (_obs(0, person=box),))
        persistent = Track(
            1, tuple(_obs(f, person=box) for f in (0, 2, 4, 6, 8, 10))
        )

        brief_score = track_scores(brief, 6)[0]
        persistent_score = track_scores(persistent, 6)[0]

        assert persistent_score > brief_score

    def test_should_raise_on_a_non_positive_sample_count(self):
        with pytest.raises(SubjectError, match="sample_count"):
            track_scores(
                Track(0, (_obs(0, person=_box(0.5, 0.5, 0.2, 0.8)),)), 0
            )


def _dialogue(frames, left_size, right_size, size=(1920, 1080)):
    """Two people facing each other; left is `left_size` tall."""
    dets = []
    for frame in frames:
        dets.append(_person(frame, 0.35, 0.55, left_size / 4, left_size,
                            size=size))
        dets.append(_person(frame, 0.65, 0.55, right_size / 4, right_size,
                            size=size))
    return dets


class TestFollowHysteresis:
    def _setup(self, left_size, right_size, frames):
        dets = _dialogue(frames, left_size, right_size)
        by_frame = {
            f: observations_for_frame(
                f, [d for d in dets if d.frame == f], 1920, 1080
            )
            for f in frames
        }
        tracks = build_tracks(frames, by_frame, max_gap=3)
        scores = {
            t.track_id: track_scores(t, len(frames)) for t in tracks
        }
        return frames, tracks, scores

    def _ramp(self):
        """Left steady; right walks toward camera from frame 40 on."""
        frames = tuple(range(0, 240, 2))
        dets = []
        for frame in frames:
            dets.append(_person(frame, 0.35, 0.55, 0.15, 0.60))
            grown = 0.30 + max(0, frame - 40) * 0.01
            dets.append(_person(frame, 0.65, 0.55, 0.15, min(0.95, grown)))
        by_frame = {
            f: observations_for_frame(
                f, [d for d in dets if d.frame == f], 1920, 1080
            )
            for f in frames
        }
        tracks = build_tracks(frames, by_frame, max_gap=3)
        scores = {t.track_id: track_scores(t, len(frames)) for t in tracks}
        return frames, tracks, scores

    def test_should_adopt_a_leader_on_the_first_sample(self):
        frames = tuple(range(0, 48, 2))
        frames, tracks, scores = self._setup(0.60, 0.50, frames)

        followed, switches = follow(frames, tracks, scores, 24.0)

        assert followed[0] is not None
        assert switches == []

    def test_should_not_ping_pong_on_a_two_person_dialogue(self):
        frames = tuple(range(0, 240, 2))
        dets = []
        for index, frame in enumerate(frames):
            wobble = 0.01 if index % 2 == 0 else -0.01
            dets.append(_person(frame, 0.35, 0.55, 0.15, 0.60 + wobble))
            dets.append(_person(frame, 0.65, 0.55, 0.15, 0.60 - wobble))
        by_frame = {
            f: observations_for_frame(
                f, [d for d in dets if d.frame == f], 1920, 1080
            )
            for f in frames
        }
        tracks = build_tracks(frames, by_frame, max_gap=3)
        scores = {t.track_id: track_scores(t, len(frames)) for t in tracks}

        _followed, switches = follow(frames, tracks, scores, 24.0)

        assert switches == []

    def test_should_not_switch_when_the_margin_is_too_small(self):
        frames = tuple(range(0, 240, 2))
        frames, tracks, scores = self._setup(0.50, 0.55, frames)

        _followed, switches = follow(frames, tracks, scores, 24.0)

        assert switches == []

    def test_should_switch_after_a_big_margin_is_sustained(self):
        frames, tracks, scores = self._ramp()

        _followed, switches = follow(frames, tracks, scores, 24.0)

        assert switches == [122]

    def test_should_not_switch_before_the_margin_has_been_held(self):
        frames, tracks, scores = self._ramp()

        followed, _switches = follow(frames, tracks, scores, 24.0)

        at_120 = followed[frames.index(120)]
        at_122 = followed[frames.index(122)]
        assert at_120 != at_122

    def test_should_switch_immediately_when_hysteresis_sec_is_zero(self):
        frames, tracks, scores = self._ramp()

        _followed, switches = follow(
            frames, tracks, scores, 24.0, hysteresis_sec=0.0
        )

        assert switches == [98]

    def test_should_hold_the_incumbent_when_the_margin_is_raised(self):
        frames, tracks, scores = self._ramp()

        _f, default_switches = follow(frames, tracks, scores, 24.0)
        _g, strict_switches = follow(
            frames, tracks, scores, 24.0, hysteresis_margin=0.35
        )

        assert default_switches == [122]
        assert strict_switches == []

    def test_should_raise_on_a_margin_outside_the_unit_interval(self):
        frames, tracks, scores = self._ramp()

        with pytest.raises(SubjectError, match="hysteresis_margin"):
            follow(frames, tracks, scores, 24.0, hysteresis_margin=1.0)

    def test_should_raise_on_a_negative_hysteresis_sec(self):
        frames, tracks, scores = self._ramp()

        with pytest.raises(SubjectError, match="hysteresis_sec"):
            follow(frames, tracks, scores, 24.0, hysteresis_sec=-1.0)

    def test_should_switch_immediately_when_the_incumbent_leaves(self):
        frames = tuple(range(0, 40, 2))
        dets = []
        for frame in frames:
            if frame < 10:
                dets.append(_person(frame, 0.30, 0.55, 0.15, 0.70))
            dets.append(_person(frame, 0.70, 0.55, 0.10, 0.40))
        by_frame = {
            f: observations_for_frame(
                f, [d for d in dets if d.frame == f], 1920, 1080
            )
            for f in frames
        }
        tracks = build_tracks(frames, by_frame, max_gap=1)
        scores = {t.track_id: track_scores(t, len(frames)) for t in tracks}

        followed, switches = follow(
            frames, tracks, scores, 24.0, max_gap=1
        )

        assert len(switches) == 1
        assert followed[-1] is not None

    def test_should_record_a_switch_across_an_unpopulated_gap(self):
        frames = tuple(range(0, 40, 2))
        dets = []
        for frame in frames:
            if frame < 10:
                dets.append(_person(frame, 0.30, 0.55, 0.15, 0.70))
            if frame >= 20:
                dets.append(_person(frame, 0.70, 0.55, 0.15, 0.70))
        by_frame = {
            f: observations_for_frame(
                f, [d for d in dets if d.frame == f], 1920, 1080
            )
            for f in frames
        }
        tracks = build_tracks(frames, by_frame, max_gap=1)
        scores = {t.track_id: track_scores(t, len(frames)) for t in tracks}

        _followed, switches = follow(
            frames, tracks, scores, 24.0, max_gap=1
        )

        assert len(switches) == 1


def _detection_set(dets, sampled, stride=2):
    return DetectionSet(
        stride=stride, sampled_frames=tuple(sampled),
        detections=tuple(dets),
    )


class TestSelectSubjects:
    def test_should_return_one_result_per_shot_in_order(self):
        shots = [Shot(0, 0, 24, 24.0), Shot(1, 24, 48, 24.0)]
        sampled = tuple(range(0, 48, 2))
        dets = [_person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled]

        results = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )

        assert [r.shot_index for r in results] == [0, 1]

    def test_should_only_use_samples_inside_the_shot(self):
        shots = [Shot(0, 0, 24, 24.0), Shot(1, 24, 48, 24.0)]
        sampled = tuple(range(0, 48, 2))
        dets = [_person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled]

        results = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )

        assert results[0].sampled_frames == tuple(range(0, 24, 2))
        assert results[1].sampled_frames == tuple(range(24, 48, 2))

    def test_should_emit_one_sample_per_sampled_frame(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [_person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled[:4]]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        assert len(result.samples) == len(sampled)
        assert tuple(s.frame for s in result.samples) == sampled

    def test_should_reset_the_incumbent_at_every_shot_boundary(self):
        shots = [Shot(0, 0, 24, 24.0), Shot(1, 24, 48, 24.0)]
        sampled = tuple(range(0, 48, 2))
        dets = []
        for f in sampled:
            if f < 24:
                dets.append(_person(f, 0.30, 0.55, 0.15, 0.70))
            else:
                dets.append(_person(f, 0.70, 0.55, 0.15, 0.70))

        results = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )

        assert results[0].switch_frames == ()
        assert results[1].switch_frames == ()
        assert results[1].samples[0].cx == pytest.approx(0.70)

    def test_should_centre_crop_a_shot_that_found_nobody(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))

        result = select_subjects(
            shots, _detection_set([], sampled), _meta()
        )[0]

        assert result.has_subject is False
        assert all(s.kind == KIND_CENTER for s in result.samples)
        assert all(s.cx == 0.5 and s.cy == 0.5 for s in result.samples)
        assert all(s.track_id == NO_TRACK for s in result.samples)

    def test_should_hold_the_last_point_through_a_mid_shot_gap(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [
            _person(f, 0.30, 0.55, 0.15, 0.70)
            for f in sampled if f < 6 or f >= 18
        ]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        middle = [s for s in result.samples if 6 <= s.frame < 18]
        assert all(not s.observed for s in middle)
        assert all(s.kind == KIND_HELD for s in middle)
        assert all(s.cx == pytest.approx(0.30) for s in middle)

    def test_should_backfill_the_first_point_through_a_leading_gap(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [
            _person(f, 0.70, 0.55, 0.15, 0.70) for f in sampled if f >= 12
        ]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        leading = [s for s in result.samples if s.frame < 12]
        assert all(not s.observed for s in leading)
        assert all(s.cx == pytest.approx(0.70) for s in leading)

    def test_should_hold_the_last_point_through_a_trailing_gap(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [
            _person(f, 0.25, 0.55, 0.15, 0.70) for f in sampled if f < 8
        ]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        trailing = [s for s in result.samples if s.frame >= 8]
        assert all(s.cx == pytest.approx(0.25) for s in trailing)

    def test_should_report_the_observed_fraction(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [
            _person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled if f < 12
        ]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        assert result.observed_fraction == pytest.approx(0.5)

    def test_should_follow_an_orphan_face_when_no_body_is_found(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [_face(f, 0.62, 0.40, 0.20, 0.36) for f in sampled]

        result = select_subjects(
            shots, _detection_set(dets, sampled), _meta()
        )[0]

        assert result.has_subject is True
        assert all(s.kind == KIND_FACE for s in result.samples)
        assert result.samples[0].cx == pytest.approx(0.62)

    def test_should_raise_when_a_shot_has_no_sampled_frames(self):
        shots = [Shot(0, 0, 2, 24.0), Shot(1, 2, 24, 24.0)]
        sampled = tuple(range(2, 24, 4))
        dets = [_person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled]

        with pytest.raises(SubjectError, match="no sampled frame"):
            select_subjects(
                shots, _detection_set(dets, sampled, stride=4), _meta()
            )

    def test_should_raise_when_a_shots_sample_list_omits_its_start_frame(
        self,
    ):
        shots = [Shot(0, 0, 10, 24.0)]
        sampled = (2, 4, 6)
        dets = [_person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled]

        with pytest.raises(SubjectError, match="start_frame"):
            select_subjects(
                shots, _detection_set(dets, sampled, stride=2), _meta()
            )

    def test_should_raise_when_a_detection_is_not_in_sampled_frames(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [_person(3, 0.5, 0.5, 0.15, 0.6)]

        with pytest.raises(SubjectError, match="not in sampled_frames"):
            select_subjects(shots, _detection_set(dets, sampled), _meta())

    def test_should_raise_when_sampled_frames_go_backwards(self):
        shots = [Shot(0, 0, 24, 24.0)]
        bad = DetectionSet(stride=2, sampled_frames=(0, 4, 2),
                           detections=())

        with pytest.raises(SubjectError, match="strictly increasing"):
            select_subjects(shots, bad, _meta())

    def test_should_raise_when_a_sample_falls_outside_the_shot_list(self):
        shots = [Shot(0, 0, 10, 24.0)]
        sampled = (0, 2, 4, 12)

        with pytest.raises(SubjectError, match="outside the shot list"):
            select_subjects(shots, _detection_set([], sampled), _meta())

    def test_should_raise_on_an_unknown_detection_class(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = (0, 2)
        bad = Detection(0, "dog", 0.0, 0.0, 10.0, 10.0, 0.9)

        with pytest.raises(SubjectError, match="unknown detection class"):
            select_subjects(shots, _detection_set([bad], sampled), _meta())

    def test_should_raise_when_the_shot_list_is_empty(self):
        with pytest.raises(SubjectError, match="must not be empty"):
            select_subjects([], _detection_set([], (0,)), _meta())

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_should_pick_the_same_subject_at_every_resolution(self, tier):
        width, height = TIERS[tier]
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = []
        for f in sampled:
            dets.append(
                _person(f, 0.30, 0.55, 0.10, 0.60, size=(width, height))
            )
            dets.append(
                _person(f, 0.70, 0.55, 0.06, 0.30, size=(width, height))
            )

        result = select_subjects(
            shots, _detection_set(dets, sampled),
            _meta(width=width, height=height),
        )[0]

        assert result.samples[0].cx == pytest.approx(0.30, abs=1e-6)
        assert result.has_subject is True

    def test_should_use_display_geometry_on_a_rotated_source(self):
        shots = [Shot(0, 0, 24, 24.0)]
        sampled = tuple(range(0, 24, 2))
        dets = [
            Detection(f, "person", 0.0, 0.0, 540.0, 480.0, 0.95)
            for f in sampled
        ]
        meta = _meta(width=1920, height=1080, rotation=90)

        result = select_subjects(
            shots, _detection_set(dets, sampled), meta
        )[0]

        assert result.samples[0].cx == pytest.approx(540.0 / 2 / 1080)


class TestWeightValidation:
    def _args(self):
        return (
            [Shot(0, 0, 8, 24.0)],
            _detection_set([], (0, 2, 4, 6)),
            _meta(),
        )

    def test_should_raise_on_a_centrality_weight_of_one(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="centrality_weight"):
            select_subjects(shots, dets, meta, centrality_weight=1.0)

    def test_should_raise_on_a_negative_confidence_weight(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="confidence_weight"):
            select_subjects(shots, dets, meta, confidence_weight=-0.1)

    def test_should_raise_on_a_persistence_weight_of_one(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="persistence_weight"):
            select_subjects(shots, dets, meta, persistence_weight=1.0)

    def test_should_raise_on_a_zero_iou_match_min(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="iou_match_min"):
            select_subjects(shots, dets, meta, iou_match_min=0.0)

    def test_should_raise_on_a_face_weight_below_one(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="face_weight"):
            select_subjects(shots, dets, meta, face_weight=0.5)

    def test_should_raise_on_a_hysteresis_margin_outside_the_unit_interval(
        self,
    ):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="hysteresis_margin"):
            select_subjects(shots, dets, meta, hysteresis_margin=1.0)

    def test_should_raise_on_a_negative_hysteresis_sec(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="hysteresis_sec"):
            select_subjects(shots, dets, meta, hysteresis_sec=-1.0)

    def test_should_raise_on_a_negative_track_gap_sec(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="track_gap_sec"):
            select_subjects(shots, dets, meta, track_gap_sec=-1.0)

    def test_should_raise_on_a_min_confidence_person_of_one(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="min_confidence_person"):
            select_subjects(shots, dets, meta, min_confidence_person=1.0)

    def test_should_raise_on_a_zero_min_confidence_face(self):
        shots, dets, meta = self._args()

        with pytest.raises(SubjectError, match="min_confidence_face"):
            select_subjects(shots, dets, meta, min_confidence_face=0.0)


class TestSubjectSummary:
    def test_should_count_shots_switches_and_observed_samples(self):
        shots = [Shot(0, 0, 8, 24.0), Shot(1, 8, 16, 24.0)]
        sampled = tuple(range(0, 16, 2))
        dets = [
            _person(f, 0.5, 0.5, 0.15, 0.6) for f in sampled if f < 8
        ]

        line = subject_summary(
            select_subjects(shots, _detection_set(dets, sampled), _meta())
        )

        assert "2 shot(s)" in line
        assert "1 with no subject" in line
        assert "0 switch(es)" in line
        assert "50.0% of samples observed" in line


class TestSubjectToJson:
    def test_should_round_coordinates_to_six_places(self):
        shots = [Shot(0, 0, 8, 24.0)]
        sampled = (0, 2, 4, 6)
        dets = [_person(f, 1.0 / 3, 0.5, 0.15, 0.6) for f in sampled]

        payload = subject_to_json(
            select_subjects(shots, _detection_set(dets, sampled), _meta())
        )

        assert payload[0]["samples"][0]["cx"] == round(1.0 / 3, 6)

    def test_should_mark_unobserved_samples(self):
        shots = [Shot(0, 0, 8, 24.0)]
        sampled = (0, 2, 4, 6)

        payload = subject_to_json(
            select_subjects(shots, _detection_set([], sampled), _meta())
        )

        assert all(
            s["observed"] is False for s in payload[0]["samples"]
        )
        assert payload[0]["has_subject"] is False


class TestObservationInvariant:
    def test_should_refuse_an_observation_with_neither_box(self):
        """anchor and point need at least one box to exist."""
        with pytest.raises(SubjectError, match="neither"):
            Observation(frame=0, person=None, face=None,
                        person_score=0.0, face_score=0.0)
