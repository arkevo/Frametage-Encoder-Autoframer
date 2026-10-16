"""Tests for ftea.reframe.detect -- strided person + face detection.

Every test in this file is GPU-free and video-free: the decoder and both
detectors are injected Callables, and frames are numpy zeros.

The one @pytest.mark.gpu test is latent: it runs only under the reframe
virtualenv, after a developer has added pytest to it by hand
(`pip install pytest`). That is the one sanctioned developer-only
exception to the install rule: pytest pulls in no torch, so it cannot
displace the CUDA build.
"""

import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from ftea import paths
from ftea.reframe.detect import (
    CLS_FACE,
    CLS_PERSON,
    DEFAULT_DEVICE,
    DEFAULT_FACE_MODEL_NAME,
    DEFAULT_MIN_CONFIDENCE_FACE,
    DEFAULT_MIN_CONFIDENCE_PERSON,
    Detection,
    DetectionSet,
    DetectError,
    clamp_box,
    decode_frames,
    detect_from_settings,
    detect_video,
    detections_to_json,
    detector_input_size,
    face_model_path,
    load_face_detector,
    load_person_detector,
    main,
    records_for_frame,
    run_detection,
    sample_frames,
    scale_box,
    torch_home,
)
from ftea.reframe.errors import ReframeError
from ftea.reframe.shots import Shot


def _shot(index, start, end, fps=24.0):
    return Shot(index=index, start_frame=start, end_frame=end, fps=fps)


def _det(frame=0, cls=CLS_PERSON, box=(0.0, 0.0, 10.0, 20.0), score=0.9):
    return Detection(frame=frame, cls=cls, x1=box[0], y1=box[1],
                     x2=box[2], y2=box[3], score=score)


class TestDetectErrorLineage:
    def test_should_be_catchable_as_a_reframe_error(self):
        with pytest.raises(ReframeError):
            raise DetectError("boom")


class TestDetectionRecord:
    def test_should_expose_exactly_the_seven_pinned_fields(self):
        """This list is pinned; properties must not become fields."""
        assert [f.name for f in fields(Detection)] == [
            "frame", "cls", "x1", "y1", "x2", "y2", "score",
        ]

    def test_should_be_frozen(self):
        det = _det()
        with pytest.raises(Exception):
            det.frame = 3

    def test_should_derive_size_and_centre(self):
        det = _det(box=(10.0, 20.0, 30.0, 60.0))
        assert det.width == 20.0
        assert det.height == 40.0
        assert det.area == 800.0
        assert det.cx == 20.0
        assert det.cy == 40.0


class TestDetectionSetQueries:
    def _set(self):
        return DetectionSet(
            stride=2,
            sampled_frames=(0, 2, 4, 6),
            detections=(
                _det(frame=0, cls=CLS_FACE, score=0.5),
                _det(frame=0, cls=CLS_PERSON, score=0.9),
                _det(frame=4, cls=CLS_PERSON, score=0.95),
            ),
        )

    def test_should_expose_exactly_the_three_pinned_fields(self):
        assert [f.name for f in fields(DetectionSet)] == [
            "stride", "sampled_frames", "detections",
        ]

    def test_should_select_by_frame(self):
        assert len(self._set().for_frame(0)) == 2
        assert self._set().for_frame(2) == ()

    def test_should_select_by_class(self):
        assert len(self._set().of_class(CLS_PERSON)) == 2
        assert len(self._set().of_class(CLS_FACE)) == 1

    def test_should_select_a_half_open_frame_range(self):
        """Matches Shot's [start, end) semantics exactly."""
        assert len(self._set().in_range(0, 4)) == 2
        assert len(self._set().in_range(0, 5)) == 3
        assert self._set().in_range(1, 4) == ()

    def test_should_report_sampled_frames_in_a_half_open_range(self):
        assert self._set().sampled_in_range(0, 4) == (0, 2)
        assert self._set().sampled_in_range(4, 8) == (4, 6)

    def test_should_count_frames_that_found_something(self):
        assert self._set().frames_with_detections == 2


class TestSampleFrames:
    def test_should_stride_within_a_single_shot(self):
        assert sample_frames([_shot(0, 0, 10)], 3) == (0, 3, 6, 9)

    def test_should_reset_the_stride_phase_at_every_cut(self):
        """Shot 1 starts its own phase at 10, not at 9 + 3."""
        shots = [_shot(0, 0, 10), _shot(1, 10, 20)]
        assert sample_frames(shots, 3) == (0, 3, 6, 9, 10, 13, 16, 19)

    def test_should_always_sample_the_first_frame_of_every_shot(self):
        shots = [_shot(0, 0, 100), _shot(1, 100, 101), _shot(2, 101, 200)]
        sampled = sample_frames(shots, 24)
        assert 100 in sampled
        assert 101 in sampled

    def test_should_sample_a_one_frame_shot(self):
        assert sample_frames([_shot(0, 7, 8)], 99) == (7,)

    def test_should_return_strictly_increasing_indices(self):
        shots = [_shot(0, 0, 5), _shot(1, 5, 9), _shot(2, 9, 13)]
        sampled = sample_frames(shots, 2)
        assert list(sampled) == sorted(set(sampled))

    def test_should_sample_every_frame_when_stride_is_one(self):
        assert sample_frames([_shot(0, 0, 4)], 1) == (0, 1, 2, 3)

    def test_should_raise_when_stride_is_below_one(self):
        with pytest.raises(DetectError, match="stride"):
            sample_frames([_shot(0, 0, 10)], 0)

    def test_should_raise_when_shots_are_empty(self):
        with pytest.raises(DetectError, match="non-empty"):
            sample_frames([], 3)

    def test_should_raise_when_shots_are_not_contiguous(self):
        with pytest.raises(DetectError, match="contiguous"):
            sample_frames([_shot(0, 0, 10), _shot(1, 12, 20)], 3)

    def test_should_raise_when_a_shot_is_empty(self):
        with pytest.raises(DetectError, match="empty"):
            sample_frames([_shot(0, 0, 0)], 3)


class TestDetectorInputSize:
    def test_should_use_native_resolution_when_long_side_is_zero(self):
        assert detector_input_size(3840, 2160, 0) == (3840, 2160)
        assert detector_input_size(1280, 720, 0) == (1280, 720)

    @pytest.mark.parametrize("width,height", [
        (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160),
        (2560, 1080), (1920, 816),
    ])
    def test_should_pass_every_landscape_geometry_through_untouched(
        self, width, height,
    ):
        """One code path, no per-resolution tiers."""
        assert detector_input_size(width, height, 0) == (width, height)

    def test_should_downscale_by_the_long_side(self):
        assert detector_input_size(3840, 2160, 1920) == (1920, 1080)
        assert detector_input_size(2560, 1080, 1280) == (1280, 540)

    def test_should_never_upscale_a_smaller_source(self):
        assert detector_input_size(1280, 720, 3840) == (1280, 720)

    def test_should_raise_below_the_detector_input_floor(self):
        with pytest.raises(DetectError, match="detect_long_side"):
            detector_input_size(3840, 2160, 200)

    def test_should_raise_on_a_negative_long_side(self):
        with pytest.raises(DetectError, match="detect_long_side"):
            detector_input_size(3840, 2160, -1)

    def test_should_raise_on_non_positive_geometry(self):
        with pytest.raises(DetectError, match="display geometry"):
            detector_input_size(0, 1080, 0)


class TestScaleBox:
    def test_should_be_identity_when_sizes_match(self):
        box = (10.5, 20.25, 30.0, 40.0, 0.9)
        assert scale_box(box, from_size=(1920, 1080),
                         to_size=(1920, 1080)) == box

    def test_should_map_a_detector_box_back_to_source_pixels(self):
        box = (100.0, 50.0, 200.0, 150.0, 0.7)
        scaled = scale_box(box, from_size=(1920, 1080),
                           to_size=(3840, 2160))
        assert scaled == (200.0, 100.0, 400.0, 300.0, 0.7)

    def test_should_use_independent_axis_ratios(self):
        """Rounded detector sizes make the two axes differ slightly."""
        box = (0.0, 0.0, 100.0, 100.0, 0.5)
        scaled = scale_box(box, from_size=(1280, 540),
                           to_size=(2560, 1080))
        assert scaled[2] == 200.0
        assert scaled[3] == 200.0

    def test_should_preserve_the_score_untouched(self):
        box = (1.0, 2.0, 3.0, 4.0, 0.4321)
        assert scale_box(box, from_size=(640, 360),
                         to_size=(1280, 720))[4] == 0.4321


class TestClampBox:
    def test_should_leave_an_inside_box_untouched(self):
        box = (10.0, 20.0, 30.0, 40.0, 0.9)
        assert clamp_box(box, 1920, 1080) == box

    def test_should_clamp_a_box_past_the_right_and_bottom_edges(self):
        box = (1900.0, 1000.0, 2100.0, 1200.0, 0.9)
        assert clamp_box(box, 1920, 1080) == (1900.0, 1000.0,
                                              1920.0, 1080.0, 0.9)

    def test_should_clamp_negative_origins_to_zero(self):
        box = (-50.0, -20.0, 100.0, 100.0, 0.9)
        assert clamp_box(box, 1920, 1080) == (0.0, 0.0, 100.0, 100.0, 0.9)

    def test_should_normalise_an_inverted_box(self):
        box = (300.0, 400.0, 100.0, 200.0, 0.9)
        assert clamp_box(box, 1920, 1080) == (100.0, 200.0,
                                              300.0, 400.0, 0.9)

    def test_should_drop_a_box_entirely_left_of_the_frame(self):
        assert clamp_box((-200.0, 10.0, -100.0, 90.0, 0.9), 1920, 1080) is None

    def test_should_drop_a_box_entirely_below_the_frame(self):
        assert clamp_box((10.0, 1200.0, 90.0, 1400.0, 0.9), 1920, 1080) is None

    def test_should_drop_a_sub_pixel_box(self):
        assert clamp_box((10.0, 10.0, 10.4, 40.0, 0.9), 1920, 1080) is None


class TestRecordsForFrame:
    def _call(self, persons, faces, **kwargs):
        options = dict(
            source_size=(1920, 1080),
            detect_size=(1920, 1080),
            min_confidence_person=DEFAULT_MIN_CONFIDENCE_PERSON,
            min_confidence_face=DEFAULT_MIN_CONFIDENCE_FACE,
        )
        options.update(kwargs)
        return records_for_frame(7, persons, faces, **options)

    def test_should_label_classes_exactly_person_and_face(self):
        out = self._call([(0.0, 0.0, 50.0, 90.0, 0.99)],
                         [(10.0, 10.0, 30.0, 30.0, 0.5)])
        assert sorted(d.cls for d in out) == [CLS_FACE, CLS_PERSON]
        assert {d.cls for d in out} <= {"person", "face"}

    def test_should_stamp_the_frame_index_on_every_record(self):
        out = self._call([(0.0, 0.0, 50.0, 90.0, 0.99)],
                         [(10.0, 10.0, 30.0, 30.0, 0.5)])
        assert {d.frame for d in out} == {7}

    def test_should_keep_a_person_above_the_person_floor(self):
        out = self._call([(0.0, 0.0, 50.0, 90.0, 0.81)], [])
        assert len(out) == 1

    def test_should_drop_a_person_below_the_person_floor(self):
        """Lace and curtains score ~0.60 with no people in frame."""
        out = self._call([(0.0, 0.0, 50.0, 90.0, 0.60)], [])
        assert out == []

    def test_should_keep_a_face_above_the_face_floor(self):
        out = self._call([], [(10.0, 10.0, 30.0, 30.0, 0.43)])
        assert len(out) == 1

    def test_should_drop_a_face_below_the_face_floor(self):
        out = self._call([], [(10.0, 10.0, 30.0, 30.0, 0.30)])
        assert out == []

    def test_should_not_apply_the_person_floor_to_faces(self):
        """Regression test: one shared floor cannot serve both classes.

        A 0.43 face is real (measured on real footage) while a
        0.60 person is fabric. A shared floor cannot keep one and drop
        the other; separate floors must.
        """
        out = self._call([(0.0, 0.0, 50.0, 90.0, 0.60)],
                         [(10.0, 10.0, 30.0, 30.0, 0.43)])
        assert [d.cls for d in out] == [CLS_FACE]
        assert out[0].score == 0.43

    def test_should_clamp_every_emitted_box_into_frame(self):
        out = self._call([(-100.0, -50.0, 2500.0, 1500.0, 0.99)], [])
        det = out[0]
        assert (det.x1, det.y1, det.x2, det.y2) == (0.0, 0.0, 1920.0, 1080.0)

    def test_should_drop_a_box_that_clamps_to_nothing(self):
        out = self._call([(-300.0, 10.0, -200.0, 90.0, 0.99)], [])
        assert out == []

    def test_should_map_boxes_back_to_source_pixels_when_downscaled(self):
        out = self._call([(50.0, 25.0, 100.0, 75.0, 0.99)], [],
                         source_size=(3840, 2160),
                         detect_size=(1920, 1080))
        det = out[0]
        assert (det.x1, det.y1, det.x2, det.y2) == (100.0, 50.0,
                                                    200.0, 150.0)

    def test_should_clamp_after_scaling_not_before(self):
        """A box legal in detector space can exceed the source frame."""
        out = self._call([(0.0, 0.0, 1920.0, 1080.0, 0.99)], [],
                         source_size=(3840, 2160),
                         detect_size=(1920, 1080))
        det = out[0]
        assert det.x2 == 3840.0
        assert det.y2 == 2160.0

    def test_should_return_an_empty_list_when_nothing_survives(self):
        assert self._call([], []) == []

    def test_should_not_associate_faces_with_bodies(self):
        """34.8% of faces have no containing person box.

        detect.py emits both classes flat; association is subject.py's,
        so a Detection carries no parent, track or group reference.
        """
        out = self._call([(0.0, 0.0, 500.0, 900.0, 0.99)],
                         [(100.0, 50.0, 200.0, 150.0, 0.5)])
        assert len(out) == 2
        for det in out:
            assert not hasattr(det, "person_index")
            assert not hasattr(det, "track_id")


def _meta(width=1920, height=1080, fps=24.0, duration_s=10.0,
          path="clip.mp4"):
    """A SourceMeta without touching ffprobe."""
    from ftea.reframe.probe import SourceMeta
    return SourceMeta(path=Path(path), width=width, height=height,
                      rotation=0, fps=fps, duration_s=duration_s,
                      has_audio=True)


def _frame(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeDecoder:
    """Yields blank frames for exactly the indices asked for.

    `skip` lets a test simulate a source that ends early; `sizes` lets a
    test simulate a decoder handing back the wrong geometry.
    """

    def __init__(self, width, height, *, skip=(), size_override=None):
        self.width = width
        self.height = height
        self.skip = set(skip)
        self.size_override = size_override
        self.calls = []

    def __call__(self, source, wanted):
        self.calls.append((source, frozenset(wanted)))
        for index in sorted(wanted):
            if index in self.skip:
                continue
            size = self.size_override or (self.width, self.height)
            yield index, _frame(size[0], size[1])


class FakeDetector:
    """Returns canned boxes per frame, in call order.

    `per_frame` maps a 0-based call counter to a list of boxes, so a
    test can say "the third frame the detector sees has two people".
    """

    def __init__(self, per_frame=None, default=()):
        self.per_frame = per_frame or {}
        self.default = list(default)
        self.seen = 0
        self.batch_sizes = []

    def __call__(self, frames):
        self.batch_sizes.append(len(frames))
        out = []
        for _ in frames:
            out.append(list(self.per_frame.get(self.seen, self.default)))
            self.seen += 1
        return out


PERSON_BOX = (100.0, 100.0, 300.0, 700.0, 0.95)
FACE_BOX = (150.0, 120.0, 230.0, 200.0, 0.5)


class TestRunDetection:
    def _run(self, meta=None, shots=None, decoder=None,
             person=None, face=None, **kwargs):
        meta = meta or _meta()
        shots = shots or [_shot(0, 0, 12)]
        decoder = decoder or FakeDecoder(meta.display_width,
                                         meta.display_height)
        person = person or FakeDetector(default=[PERSON_BOX])
        face = face or FakeDetector(default=[FACE_BOX])
        options = dict(stride=3, batch_size=2, decoder_fn=decoder,
                       person_fn=person, face_fn=face)
        options.update(kwargs)
        return run_detection(meta, shots, **options)

    def test_should_sample_every_shot_and_emit_a_detection_set(self):
        dset, stats = self._run()
        assert dset.stride == 3
        assert dset.sampled_frames == (0, 3, 6, 9)
        assert stats.sampled == 4
        assert stats.decoded == 4

    def test_should_emit_both_classes_for_every_sampled_frame(self):
        dset, _ = self._run()
        assert len(dset.detections) == 8
        assert len(dset.of_class(CLS_PERSON)) == 4
        assert len(dset.of_class(CLS_FACE)) == 4

    def test_should_record_a_sampled_frame_that_found_nothing(self):
        """Sampled-and-empty is not never-decoded."""
        dset, _ = self._run(person=FakeDetector(default=[]),
                            face=FakeDetector(default=[]))
        assert dset.sampled_frames == (0, 3, 6, 9)
        assert dset.detections == ()

    def test_should_never_reference_a_frame_it_did_not_decode(self):
        dset, _ = self._run()
        assert {d.frame for d in dset.detections} <= set(dset.sampled_frames)

    def test_should_pass_the_exact_sample_set_to_the_decoder(self):
        decoder = FakeDecoder(1920, 1080)
        self._run(decoder=decoder)
        assert decoder.calls[0][1] == frozenset({0, 3, 6, 9})

    def test_should_batch_at_the_configured_size(self):
        person = FakeDetector(default=[PERSON_BOX])
        self._run(person=person, batch_size=2)
        assert person.batch_sizes == [2, 2]

    def test_should_flush_a_short_tail_batch(self):
        person = FakeDetector(default=[PERSON_BOX])
        self._run(shots=[_shot(0, 0, 9)], person=person, batch_size=2)
        assert person.batch_sizes == [2, 1]

    def test_should_run_the_face_pass_on_frames_with_no_person(self):
        """Regression: running faces only where a person was found
        lost 47 of 204 faces, because faces occur where the person
        detector finds nobody."""
        face = FakeDetector(default=[FACE_BOX])
        dset, _ = self._run(person=FakeDetector(default=[]), face=face)
        assert face.seen == 4
        assert len(dset.of_class(CLS_FACE)) == 4

    def test_should_clamp_every_emitted_box_into_frame(self):
        person = FakeDetector(default=[(-50.0, -50.0, 5000.0, 5000.0, 0.9)])
        dset, _ = self._run(person=person)
        for det in dset.detections:
            assert 0.0 <= det.x1 < det.x2 <= 1920.0
            assert 0.0 <= det.y1 < det.y2 <= 1080.0

    def test_should_label_classes_exactly_person_and_face(self):
        dset, _ = self._run()
        assert {d.cls for d in dset.detections} == {"person", "face"}

    def test_should_order_detections_deterministically(self):
        """Threading must not leak into the emitted order."""
        first, _ = self._run()
        second, _ = self._run()
        assert first.detections == second.detections
        keys = [(d.frame, d.cls) for d in first.detections]
        assert keys == sorted(keys)

    def test_should_tolerate_a_two_frame_shortfall_at_the_tail(self):
        decoder = FakeDecoder(1920, 1080, skip=(9,))
        dset, stats = self._run(decoder=decoder)
        assert dset.sampled_frames == (0, 3, 6)
        assert stats.decoded == 3
        assert stats.sampled == 4

    def test_should_raise_when_the_decoder_stops_early_beyond_tolerance(self):
        decoder = FakeDecoder(1920, 1080, skip=(3, 6, 9))
        with pytest.raises(DetectError, match="ended early"):
            self._run(decoder=decoder)

    def test_should_raise_a_mid_clip_message_when_the_gap_is_not_a_tail(
        self,
    ):
        """A non-suffix gap is a different failure, and
        the message must say so rather than claiming the source ended."""
        decoder = FakeDecoder(1920, 1080, skip=(0, 3, 9))
        with pytest.raises(DetectError, match="mid-clip"):
            self._run(decoder=decoder)

    def test_should_log_a_mid_clip_warning_distinctly_when_tolerated(
        self, capsys,
    ):
        decoder = FakeDecoder(1920, 1080, skip=(3,))
        self._run(decoder=decoder)
        assert "mid-clip" in capsys.readouterr().err

    def test_should_raise_when_a_shot_has_no_sampled_frame(self):
        shots = [_shot(0, 0, 6), _shot(1, 6, 12)]
        decoder = FakeDecoder(1920, 1080, skip=(6, 9))
        with pytest.raises(DetectError, match="shot 1"):
            self._run(shots=shots, decoder=decoder, missing_tolerance=99)

    def test_should_raise_when_frame_geometry_disagrees_with_meta(self):
        decoder = FakeDecoder(1920, 1080, size_override=(1280, 720))
        with pytest.raises(DetectError, match="display geometry"):
            self._run(decoder=decoder)

    def test_should_raise_when_a_detector_returns_the_wrong_list_count(self):
        class ShortDetector:
            def __call__(self, frames):
                return [[]]

        with pytest.raises(DetectError, match="one list per frame"):
            self._run(person=ShortDetector())

    def test_should_wrap_a_decoder_exception_as_a_detect_error(self):
        class BoomDecoder:
            def __call__(self, source, wanted):
                raise OSError("disk went away")
                yield  # pragma: no cover - makes it a generator

        with pytest.raises(DetectError, match="decode failed"):
            self._run(decoder=BoomDecoder())

    def test_should_wrap_a_cuda_oom_with_an_actionable_message(self):
        class BoomDetector:
            def __call__(self, frames):
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 2.00 GiB")

        with pytest.raises(DetectError, match="REFRAME_BATCH_SIZE"):
            self._run(person=BoomDetector())

    def test_should_propagate_a_non_oom_detector_exception_untouched(self):
        class BoomDetector:
            def __call__(self, frames):
                raise ValueError("bug")

        with pytest.raises(ValueError, match="bug"):
            self._run(person=BoomDetector())

    def test_should_name_the_configured_batch_size_on_a_tail_batch(self):
        """The short tail batch must not be reported as the setting."""
        class TailOom:
            def __call__(self, frames):
                if len(frames) < 3:
                    raise RuntimeError("CUDA out of memory")
                return [[] for _ in frames]

        with pytest.raises(DetectError) as excinfo:
            self._run(shots=[_shot(0, 0, 12)], person=TailOom(),
                      batch_size=3)

        message = str(excinfo.value)
        assert "REFRAME_BATCH_SIZE (currently 3)" in message
        assert "batch of 1 frame(s)" in message

    def test_should_wrap_a_face_oom_with_a_cpu_remedy(self):
        """YuNet runs on the CPU, so moving detection to the CPU is no
        remedy for it running out of memory."""
        class FaceOom:
            def __call__(self, frames):
                raise MemoryError("Failed to allocate 99 bytes")

        with pytest.raises(DetectError) as excinfo:
            self._run(face=FaceOom())

        message = str(excinfo.value)
        assert "face detection failed" in message
        assert "REFRAME_BATCH_SIZE (currently 2)" in message
        assert "REFRAME_DETECT_LONG_SIDE" in message
        assert "REFRAME_DEVICE" not in message

    def test_should_propagate_a_non_oom_face_exception_untouched(self):
        class FaceBug:
            def __call__(self, frames):
                raise ValueError("face bug")

        with pytest.raises(ValueError, match="face bug"):
            self._run(face=FaceBug())

    def test_should_raise_when_batch_size_is_below_one(self):
        with pytest.raises(DetectError, match="batch_size"):
            self._run(batch_size=0)

    @pytest.mark.parametrize("width,height", [
        (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160),
        (2560, 1080), (1920, 816),
    ])
    def test_should_work_at_every_landscape_geometry(self, width, height):
        """One code path for every resolution, no per-tier branch."""
        meta = _meta(width=width, height=height)
        dset, stats = self._run(meta=meta)
        assert stats.detect_width == width
        assert stats.detect_height == height
        assert len(dset.detections) == 8
        for det in dset.detections:
            assert 0.0 <= det.x2 <= float(width)
            assert 0.0 <= det.y2 <= float(height)


class TestDetectStats:
    def test_should_report_per_frame_cost(self):
        _, stats = TestRunDetection()._run()
        assert stats.elapsed_s > 0.0
        assert stats.ms_per_sampled_frame > 0.0
        assert stats.batches == 2

    def test_should_report_a_realtime_ratio_against_clip_length(self):
        _, stats = TestRunDetection()._run(meta=_meta(duration_s=10.0))
        assert stats.realtime_ratio == pytest.approx(
            10.0 / stats.elapsed_s)

    def test_should_report_zero_ratio_for_a_zero_length_clip(self):
        _, stats = TestRunDetection()._run(meta=_meta(duration_s=0.0))
        assert stats.realtime_ratio == 0.0


class TestDetectVideo:
    def test_should_return_only_the_detection_set(self):
        meta = _meta()
        dset = detect_video(
            meta, [_shot(0, 0, 12)], stride=3, batch_size=2,
            decoder_fn=FakeDecoder(1920, 1080),
            person_fn=FakeDetector(default=[PERSON_BOX]),
            face_fn=FakeDetector(default=[FACE_BOX]),
        )
        assert isinstance(dset, DetectionSet)
        assert not isinstance(dset, tuple)


class TestFaceModelPath:
    def test_should_default_to_the_yunet_file_under_models_dir(self):
        assert face_model_path(Path("models"), "") == \
            Path("models") / DEFAULT_FACE_MODEL_NAME

    def test_should_honour_an_explicit_override(self):
        assert face_model_path(Path("models"), "C:/x/y.onnx") == \
            Path("C:/x/y.onnx")


class TestRealDecoderErrors:
    def test_should_raise_when_the_source_cannot_be_opened(self, tmp_path):
        """VideoCapture fails SILENTLY -- isOpened() is the check."""
        pytest.importorskip("cv2")
        missing = tmp_path / "nope.mp4"
        with pytest.raises(DetectError, match="cannot open"):
            list(decode_frames(missing, frozenset({0, 1})))

    def test_should_yield_nothing_when_no_frame_is_wanted(self, tmp_path):
        assert list(decode_frames(tmp_path / "nope.mp4", frozenset())) == []


class TestRealLoaderErrors:
    def test_should_raise_a_named_error_when_the_face_model_is_absent(
        self, tmp_path,
    ):
        pytest.importorskip("cv2")
        with pytest.raises(DetectError, match="install.sh"):
            load_face_detector(model_path=tmp_path / "absent.onnx",
                               input_size=(640, 360))

    def test_should_raise_a_named_error_when_person_weights_are_absent(
        self, tmp_path,
    ):
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        with pytest.raises(DetectError, match="REFRAME_PERSON_MODEL"):
            load_person_detector(device="cpu",
                                 weights_path=str(tmp_path / "absent.pth"))


class TestRealDefaultsAreWired:
    """Injectable seams hide a disconnected default; assert it directly.

    Behavioural, not textual: a substring check on the
    source only proves the default is *mentioned*, not that it is
    reachable or that the right keyword reaches the right loader (e.g.
    the face floor landing on the person loader would sail past a
    substring check). This rebinds the plain module globals
    `run_detection` resolves at call time -- the same seam already
    injected everywhere else in this module via kwargs, not a
    `unittest.mock.patch` on a torch model.
    """

    def test_should_wire_the_real_decoder_and_loaders_by_default(
        self, monkeypatch,
    ):
        import ftea.reframe.detect as module

        meta = _meta()
        decode_calls = []
        person_calls = []
        face_calls = []

        def recorder(source, wanted):
            decode_calls.append((source, frozenset(wanted)))
            for index in sorted(wanted):
                yield index, _frame(meta.display_width,
                                    meta.display_height)

        def fake_person_loader(*, device, models_dir, weights_path,
                               score_floor):
            person_calls.append(
                (device, models_dir, weights_path, score_floor))
            return FakeDetector(default=[PERSON_BOX])

        def fake_face_loader(*, model_path, input_size, score_floor):
            face_calls.append((model_path, input_size, score_floor))
            return FakeDetector(default=[FACE_BOX])

        monkeypatch.setattr(module, "decode_frames", recorder)
        monkeypatch.setattr(module, "load_person_detector",
                            fake_person_loader)
        monkeypatch.setattr(module, "load_face_detector",
                            fake_face_loader)

        run_detection(meta, [_shot(0, 0, 12)], stride=3, batch_size=2)

        assert decode_calls == [(meta.path, frozenset({0, 3, 6, 9}))]
        device, models_dir, weights_path, score_floor = person_calls[0]
        assert score_floor == DEFAULT_MIN_CONFIDENCE_PERSON
        assert device == DEFAULT_DEVICE
        assert models_dir == paths.models_dir()
        assert weights_path == ""
        model_path, input_size, face_floor = face_calls[0]
        assert model_path == paths.models_dir() / DEFAULT_FACE_MODEL_NAME
        assert input_size == (meta.display_width, meta.display_height)
        assert face_floor == DEFAULT_MIN_CONFIDENCE_FACE


@pytest.mark.gpu
class TestGpuEndToEnd:
    """Latent by design: it needs a CUDA GPU, the real weights, and the
    reframe virtualenv with pytest added by hand (`pip install pytest`).
    That is the one sanctioned developer-only exception to the install
    rule: pytest pulls in no torch, so it cannot displace the CUDA build.
    It skips everywhere else. Point FTEA_TEST_CLIP at a landscape clip
    that shows people."""

    def _guard(self) -> Path:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        if Path(sys.prefix).resolve() != paths.reframe_venv_dir().resolve():
            pytest.skip("not running under the reframe virtualenv")
        weights = paths.models_dir() / DEFAULT_FACE_MODEL_NAME
        if not weights.exists():
            pytest.skip("YuNet weights absent; run install.sh")
        clip = os.environ.get("FTEA_TEST_CLIP", "")
        if not clip or not Path(clip).is_file():
            pytest.skip("set FTEA_TEST_CLIP to a landscape clip with people")
        return Path(clip)

    def test_should_detect_people_on_a_real_clip(self, real_ftea_home):
        clip = self._guard()
        from ftea.reframe.probe import probe_source
        from ftea.reframe.shots import detect_shots

        meta = probe_source(clip)
        dset, stats = run_detection(meta, detect_shots(meta), stride=24)
        assert stats.decoded > 0
        persons = dset.of_class(CLS_PERSON)
        assert persons, "FTEA_TEST_CLIP should show at least one person"
        assert min(d.score for d in persons) >= DEFAULT_MIN_CONFIDENCE_PERSON


class TestDetectionsToJson:
    def _set(self):
        return DetectionSet(
            stride=3,
            sampled_frames=(0, 3, 6),
            detections=(
                _det(frame=0, cls=CLS_PERSON,
                     box=(1.234567, 2.0, 3.0, 4.0), score=0.987654),
            ),
        )

    def test_should_emit_the_sampled_set_separately_from_detections(self):
        payload = detections_to_json(self._set())
        assert payload["stride"] == 3
        assert payload["sampled_frames"] == [0, 3, 6]
        assert len(payload["detections"]) == 1

    def test_should_round_coordinates_and_scores(self):
        box = detections_to_json(self._set())["detections"][0]
        assert box["x1"] == 1.23
        assert box["score"] == 0.9877
        assert box["cls"] == "person"

    def test_should_omit_stats_when_none_is_given(self):
        assert "stats" not in detections_to_json(self._set())

    def test_should_include_stats_when_given(self):
        _, stats = TestRunDetection()._run()
        payload = detections_to_json(self._set(), stats=stats)
        assert payload["stats"]["detect_width"] == 1920
        assert payload["stats"]["sampled"] == 4
        assert "ms_per_sampled_frame" in payload["stats"]

    def test_should_be_json_serialisable(self):
        import json

        _, stats = TestRunDetection()._run()
        json.dumps(detections_to_json(self._set(), stats=stats))


class TestDetectFromSettings:
    def _settings(self):
        from ftea.reframe.settings import load_settings
        return load_settings(paths.config_path())

    def test_should_thread_every_detect_key_through(self):
        settings = self._settings()
        meta = _meta()
        person = FakeDetector(default=[PERSON_BOX])
        dset, stats = detect_from_settings(
            meta, [_shot(0, 0, 12)], settings,
            decoder_fn=FakeDecoder(1920, 1080),
            person_fn=person, face_fn=FakeDetector(default=[FACE_BOX]),
        )
        assert dset.stride == settings.stride
        assert stats.detect_width == 1920

    def test_should_let_an_override_win_over_the_settings_value(self):
        settings = self._settings()
        dset, _ = detect_from_settings(
            _meta(), [_shot(0, 0, 12)], settings, stride=4,
            decoder_fn=FakeDecoder(1920, 1080),
            person_fn=FakeDetector(default=[]),
            face_fn=FakeDetector(default=[]),
        )
        assert dset.stride == 4


class TestTorchHome:
    def test_should_keep_the_torch_cache_inside_the_models_dir(self):
        assert torch_home(Path("models")) == Path("models") / "torch"


class TestCli:
    def test_should_read_the_data_root_config_env_by_default(
        self, tmp_path, monkeypatch, capsys,
    ):
        from ftea.reframe.probe import EXIT_ERROR

        paths.config_path().write_text("REFRAME_STRIDE=0\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "config.env").write_text("REFRAME_BATCH_SIZE=0\n")
        monkeypatch.chdir(elsewhere)

        code = main([str(tmp_path / "clip.mp4")])

        err = capsys.readouterr().err
        assert code == EXIT_ERROR
        assert "REFRAME_STRIDE" in err
        assert "REFRAME_BATCH_SIZE" not in err

    def test_should_exit_error_when_the_source_cannot_be_probed(
        self, tmp_path, capsys,
    ):
        from ftea.reframe.probe import EXIT_ERROR

        code = main([str(tmp_path / "absent.mp4")])
        captured = capsys.readouterr()
        assert code == EXIT_ERROR
        assert captured.out == ""
        assert "reframe: detection failed:" in captured.err
        assert "Traceback" not in captured.err

    def test_should_exit_reframe_on_a_successful_run(
        self, monkeypatch, tmp_path,
    ):
        """Behavioural, not a substring check on the source
        -- `return int(EXIT_ERROR)` would pass the old text-only test
        while still returning a literal-typed value."""
        import ftea.reframe.detect as module
        from ftea.reframe.probe import EXIT_REFRAME

        clip = tmp_path / "clip.mp4"
        meta = _meta(path=str(clip))
        fake_dset = DetectionSet(stride=3, sampled_frames=(0,),
                                 detections=())
        _, fake_stats = TestRunDetection()._run()

        monkeypatch.setattr(
            "ftea.reframe.probe.probe_source", lambda source: meta)
        monkeypatch.setattr(
            "ftea.reframe.shots.detect_shots",
            lambda meta, **kw: [_shot(0, 0, 12)])
        monkeypatch.setattr(
            module, "run_detection",
            lambda *a, **k: (fake_dset, fake_stats))

        assert main([str(clip)]) == EXIT_REFRAME

    def test_should_apply_the_env_overlay_when_no_config_flag_is_given(
        self, monkeypatch, tmp_path,
    ):
        """The CLI reproduces the production overlay by
        default -- REFRAME_STRIDE in the environment must reach
        run_detection, and an explicit --stride must still win."""
        import ftea.reframe.detect as module

        clip = tmp_path / "clip.mp4"
        meta = _meta(path=str(clip))
        seen_strides = []
        _, fake_stats = TestRunDetection()._run()

        def fake_run_detection(meta, shots, *, stride, **kwargs):
            seen_strides.append(stride)
            return (DetectionSet(stride=stride, sampled_frames=(),
                                 detections=()), fake_stats)

        monkeypatch.setattr(
            "ftea.reframe.probe.probe_source", lambda source: meta)
        monkeypatch.setattr(
            "ftea.reframe.shots.detect_shots",
            lambda meta, **kw: [_shot(0, 0, 12)])
        monkeypatch.setattr(module, "run_detection", fake_run_detection)
        monkeypatch.setenv("REFRAME_STRIDE", "7")

        main([str(clip)])
        assert seen_strides == [7]

        main([str(clip), "--stride", "3"])
        assert seen_strides == [7, 3]

    def test_should_detect_shots_with_the_configured_thresholds(
        self, monkeypatch, tmp_path,
    ):
        """The stage cuts shots with REFRAME_SCENE_THRESHOLD and
        REFRAME_MIN_SHOT_SEC, and striding restarts at every cut, so the
        CLI must cut the same way to sample the same frames."""
        import ftea.reframe.detect as module

        clip = tmp_path / "clip.mp4"
        meta = _meta(path=str(clip))
        seen = {}
        _, fake_stats = TestRunDetection()._run()

        def fake_detect_shots(meta, **kwargs):
            seen.update(kwargs)
            return [_shot(0, 0, 12)]

        monkeypatch.setattr(
            "ftea.reframe.probe.probe_source", lambda source: meta)
        monkeypatch.setattr(
            "ftea.reframe.shots.detect_shots", fake_detect_shots)
        monkeypatch.setattr(
            module, "run_detection",
            lambda *a, **k: (DetectionSet(stride=3, sampled_frames=(),
                                          detections=()), fake_stats))
        monkeypatch.setenv("REFRAME_SCENE_THRESHOLD", "0.55")
        monkeypatch.setenv("REFRAME_MIN_SHOT_SEC", "1.25")

        main([str(clip)])

        assert seen == {"threshold": 0.55, "min_shot_sec": 1.25}

    def test_should_exit_error_when_an_explicit_config_is_missing(
        self, tmp_path, capsys,
    ):
        """A mistyped --config must not quietly run on the defaults."""
        from ftea.reframe.probe import EXIT_ERROR

        typo = tmp_path / "confg.env"
        code = main([str(tmp_path / "clip.mp4"), "--config", str(typo)])

        err = capsys.readouterr().err
        assert code == EXIT_ERROR
        assert str(typo) in err
        assert "does not exist" in err


class _FakeYuNet:
    """Stands in for cv2.FaceDetectorYN: canned rows per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, frame):
        return 1, self.answers.pop(0)


def _yunet_row(x, y, w, h, score):
    """A 15-column YuNet row: box, five landmarks, then the score."""
    return [x, y, w, h] + [0.0] * 10 + [score]


class TestFaceDetectorConversion:
    def _load(self, monkeypatch, tmp_path, yunet, size=(1920, 1080)):
        from types import ModuleType

        model = tmp_path / "yunet.onnx"
        model.write_bytes(b"onnx")
        created = []
        cv2 = ModuleType("cv2")

        def create(path, config, size_in, **kwargs):
            created.append((path, size_in, kwargs))
            return yunet

        cv2.FaceDetectorYN_create = create
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        run = load_face_detector(model_path=model, input_size=size,
                                 score_floor=0.35)
        return run, created

    def test_should_turn_xywh_rows_into_corner_boxes(
        self, monkeypatch, tmp_path,
    ):
        yunet = _FakeYuNet([np.array([_yunet_row(100, 50, 40, 60, 0.7)])])
        run, created = self._load(monkeypatch, tmp_path, yunet)

        boxes = run([_frame(1920, 1080)])

        assert boxes == [[(100.0, 50.0, 140.0, 110.0, 0.7)]]
        assert created[0][2]["score_threshold"] == 0.35

    def test_should_read_none_as_no_faces(self, monkeypatch, tmp_path):
        yunet = _FakeYuNet([None, np.array([_yunet_row(1, 2, 3, 4, 0.5)])])
        run, _ = self._load(monkeypatch, tmp_path, yunet)

        boxes = run([_frame(1920, 1080), _frame(1920, 1080)])

        assert boxes == [[], [(1.0, 2.0, 4.0, 6.0, 0.5)]]

    def test_should_resize_the_input_only_when_the_frame_size_changes(
        self, monkeypatch, tmp_path,
    ):
        yunet = _FakeYuNet([None, None, None])
        run, _ = self._load(monkeypatch, tmp_path, yunet)

        run([_frame(1920, 1080), _frame(1280, 720), _frame(1280, 720)])

        assert yunet.sizes == [(1920, 1080), (1280, 720)]


class _FakeTensor:
    """A numpy-backed stand-in for the few tensor calls detect.py makes."""

    def __init__(self, array, log=None):
        self.array = np.asarray(array)
        self.log = log if log is not None else []

    def _wrap(self, array):
        return _FakeTensor(array, self.log)

    def to(self, *args, **kwargs):
        return self

    def permute(self, *dims):
        return self._wrap(self.array.transpose(dims))

    def flip(self, dim):
        self.log.append(("flip", dim))
        return self._wrap(np.flip(self.array, dim))

    def div_(self, value):
        return self._wrap(self.array / value)

    def detach(self):
        return self

    def tolist(self):
        return self.array.tolist()

    def __eq__(self, other):
        return self._wrap(self.array == other)

    def __getitem__(self, mask):
        return self._wrap(self.array[mask.array])


def _fake_torch_stack(monkeypatch, outputs, seen):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    class Model:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def eval(self):
            return self

        def to(self, device):
            seen["device"] = device
            return self

        def __call__(self, batch):
            seen["batch"] = batch
            return outputs

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.device = lambda name: ("device", name)
    torch.float32 = "float32"
    torch.inference_mode = nullcontext
    torch.from_numpy = lambda frame: _FakeTensor(frame, seen["log"])
    detection = ModuleType("torchvision.models.detection")
    detection.fasterrcnn_mobilenet_v3_large_fpn = Model
    detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights = SimpleNamespace(
        COCO_V1="COCO_V1")
    models = ModuleType("torchvision.models")
    models.detection = detection
    torchvision = ModuleType("torchvision")
    torchvision.models = models
    for name, module in (("torch", torch), ("torchvision", torchvision),
                         ("torchvision.models", models),
                         ("torchvision.models.detection", detection)):
        monkeypatch.setitem(sys.modules, name, module)


class TestPersonDetectorConversion:
    def test_should_keep_only_coco_people_with_their_scores(
        self, monkeypatch, tmp_path,
    ):
        """Label 1 is a person in COCO; a dog (18) must not become a
        subject, and each kept box keeps its own score."""
        seen = {"log": []}
        outputs = [{
            "labels": _FakeTensor([1, 18, 1]),
            "boxes": _FakeTensor([[0, 0, 10, 20], [5, 5, 6, 6],
                                  [30, 40, 50, 60]]),
            "scores": _FakeTensor([0.9, 0.99, 0.85]),
        }]
        _fake_torch_stack(monkeypatch, outputs, seen)
        monkeypatch.delenv("TORCH_HOME", raising=False)

        run = load_person_detector(device="cuda", models_dir=tmp_path,
                                   score_floor=0.8)
        boxes = run([_frame(64, 48)])

        assert boxes == [[(0.0, 0.0, 10.0, 20.0, 0.9),
                          (30.0, 40.0, 50.0, 60.0, 0.85)]]
        assert seen["kwargs"]["box_score_thresh"] == 0.8
        assert seen["device"] == ("device", "cuda")
        assert ("flip", 0) in seen["log"]
        assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")

    def test_should_return_nothing_for_an_empty_batch(
        self, monkeypatch, tmp_path,
    ):
        seen = {"log": []}
        _fake_torch_stack(monkeypatch, [], seen)
        monkeypatch.delenv("TORCH_HOME", raising=False)

        run = load_person_detector(device="cpu", models_dir=tmp_path)

        assert run([]) == []
        assert "batch" not in seen
