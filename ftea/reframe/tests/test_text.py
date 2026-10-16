"""Unit tests for ftea/reframe/text.py -- the pure half.

No model, no video, no OpenCV. Every box here is hand-built, so a
failure names a rule rather than a detector.
"""
import math
from pathlib import Path

import pytest

from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import SourceMeta
from ftea.reframe.shots import Shot
from ftea.reframe.text import (
    DEFAULT_TEXT_MODEL_NAME,
    FrameTitle,
    TextBox,
    TextError,
    box_from_quad,
    card_from_frames,
    detector_input_size,
    frame_title,
    load_text_detector,
    probe_clip,
    probe_frames,
    qualifying_lines,
    required_frames,
    scale_to_source,
    skew_degrees,
    text_model_path,
    title_summary,
)

FRAME_W = 1920
FRAME_H = 1080


def line(centre_x, top, height, *, width=700.0, skew=0.0):
    """One horizontal text line, centred on centre_x."""
    return TextBox(
        x1=centre_x - width / 2.0, y1=float(top),
        x2=centre_x + width / 2.0, y2=float(top) + height,
        skew_deg=skew,
    )


def credits(**overrides):
    """The five credit lines of a measured 1080p title card.

    Real measured geometry: centres 958..966, heights 84..98, spanning
    x 462..1458 and y 222..841.

    The measurement covered the UNION (x 462..1458, width 996, centre
    960), the per-line heights and the per-line centres -- not each
    line's width separately. The two outermost widths below are
    therefore chosen so this block reproduces the measured union
    exactly: line 1 spans 462..1454 and line 2 spans 466..1458.
    """
    rows = [
        (958.0, 222.0, 98.0, 992.0),
        (962.0, 357.0, 98.0, 992.0),
        (964.0, 487.0, 96.0, 687.0),
        (964.0, 624.0, 87.0, 687.0),
        (962.0, 757.0, 84.0, 990.0),
    ]
    return [
        line(cx, top, height, width=width, **overrides)
        for cx, top, height, width in rows
    ]


class TestSkewDegrees:
    def test_should_report_zero_for_an_axis_aligned_box(self):
        quad = ((100.0, 200.0), (400.0, 200.0),
                (400.0, 260.0), (100.0, 260.0))

        assert skew_degrees(quad) == pytest.approx(0.0, abs=1e-9)

    def test_should_report_zero_when_the_winding_starts_on_a_vertical_edge(
        self,
    ):
        """DB does not promise a starting corner -- a real quad has been
        seen whose first edge is vertical. Both edge directions are
        measured and the smaller distance from horizontal wins."""
        quad = ((156.0, 299.0), (156.0, 269.0),
                (486.0, 269.0), (486.0, 299.0))

        assert skew_degrees(quad) == pytest.approx(0.0, abs=1e-9)

    def test_should_report_the_tilt_of_a_rotated_box(self):
        angle = math.radians(15.0)
        length = 300.0
        quad = (
            (0.0, 0.0),
            (length * math.cos(angle), length * math.sin(angle)),
            (length * math.cos(angle) - 40.0 * math.sin(angle),
             length * math.sin(angle) + 40.0 * math.cos(angle)),
            (-40.0 * math.sin(angle), 40.0 * math.cos(angle)),
        )

        assert skew_degrees(quad) == pytest.approx(15.0, abs=0.01)

    def test_should_report_ninety_for_a_degenerate_quad(self):
        """Four identical points have no direction, so nothing can be
        called horizontal and the box is rejected by the skew gate."""
        quad = ((10.0, 10.0),) * 4

        assert skew_degrees(quad) == pytest.approx(90.0)

    def test_should_reject_a_quad_without_four_points(self):
        with pytest.raises(TextError) as excinfo:
            skew_degrees(((0.0, 0.0), (1.0, 1.0)))

        assert "4 points" in str(excinfo.value)

    def test_should_be_a_reframe_error(self):
        assert issubclass(TextError, ReframeError)


class TestBoxFromQuad:
    def test_should_take_the_axis_aligned_bounds_of_the_quad(self):
        quad = ((156.0, 299.0), (156.0, 269.0),
                (486.0, 269.0), (486.0, 299.0))

        box = box_from_quad(quad)

        assert (box.x1, box.y1, box.x2, box.y2) == (156.0, 269.0,
                                                    486.0, 299.0)
        assert box.width == 330.0
        assert box.height == 30.0
        assert box.centre_x == 321.0


class TestProbeFrames:
    def test_should_spread_five_samples_across_a_240_frame_shot(self):
        shot = Shot(index=0, start_frame=0, end_frame=240, fps=24.0)

        assert probe_frames(shot, 5) == (24, 72, 120, 168, 216)

    def test_should_never_sample_the_first_or_last_frame(self):
        """A cut's residue and a fade both live at the shot edges."""
        shot = Shot(index=0, start_frame=0, end_frame=240, fps=24.0)

        picks = probe_frames(shot, 5)

        assert 0 not in picks
        assert 239 not in picks

    def test_should_offset_by_the_shots_start_frame(self):
        shot = Shot(index=3, start_frame=480, end_frame=720, fps=24.0)

        assert probe_frames(shot, 5) == (504, 552, 600, 648, 696)

    def test_should_deduplicate_on_a_shot_shorter_than_the_count(self):
        shot = Shot(index=0, start_frame=10, end_frame=13, fps=24.0)

        assert probe_frames(shot, 5) == (10, 11, 12)

    def test_should_reject_a_count_below_one(self):
        shot = Shot(index=0, start_frame=0, end_frame=240, fps=24.0)

        with pytest.raises(TextError) as excinfo:
            probe_frames(shot, 0)

        assert ">= 1" in str(excinfo.value)


class TestQualifyingLines:
    def test_should_keep_lines_at_or_above_the_height_floor(self):
        boxes = credits()

        kept = qualifying_lines(boxes, FRAME_H, min_height=0.04,
                                max_skew=10.0)

        assert len(kept) == 5

    def test_should_drop_the_stray_short_box_beside_the_credits(self):
        """A 20 px box at x 462..507 appears on two sampled frames.
        0.04 * 1080 = 43.2 px, so the floor rejects it."""
        boxes = credits() + [line(484.0, 231.0, 20.0, width=45.0)]

        kept = qualifying_lines(boxes, FRAME_H, min_height=0.04,
                                max_skew=10.0)

        assert len(kept) == 5

    def test_should_drop_a_skewed_box(self):
        boxes = credits() + [line(960.0, 900.0, 90.0, skew=11.0)]

        kept = qualifying_lines(boxes, FRAME_H, min_height=0.04,
                                max_skew=10.0)

        assert len(kept) == 5

    def test_should_keep_a_box_exactly_at_both_limits(self):
        """Both gates are inclusive, so a line sitting exactly on the
        floor is a line. 0.04 * 1080 = 43.2.

        The box is built at top 0.0 so its height is EXACTLY 43.2:
        (100.0 + 43.2) - 100.0 is 43.19999999999999 in binary floating
        point, which would test the float representation rather than
        the inclusive gate this test exists to pin.
        """
        boxes = [line(960.0, 0.0, 43.2, skew=10.0)] * 2

        kept = qualifying_lines(boxes, FRAME_H, min_height=0.04,
                                max_skew=10.0)

        assert len(kept) == 2


class TestFrameTitle:
    def test_should_accept_the_measured_credit_block(self):
        found = frame_title(credits(), FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is not None
        assert found.line_count == 5
        assert found.x1 == pytest.approx(462.0, abs=1.0)
        assert found.x2 == pytest.approx(1458.0, abs=1.0)
        assert found.y1 == pytest.approx(222.0)
        assert found.y2 == pytest.approx(841.0)

    def test_should_reject_a_single_line(self):
        found = frame_title([line(960.0, 400.0, 90.0)], FRAME_W, FRAME_H,
                            min_height=0.04, min_lines=2,
                            centre_tol=0.05, max_skew=10.0)

        assert found is None

    def test_should_reject_two_lines_whose_centres_disagree(self):
        boxes = [line(300.0, 300.0, 90.0), line(1600.0, 500.0, 90.0)]

        found = frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is None

    def test_should_reject_when_one_qualifying_line_breaks_the_centre(self):
        """The centre test is STRICT. A title card sharing a frame
        with one large off-centre piece of scene text is rejected -- no
        subset search, no clustering. That is a deliberate bias to false
        negatives, stated as code."""
        boxes = credits() + [line(200.0, 950.0, 120.0, width=300.0)]

        found = frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is None

    def test_should_reject_lines_below_the_height_floor(self):
        boxes = [line(960.0, 300.0, 40.0), line(960.0, 400.0, 40.0)]

        found = frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is None

    def test_should_reject_skewed_lines(self):
        boxes = [line(960.0, 300.0, 90.0, skew=20.0),
                 line(960.0, 450.0, 90.0, skew=20.0)]

        found = frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is None

    def test_should_reject_an_empty_frame(self):
        assert frame_title([], FRAME_W, FRAME_H, min_height=0.04,
                           min_lines=2, centre_tol=0.05,
                           max_skew=10.0) is None

    def test_should_honour_a_raised_minimum_line_count(self):
        boxes = [line(960.0, 300.0, 90.0), line(960.0, 450.0, 90.0)]

        assert frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                           min_lines=3, centre_tol=0.05,
                           max_skew=10.0) is None

    def test_should_accept_the_dialogue_fixtures_frame_228_pair(self):
        """This frame really does satisfy every PER-FRAME rule.
        Only the majority below rejects the shot -- which is exactly why
        the majority is fixed and not configurable.

        Four boxes were measured on this frame, of which exactly TWO
        clear the height floor, at centres 1806 and 1828. The other two
        are therefore built below 43.2 px; if they cleared the floor the
        strict centre test would reject the frame and the point of
        the example -- that only the majority saves the shot -- would be
        lost.
        """
        boxes = [
            TextBox(x1=1686.0, y1=129.0, x2=1926.0, y2=326.0,
                    skew_deg=0.0),
            TextBox(x1=1704.0, y1=202.0, x2=1953.0, y2=397.0,
                    skew_deg=0.0),
            TextBox(x1=357.0, y1=385.0, x2=432.0, y2=418.0,
                    skew_deg=0.0),
            TextBox(x1=378.0, y1=456.0, x2=450.0, y2=492.0,
                    skew_deg=0.0),
        ]

        found = frame_title(boxes, FRAME_W, FRAME_H, min_height=0.04,
                            min_lines=2, centre_tol=0.05, max_skew=10.0)

        assert found is not None
        assert found.line_count == 2


class TestRequiredFrames:
    @pytest.mark.parametrize("sampled,want", [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (9, 5),
    ])
    def test_should_be_the_ceiling_of_half_the_samples(self, sampled,
                                                       want):
        assert required_frames(sampled) == want

    def test_should_reject_a_shot_with_no_samples(self):
        with pytest.raises(TextError):
            required_frames(0)


class TestCardFromFrames:
    def _hit(self, x1=462.0, y1=222.0, x2=1458.0, y2=841.0, lines=5):
        return frame_title(
            [TextBox(x1=x1, y1=y1, x2=x2, y2=y2, skew_deg=0.0)] * lines
            if lines else [],
            FRAME_W, FRAME_H, min_height=0.04, min_lines=1,
            centre_tol=0.05, max_skew=10.0)

    def test_should_build_a_card_when_the_majority_is_met(self):
        card = card_from_frames([self._hit()] * 4, 5)

        assert card is not None
        assert card.frames_seen == 4
        assert card.frames_sampled == 5
        assert card.centre_x == pytest.approx(960.0)
        assert card.width == pytest.approx(996.0)

    def test_should_return_none_below_the_majority(self):
        """1 accepting frame of 5 is what a measured dialogue shot
        produces, and it must not letterbox."""
        assert card_from_frames([self._hit()], 5) is None

    def test_should_return_none_when_no_frame_accepted(self):
        assert card_from_frames([], 5) is None

    def test_should_union_only_the_accepting_frames(self):
        """A frame that failed the per-frame rule may still have boxed a
        piece of graffiti. Letting it widen the card would move the
        centroid the crop is aimed at."""
        narrow = self._hit(x1=800.0, x2=1120.0)
        wide = self._hit(x1=462.0, x2=1458.0)

        card = card_from_frames([narrow, wide, narrow], 5)

        assert card.x1 == pytest.approx(462.0)
        assert card.x2 == pytest.approx(1458.0)

    def test_should_report_the_largest_line_count_seen(self):
        card = card_from_frames(
            [self._hit(lines=3), self._hit(lines=5),
             self._hit(lines=4)], 5)

        assert card.line_count == 5

    def test_should_serialise_to_ascii_json_safe_values(self):
        card = card_from_frames([self._hit()] * 3, 5)

        doc = card.to_json()

        assert doc == {
            "x1": 462.0, "y1": 222.0, "x2": 1458.0, "y2": 841.0,
            "centre_x": 960.0, "width": 996.0, "line_count": 5,
            "frames_seen": 3, "frames_sampled": 5,
        }
        assert all(isinstance(k, str) for k in doc)


# --- the model seam and probe_clip ----------------------------------


def meta_for(width=1920, height=1080, *, fps=24.0, rotation=0):
    return SourceMeta(
        path=Path("clip.mp4"), width=width, height=height,
        rotation=rotation, fps=fps, duration_s=10.0, has_audio=True,
    )


class FakeSettings:
    """The nine title fields, and nothing else probe_clip may read."""

    def __init__(self, **overrides):
        self.title_enabled = True
        self.title_model = ""
        self.title_long_side = 640
        self.title_probe_frames = 5
        self.title_min_height = 0.04
        self.title_min_lines = 2
        self.title_centre_tol = 0.05
        self.title_max_skew = 10.0
        self.title_edge_margin = 0.01
        self.models_dir = Path("models")
        for key, value in overrides.items():
            setattr(self, key, value)


def fake_decoder(frames):
    """A decoder that yields a canned object per wanted frame index.

    The default payload is the frame INDEX itself, because that is what
    probe_clip hands the detector and credit_detector below is keyed by
    index. A real decoder yields an image here; nothing in this module
    inspects it, so the index stands in for one and lets a fake
    detector decide per frame.
    """
    def decode(source, wanted):
        for index in sorted(wanted):
            yield index, frames.get(index, index)
    return decode


def credit_detector(hit_frames):
    """A detector returning the five credit lines on chosen frames.

    Boxes are in DETECTOR space (640x384), so the test also exercises
    scale_to_source on the way back.
    """
    # (centre_x, top, height, half_width), all in 640x384 detector
    # space -- the measured source values divided by 3.0 on x and
    # 2.8125 on y.
    # The half-widths DIFFER per line; a fixed one would make the union
    # 664px wide instead of 990 and the assertions below meaningless.
    rows = [(319.3, 79.0, 34.8, 108.5), (320.7, 127.0, 34.8, 141.5),
            (321.3, 173.0, 34.1, 114.5), (321.3, 222.0, 30.9, 114.5),
            (320.7, 269.0, 29.9, 165.0)]

    def detect(frame):
        if frame not in hit_frames:
            return []
        return [
            TextBox(x1=cx - half, y1=top, x2=cx + half,
                    y2=top + height, skew_deg=0.0)
            for cx, top, height, half in rows
        ]
    return detect


class TestTextModelPath:
    def test_should_use_the_models_dir_default_when_unset(self):
        path = text_model_path(Path("models"), "")

        assert path == Path("models") / DEFAULT_TEXT_MODEL_NAME

    def test_should_use_an_explicit_override_verbatim(self):
        path = text_model_path(Path("models"), r"C:\w\other.onnx")

        assert path == Path(r"C:\w\other.onnx")


class TestDetectorInputSize:
    @pytest.mark.parametrize("width,height", [
        (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160),
    ])
    def test_should_give_640x384_for_every_16_9_tier(self, width,
                                                     height):
        """640x384 is what was measured against. Every 16:9 tier lands on
        the same detector input, so the rule behaves identically at
        720p and 4K without a per-tier branch."""
        assert detector_input_size(width, height, 640) == (640, 384)

    def test_should_round_both_axes_up_to_a_multiple_of_32(self):
        width, height = detector_input_size(2560, 1080, 640)

        assert width % 32 == 0 and height % 32 == 0
        assert (width, height) == (640, 288)

    def test_should_not_upscale_a_source_below_the_long_side(self):
        assert detector_input_size(480, 270, 640) == (480, 288)

    def test_should_reject_a_long_side_below_the_floor(self):
        with pytest.raises(TextError) as excinfo:
            detector_input_size(1920, 1080, 128)

        assert ">= 320" in str(excinfo.value)

    def test_should_reject_a_non_positive_frame(self):
        with pytest.raises(TextError):
            detector_input_size(0, 1080, 640)


class TestScaleToSource:
    def test_should_map_a_detector_box_to_source_pixels(self):
        box = TextBox(x1=156.0, y1=269.0, x2=486.0, y2=299.0,
                      skew_deg=3.0)

        scaled = scale_to_source(box, from_size=(640, 384),
                                 to_size=(1920, 1080))

        assert scaled.x1 == pytest.approx(468.0)
        assert scaled.x2 == pytest.approx(1458.0)
        assert scaled.y1 == pytest.approx(756.5625)
        assert scaled.y2 == pytest.approx(840.9375)

    def test_should_preserve_the_skew_which_scaling_does_not_change(self):
        box = TextBox(x1=0.0, y1=0.0, x2=10.0, y2=5.0, skew_deg=7.5)

        scaled = scale_to_source(box, from_size=(640, 384),
                                 to_size=(1920, 1080))

        assert scaled.skew_deg == 7.5

    def test_should_be_an_identity_when_the_sizes_match(self):
        box = TextBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0, skew_deg=0.0)

        assert scale_to_source(box, from_size=(640, 384),
                               to_size=(640, 384)) is box


class TestProbeClip:
    def test_should_return_none_when_the_feature_is_disabled(self):
        """None, not a tuple of Nones: camera.plan_camera(titles=None)
        is the exact value that reproduces the behaviour without title
        detection."""
        result = probe_clip(
            Path("clip.mp4"), [Shot(0, 0, 240, 24.0)], meta_for(),
            FakeSettings(title_enabled=False),
            detector=credit_detector(set()), decoder=fake_decoder({}),
        )

        assert result is None

    def test_should_find_the_card_when_every_sample_carries_it(self):
        shots = [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=credit_detector({24, 72, 120, 168, 216}),
            decoder=fake_decoder({}),
        )

        assert len(cards) == 1
        assert cards[0] is not None
        assert cards[0].frames_seen == 5
        assert cards[0].frames_sampled == 5
        assert cards[0].line_count == 5
        assert cards[0].centre_x == pytest.approx(961.0, abs=3.0)
        assert cards[0].width == pytest.approx(990.0, abs=10.0)

    def test_should_find_the_card_on_a_bare_majority(self):
        shots = [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=credit_detector({24, 120, 216}),
            decoder=fake_decoder({}),
        )

        assert cards[0] is not None
        assert cards[0].frames_seen == 3

    def test_should_reject_a_shot_below_the_majority(self):
        """The measured dialogue case: one accepting frame of five."""
        shots = [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=credit_detector({120}), decoder=fake_decoder({}),
        )

        assert cards[0] is None

    def test_should_judge_each_shot_independently(self):
        shots = [
            Shot(index=0, start_frame=0, end_frame=240, fps=24.0),
            Shot(index=1, start_frame=240, end_frame=480, fps=24.0),
        ]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=credit_detector({24, 72, 120, 168, 216}),
            decoder=fake_decoder({}),
        )

        assert cards[0] is not None
        assert cards[1] is None

    def test_should_count_a_detector_failure_as_no_text(self):
        """A detector failure on one frame must not fail the clip.
        The shot is judged on the remaining samples."""
        calls = {"n": 0}

        def flaky(frame):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("cuda ran out of everything")
            return credit_detector({24, 72, 120, 168, 216})(24)

        shots = [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=flaky, decoder=fake_decoder({}),
        )

        assert cards[0] is not None
        assert cards[0].frames_seen == 4
        assert cards[0].frames_sampled == 5

    def test_should_fail_the_clip_when_every_frame_fails(self):
        """A detector that fails on every frame is broken, not unsure.
        Reading that as 'no title anywhere' would switch title framing
        off for the whole batch while every clip reported success."""
        def broken(frame):
            raise ValueError("detect() rejected the input size")

        shots = [
            Shot(index=0, start_frame=0, end_frame=240, fps=24.0),
            Shot(index=1, start_frame=240, end_frame=480, fps=24.0),
        ]

        with pytest.raises(TextError) as excinfo:
            probe_clip(
                Path("clip.mp4"), shots, meta_for(), FakeSettings(),
                detector=broken, decoder=fake_decoder({}),
            )

        message = str(excinfo.value)
        assert "all 10 sampled frame(s)" in message
        assert "detect() rejected the input size" in message

    def test_should_report_how_many_frames_failed(self, capsys):
        calls = {"n": 0}

        def flaky(frame):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("transient")
            return []

        probe_clip(
            Path("clip.mp4"),
            [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)],
            meta_for(), FakeSettings(),
            detector=flaky, decoder=fake_decoder({}),
        )

        assert "2 of 5 sampled frame(s) failed" in capsys.readouterr().err

    def test_should_not_report_failures_when_none_happened(self, capsys):
        probe_clip(
            Path("clip.mp4"),
            [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)],
            meta_for(), FakeSettings(),
            detector=credit_detector(set()), decoder=fake_decoder({}),
        )

        assert "failed" not in capsys.readouterr().err

    def test_should_convert_a_decode_failure_into_a_text_error(self):
        from ftea.reframe.detect import DetectError

        def broken(source, wanted):
            raise DetectError("cannot open for decoding: clip.mp4")
            yield  # pragma: no cover

        with pytest.raises(TextError) as excinfo:
            probe_clip(
                Path("clip.mp4"),
                [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)],
                meta_for(), FakeSettings(),
                detector=credit_detector(set()), decoder=broken,
            )

        assert "cannot open" in str(excinfo.value)

    def test_should_return_none_for_a_shot_that_decoded_nothing(self):
        """A decoder that stops early must not crash the probe, and the
        shot it never reached must not be called a title card."""
        def short(source, wanted):
            yield 24, "frame"

        shots = [
            Shot(index=0, start_frame=0, end_frame=240, fps=24.0),
            Shot(index=1, start_frame=240, end_frame=480, fps=24.0),
        ]

        cards = probe_clip(
            Path("clip.mp4"), shots, meta_for(), FakeSettings(),
            detector=credit_detector({24}), decoder=short,
        )

        assert cards[1] is None


class TestTitleSummary:
    def test_should_say_disabled_when_titles_are_none(self):
        assert "disabled" in title_summary(None)

    def test_should_count_the_shots_carrying_a_card(self):
        card = card_from_frames(
            [FrameTitle(462.0, 222.0, 1458.0, 841.0, 5)] * 3, 5)

        text = title_summary([card, None, card])

        assert "2 of 3" in text
        assert text.isascii()


class TestLoadTextDetectorErrors:
    def test_should_name_install_sh_when_the_model_is_missing(
        self, tmp_path,
    ):
        pytest.importorskip("cv2")
        with pytest.raises(TextError) as excinfo:
            load_text_detector(model_path=tmp_path / "absent.onnx",
                               input_size=(640, 384))

        assert "install.sh" in str(excinfo.value)
        assert "absent.onnx" in str(excinfo.value)


class TestRealDefaultsAreWired:
    """Injectable seams hide a disconnected default; assert it directly,
    as detect.py's tests do. Production calls probe_clip with neither a
    detector nor a decoder."""

    def test_should_load_the_configured_model_and_decode_by_default(
        self, monkeypatch,
    ):
        import ftea.reframe.detect as detect_module
        import ftea.reframe.text as text_module

        loads = []
        decodes = []

        def fake_loader(*, model_path, input_size):
            loads.append((model_path, input_size))
            return credit_detector(set())

        def fake_decode(source, wanted):
            decodes.append((source, frozenset(wanted)))
            for index in sorted(wanted):
                yield index, index

        monkeypatch.setattr(text_module, "load_text_detector", fake_loader)
        monkeypatch.setattr(detect_module, "decode_frames", fake_decode)
        shots = [Shot(index=0, start_frame=0, end_frame=240, fps=24.0)]
        settings = FakeSettings(models_dir=Path("weights"))

        cards = probe_clip(Path("clip.mp4"), shots, meta_for(), settings)

        assert cards == (None,)
        assert loads == [(text_model_path(Path("weights"), ""),
                          detector_input_size(1920, 1080, 640))]
        assert decodes == [(Path("clip.mp4"),
                            frozenset({24, 72, 120, 168, 216}))]


class _FakeTextNet:
    def __init__(self, quads):
        self.quads = quads
        self.settings = {}
        self.inputs = []

    def __getattr__(self, name):
        if not name.startswith("set"):
            raise AttributeError(name)

        def setter(value):
            self.settings[name] = value
        return setter

    def detect(self, image):
        self.inputs.append(image)
        return self.quads, [0.9] * len(self.quads)


def _fake_cv2(net, resized):
    from types import ModuleType, SimpleNamespace

    cv2 = ModuleType("cv2")
    cv2.error = type("error", (Exception,), {})
    cv2.INTER_AREA = 3
    cv2.dnn = SimpleNamespace(
        readNet=lambda path: ("weights", path),
        TextDetectionModel_DB=lambda model: net,
    )

    def resize(frame, size, interpolation=None):
        resized.append((frame, size, interpolation))
        return "small"

    cv2.resize = resize
    return cv2


class TestLoadTextDetectorClosure:
    def test_should_resize_then_turn_each_quad_into_a_box(
        self, tmp_path, monkeypatch,
    ):
        import sys

        model = tmp_path / "text.onnx"
        model.write_bytes(b"onnx")
        quad = ((10.0, 40.0), (10.0, 20.0), (110.0, 20.0), (110.0, 40.0))
        net = _FakeTextNet([quad])
        resized = []
        monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(net, resized))

        run = load_text_detector(model_path=model, input_size=(640, 384))
        boxes = run("frame")

        assert resized == [("frame", (640, 384), 3)]
        assert net.inputs == ["small"]
        assert net.settings["setInputSize"] == (640, 384)
        assert boxes == [box_from_quad(quad)]
        assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (
            10.0, 20.0, 110.0, 40.0)

    def test_should_return_no_boxes_when_the_net_finds_none(
        self, tmp_path, monkeypatch,
    ):
        import sys

        model = tmp_path / "text.onnx"
        model.write_bytes(b"onnx")
        monkeypatch.setitem(
            sys.modules, "cv2", _fake_cv2(_FakeTextNet([]), []))

        run = load_text_detector(model_path=model, input_size=(640, 384))

        assert run("frame") == []
