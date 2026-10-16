"""Per-shot title-card detection for the auto-reframe stage.

A "title card" here is a STRUCTURE, not a string: two or more
horizontally-aligned text lines sharing one horizontal centre,
axis-aligned, each at least a few percent of frame height, present in at
least half of a shot's sampled frames. Nothing in this module reads
text; framing needs geometry and strings would only invite alphabet
assumptions.

The strictness is deliberate and is biased to FALSE NEGATIVES. A missed
title is framed as it would be without title detection; a false
letterbox is a regression on a shot that was fine. Real film footage is
full of incidental text -- graffiti on a wall in one test clip measured
10-12% of frame height, LARGER than the title, and on a locked-off shot
it is static and axis-aligned too -- so size and stability cannot
separate a title card from a wall. The multi-line shared-centre
structure is what can.

The module has two halves and they are separable on purpose. The first
is pure geometry over hand-built boxes: no model, no video, no OpenCV,
runs on any machine. The second is a single lazy-cv2 seam, so swapping
the text model later touches exactly one function -- mirroring how
detect.py isolates the person and face models.

cv2 is imported INSIDE load_text_detector, not at module scope, so this
module's unit suite runs on an interpreter with no OpenCV at all (the
same seam detect.py uses at its three loaders). numpy is never imported:
every cv2 call this module makes accepts plain Python tuples.

A title card is accepted per SHOT, never per frame: see qualifying_lines
and card_from_frames.

ASCII only -- a sibling prints title_summary() through a stdout Windows
hands Python as cp1252.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, List, Optional, Sequence, Tuple,
)

from .errors import ReframeError
from .shots import Shot


class TextError(ReframeError):
    """Raised when a title probe cannot be trusted.

    Lives here rather than in errors.py because every sibling module in
    this package defines its own ReframeError subclass locally --
    CameraError, RenderError, DetectError, ShotsError, DebugVideoError --
    and errors.py holds only the shared base.
    """


DEFAULT_TEXT_MODEL_NAME = "text_detection_en_ppocrv3_2023may.onnx"

DEFAULT_TITLE_ENABLED = True
DEFAULT_TITLE_LONG_SIDE = 640
DEFAULT_TITLE_PROBE_FRAMES = 5
DEFAULT_TITLE_MIN_HEIGHT = 0.04
DEFAULT_TITLE_MIN_LINES = 2
DEFAULT_TITLE_CENTRE_TOL = 0.05
DEFAULT_TITLE_MAX_SKEW = 10.0

MIN_TITLE_LONG_SIDE = 320
# DB is fully convolutional with a stride of 32, so an input whose sides
# are not multiples of 32 is either rejected or silently padded.
SIZE_MULTIPLE = 32

# PP-OCRv3 preprocessing, copied from OpenCV Zoo's own ppocr_det.py so
# the weights see what they were trained on. Verified against real
# title credits: 5/5 lines found.
DB_BINARY_THRESHOLD = 0.3
DB_POLYGON_THRESHOLD = 0.5
DB_UNCLIP_RATIO = 2.0
DB_MAX_CANDIDATES = 200
DB_MEAN = (123.675, 116.28, 103.53)
DB_STD = (0.229, 0.224, 0.225)

FrameArray = Any
Point = Tuple[float, float]
Quad = Sequence[Point]
DetectorFn = Callable[[FrameArray], List["TextBox"]]


def _log(message: str) -> None:
    """Human-readable progress. stdout is reserved for JSON."""
    print(f"  reframe: {message}", file=sys.stderr)


@dataclass(frozen=True)
class TextBox:
    """One detected text box, as an axis-aligned bound plus its tilt.

    Coordinates are whatever space the producer was working in; the
    seam maps them to DISPLAY-space source pixels before any acceptance
    test runs, exactly as detect.scale_box does.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    skew_deg: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centre_x(self) -> float:
        return (self.x1 + self.x2) / 2.0


@dataclass(frozen=True)
class FrameTitle:
    """A title card as seen on ONE frame: its bound and its line count."""

    x1: float
    y1: float
    x2: float
    y2: float
    line_count: int


@dataclass(frozen=True)
class TitleCard:
    """A title card for a WHOLE shot, in display-space source pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    centre_x: float
    line_count: int
    frames_seen: int
    frames_sampled: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    def to_json(self) -> dict:
        """The crops.json payload. Rounded so a reader is not shown
        seventeen digits of detector noise."""
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "centre_x": round(self.centre_x, 2),
            "width": round(self.width, 2),
            "line_count": self.line_count,
            "frames_seen": self.frames_seen,
            "frames_sampled": self.frames_sampled,
        }


def skew_degrees(quad: Quad) -> float:
    """Angular distance from horizontal of a 4-point text box, degrees.

    BOTH edge directions are measured and the smaller distance from
    horizontal wins. DB does not promise which corner a quad starts at --
    a real one has been seen whose first edge is vertical -- so measuring
    a single edge would call an axis-aligned box 90 degrees skewed half
    the time.

    A degenerate quad (no direction at all) returns 90.0, which every
    caller's skew gate rejects.
    """
    points = tuple(quad)
    if len(points) != 4:
        raise TextError(f"a text quad needs 4 points, got {len(points)}")
    best = 90.0
    for start in (0, 1):
        (ax, ay), (bx, by) = points[start], points[(start + 1) % 4]
        dx, dy = float(bx) - float(ax), float(by) - float(ay)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        best = min(best, min(angle, 180.0 - angle))
    return best


def box_from_quad(quad: Quad) -> TextBox:
    """The axis-aligned bound of a detector quad, carrying its tilt."""
    points = tuple(quad)
    skew = skew_degrees(points)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return TextBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys),
                   skew_deg=skew)


def probe_frames(shot: Shot, count: int) -> Tuple[int, ...]:
    """`count` frame indices spread evenly across one shot.

    Sample i sits at the CENTRE of the i-th of `count` equal slices, so
    no sample lands on the shot's first or last frame -- a cut's residue
    and a fade both live there. Duplicates are removed, so a shot
    shorter than `count` frames yields fewer samples and the majority
    rule scales with what was actually sampled rather than with what was
    asked for.

    Sampling across the shot and not only at its start is not caution:
    a DIFFERENT credit state was measured at frame 12 of a shot whose
    frames 48..228 all carry the five-line block, so one shot can hold
    more than one card.
    """
    if count < 1:
        raise TextError(f"probe frame count must be >= 1, got {count}")
    span = shot.end_frame - shot.start_frame
    if span < 1:
        raise TextError(
            f"shot {shot.index} is empty: "
            f"[{shot.start_frame}, {shot.end_frame})")
    picks: List[int] = []
    for index in range(count):
        offset = min(int((index + 0.5) * span / count), span - 1)
        frame = shot.start_frame + offset
        if frame not in picks:
            picks.append(frame)
    return tuple(picks)


def qualifying_lines(
    boxes: Sequence[TextBox],
    frame_height: int,
    *,
    min_height: float = DEFAULT_TITLE_MIN_HEIGHT,
    max_skew: float = DEFAULT_TITLE_MAX_SKEW,
) -> List[TextBox]:
    """Boxes tall enough and flat enough to count as a title LINE.

    Both gates are INCLUSIVE: a line sitting exactly on the floor is a
    line. The height floor is a fraction of frame height, never pixels,
    so one value behaves identically at 720p and 4K.
    """
    floor_px = min_height * float(frame_height)
    return [
        box for box in boxes
        if box.height >= floor_px and box.skew_deg <= max_skew
    ]


def frame_title(
    boxes: Sequence[TextBox],
    frame_width: int,
    frame_height: int,
    *,
    min_height: float = DEFAULT_TITLE_MIN_HEIGHT,
    min_lines: int = DEFAULT_TITLE_MIN_LINES,
    centre_tol: float = DEFAULT_TITLE_CENTRE_TOL,
    max_skew: float = DEFAULT_TITLE_MAX_SKEW,
) -> Optional[FrameTitle]:
    """The per-frame test. None when this frame is not a title card.

    The centre test is STRICT: EVERY qualifying line must agree, not
    merely some subset of them. A title card sharing a frame with one
    large off-centre piece of scene text is therefore rejected. That is
    a deliberate bias to false negatives expressed as code, and it needs
    no subset search and no second threshold.
    """
    lines = qualifying_lines(boxes, frame_height, min_height=min_height,
                             max_skew=max_skew)
    if len(lines) < min_lines:
        return None
    centres = [box.centre_x for box in lines]
    if (max(centres) - min(centres)) > centre_tol * float(frame_width):
        return None
    return FrameTitle(
        x1=min(box.x1 for box in lines),
        y1=min(box.y1 for box in lines),
        x2=max(box.x2 for box in lines),
        y2=max(box.y2 for box in lines),
        line_count=len(lines),
    )


def required_frames(sampled: int) -> int:
    """The majority: ceil(sampled / 2). FIXED, never configurable.

    One more knob on a rule already carrying five is not worth the
    surface -- and this particular knob is load-bearing. A two-person
    dialogue clip was measured accepting on exactly 1 of 5 sampled
    frames; only this majority stops that shot letterboxing.
    """
    if sampled < 1:
        raise TextError("a shot must have at least one sampled frame")
    return (sampled + 1) // 2


def card_from_frames(
    hits: Sequence[FrameTitle], frames_sampled: int,
) -> Optional[TitleCard]:
    """Union the ACCEPTING frames into one TitleCard, or None.

    Only accepting frames contribute. A frame that failed the per-frame
    rule may still have boxed a piece of graffiti, and letting it widen
    the card would move the very centroid the crop is aimed at.
    """
    if not hits or len(hits) < required_frames(frames_sampled):
        return None
    x1 = min(hit.x1 for hit in hits)
    y1 = min(hit.y1 for hit in hits)
    x2 = max(hit.x2 for hit in hits)
    y2 = max(hit.y2 for hit in hits)
    return TitleCard(
        x1=x1, y1=y1, x2=x2, y2=y2, centre_x=(x1 + x2) / 2.0,
        line_count=max(hit.line_count for hit in hits),
        frames_seen=len(hits), frames_sampled=frames_sampled,
    )


def text_model_path(models_dir: Path, title_model: str) -> Path:
    """Resolve REFRAME_TITLE_MODEL, or the default PP-OCRv3 file in the
    models directory (fetched there, not shipped in the repository)."""
    if title_model:
        return Path(title_model)
    return Path(models_dir) / DEFAULT_TEXT_MODEL_NAME


def _multiple_of_32(value: float) -> int:
    """Round UP to a multiple of 32, with a floor of one step.

    Up rather than nearest, so neither axis is ever quietly submitted
    below the resolution the operator asked for.
    """
    steps = max(1, int(math.ceil(value / float(SIZE_MULTIPLE))))
    return steps * SIZE_MULTIPLE


def detector_input_size(display_w: int, display_h: int,
                        long_side: int) -> Tuple[int, int]:
    """Frame size submitted to the DB detector, both axes /32.

    Unlike detect.detector_input_size there is no 0 == native mode:
    running DB at native 4K would cost far more than the whole title
    feature is worth, and at 640 the detector found 5/5 credit lines at
    1920x1080 with the glyphs at 7.4% of frame height. A source already
    shorter than `long_side` is NOT upscaled -- upscaling costs detector
    time and invents no detail.

    The aspect is only approximately preserved, because both axes round
    up independently. That costs nothing: boxes come back through
    scale_to_source with independent per-axis ratios, exactly as
    detect.scale_box does.
    """
    if display_w < 1 or display_h < 1:
        raise TextError(
            "display geometry must be positive, got "
            f"{display_w}x{display_h}")
    if long_side < MIN_TITLE_LONG_SIDE:
        raise TextError(
            f"title_long_side must be >= {MIN_TITLE_LONG_SIDE}, got "
            f"{long_side}")
    source_long = max(display_w, display_h)
    scale = min(1.0, long_side / float(source_long))
    return (_multiple_of_32(display_w * scale),
            _multiple_of_32(display_h * scale))


def scale_to_source(
    box: TextBox, *, from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> TextBox:
    """Map a detector-space box back to display-space source pixels.

    Per-axis ratios, because the /32-rounded detector size is not
    exactly proportional on both. `skew_deg` is carried through
    unchanged: a non-uniform scale does change a real angle, but the
    box has already been reduced to an axis-aligned bound by then, and
    re-deriving a tilt from that bound would invent a number.
    """
    if tuple(from_size) == tuple(to_size):
        return box
    from_w, from_h = from_size
    if from_w < 1 or from_h < 1:
        raise TextError(f"cannot scale from {from_size}")
    ratio_x = to_size[0] / float(from_w)
    ratio_y = to_size[1] / float(from_h)
    return TextBox(
        x1=box.x1 * ratio_x, y1=box.y1 * ratio_y,
        x2=box.x2 * ratio_x, y2=box.y2 * ratio_y,
        skew_deg=box.skew_deg,
    )


def load_text_detector(
    *, model_path: Path, input_size: Tuple[int, int],
) -> DetectorFn:
    """OpenCV's DB detector over PP-OCRv3 weights (Apache-2.0).

    cv2 is imported HERE, not at module scope, so this module's unit
    suite runs on an interpreter with no OpenCV -- the same seam
    detect.py uses at its three loaders.

    The preprocessing constants come from OpenCV Zoo's own ppocr_det.py
    so the weights see what they were trained on. They are passed as
    plain tuples, not numpy arrays: cv2 was verified to accept that, and
    this module must not acquire numpy.

    The returned callable OWNS the resize, so the caller never touches
    cv2 and a fake detector in a test receives whatever the fake decoder
    yielded.
    """
    import cv2

    path = Path(model_path)
    if not path.exists():
        raise TextError(
            f"text model not found: {path} -- run "
            "install.sh to fetch "
            f"{DEFAULT_TEXT_MODEL_NAME}")
    try:
        net = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(str(path)))
    except cv2.error as exc:
        raise TextError(
            f"could not load the text model {path}: {exc}") from exc
    net.setBinaryThreshold(DB_BINARY_THRESHOLD)
    net.setPolygonThreshold(DB_POLYGON_THRESHOLD)
    net.setUnclipRatio(DB_UNCLIP_RATIO)
    net.setMaxCandidates(DB_MAX_CANDIDATES)
    size = (int(input_size[0]), int(input_size[1]))
    net.setInputSize(size)
    net.setInputMean(DB_MEAN)
    net.setInputScale(tuple(1.0 / 255.0 / value for value in DB_STD))

    def run(frame: FrameArray) -> List[TextBox]:
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        # detect() returns (boxes, confidences); the confidences are
        # ignored on purpose. DB's own polygon threshold already
        # rejected the weak candidates, and the per-frame test's
        # discriminator is STRUCTURE -- a confident box on a wall is
        # exactly the failure mode a score gate cannot see.
        boxes, _confidences = net.detect(small)
        return [
            box_from_quad(tuple((float(p[0]), float(p[1]))
                                for p in quad))
            for quad in boxes
        ]

    return run


def probe_clip(
    source: Path,
    shots: Sequence[Shot],
    meta: Any,
    settings: Any,
    *,
    detector: Optional[DetectorFn] = None,
    decoder: Optional[Callable] = None,
) -> Optional[Tuple[Optional[TitleCard], ...]]:
    """One Optional[TitleCard] per shot, aligned to `shots`.

    Returns None -- not a tuple of Nones -- when REFRAME_TITLE_ENABLED
    is false, because None is the exact value camera.plan_camera needs
    to reproduce the behaviour without title detection bit-for-bit.

    The model is loaded ONCE and the clip is decoded ONCE. The decode
    reuses detect.decode_frames -- grab() every frame, retrieve() only
    the wanted ones -- because that is the mechanism that was measured
    at 3.482 ms per decoded frame against 32.209 for an ffmpeg
    rawvideo pipe, and a second copy of it here would eventually
    disagree with the first.

    Every uncertainty resolves to "no title", which means normal
    people-framing: a detector failure on a frame counts as no text and
    the shot is judged on the remaining samples, and a shot whose
    samples never arrived is not a title card. A detector that fails on
    every sampled frame is not uncertainty but a broken setup, and
    raises TextError.
    """
    if not getattr(settings, "title_enabled", DEFAULT_TITLE_ENABLED):
        return None
    if not shots:
        raise TextError("shot list is empty; shots.py never returns one")

    frame_width = int(meta.display_width)
    frame_height = int(meta.display_height)
    detect_size = detector_input_size(
        frame_width, frame_height, int(settings.title_long_side))
    source_size = (frame_width, frame_height)

    run = detector
    if run is None:
        run = load_text_detector(
            model_path=text_model_path(settings.models_dir,
                                       settings.title_model),
            input_size=detect_size,
        )

    owner: dict = {}
    for shot in shots:
        for frame in probe_frames(shot,
                                  int(settings.title_probe_frames)):
            owner[frame] = shot.index
    hits: dict = {shot.index: [] for shot in shots}
    sampled: dict = {shot.index: 0 for shot in shots}
    failures: List[str] = []

    decode = decoder
    if decode is None:
        from .detect import decode_frames

        decode = decode_frames

    try:
        stream = decode(source, set(owner))
        for frame_index, frame in stream:
            shot_index = owner[frame_index]
            sampled[shot_index] += 1
            try:
                boxes = run(frame)
            except Exception as exc:          # noqa: BLE001 -- see below
                # A detector failure on one frame counts that frame as
                # 'no text' and the shot is judged on the remaining
                # samples: it must not fail the clip. The failure is
                # LOGGED, never silent, and it can only ever move the
                # answer toward people-framing. A detector that fails
                # on EVERY frame is caught after the loop.
                failures.append(f"{type(exc).__name__}: {exc}")
                _log(f"title probe: frame {frame_index} failed "
                     f"({failures[-1]}); counted as no text")
                boxes = []
            found = frame_title(
                [scale_to_source(box, from_size=detect_size,
                                 to_size=source_size)
                 for box in boxes],
                frame_width, frame_height,
                min_height=float(settings.title_min_height),
                min_lines=int(settings.title_min_lines),
                centre_tol=float(settings.title_centre_tol),
                max_skew=float(settings.title_max_skew),
            )
            if found is not None:
                hits[shot_index].append(found)
    except ReframeError as exc:
        # decode_frames raises DetectError. Converted here so this
        # module keeps ONE exception type across its public surface.
        raise TextError(
            f"title probe could not read {source}: {exc}") from exc

    _check_failures(failures, sum(sampled.values()))
    return tuple(
        card_from_frames(hits[shot.index], sampled[shot.index])
        if sampled[shot.index] else None
        for shot in shots
    )


def _check_failures(failures: Sequence[str], frames_sampled: int) -> None:
    """Fail the clip when the detector failed on every sampled frame.

    One bad frame is noise and costs that frame only. Every frame
    failing means the detector itself is broken (a model file OpenCV
    loads but cannot run, an OpenCV build mismatch), and reading that
    as "no title in any shot" would quietly switch title framing off
    for the whole batch while each clip reported success.
    """
    if not failures:
        return
    if len(failures) >= frames_sampled:
        raise TextError(
            f"title detector failed on all {frames_sampled} sampled "
            f"frame(s); first error: {failures[0]}")
    _log(f"title probe: {len(failures)} of {frames_sampled} sampled "
         "frame(s) failed and were counted as no text")


def title_summary(
    cards: Optional[Sequence[Optional[TitleCard]]],
) -> str:
    """One ASCII line for run_stages to print. No I/O here."""
    if cards is None:
        return "title probe disabled (REFRAME_TITLE_ENABLED=false)"
    found = [card for card in cards if card is not None]
    if not found:
        return f"no title card in {len(cards)} shot(s)"
    widest = max(card.width for card in found)
    return (
        f"title card in {len(found)} of {len(cards)} shot(s), "
        f"widest {widest:.0f}px, "
        f"{max(card.line_count for card in found)} line(s) max")
