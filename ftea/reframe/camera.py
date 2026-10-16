"""Crop-trajectory planning for the auto-reframe stage.

Turns subject.py's per-sample followed points into a dense, per-frame,
even-pixel crop trajectory: a latched dead zone, a critically-damped
filter, a normalized pan-speed cap, edge clamping, and a hard reset at
every shot boundary.

Every parameter is a fraction of frame size, or a dimensionless fraction
of one. On the people-following path pixels appear in exactly two places:
crop_window(), which sizes the crop rectangle once per clip, and the
quantization line in plan_shot that turns a normalized centre into an
even integer x. The title and letterbox helpers (canvas_for, title_x,
title_treatment, _plan_title_shot) work in pixels throughout, because
the text box arrives from the detector in pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ReframeError
from .probe import SourceMeta
from .subject import ShotSubject


class CameraError(ReframeError):
    """Raised when a crop trajectory cannot be planned."""


DEFAULT_ASPECT_W = 9                  # REFRAME_ASPECT_W
DEFAULT_ASPECT_H = 16                 # REFRAME_ASPECT_H
DEFAULT_DEAD_ZONE = 0.9               # REFRAME_DEAD_ZONE
DEFAULT_SMOOTHING_HZ = 0.6            # REFRAME_SMOOTHING_HZ
DEFAULT_MAX_PAN_FRAC_PER_SEC = 0.20   # REFRAME_MAX_PAN_FRAC_PER_SEC
DEFAULT_DEAD_ZONE_RELEASE = 0.6       # REFRAME_DEAD_ZONE_RELEASE
# These camera values were chosen from side-by-side renders on real
# feature footage: the crop behaves like a second camera stacked on the
# operator's, so it holds until the subject nears the crop edge, then
# nudges the minimum back rather than recentring. MAX_PAN stays 0.20
# deliberately -- raising it measurably increased jerk. The earlier,
# livelier defaults were dead_zone 0.3, smoothing 1.5, release 0.25.

DEFAULT_TITLE_EDGE_MARGIN = 0.01      # REFRAME_TITLE_EDGE_MARGIN

TREATMENT_CROP = "crop"
TREATMENT_LETTERBOX = "letterbox"

# The dead-zone metric: the subject should be inside the dead zone in
# at least 95% of observed samples. crops.json publishes it under the
# key "success_criterion_2".
SC2_THRESHOLD = 0.95


def round_up_even(value: float) -> int:
    """Round up to the next even integer. Never //2*2."""
    return int(math.ceil(value / 2.0)) * 2


def even_floor(value: float) -> int:
    """Largest even integer <= value."""
    floored = int(math.floor(value))
    return floored - (floored % 2)


def even_round(value: float) -> int:
    """Nearest even integer, ties rounded UP. Never banker's rounding."""
    return int(math.floor(value / 2.0 + 0.5)) * 2


@dataclass(frozen=True)
class CropWindow:
    """The crop rectangle, constant for the whole clip. PIXELS."""

    frame_width: int
    frame_height: int
    crop_w: int
    crop_h: int
    y: int

    @property
    def max_x(self) -> int:
        return self.frame_width - self.crop_w

    @property
    def centre_x(self) -> int:
        return even_round(self.max_x / 2.0)

    @property
    def half_crop_norm(self) -> float:
        return self.crop_w / (2.0 * self.frame_width)


def crop_window(
    meta: SourceMeta,
    *,
    aspect_w: int = DEFAULT_ASPECT_W,
    aspect_h: int = DEFAULT_ASPECT_H,
) -> CropWindow:
    """Compute the crop rectangle from the source's DISPLAY geometry.

    crop_w = round_up_even(crop_h * aspect_w / aspect_h); crop_h is the
    display height floored to even. Computed, never tabulated, never
    branched on resolution.

    An aspect wider than the source would need a crop wider than the
    frame. That raises: clamping it would render a file in a different
    aspect from the one configured, with nothing to say so.
    """
    if aspect_w < 1:
        raise CameraError(
            f"REFRAME_ASPECT_W must be >= 1, got {aspect_w}")
    if aspect_h < 1:
        raise CameraError(
            f"REFRAME_ASPECT_H must be >= 1, got {aspect_h}")
    frame_width = even_floor(meta.display_width)
    frame_height = even_floor(meta.display_height)
    if frame_width < 2 or frame_height < 2:
        raise CameraError(
            "source has no usable frame area: display "
            f"{meta.display_width}x{meta.display_height} floors to "
            f"{frame_width}x{frame_height}")
    crop_h = frame_height
    crop_w = round_up_even(crop_h * aspect_w / aspect_h)
    if crop_w > frame_width:
        raise CameraError(
            f"REFRAME_ASPECT_W:REFRAME_ASPECT_H {aspect_w}:{aspect_h} "
            f"needs a crop {crop_w} px wide but the source is only "
            f"{frame_width} px wide; set an aspect narrower than the "
            "source's")
    return CropWindow(
        frame_width=frame_width, frame_height=frame_height,
        crop_w=crop_w, crop_h=crop_h, y=0,
    )


@dataclass(frozen=True)
class Canvas:
    """The static padded canvas a letterbox shot crops out of. PIXELS.

    `pad` is NOT runtime-commandable in ffmpeg while `crop`'s w/h/x/y
    are, so the padding has to be static and applied ONCE,
    before the commandable crop. That asymmetry is the entire reason
    the design pads to a canvas instead of doing crop -> scale -> pad.
    """

    width: int
    height: int
    pad_x: int
    pad_y: int


def canvas_for(window: CropWindow) -> Canvas:
    """The canvas for a clip that letterboxes at least one shot.

    The canvas carries the OUTPUT aspect, not the source's: the
    letterbox state crops the WHOLE canvas and a static scale then fits
    it to crop_w x crop_h, so a canvas of any other shape would stretch
    the picture.

    `even_round`, specifically -- NOT `round_up_even`. It was measured
    that round-half-up-to-even is exactly what ffmpeg's
    `scale=-2` does (2560x1080 -> 1706, not 1708), so using the same
    helper keeps the canvas and any downstream scale agreeing by
    construction rather than by coincidence.

    `pad_y` may be ODD and that is deliberate. A test render of
    pad=1920:3410:0:1165 through yuv420p was accepted by ffmpeg, exit 0,
    with the credits legible, and a second check confirmed it. Do not
    "fix" this to an even value on the theory that chroma siting
    requires it -- the measurement says otherwise, and an even pad would
    move the image off the canvas centre.
    """
    if window.crop_w >= window.frame_width:
        raise CameraError(
            "cannot letterbox a source whose crop already spans the "
            f"frame: crop_w {window.crop_w}, frame_width "
            f"{window.frame_width}")
    height = even_round(
        window.frame_width * window.crop_h / float(window.crop_w))
    if height < window.frame_height:
        raise CameraError(
            f"canvas height {height} is shorter than the frame "
            f"{window.frame_height}; the source cannot be letterboxed")
    return Canvas(width=window.frame_width, height=height, pad_x=0,
                  pad_y=(height - window.frame_height) // 2)


def title_x(title: Any, window: CropWindow) -> int:
    """The constant crop left edge for a title shot, in EVEN pixels.

    Centred on the text centroid and clamped to the pan range. No dead
    zone, no damping, no pan cap: there is nothing to follow and
    nothing to smooth.
    """
    left = even_round(title.centre_x - window.crop_w / 2.0)
    return min(max(left, 0), window.max_x)


def title_treatment(
    title: Any, window: CropWindow,
    edge_margin: float = DEFAULT_TITLE_EDGE_MARGIN,
) -> str:
    """"letterbox" iff the text will not fit with a margin each side.

    The margin exists so a title that JUST fits does not render flush
    against both crop edges, which reads as a mistake even though no
    pixel is lost. It also settles the degenerate case: without it, a
    2px overflow would trade 68% of the picture for two pixels of
    glyph. The default is a judgement, not a measurement -- the
    measured title card's text overflows by nearly 400px and letterboxes
    under any sane value, so nothing here is tuned to it.

    A crop that already spans the whole frame width (an aspect as wide
    as the source) shows all the text there is, so it never letterboxes;
    canvas_for could not build that canvas anyway.
    """
    if window.crop_w >= window.frame_width:
        return TREATMENT_CROP
    margin_px = edge_margin * float(window.frame_width)
    if title.width + 2.0 * margin_px > window.crop_w:
        return TREATMENT_LETTERBOX
    return TREATMENT_CROP


def dead_zone_half(window: CropWindow, dead_zone: float) -> float:
    """Half the dead-zone width, NORMALIZED to the frame width.

    The dead zone is a share of the CROP, not of the frame, so the same
    config value frames a subject identically at 720p and at 4K.
    """
    if not 0.0 < dead_zone < 1.0:
        raise CameraError(
            "REFRAME_DEAD_ZONE must be between 0 and 1 exclusive, got "
            f"{dead_zone}")
    return dead_zone * window.crop_w / (2.0 * window.frame_width)


def smooth_step(
    position: float,
    velocity: float,
    target: float,
    dt: float,
    smoothing_hz: float,
) -> Tuple[float, float]:
    """One critically-damped step, implicit Euler.

    Unconditionally stable: the denominator grows with dt, so no frame
    rate and no smoothing_hz can make this diverge. Monotone and
    overshoot-free at every configuration measured.
    """
    if smoothing_hz <= 0.0:
        raise CameraError(
            f"REFRAME_SMOOTHING_HZ must be > 0, got {smoothing_hz}")
    if dt <= 0.0:
        raise CameraError(f"frame interval must be > 0, got {dt}")
    omega = 2.0 * math.pi * smoothing_hz
    denom = (1.0 + dt * omega) ** 2
    next_velocity = (
        velocity + dt * omega * omega * (target - position)) / denom
    return position + dt * next_velocity, next_velocity


def _validate_shot_subject(subject: ShotSubject) -> None:
    """Reject anything plan_shot is about to trust."""
    if subject.fps <= 0.0:
        raise CameraError(
            f"shot {subject.shot_index}: fps must be > 0, got "
            f"{subject.fps}")
    if subject.stride < 1:
        raise CameraError(
            f"shot {subject.shot_index}: stride must be >= 1, got "
            f"{subject.stride}")
    if subject.end_frame <= subject.start_frame:
        raise CameraError(
            f"shot {subject.shot_index}: empty frame range "
            f"[{subject.start_frame}, {subject.end_frame})")
    if len(subject.samples) != len(subject.sampled_frames):
        raise CameraError(
            f"shot {subject.shot_index}: {len(subject.samples)} samples "
            f"for {len(subject.sampled_frames)} sampled_frames")
    if not subject.sampled_frames:
        raise CameraError(
            f"shot {subject.shot_index}: no sampled frames")
    previous = -1
    for frame, sample in zip(subject.sampled_frames, subject.samples):
        if frame <= previous:
            raise CameraError(
                f"shot {subject.shot_index}: sampled_frames must be "
                f"strictly increasing, got {frame} after {previous}")
        previous = frame
        if not subject.start_frame <= frame < subject.end_frame:
            raise CameraError(
                f"shot {subject.shot_index}: sampled frame {frame} is "
                f"outside [{subject.start_frame}, {subject.end_frame})")
        if sample.frame != frame:
            raise CameraError(
                f"shot {subject.shot_index}: sample frame "
                f"{sample.frame} does not match sampled frame {frame}")
        if not 0.0 <= sample.cx <= 1.0:
            raise CameraError(
                f"shot {subject.shot_index}: frame {frame} has cx "
                f"{sample.cx}, which is outside [0, 1]")


def expand_targets(
    subject: ShotSubject,
) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
    """Zero-order hold: one target and one observed flag per FRAME.

    Frames in [sampled[j], sampled[j+1]) take sample j's point; frames
    before the first sample BACK-FILL it; frames after the last HOLD it.
    observed is True only at a sampled frame whose sample was itself
    observed.
    """
    _validate_shot_subject(subject)
    targets: List[float] = []
    observed: List[bool] = []
    index = 0
    by_frame = {s.frame: s for s in subject.samples}
    for frame in range(subject.start_frame, subject.end_frame):
        while (index + 1 < len(subject.sampled_frames)
               and subject.sampled_frames[index + 1] <= frame):
            index += 1
        targets.append(subject.samples[index].cx)
        sample = by_frame.get(frame)
        observed.append(bool(sample is not None and sample.observed))
    return tuple(targets), tuple(observed)


@dataclass(frozen=True)
class ShotTrajectory:
    """The per-frame crop track for ONE shot. All tuples same length."""

    shot_index: int
    start_frame: int
    end_frame: int
    fps: float
    x: Tuple[int, ...]
    subject_x: Tuple[int, ...]
    observed: Tuple[bool, ...]
    inside: Tuple[bool, ...]
    clamped: Tuple[bool, ...]
    has_subject: bool
    switch_frames: Tuple[int, ...]
    engagements: int
    # --- title detection. Appended at the END with defaults so existing
    # positional and keyword constructions stay valid. `title` is typed
    # Any rather than text.TitleCard on purpose: camera.py only ever
    # touches .centre_x, .width and .to_json(), and a type-only import
    # would put text.py -- which lazily reaches for cv2 -- on this
    # module's import path for no benefit.
    treatment: str = TREATMENT_CROP
    title: Any = None

    def __post_init__(self) -> None:
        """treatment and title must agree; every per-frame tuple must
        hold one value per frame of the shot."""
        if self.treatment not in (TREATMENT_CROP, TREATMENT_LETTERBOX):
            raise CameraError(
                f"shot {self.shot_index}: unknown treatment "
                f"{self.treatment!r}")
        if self.treatment == TREATMENT_LETTERBOX and self.title is None:
            raise CameraError(
                f"shot {self.shot_index}: a letterbox shot needs the "
                "title card that caused it")
        for name in ("x", "subject_x", "observed", "inside", "clamped"):
            count = len(getattr(self, name))
            if count != self.frame_count:
                raise CameraError(
                    f"shot {self.shot_index}: {name} has {count} values "
                    f"for {self.frame_count} frames")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def x_at(self, frame: int) -> int:
        if not self.start_frame <= frame < self.end_frame:
            raise CameraError(
                f"frame {frame} is outside shot {self.shot_index} "
                f"[{self.start_frame}, {self.end_frame})")
        return self.x[frame - self.start_frame]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _validate_motion(
    dead_zone_release: float, max_pan_frac_per_sec: float,
    smoothing_hz: float,
) -> None:
    if not 0.0 <= dead_zone_release < 1.0:
        raise CameraError(
            "REFRAME_DEAD_ZONE_RELEASE must be in [0, 1), got "
            f"{dead_zone_release}")
    if not 0.0 < max_pan_frac_per_sec <= 1.0:
        raise CameraError(
            "REFRAME_MAX_PAN_FRAC_PER_SEC must be in (0, 1], got "
            f"{max_pan_frac_per_sec}")
    if smoothing_hz <= 0.0:
        raise CameraError(
            f"REFRAME_SMOOTHING_HZ must be > 0, got {smoothing_hz}")


def _plan_title_shot(
    subject: ShotSubject, window: CropWindow,
    targets: Tuple[float, ...], observed: Tuple[bool, ...],
    title: Any, half_dead: float, edge_margin: float,
) -> ShotTrajectory:
    """A title shot: constant x at the text centroid, nothing else.

    The subject track is still expanded and recorded. The centroid
    governs only the "crop" treatment -- a letterbox shot crops the
    whole canvas, so its x never reaches the pixels -- but recording it
    is what makes the decision auditable after the fact, and what a
    later "was letterboxing actually needed?" review reads.

    `inside` and `clamped` are computed against the planned window
    exactly as a normal shot would, so the per-frame arrays keep
    meaning something. dead_zone_report excludes them from the dead-zone
    aggregate instead; nothing is falsified here.
    """
    frame_width = window.frame_width
    left = title_x(title, window)
    centre = (left + window.crop_w / 2.0) / frame_width
    xs: List[int] = []
    subject_px: List[int] = []
    insides: List[bool] = []
    clampeds: List[bool] = []
    for point in targets:
        xs.append(left)
        subject_px.append(
            min(max(int(round(point * frame_width)), 0), frame_width))
        error = point - centre
        is_clamped = (
            (left == 0 and error < 0.0)
            or (left == window.max_x and error > 0.0))
        clampeds.append(is_clamped)
        insides.append(abs(error) <= half_dead + 1e-12 or is_clamped)
    return ShotTrajectory(
        shot_index=subject.shot_index,
        start_frame=subject.start_frame,
        end_frame=subject.end_frame,
        fps=subject.fps,
        x=tuple(xs),
        subject_x=tuple(subject_px),
        observed=observed,
        inside=tuple(insides),
        clamped=tuple(clampeds),
        has_subject=subject.has_subject,
        switch_frames=tuple(subject.switch_frames),
        engagements=0,
        treatment=title_treatment(title, window, edge_margin),
        title=title,
    )


def plan_shot(
    subject: ShotSubject,
    window: CropWindow,
    *,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    dead_zone_release: float = DEFAULT_DEAD_ZONE_RELEASE,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
    max_pan_frac_per_sec: float = DEFAULT_MAX_PAN_FRAC_PER_SEC,
    title: Any = None,
    edge_margin: float = DEFAULT_TITLE_EDGE_MARGIN,
) -> ShotTrajectory:
    """Plan the crop trajectory for ONE shot.

    Hard reset at the shot start, latched dead zone, critically-damped
    step, pan cap with anti-wind-up, edge clamp that kills the velocity,
    even-pixel output.

    A `title` short-circuits all of that to a constant x at the text
    centroid: there is nothing to follow and nothing to smooth.
    """
    _validate_motion(
        dead_zone_release, max_pan_frac_per_sec, smoothing_hz)
    half_dead = dead_zone_half(window, dead_zone)
    targets, observed = expand_targets(subject)

    if title is not None:
        return _plan_title_shot(
            subject, window, targets, observed, title, half_dead,
            edge_margin)

    frame_width = window.frame_width
    half_crop = window.half_crop_norm
    low, high = half_crop, 1.0 - half_crop
    release = dead_zone_release * half_dead
    dt = 1.0 / subject.fps
    max_delta = max_pan_frac_per_sec / subject.fps

    centre = _clamp(targets[0], low, high)     # the hard reset
    velocity = 0.0
    engaged = False
    engagements = 0
    xs: List[int] = []
    subject_px: List[int] = []
    insides: List[bool] = []
    clampeds: List[bool] = []

    for index, point in enumerate(targets):
        if index > 0:
            error = point - centre
            if engaged:
                if abs(error) <= release:
                    engaged = False
            elif abs(error) > half_dead:
                engaged = True
                engagements += 1
            target = _clamp(point, low, high) if engaged else centre
            moved, next_velocity = smooth_step(
                centre, velocity, target, dt, smoothing_hz)
            delta = moved - centre
            if abs(delta) > max_delta:
                delta = math.copysign(max_delta, delta)
                moved = centre + delta
                next_velocity = delta / dt
            if moved < low:
                moved, next_velocity = low, 0.0
            elif moved > high:
                moved, next_velocity = high, 0.0
            centre, velocity = moved, next_velocity

        left = even_round((centre - half_crop) * frame_width)
        left = min(max(left, 0), window.max_x)
        xs.append(left)
        subject_px.append(
            min(max(int(round(point * frame_width)), 0), frame_width))
        error = point - centre
        is_clamped = (
            (left == 0 and error < 0.0)
            or (left == window.max_x and error > 0.0))
        clampeds.append(is_clamped)
        insides.append(abs(error) <= half_dead + 1e-12 or is_clamped)

    return ShotTrajectory(
        shot_index=subject.shot_index,
        start_frame=subject.start_frame,
        end_frame=subject.end_frame,
        fps=subject.fps,
        x=tuple(xs),
        subject_x=tuple(subject_px),
        observed=observed,
        inside=tuple(insides),
        clamped=tuple(clampeds),
        has_subject=subject.has_subject,
        switch_frames=tuple(subject.switch_frames),
        engagements=engagements,
    )


@dataclass(frozen=True)
class CropTrack:
    """Everything render.py needs for the whole clip."""

    window: CropWindow
    fps: float
    stride: int
    total_frames: int
    dead_zone: float
    dead_zone_release: float
    smoothing_hz: float
    max_pan_frac_per_sec: float
    shots: Tuple[ShotTrajectory, ...]
    # --- title detection. None when no shot letterboxes, which is the
    # common case and the one render.py keeps its existing graph for.
    canvas: Optional[Canvas] = None

    def __post_init__(self) -> None:
        """A canvas exists exactly when some shot letterboxes: render.py
        picks its filtergraph from the canvas alone."""
        boxed = any(shot.treatment == TREATMENT_LETTERBOX
                    for shot in self.shots)
        if boxed and self.canvas is None:
            raise CameraError(
                "a track with a letterbox shot needs a canvas")
        if not boxed and self.canvas is not None:
            raise CameraError(
                "a canvas was given but no shot letterboxes")

    def x_at(self, frame: int) -> int:
        for shot in self.shots:
            if shot.start_frame <= frame < shot.end_frame:
                return shot.x[frame - shot.start_frame]
        raise CameraError(
            f"frame {frame} is outside the clip [0, {self.total_frames})")

    def flatten(self) -> Tuple[int, ...]:
        values: List[int] = []
        for shot in self.shots:
            values.extend(shot.x)
        return tuple(values)


def _validate_tiling(subjects: Sequence[ShotSubject]) -> None:
    """The tiling shots.py guarantees, re-checked at the boundary."""
    if not subjects:
        raise CameraError("subject list is empty; shots.py never "
                          "returns an empty shot list")
    if subjects[0].start_frame != 0:
        raise CameraError(
            "first shot must start at frame 0, got "
            f"{subjects[0].start_frame}")
    fps = subjects[0].fps
    stride = subjects[0].stride
    for index, subject in enumerate(subjects):
        if subject.shot_index != index:
            raise CameraError(
                f"shot_index {subject.shot_index} at position {index}; "
                "shot_index must be 0..n-1 in order")
        if subject.fps != fps:
            raise CameraError(
                f"shot {subject.shot_index} has fps {subject.fps}, but "
                f"shot 0 has {fps}; one clip has one frame rate")
        if subject.stride != stride:
            raise CameraError(
                f"shot {subject.shot_index} has stride {subject.stride}, "
                f"but shot 0 has stride {stride}; one clip has one "
                "detection stride")
        if index and subject.start_frame != subjects[index - 1].end_frame:
            raise CameraError(
                f"shot {subject.shot_index} starts at "
                f"{subject.start_frame} but shot "
                f"{subjects[index - 1].shot_index} ends at "
                f"{subjects[index - 1].end_frame}; shots must tile the "
                "clip with no gap and no overlap")


def plan_camera(
    subjects: Sequence[ShotSubject],
    meta: SourceMeta,
    *,
    aspect_w: int = DEFAULT_ASPECT_W,
    aspect_h: int = DEFAULT_ASPECT_H,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    dead_zone_release: float = DEFAULT_DEAD_ZONE_RELEASE,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
    max_pan_frac_per_sec: float = DEFAULT_MAX_PAN_FRAC_PER_SEC,
    titles: Optional[Sequence[Any]] = None,
    edge_margin: float = DEFAULT_TITLE_EDGE_MARGIN,
) -> CropTrack:
    """Plan the crop trajectory for a whole clip. THE entry point.

    `titles` is one Optional[TitleCard] per shot, aligned to
    `subjects`, exactly as text.probe_clip returns. Passing None --
    the default -- reproduces the behaviour without title detection
    bit-for-bit; that is this module's regression contract.
    """
    _validate_tiling(subjects)
    if abs(subjects[0].fps - meta.fps) > 1e-6:
        raise CameraError(
            f"shot fps {subjects[0].fps} disagrees with source fps "
            f"{meta.fps}")
    window = crop_window(meta, aspect_w=aspect_w, aspect_h=aspect_h)
    if titles is not None and len(titles) != len(subjects):
        raise CameraError(
            f"{len(titles)} title(s) for {len(subjects)} shot(s); "
            "text.probe_clip returns one entry per shot")
    per_shot = tuple(titles) if titles is not None \
        else (None,) * len(subjects)
    shots = tuple(
        plan_shot(
            subject, window,
            dead_zone=dead_zone,
            dead_zone_release=dead_zone_release,
            smoothing_hz=smoothing_hz,
            max_pan_frac_per_sec=max_pan_frac_per_sec,
            title=title,
            edge_margin=edge_margin,
        )
        for subject, title in zip(subjects, per_shot)
    )
    canvas = (canvas_for(window)
              if any(shot.treatment == TREATMENT_LETTERBOX
                     for shot in shots) else None)
    return CropTrack(
        window=window,
        fps=subjects[0].fps,
        stride=subjects[0].stride,
        total_frames=subjects[-1].end_frame,
        dead_zone=dead_zone,
        dead_zone_release=dead_zone_release,
        smoothing_hz=smoothing_hz,
        max_pan_frac_per_sec=max_pan_frac_per_sec,
        shots=shots,
        canvas=canvas,
    )


def crop_changes(track: CropTrack) -> List[Tuple[int, int]]:
    """(frame, x) only where x changes. Lossless; first entry is frame 0.

    render.py turns these into sendcmd lines. A static shot collapses to
    one entry; replaying them reconstructs flatten() exactly.
    """
    flat = track.flatten()
    if not flat:
        raise CameraError("crop track has no frames")
    changes = [(0, flat[0])]
    for frame in range(1, len(flat)):
        if flat[frame] != flat[frame - 1]:
            changes.append((frame, flat[frame]))
    return changes


def crop_rects(
    track: CropTrack,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """One (w, h, x, y) per FRAME, in CANVAS coordinates.

    Only defined for a clip that HAS a canvas. Without one there is no
    padding, w/h/y never move, and render.py keeps emitting the x-only
    command stream it emits today -- which is what makes that stream
    byte-identical by construction rather than by test.

    A "crop" shot reads the real image where the pad put it, at
    y = pad_y. A "letterbox" shot takes the whole canvas at (0, 0), and
    the static scale afterwards fits it into crop_w x crop_h.
    """
    canvas = track.canvas
    if canvas is None:
        raise CameraError(
            "crop_rects needs a canvas; this clip has no letterbox "
            "shot and render.py uses the x-only command stream for it")
    window = track.window
    rects: List[Tuple[int, int, int, int]] = []
    for shot in track.shots:
        if shot.treatment == TREATMENT_LETTERBOX:
            rects.extend([(canvas.width, canvas.height, 0, 0)]
                         * shot.frame_count)
        else:
            rects.extend(
                (window.crop_w, window.crop_h, value, canvas.pad_y)
                for value in shot.x)
    if len(rects) != track.total_frames:
        raise CameraError(
            f"crop_rects produced {len(rects)} rectangles for "
            f"{track.total_frames} frames")
    return tuple(rects)


def rect_changes(track: CropTrack) -> List[Tuple[int, Dict[str, int]]]:
    """(frame, {changed component: value}). Lossless; frame 0 is full.

    Only the components that actually MOVED are emitted, because
    sendcmd's parse cost is super-linear -- 106 us per command at 36 000
    commands against 10 us at 1 000 -- and re-stating a static w/h/y on
    every pan would quadruple the stream for nothing.
    """
    rects = crop_rects(track)
    first = rects[0]
    changes: List[Tuple[int, Dict[str, int]]] = [
        (0, {"w": first[0], "h": first[1], "x": first[2],
             "y": first[3]})]
    for frame in range(1, len(rects)):
        delta = {
            key: rects[frame][index]
            for key, index in (("w", 0), ("h", 1), ("x", 2), ("y", 3))
            if rects[frame][index] != rects[frame - 1][index]
        }
        if delta:
            changes.append((frame, delta))
    return changes


@dataclass(frozen=True)
class DeadZoneReport:
    """The dead-zone metric (subject inside the dead zone), as data."""

    observed_samples: int
    inside_samples: int
    clamped_samples: int
    switch_samples: int
    # --- title detection. Observed samples belonging to a TITLE shot,
    # excluded from observed_samples / inside_samples above. The
    # dead-zone metric measures people-following, and a title shot
    # deliberately stops following people -- so counting those frames
    # would make a correct letterbox read as a 95%-threshold miss. The
    # count is published so the exclusion is visible rather than
    # assumed, exactly as switch_samples and `measurable` already are.
    title_samples: int = 0

    @property
    def fraction(self) -> float:
        if self.observed_samples == 0:
            return 1.0
        return round(self.inside_samples / self.observed_samples, 6)

    @property
    def passes(self) -> bool:
        return self.fraction >= SC2_THRESHOLD

    @property
    def measurable(self) -> bool:
        """False when nothing was observed -- distinguishes "the
        dead-zone metric passed" from "there was nothing to measure".
        fraction and passes are unaffected: a shot with no subject
        still passes, correctly."""
        return self.observed_samples > 0


def _switch_recovery_samples(shot: ShotTrajectory) -> int:
    """Observed samples spent recovering from a subject switch.

    For each switch_frame, the recovering
    window runs from that frame up to and including the first later
    frame where `inside` is True again, or the shot end if the camera
    never recovers within the shot. Windows are unioned across
    switches so two switches close together are not double counted.
    """
    recovering = [False] * shot.frame_count
    for switch_frame in shot.switch_frames:
        start = switch_frame - shot.start_frame
        end = start
        while end < shot.frame_count - 1 and not shot.inside[end]:
            end += 1
        for index in range(start, end + 1):
            recovering[index] = True
    return sum(
        1 for index in range(shot.frame_count)
        if recovering[index] and shot.observed[index]
    )


def dead_zone_report(track: CropTrack) -> DeadZoneReport:
    """Count the dead-zone metric over OBSERVED samples only.

    A pure counter: `clamped` is recorded per frame by plan_shot at the
    one place that knows whether the edge clamp fired, and this
    function performs no geometry of its own. `switch_samples` counts
    only the recovery window after a subject switch, not the whole
    remaining shot.
    """
    observed = inside = clamped = switches = titles = 0
    for shot in track.shots:
        is_title = shot.title is not None
        # A title shot's samples are left out of observed_samples, so
        # its switches are left out too: switch_samples explains misses
        # among the samples the criterion counts, and no others.
        if shot.switch_frames and not is_title:
            switches += _switch_recovery_samples(shot)
        for index in range(shot.frame_count):
            if not shot.observed[index]:
                continue
            if is_title:
                titles += 1
                continue
            observed += 1
            if shot.inside[index]:
                inside += 1
                if shot.clamped[index]:
                    clamped += 1
    return DeadZoneReport(
        observed_samples=observed, inside_samples=inside,
        clamped_samples=clamped, switch_samples=switches,
        title_samples=titles,
    )


def camera_summary(track: CropTrack) -> str:
    """One ASCII line for run_stages to print. No I/O here."""
    report = dead_zone_report(track)
    switches = sum(len(shot.switch_frames) for shot in track.shots)
    titles = sum(1 for shot in track.shots if shot.title is not None)
    extra = ""
    if titles:
        boxed = sum(1 for shot in track.shots
                    if shot.treatment == TREATMENT_LETTERBOX)
        extra = f", {titles} title shot(s) ({boxed} letterboxed)"
    return (
        f"crop {track.window.crop_w}x{track.window.crop_h}, "
        f"{len(track.shots)} shot(s), {track.total_frames} frame(s), "
        f"dead-zone {report.fraction * 100.0:.1f}% of "
        f"{report.observed_samples} observed sample(s) "
        f"({report.clamped_samples} clamped), {switches} switch(es)"
        f"{extra}")


def crop_track_to_json(track: CropTrack) -> dict:
    """The trajectory payload. render.py owns crops.json's envelope."""
    report = dead_zone_report(track)
    return {
        "crop_w": track.window.crop_w,
        "crop_h": track.window.crop_h,
        "y": track.window.y,
        "frame_width": track.window.frame_width,
        "frame_height": track.window.frame_height,
        "fps": track.fps,
        # The dead-zone metric is stride-sensitive, so the retained
        # crops.json has to name the stride that produced it.
        "stride": track.stride,
        "total_frames": track.total_frames,
        "dead_zone": track.dead_zone,
        "dead_zone_release": track.dead_zone_release,
        "smoothing_hz": track.smoothing_hz,
        "max_pan_frac_per_sec": track.max_pan_frac_per_sec,
        "dead_zone_fraction": report.fraction,
        "sc2_measurable": report.measurable,
        "observed_samples": report.observed_samples,
        "inside_samples": report.inside_samples,
        "clamped_samples": report.clamped_samples,
        "switch_samples": report.switch_samples,
        "title_samples": report.title_samples,
        "shots": [
            {
                "index": shot.shot_index,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "has_subject": shot.has_subject,
                "engagements": shot.engagements,
                "switch_frames": list(shot.switch_frames),
                "x": list(shot.x),
                "subject_x": list(shot.subject_x),
                "observed": [bool(v) for v in shot.observed],
                "inside": [bool(v) for v in shot.inside],
                "treatment": shot.treatment,
                **({"title": shot.title.to_json()}
                   if shot.title is not None else {}),
            }
            for shot in track.shots
        ],
    }
