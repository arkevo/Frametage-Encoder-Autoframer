"""The reframe stage sequence: shots -> detect -> subject -> camera -> render.

This module is the single seam between the entry point (which owns the
CLI, the input gate, and the artifact contract) and the compute stages
(which own the actual reframing). main.py and its tests are written
against run_stages' signature, which therefore never changes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .camera import camera_summary, plan_camera
from .detect import detect_from_settings
from .probe import SourceMeta
from .render import render_summary, render_vertical
from .settings import ReframeSettings
from .shots import detect_shots
from .subject import select_subjects, subject_summary
from .text import probe_clip, title_summary


@dataclass(frozen=True)
class Stages:
    """The five stage entry points run_stages calls, as data.

    Private — not part of the published contract. Tests build a Stages of
    plain local functions instead of patching this module's attributes,
    which would couple every test to its import style.
    """

    shots: Callable
    detect: Callable
    subjects: Callable
    camera: Callable
    render: Callable
    # --- title detection (text.py). Appended at the END with a default
    # so every existing five-field construction in the tests stays
    # valid. The field ORDER here has nothing to do with the call order
    # below.
    titles: Callable = probe_clip


DEFAULT_STAGES = Stages(
    shots=detect_shots, detect=detect_from_settings,
    subjects=select_subjects, camera=plan_camera, render=render_vertical,
    titles=probe_clip,
)


def run_stages(
    source: Path,
    meta: SourceMeta,
    out_path: Path,
    crops_path: Path,
    settings: ReframeSettings,
) -> None:
    """Reframe `source` to `out_path`, writing the crop track to `crops_path`.

    `meta` is already probed — do not probe again. All geometry is in
    DISPLAY space: use meta.display_width / meta.display_height, never
    meta.width / meta.height.

    `out_path` and `crops_path` are FINAL destinations. Each is written to
    its sibling temp name — `render._partial(path)`, which keeps the real
    extension (`vertical.partial.mp4`, not `vertical.mp4.partial`) so
    ffmpeg can infer the muxer — and renamed on success, so a partially
    written intermediate can never be picked up downstream.

    Stdout is reserved for the entry point's single JSON payload. This
    function does not redirect it: the caller does (main.py wraps the
    call in contextlib.redirect_stdout(sys.stderr)), so the stage
    modules may print() progress freely. Any other caller that needs a
    clean stdout must redirect it the same way.

    Raises ReframeError (or a subclass) for an expected stage failure.
    A programming error, such as a non-memory failure inside a detector
    call, propagates unchanged so its traceback survives. On success
    BOTH files exist.
    """
    _run_stages(source, meta, out_path, crops_path, settings)


def _run_stages(
    source: Path,
    meta: SourceMeta,
    out_path: Path,
    crops_path: Path,
    settings: ReframeSettings,
    *,
    stages: Stages = DEFAULT_STAGES,
) -> None:
    """The real five-stage sequence, injectable via `stages` for tests.

    The sequence is flat and unconditional: every stage guarantees a
    usable result or raises. detect_shots yields at least one shot,
    select_subjects yields one ShotSubject per shot (a static centre crop
    when there is nobody to follow), plan_camera yields a dense per-frame
    trajectory, and render_vertical yields both artifacts.

    Each stage but detect prints its own summary as it completes, so a
    clip that dies in detection has already reported how many shots were
    found; detect logs its own richer line from inside
    detect_from_settings — carrying the realtime ratio (clip seconds
    per wall-clock second) — and is not duplicated here.

    The title probe runs BEFORE detection, not after: it loads a model
    and decodes the clip, so a missing or unreadable weights file should
    fail the clip in a second rather than after the expensive detect
    pass. It returns None -- not a tuple of Nones -- when
    REFRAME_TITLE_ENABLED is false, and None is exactly what
    plan_camera needs to reproduce the trajectory without title
    detection bit-for-bit.
    """
    shots = stages.shots(
        meta,
        threshold=settings.scene_threshold,
        min_shot_sec=settings.min_shot_sec,
    )
    print(f"  reframe: {len(shots)} shot(s)")

    titles = stages.titles(source, shots, meta, settings)
    print(f"  reframe: {title_summary(titles)}")

    detections, _stats = stages.detect(meta, shots, settings)

    subjects = stages.subjects(
        shots, detections, meta,
        face_weight=settings.face_weight,
        hysteresis_margin=settings.hysteresis_margin,
        hysteresis_sec=settings.hysteresis_sec,
        centrality_weight=settings.centrality_weight,
        confidence_weight=settings.confidence_weight,
        persistence_weight=settings.persistence_weight,
        min_confidence_person=settings.min_confidence_person,
        min_confidence_face=settings.min_confidence_face,
        iou_match_min=settings.iou_match_min,
        track_gap_sec=settings.track_gap_sec,
    )
    print(f"  reframe: {subject_summary(subjects)}")

    track = stages.camera(
        subjects, meta,
        aspect_w=settings.aspect_w,
        aspect_h=settings.aspect_h,
        dead_zone=settings.dead_zone,
        dead_zone_release=settings.dead_zone_release,
        smoothing_hz=settings.smoothing_hz,
        max_pan_frac_per_sec=settings.max_pan_frac_per_sec,
        titles=titles,
        edge_margin=settings.title_edge_margin,
    )
    print(f"  reframe: {camera_summary(track)}")

    result = stages.render(
        source=source,
        meta=meta,
        track=track,
        shots=shots,
        out_path=out_path,
        crops_path=crops_path,
        settings=settings,
    )
    print(f"  reframe: {render_summary(result)}")
