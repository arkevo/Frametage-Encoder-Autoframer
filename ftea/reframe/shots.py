"""Shot boundaries for the auto-reframe stage.

Runs the encoder's ffmpeg scene-detection command
(`ftea.encode.scene.detect_scene_timestamps`, kept identical by a test)
and turns its second-based cut list into absolute source frame ranges.
Every downstream reframe module works strictly per shot: subject
selection, hysteresis and the crop trajectory all reset at a cut and
never smooth across one.

Ranges are HALF-OPEN: `start_frame` is included, `end_frame` is not, so
`end_frame - start_frame` is the frame count and one shot's `end_frame`
is the next shot's `start_frame`. The returned shots always tile
`[0, total_frames)` exactly, with no gaps and no overlap.

stdlib + the ffmpeg/ffprobe binaries only. No GPU, no torch: the whole
module is unit-testable on any machine.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .errors import ReframeError
from .probe import EXIT_ERROR, ProbeError, SourceMeta, probe_source

# Same value as the encoder's SCENE_THRESHOLD default in
# config.env.example. Not read from config.env here: settings.py reads
# REFRAME_SCENE_THRESHOLD and run_stages passes it in as an argument.
DEFAULT_SCENE_THRESHOLD = 0.3

# Shots shorter than this are treated as false cuts (a flash, a one-frame
# transition, a whip pan) and merged into their neighbour. Deliberately
# far below any real narrative shot and far above a strobe. This is NOT
# the encoder's MAX_SCENE_SEC splitting (encode.scene.split_long_scenes)
# -- nothing here ever ADDS a boundary.
DEFAULT_MIN_SHOT_SEC = 0.4

# How far from EOF the decode-integrity probe starts. One second is long
# enough to force a real decode and short enough that the probe costs a
# few percent of the scene-detection pass.
INTEGRITY_TAIL_SEC = 1.0


class ShotsError(ReframeError):
    """Raised when shot boundaries cannot be trusted for a source."""


@dataclass(frozen=True)
class Shot:
    """One continuous take, as an absolute source frame range.

    The range is half-open: frames `start_frame` through `end_frame - 1`
    belong to this shot. `fps` is carried so a shot is self-describing --
    a crops.json entry or a log line never needs the SourceMeta beside it.
    """

    index: int
    start_frame: int
    end_frame: int
    fps: float

    @property
    def frame_count(self) -> int:
        """Number of frames in the shot."""
        return self.end_frame - self.start_frame

    @property
    def start_sec(self) -> float:
        """Presentation time of the first frame, in seconds."""
        return round(self.start_frame / self.fps, 3)

    @property
    def end_sec(self) -> float:
        """Exclusive end time, in seconds -- the next shot's start_sec."""
        return round(self.end_frame / self.fps, 3)

    @property
    def duration_s(self) -> float:
        """Shot length in seconds.

        Derived from the two published second boundaries -- NOT from
        frame_count / fps independently -- so start_sec + duration_s
        always equals end_sec exactly, even at non-integer frame rates
        (e.g. 29.97) where three separate roundings could disagree.
        """
        return round(self.end_sec - self.start_sec, 3)

    def contains(self, frame: int) -> bool:
        """Report whether an absolute source frame falls in this shot."""
        return self.start_frame <= frame < self.end_frame


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Convert a presentation time to its absolute source frame index.

    Rounds rather than truncates: a scene cut's pts_time is the exact
    presentation time of a real frame, so 2.0s at 25fps is frame 50, but
    float noise can render it 1.9999999 and truncation would give 49.
    """
    if fps <= 0:
        raise ShotsError(f"non-positive fps {fps}")
    return int(round(seconds * fps))


def total_frame_count(meta: SourceMeta) -> int:
    """Total number of frames in the source.

    Derived from duration x fps rather than ffprobe's nb_frames, which is
    absent for Matroska and other containers.
    """
    return seconds_to_frame(meta.duration_s, meta.fps)


def cut_frames(
    timestamps: list[float], fps: float, total_frames: int,
) -> list[int]:
    """Turn scene-cut seconds into sorted, de-duplicated frame indices.

    Always starts at frame 0 whatever the input, and drops any cut at or
    past the end of the clip -- a boundary there would create an empty
    trailing shot.
    """
    frames = {0}
    for ts in timestamps:
        frame = seconds_to_frame(ts, fps)
        if 0 < frame < total_frames:
            frames.add(frame)
    return sorted(frames)


def drop_short_shots(
    boundaries: list[int], total_frames: int, min_frames: int,
) -> list[int]:
    """Remove boundaries that would create a shot shorter than min_frames.

    Only ever REMOVES boundaries, so a real take is never split. A too
    short piece merges into the take before it; a too short FIRST piece
    merges into the take after it, which falls out of keeping frame 0 and
    dropping the offending boundary.

    Written here rather than reusing encode.scene.merge_short_scenes because
    that function works in seconds, needs the total duration appended to
    its input, and leaves a short FINAL scene in place (its tail branch
    appends the last timestamp unconditionally).
    """
    if min_frames <= 1 or len(boundaries) <= 1:
        return list(boundaries)

    kept = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - kept[-1] >= min_frames:
            kept.append(boundary)

    while len(kept) > 1 and total_frames - kept[-1] < min_frames:
        kept.pop()

    return kept


def boundaries_to_shots(
    boundaries: list[int], total_frames: int, fps: float,
) -> list[Shot]:
    """Expand start-frame boundaries into half-open Shot ranges."""
    edges = list(boundaries) + [total_frames]
    return [
        Shot(index=i, start_frame=edges[i], end_frame=edges[i + 1], fps=fps)
        for i in range(len(edges) - 1)
    ]


@dataclass(frozen=True)
class TailProbe:
    """Result of decoding a source's tail: what came out, and when.

    `last_pts` is the ABSOLUTE source presentation time (seconds) of the
    last decoded frame -- meaningful only because tail_decode_probe
    passes -copyts, without which -sseof reports seek-relative
    timestamps that cannot be compared against total_frame_count(meta).
    """

    frames: int
    last_pts: float


IntegrityFn = Callable[[SourceMeta], TailProbe]


def _last_error_line(stderr: str) -> str:
    """Pick the most useful line out of an ffmpeg stderr dump."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "no stderr"


def tail_decode_probe(
    meta: SourceMeta, *, tail_sec: float = INTEGRITY_TAIL_SEC,
) -> TailProbe:
    """Decode a source's tail and report the frame count and last pts.

    This is the integrity guard that detect_scene_timestamps cannot
    provide. That function discards its CompletedProcess, so a caller
    cannot see the return code, and even the return code is not enough:
    an mp4 whose moov atom survives but whose media data is truncated
    makes ffmpeg exit 0 having decoded nothing at all.

    -copyts makes -sseof report SOURCE-absolute presentation times
    rather than seek-relative ones -- required so the returned last_pts
    is directly comparable to total_frame_count(meta) in
    assert_frame_count_consistent.

    Costs roughly one GOP of decoding -- measured at 0.08s against 1.41s
    for the full scene-detection pass on a 60s 1080p GOP-300 clip.

    The frame cap spans the whole tail window rather than a single
    frame. `-frames:v 1` stops at the FIRST frame after the seek, so
    last_pts would report where the tail window STARTS, not the clip's
    final frame -- measured on Windows ffmpeg 8.1 as 3.04s on a clip
    whose last frame is at 3.96s, which makes
    assert_frame_count_consistent reject every healthy source. The cap
    is still bounded (one window plus two frames of slack), so the decode
    cost stays at roughly one GOP.

    Raises ShotsError when ffmpeg cannot open or decode the source.
    """
    max_frames = seconds_to_frame(tail_sec, meta.fps) + 2
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-copyts",
        "-sseof", f"-{tail_sec:g}",
        "-i", str(meta.path),
        "-vf", "showinfo",
        "-frames:v", str(max_frames),
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ShotsError(
            f"ffmpeg cannot decode {meta.path}: "
            f"{_last_error_line(exc.stderr or '')}"
        ) from None
    except FileNotFoundError:
        raise ShotsError("ffmpeg not found on PATH") from None

    frames = 0
    last_pts = 0.0
    for line in result.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        frames += 1
        for token in line.split():
            if token.startswith("pts_time:"):
                last_pts = max(last_pts, float(token.split(":", 1)[1]))
    return TailProbe(frames=frames, last_pts=last_pts)


def assert_decodable(
    meta: SourceMeta, *, integrity_fn: Optional[IntegrityFn] = None,
) -> TailProbe:
    """Fail loudly when a source cannot actually be decoded.

    Run BEFORE scene detection so a corrupt clip costs one GOP instead of
    a full decode pass. Returns the TailProbe so a caller can also run
    assert_frame_count_consistent without paying for a second decode.
    """
    probe = (integrity_fn or tail_decode_probe)(meta)
    if probe.frames < 1:
        raise ShotsError(
            f"ffmpeg decoded no frames near the end of {meta.path}; "
            "the source is truncated or has no usable video stream"
        )
    return probe


def assert_frame_count_consistent(
    meta: SourceMeta, probe: TailProbe, *, tolerance_frames: int = 2,
) -> None:
    """Cross-check total_frame_count(meta) against a real decoded pts.

    duration_s x fps assumes a constant frame rate. For a variable-frame-
    rate source, SourceMeta.fps (r_frame_rate) is the nominal MAXIMUM
    rate, so the derived count can silently exceed the true decoded
    count. total_frame_count is the anchor every downstream reframe
    module sizes its per-frame arrays from, so a mismatch here would
    desync the crop track from the real frames with no error anywhere
    else. Also catches a wrong format.duration and a truncated-but-
    openable container.

    Raises ShotsError naming both numbers when they disagree by more
    than tolerance_frames.
    """
    derived = round(probe.last_pts * meta.fps) + 1
    expected = total_frame_count(meta)
    if abs(derived - expected) > tolerance_frames:
        raise ShotsError(
            f"frame count mismatch for {meta.path}: total_frame_count "
            f"reports {expected} frames but the last decoded frame's "
            f"presentation time ({probe.last_pts}s) implies {derived} "
            "-- the source may be variable-frame-rate or have the "
            "wrong duration"
        )


SceneFn = Callable[[Path, float], list[float]]


def _scene_timestamps_checked(
    source: Path, threshold: float,
) -> list[float]:
    """Run the encoder's scene-detection command with check=True.

    Same argv as ftea.encode.scene.detect_scene_timestamps -- copied
    deliberately for returncode visibility;
    TestSceneCommandMatchesTheEncoder keeps the two identical. That
    function discards its CompletedProcess, so a killed or otherwise
    failed detection pass -- even after a healthy tail-decode guard --
    would still return [0.0], indistinguishable from a legitimate
    single-take clip. This is the default scene_fn: the detector's
    COMMAND is reused, its RESULT HANDLING is owned here.
    """
    cmd = [
        "ffmpeg", "-i", str(source),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ShotsError(
            f"scene detection failed for {source}: "
            f"{_last_error_line(exc.stderr or '')}"
        ) from None
    except FileNotFoundError:
        raise ShotsError("ffmpeg not found on PATH") from None
    return [0.0] + [
        float(line.split("pts_time:")[1].split()[0])
        for line in result.stderr.splitlines()
        if "pts_time:" in line
    ]


def _validate_tiling(shots: list[Shot], total_frames: int) -> None:
    """Assert the shots tile [0, total_frames) with no gap or overlap."""
    if not shots:
        raise ShotsError("shot detection produced no shots")
    if shots[0].start_frame != 0:
        raise ShotsError(
            f"first shot starts at {shots[0].start_frame}, not 0"
        )
    if shots[-1].end_frame != total_frames:
        raise ShotsError(
            f"last shot ends at {shots[-1].end_frame}, not {total_frames}"
        )
    for previous, current in zip(shots, shots[1:]):
        if previous.end_frame != current.start_frame:
            raise ShotsError(
                f"gap between shot {previous.index} and {current.index}"
            )
    for shot in shots:
        if shot.frame_count <= 0:
            raise ShotsError(f"shot {shot.index} is empty")


def detect_shots(
    meta: SourceMeta,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_shot_sec: float = DEFAULT_MIN_SHOT_SEC,
    scene_fn: Optional[SceneFn] = None,
    integrity_fn: Optional[IntegrityFn] = None,
) -> list[Shot]:
    """Return the source's shots as absolute, half-open frame ranges.

    Takes an already-probed SourceMeta so the file is read once for the
    whole stage; fps and duration come from there rather than from extra
    ffprobe spawns.

    A clip with no detected cut yields exactly one shot spanning the
    whole source -- which is also what a silently failed ffmpeg run would
    look like, so decodability is asserted first and a corrupt source
    raises.

    Pass scene_fn / integrity_fn to substitute fakes in tests.

    Raises ShotsError for an undecodable source, a zero-length source, a
    total_frame_count that disagrees with a real decoded frame (e.g. a
    variable-frame-rate source), or boundaries that fail the tiling
    invariant.
    """
    total = total_frame_count(meta)
    if total < 1:
        raise ShotsError(
            f"source {meta.path} has no frames "
            f"({meta.duration_s}s at {meta.fps}fps)"
        )

    tail_probe = assert_decodable(meta, integrity_fn=integrity_fn)
    assert_frame_count_consistent(meta, tail_probe)

    timestamps = (scene_fn or _scene_timestamps_checked)(
        meta.path, threshold,
    )

    out_of_range = [
        t for t in timestamps
        if seconds_to_frame(t, meta.fps) >= total
    ]
    if len(out_of_range) > 1:
        raise ShotsError(
            f"{len(out_of_range)} detected cuts fall at or past the "
            f"clip end (last at {max(out_of_range)}s); total is "
            f"{total} frames -- this usually means duration_s is wrong"
        )

    boundaries = cut_frames(timestamps, meta.fps, total)
    boundary_count = len(boundaries)
    boundaries = drop_short_shots(
        boundaries, total, seconds_to_frame(min_shot_sec, meta.fps),
    )
    merged = boundary_count - len(boundaries)
    if merged > 0:
        print(
            f"  reframe: merged {merged} short shot(s) under "
            f"{min_shot_sec}s",
            file=sys.stderr,
        )

    shots = boundaries_to_shots(boundaries, total, meta.fps)
    _validate_tiling(shots, total)
    return shots


def shots_to_json(shots: list[Shot]) -> list[dict]:
    """Serialize shots for crops.json and this module's CLI."""
    return [
        {
            "index": shot.index,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "frame_count": shot.frame_count,
            "start_sec": shot.start_sec,
            "end_sec": shot.end_sec,
        }
        for shot in shots
    ]


_USAGE = "Usage: python -m ftea.reframe.shots <video>"


def main(argv: Optional[list[str]] = None) -> int:
    """Print one source's shots as JSON. Run from the repo root."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(_USAGE, file=sys.stderr)
        return EXIT_ERROR

    try:
        meta = probe_source(Path(args[0]))
        shots = detect_shots(meta)
    except (ProbeError, ShotsError) as exc:
        print(f"  reframe: shot detection failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(shots_to_json(shots), indent=2))
    print(
        f"  reframe: {len(shots)} shot(s), "
        f"{total_frame_count(meta)} frames at {meta.fps}fps",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
