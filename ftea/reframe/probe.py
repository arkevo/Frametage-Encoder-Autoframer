"""Input gate for the auto-reframe stage.

Reads a source clip's geometry with a single ffprobe call and decides
whether it needs reframing. Landscape sources proceed to the reframe
pipeline; vertical and square sources pass through untouched.

Deliberately depends on nothing outside the standard library, the
ffprobe binary and the package's own errors.py (which imports nothing)
— no GPU, no torch, no OpenCV — so the whole gate is unit-testable on
any machine.

ffprobe reports CODED dimensions, which for a rotated file are not the
dimensions a viewer (or ffmpeg's autorotating decoder) sees. Everything
downstream of this module works in DISPLAY space: use display_width and
display_height, never width and height.

stdout is reserved for machine-readable JSON; every human-readable line
goes to stderr (callers parse this module's stdout).
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .errors import ReframeError


def normalize_rotation(raw: object) -> int:
    """Normalize an ffprobe rotation value to 0, 90, 180, or 270 degrees.

    ffprobe reports the display-matrix rotation as a signed number of
    degrees; portrait phone footage most commonly carries -90. Values are
    rounded to the nearest quarter turn (exact .5 ties round to even) and
    wrapped into [0, 360). Anything unparseable is treated as no rotation,
    which is the safe default: the clip is judged on its coded dimensions.
    """
    if raw is None:
        return 0
    try:
        degrees = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        print(f"  reframe: unparseable rotation {raw!r}, assuming 0",
              file=sys.stderr)
        return 0
    return int(round(degrees / 90.0)) * 90 % 360


def is_quarter_turn(rotation: int) -> bool:
    """Report whether a rotation swaps the width and height axes."""
    return rotation % 180 == 90


@dataclass(frozen=True)
class SourceMeta:
    """Geometry of one source clip, as read by a single ffprobe call.

    Shared by every reframe module so the file is probed once. `width` and
    `height` are the CODED values straight from ffprobe; `display_width`
    and `display_height` apply the rotation and are what every consumer
    should actually use.
    """

    path: Path
    width: int
    height: int
    rotation: int
    fps: float
    duration_s: float
    has_audio: bool

    @property
    def display_width(self) -> int:
        """Width as displayed, after applying the display matrix."""
        return self.height if is_quarter_turn(self.rotation) else self.width

    @property
    def display_height(self) -> int:
        """Height as displayed, after applying the display matrix."""
        return self.width if is_quarter_turn(self.rotation) else self.height

    @property
    def orientation(self) -> str:
        """Display-space orientation: landscape, portrait, or square.

        Unlike `encode.scene.detect_orientation`, which is two-way and
        reads a square frame as "portrait", this is three-way. Do NOT
        pass this value to `PipelineConfig.build_variants()`, which
        accepts only "portrait" or "landscape" and raises ValueError for
        "square".
        """
        if self.display_width > self.display_height:
            return "landscape"
        if self.display_width < self.display_height:
            return "portrait"
        return "square"


class ProbeError(ReframeError):
    """Raised when ffprobe fails or the source has no usable video stream.

    A ReframeError like every other reframe failure, so one
    `except ReframeError` tells a domain failure from a crash.
    """


# A per-call bound, so a hung ffprobe (a slow network share or a
# cloud-synced input folder) fails the clip with a clear ProbeError in
# a minute, instead of using up the whole REFRAME_TIMEOUT and leaving the
# encoder to report only a timeout.
FFPROBE_TIMEOUT_SEC = 60

MAX_PLAUSIBLE_FPS = 1000

# One call returns width, height, rotation, both frame rates, and both
# durations, for every stream — including audio, so has_audio can be
# derived from this same payload without a second spawn. Deliberately NOT
# -select_streams v:0 or -show_entries: an ffprobe build that does not know
# a requested section name exits 1 with "No match for section", and the
# user's ffprobe version is not pinned.
FFPROBE_ARGS = (
    "-v", "error",
    "-show_streams",
    "-show_format",
    "-of", "json",
)

FFProbeRunner = Callable[[Path], dict]


def _run_ffprobe(source: Path) -> dict:
    """Run ffprobe once and return its parsed JSON payload.

    Checks the return code by hand rather than passing check=True so the
    raised message carries ffprobe's own stderr — this string ends up in
    the batch report as the clip's failure reason. Also converts a missing
    binary and a hang into ProbeError (see FFPROBE_TIMEOUT_SEC above) so
    every caller — the batch report, the entry point's stdout payload,
    and in-process callers such as shots.py — sees exactly one
    exception type, never a bare OSError/TimeoutExpired/UnicodeDecodeError.
    """
    try:
        result = subprocess.run(
            ["ffprobe", *FFPROBE_ARGS, str(source)],
            capture_output=True,
            timeout=FFPROBE_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise ProbeError(
            f"ffprobe timed out after {FFPROBE_TIMEOUT_SEC}s for {source}"
        ) from None
    except OSError as exc:
        raise ProbeError(
            f"could not run ffprobe for {source}: {exc}"
        ) from None
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise ProbeError(f"ffprobe failed for {source}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            f"ffprobe emitted invalid JSON for {source}: {exc}"
        ) from None


def _parse_frame_rate(value: Optional[str]) -> Optional[float]:
    """Parse an ffprobe rational frame rate such as '30000/1001'."""
    if not value:
        return None
    numerator, _, denominator = str(value).partition("/")
    try:
        num = int(numerator)
        den = int(denominator) if denominator else 1
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return round(num / den, 3)


def _parse_duration(value: Optional[str]) -> Optional[float]:
    """Parse an ffprobe duration, rejecting 'N/A' and non-positive values."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _extract_rotation(stream: dict) -> int:
    """Read rotation from the display matrix, falling back to the tag.

    Modern muxers write a Display Matrix side-data entry; older files
    carry only a `rotate` stream tag. ffmpeg 8 no longer writes the tag,
    so the side data wins when both are present.
    """
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return normalize_rotation(side_data["rotation"])
    return normalize_rotation((stream.get("tags") or {}).get("rotate"))


def probe_source(
    source: Path, *, runner: Optional[FFProbeRunner] = None,
) -> SourceMeta:
    """Read one source clip's geometry.

    Pass `runner` to supply a captured ffprobe payload instead of spawning
    the real binary — that is how every parsing test stays hermetic.

    Raises ProbeError when ffprobe fails, when the file has no usable
    video stream (ffprobe exits 0 with no video-typed stream for an
    audio-only file, or with an attached-picture stream masquerading as
    video), or when any required field is missing or nonsensical.
    """
    source = Path(source)
    data = (runner or _run_ffprobe)(source)

    streams = data.get("streams") or []
    stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if stream is None:
        raise ProbeError(f"no video stream in {source}")
    if (stream.get("disposition") or {}).get("attached_pic"):
        raise ProbeError(
            f"no video stream in {source} (attached picture only)"
        )

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError):
        raise ProbeError(f"missing dimensions for {source}") from None
    if width <= 0 or height <= 0:
        raise ProbeError(
            f"non-positive dimensions {width}x{height} for {source}"
        )

    fps = _parse_frame_rate(stream.get("r_frame_rate"))
    if fps is None:
        fps = _parse_frame_rate(stream.get("avg_frame_rate"))
    if fps is None:
        raise ProbeError(f"could not determine frame rate for {source}")
    if fps > MAX_PLAUSIBLE_FPS:
        raise ProbeError(f"implausible frame rate {fps} for {source}")

    duration = _parse_duration(stream.get("duration"))
    if duration is None:
        duration = _parse_duration((data.get("format") or {}).get("duration"))
    if duration is None:
        raise ProbeError(f"could not determine duration for {source}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return SourceMeta(
        path=source,
        width=width,
        height=height,
        rotation=_extract_rotation(stream),
        fps=fps,
        duration_s=round(duration, 3),
        has_audio=has_audio,
    )


# Gate outcomes. Plain strings, matching the report vocabulary's own idiom
# ("passed" / "failed" / "warning") and serializing to JSON with no
# encoder support required.
REASON_LANDSCAPE = "landscape"
REASON_ALREADY_VERTICAL = "already_vertical"
REASON_SQUARE = "square"
_REASONS = (REASON_LANDSCAPE, REASON_ALREADY_VERTICAL, REASON_SQUARE)


@dataclass(frozen=True)
class GateDecision:
    """Whether one clip needs reframing, and why.

    should_reframe is True exactly when the reason is "landscape"; a
    decision where the two disagree is refused with ValueError.
    """

    should_reframe: bool
    reason: str
    meta: SourceMeta

    def __post_init__(self) -> None:
        if self.reason not in _REASONS:
            raise ValueError(f"unknown gate reason {self.reason!r}")
        if self.should_reframe != (self.reason == REASON_LANDSCAPE):
            raise ValueError(
                f"should_reframe={self.should_reframe} contradicts the "
                f"{self.reason!r} reason"
            )

    @property
    def log_line(self) -> str:
        """One-line summary for the batch console.

        Reports DISPLAY geometry — what a viewer sees — because that is
        what the decision was made on. Plain ASCII: this prints to a
        Windows console whose code page is not guaranteed to be UTF-8.
        """
        meta = self.meta
        geometry = f"{meta.display_width}x{meta.display_height}"
        if meta.rotation:
            geometry += f", rot {meta.rotation}"
        if self.reason == REASON_ALREADY_VERTICAL:
            return f"  reframe: skipped: already vertical ({geometry})"
        if self.reason == REASON_SQUARE:
            return f"  reframe: skipped: square ({geometry})"
        return f"  reframe: {geometry} landscape - reframing"


def decide(
    source: "Path | SourceMeta", *,
    runner: Optional[FFProbeRunner] = None,
) -> GateDecision:
    """Decide whether a clip needs reframing.

    Accepts a path (probed here) or an already-probed SourceMeta, so a
    caller that has probed the file for another reason does not pay for a
    second ffprobe.

    Landscape sources reframe. Vertical and square sources pass through
    untouched — "pass through" means the caller keeps using the ORIGINAL
    file; nothing is copied and nothing is written.

    Raises ProbeError when given a path that cannot be probed.
    """
    meta = (
        source if isinstance(source, SourceMeta)
        else probe_source(Path(source), runner=runner)
    )
    orientation = meta.orientation
    if orientation == "landscape":
        return GateDecision(True, REASON_LANDSCAPE, meta)
    reason = (
        REASON_SQUARE if orientation == "square" else REASON_ALREADY_VERTICAL
    )
    return GateDecision(False, reason, meta)


OUTCOME_REFRAMED = "reframed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_FAILED = "failed"
OUTCOME_GATE_ONLY = "gate_only"
_OUTCOMES = (
    OUTCOME_REFRAMED, OUTCOME_SKIPPED, OUTCOME_FAILED, OUTCOME_GATE_ONLY,
)


def reframe_report_entry(
    decision: GateDecision, *, applied: bool, elapsed_sec: float = 0.0,
    outcome: Optional[str] = None,
) -> dict:
    """Build the `ClipReport.reframe` payload for one clip.

    The batch report's status vocabulary is deliberately unchanged: a
    reframe skip is NOT a new ClipReport status. A skipped clip is still
    encoded and packaged, so its status stays "passed" -- a "skipped"
    status would count a correctly handled clip as not passing. The
    skip is recorded here instead, in an additive field.

    `applied` is the caller's statement of what actually happened: False
    for every skip, and False for a clip whose reframe was attempted but
    did not finish (that case also sets the clip's status to "failed"
    through the existing error path). Passing applied=True for a decision
    that said not to reframe is a programming error and raises ValueError.

    `outcome` disambiguates the two situations that both produce
    `applied=False` for a landscape decision: a purely diagnostic run
    (ftea/reframe/main.py builds this entry before calling `run_stages`,
    giving "gate_only") versus a stage that actually failed (main.py
    passes outcome=OUTCOME_FAILED explicitly when `run_stages` raises). Left as
    None, it is derived: "reframed" when applied, "skipped" when the gate
    said not to reframe, otherwise "gate_only". A given outcome must be
    one of the four, "reframed" exactly when applied is True, and
    "skipped" exactly when the gate said not to reframe; anything else
    is a programming error and raises ValueError, matching the
    applied/should_reframe guard.
    """
    if applied and not decision.should_reframe:
        raise ValueError(
            f"applied=True contradicts a {decision.reason} skip decision"
        )
    if outcome is not None:
        _check_outcome(decision, applied, outcome)
    if outcome is None:
        if applied:
            outcome = OUTCOME_REFRAMED
        elif not decision.should_reframe:
            outcome = OUTCOME_SKIPPED
        else:
            outcome = OUTCOME_GATE_ONLY
    meta = decision.meta
    return {
        "applied": applied,
        "outcome": outcome,
        "reason": decision.reason,
        "elapsed_sec": round(elapsed_sec, 1),
        "source": {
            "width": meta.width,
            "height": meta.height,
            "rotation": meta.rotation,
            "display_width": meta.display_width,
            "display_height": meta.display_height,
            "orientation": meta.orientation,
            "fps": meta.fps,
            "duration_s": meta.duration_s,
        },
    }


def _check_outcome(
    decision: GateDecision, applied: bool, outcome: str,
) -> None:
    """Refuse an outcome that is unknown or contradicts the entry."""
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    if applied != (outcome == OUTCOME_REFRAMED):
        raise ValueError(
            f"outcome={outcome!r} contradicts applied={applied}"
        )
    if (outcome == OUTCOME_SKIPPED) == decision.should_reframe:
        raise ValueError(
            f"outcome={outcome!r} contradicts the {decision.reason!r} "
            "gate decision"
        )


# CLI exit codes, shared with ftea/reframe/main.py: 0 reframe/reframed,
# 1 error/failed, 2 skip/skipped -- one vocabulary for the whole package.
EXIT_REFRAME = 0
EXIT_ERROR = 1
EXIT_SKIP = 2

_USAGE = "Usage: python -m ftea.reframe.probe <video>"


def main(argv: Optional[list[str]] = None) -> int:
    """Probe one clip and report the gate decision.

    Run it with the checkout importable (from the checkout, or with
    PYTHONPATH set to it):

        python -m ftea.reframe.probe input/clip.mp4

    Prints the report entry as JSON on stdout and the human summary on
    stderr, so a caller can pipe stdout into jq while still seeing the log.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(_USAGE, file=sys.stderr)
        return EXIT_ERROR

    try:
        decision = decide(Path(args[0]))
    except ProbeError as exc:
        print(f"  reframe: probe failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(reframe_report_entry(decision, applied=False), indent=2))
    print(decision.log_line, file=sys.stderr)
    return EXIT_REFRAME if decision.should_reframe else EXIT_SKIP


if __name__ == "__main__":
    sys.exit(main())
