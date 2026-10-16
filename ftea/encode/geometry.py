"""Display geometry of a clip: coded size, rotation and shown size.

ffprobe reports a stream's CODED width and height. Phones usually store
a vertical video as landscape pixels plus a rotation flag, and ffmpeg
turns the frames upright as it decodes them. Every decision the encoder
makes about shape therefore uses the DISPLAY size, rotation applied.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .tools import ToolError, run_tool

FFPROBE_TIMEOUT_SEC = 60
FFPROBE_ARGS = (
    "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
)


def normalize_rotation(raw: object) -> int:
    """An ffprobe rotation value as 0, 90, 180 or 270 degrees.

    Phone footage usually carries -90. Values are rounded to the nearest
    quarter turn and wrapped into [0, 360); anything unreadable counts
    as no rotation.
    """
    if raw is None:
        return 0
    try:
        degrees = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return int(round(degrees / 90.0)) * 90 % 360


@dataclass(frozen=True)
class Geometry:
    """Coded size and rotation of a clip's first video stream."""

    width: int
    height: int
    rotation: int = 0

    @property
    def quarter_turned(self) -> bool:
        return self.rotation % 180 == 90

    @property
    def display_width(self) -> int:
        return self.height if self.quarter_turned else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.quarter_turned else self.height

    @property
    def label(self) -> str:
        return f"{self.display_width}x{self.display_height}"


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _rotation(stream: dict) -> int:
    """Rotation from the display matrix, falling back to the old tag."""
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return normalize_rotation(side_data["rotation"])
    return normalize_rotation((stream.get("tags") or {}).get("rotate"))


def probe_geometry(source: Path) -> Geometry:
    """Read a clip's coded size and rotation with one ffprobe call."""
    result = run_tool(
        ["ffprobe", *FFPROBE_ARGS, str(source)], timeout=FFPROBE_TIMEOUT_SEC,
    )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        raise ToolError(
            f"ffprobe returned unreadable output for {source.name}"
        ) from None
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not streams:
        raise ToolError(f"{source.name} has no video stream")
    stream = streams[0]
    width = _as_int(stream.get("width"))
    height = _as_int(stream.get("height"))
    if width <= 0 or height <= 0:
        raise ToolError(f"{source.name} reports no frame size")
    return Geometry(width=width, height=height, rotation=_rotation(stream))


# --- Conforming odd shapes ----------------------------------------------------
# The encode ladder has two frame shapes, 1080x1920 (vertical) and
# 1920x1080 (landscape). A clip whose display shape is neither, such as a
# square or 4:5 clip or a tall screen recording, is scaled to fit inside
# its target frame without distortion and centred on black bars, in a
# near-lossless intermediate. A square clip becomes vertical. Clips within
# ASPECT_TOLERANCE of a frame shape (1920x1088, say) are encoded as they
# are, without conforming.

PORTRAIT_FRAME = (1080, 1920)
LANDSCAPE_FRAME = (1920, 1080)
ASPECT_TOLERANCE = 0.01
CONFORM_NAME = "conformed.mp4"
# Near-lossless, like the reframe stage's own intermediate.
CONFORM_VIDEO_ARGS = (
    "-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "10",
    "-b:v", "0",
)
CONFORM_AUDIO_ARGS = ("-c:a", "aac", "-b:a", "320k")


def target_frame(geometry: Geometry) -> tuple[int, int]:
    """1080x1920 for vertical and square clips, 1920x1080 for landscape."""
    if geometry.display_height >= geometry.display_width:
        return PORTRAIT_FRAME
    return LANDSCAPE_FRAME


def needs_conform(
    geometry: Geometry, tolerance: float = ASPECT_TOLERANCE,
) -> bool:
    """Whether the display shape is more than `tolerance` off its frame."""
    width, height = target_frame(geometry)
    wanted = width / height
    actual = geometry.display_width / geometry.display_height
    return abs(actual - wanted) / wanted > tolerance


def conform_filter(width: int, height: int) -> str:
    """Scale to fit inside width x height, centred on black bars."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        ":force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def build_conform_cmd(
    source: Path, output: Path, frame: tuple[int, int],
    video_args: Sequence[str] = CONFORM_VIDEO_ARGS,
) -> list[str]:
    """The ffmpeg command that writes the conformed intermediate."""
    width, height = frame
    return [
        "ffmpeg", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", conform_filter(width, height),
        *video_args, *CONFORM_AUDIO_ARGS,
        str(output),
    ]


def conform(
    source: Path, work_dir: Path, geometry: Geometry,
    *, video_args: Sequence[str] = CONFORM_VIDEO_ARGS,
) -> Path:
    """Write the conformed intermediate into work_dir; return its path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / CONFORM_NAME
    run_tool(build_conform_cmd(
        source, output, target_frame(geometry), video_args,
    ))
    return output


def conform_entry(geometry: Geometry) -> dict:
    """What a clip's report records about its conform."""
    width, height = target_frame(geometry)
    return {
        "applied": True,
        "source": geometry.label,
        "rotation": geometry.rotation,
        "frame": f"{width}x{height}",
    }
