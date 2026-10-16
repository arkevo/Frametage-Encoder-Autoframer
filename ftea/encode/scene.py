"""Scene detection, B&W detection, and source splitting.

Standard library only. The reframe stage keeps its own copy of the
detect_scene_timestamps ffmpeg command, and its tests import this module
inside the reframe virtualenv to check that the two stay identical. So
nothing here may import ftea.encode.main or anything heavier than the
standard library, and that command must not change in one place only.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, nvenc_supports_film_grain
from .geometry import probe_geometry
from .tools import run_tool


@dataclass
class SceneChunk:
    """A single scene chunk split from the source."""

    index: int
    path: Path
    start_time: float
    duration: float


@dataclass
class SourceAnalysis:
    """Analysis results for a source video."""

    orientation: str
    fps: float
    keyframe_interval: int
    is_bw: bool
    film_grain: int
    has_audio: bool
    scenes: list[SceneChunk]


def detect_orientation(width: int, height: int) -> str:
    """Landscape when wider than tall, otherwise portrait.

    A square frame is portrait, the same rule as geometry.target_frame,
    which pads a square clip onto the vertical frame. Pass DISPLAY
    dimensions, which is what detect_dimensions returns.
    """
    return "landscape" if width > height else "portrait"


def detect_bw(saturation: float, threshold: int = 10) -> bool:
    """Determine if content is black & white from average saturation."""
    return saturation < threshold


def merge_short_scenes(
    timestamps: list[float], min_sec: float, total_duration: float,
) -> list[float]:
    """Merge scenes shorter than min_sec with their neighbors.

    A short scene joins the one before it; a short first scene joins the
    one after it, and a short last scene the one before it. A clip
    shorter than min_sec stays one scene.
    """
    if len(timestamps) <= 2:
        return timestamps

    all_ts = (
        timestamps if timestamps[-1] == total_duration
        else timestamps + [total_duration]
    )
    merged = [all_ts[0]]

    for i in range(1, len(all_ts) - 1):
        gap = all_ts[i] - merged[-1]
        if gap >= min_sec:
            merged.append(all_ts[i])

    if len(merged) > 1 and all_ts[-1] - merged[-1] < min_sec:
        merged.pop()
    if merged[-1] != all_ts[-1]:
        merged.append(all_ts[-1])

    return merged


def split_long_scenes(
    timestamps: list[float], max_sec: float,
) -> list[float]:
    """Split scenes longer than max_sec into equal parts."""
    if len(timestamps) < 2:
        return timestamps

    result = [timestamps[0]]
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        duration = end - start
        if duration > max_sec:
            num_splits = int(duration / max_sec) + 1
            split_dur = duration / num_splits
            for j in range(1, num_splits):
                result.append(round(start + j * split_dur, 3))
        result.append(end)

    return sorted(set(result))


def detect_fps(source: Path) -> float:
    """Get framerate from source via ffprobe."""
    result = run_tool([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(source),
    ])
    # A clip with rotation side data gets a trailing empty csv field
    # ("24/1,"), so read the first field only.
    rate = result.stdout.strip().split(",")[0]
    num, den = rate.split("/")
    return round(int(num) / int(den), 3)


def detect_dimensions(source: Path) -> tuple[int, int]:
    """Display width and height of the source, rotation applied.

    A phone stores most vertical video as landscape pixels plus a
    rotation flag, and ffmpeg turns the frames upright while decoding.
    The coded size would call such a clip landscape and every variant
    would come out squashed. See ftea.encode.geometry.
    """
    shape = probe_geometry(source)
    return shape.display_width, shape.display_height


def detect_duration(source: Path) -> float:
    """Get duration in seconds from source via ffprobe."""
    result = run_tool([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(source),
    ])
    return float(result.stdout.strip())


def detect_has_audio(source: Path) -> bool:
    """Check if source has an audio stream.

    A file without one makes ffprobe print nothing and exit 0. A failed
    probe raises ToolError: read as "no audio", it would package the
    clip silent and disable the silent-intermediate checks.
    """
    result = run_tool([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(source),
    ])
    return result.stdout.strip() == "audio"


def detect_saturation(source: Path) -> Optional[float]:
    """Get average saturation via ffmpeg signalstats.

    Samples first 5 seconds to avoid processing entire clip.
    Returns average SATAVG across sampled frames, or None when ffmpeg
    printed none: unknown is not 0, which would read as black and white.
    A failed ffmpeg run raises ToolError.
    """
    result = run_tool([
        "ffmpeg", "-t", "5", "-i", str(source),
        "-vf", "signalstats,metadata=print:key=lavfi.signalstats.SATAVG",
        "-f", "null", "-",
    ])
    values = []
    for line in result.stderr.splitlines():
        if "lavfi.signalstats.SATAVG" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    values.append(float(parts[-1].strip()))
                except ValueError:
                    pass
    return sum(values) / len(values) if values else None


def detect_scene_timestamps(source: Path, threshold: float) -> list[float]:
    """Detect scene change timestamps using ffmpeg scene filter.

    Returns list of timestamps (in seconds) where scene changes occur.
    Always includes 0.0 as the first timestamp. A failed ffmpeg run
    raises ToolError instead of passing for a clip with no cuts.
    """
    result = run_tool([
        "ffmpeg", "-i", str(source),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ])
    timestamps = [0.0]
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            for part in line.split():
                if part.startswith("pts_time:"):
                    ts = float(part.split(":")[1])
                    if ts > 0.0:
                        timestamps.append(ts)
    return sorted(set(timestamps))


def cut_time(seconds: float) -> str:
    """A cut time as ffmpeg should read it: rounded DOWN to the
    millisecond. Rounded to nearest, a cut on frame 226 at 24 fps
    (9.41667 s) becomes 9.417, after that frame's keyframe, and stream
    copy skips to the next one. A millisecond is far below one frame.
    """
    return f"{math.floor(seconds * 1000 + 1e-6) / 1000:.3f}"


def split_source(
    source: Path, timestamps: list[float], work_dir: Path,
) -> list[SceneChunk]:
    """Split source video at scene boundaries into chunk files.

    The scenes folder is emptied first. The chunks are found by listing
    it, and a failed earlier attempt at the same clip leaves its own
    chunks there; listed with the new ones, they would be encoded onto
    the new clip. Stream copy cuts only at keyframes, so ffmpeg can write
    fewer chunks than there are cut points, never more. Each chunk is
    timed from its own file.

    The segment muxer reads its whole output path as a template in which
    a '%' must start a frame number, and the work folder is named after
    the clip (100% Pure.mp4). So ffmpeg runs inside the scenes folder and
    the pattern names no folder.
    """
    scenes_dir = work_dir / "scenes"
    if scenes_dir.exists():
        shutil.rmtree(scenes_dir)
    scenes_dir.mkdir(parents=True)

    if len(timestamps) <= 2:
        chunk_path = scenes_dir / "chunk_000.mp4"
        run_tool([
            "ffmpeg", "-y", "-i", str(source),
            "-c", "copy", "-an",
            str(chunk_path),
        ])
        duration = detect_duration(source)
        return [SceneChunk(
            index=0, path=chunk_path, start_time=0.0, duration=duration,
        )]

    segment_times = ",".join(cut_time(ts) for ts in timestamps[1:-1])

    run_tool(
        [
            "ffmpeg", "-y", "-i", str(Path(source).absolute()),
            "-f", "segment",
            "-segment_times", segment_times,
            "-c", "copy", "-an",
            "-reset_timestamps", "1",
            "chunk_%03d.mp4",
        ],
        cwd=scenes_dir,
    )

    chunk_files = sorted(scenes_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        raise RuntimeError(f"ffmpeg cut no scene chunks from {source.name}")
    # Stream copy cuts only at keyframes, so the chunks are not the
    # planned scenes: time each one from its own file and lay them end
    # to end. Pairing them with the plan by position reported 31 s for
    # a 50 s clip.
    chunks = []
    start = 0.0
    for i, chunk_path in enumerate(chunk_files):
        duration = round(detect_duration(chunk_path), 3)
        chunks.append(SceneChunk(
            index=i, path=chunk_path, start_time=round(start, 3),
            duration=duration,
        ))
        start += duration
    return chunks


def analyze_and_split(
    source: Path, config: PipelineConfig, work_dir: Path,
) -> SourceAnalysis:
    """Full source analysis: display orientation, fps, B&W, scene split."""
    width, height = detect_dimensions(source)
    orientation = detect_orientation(width, height)
    fps = detect_fps(source)
    keyframe_interval = round(fps * config.hls_segment_sec)
    has_audio = detect_has_audio(source)
    total_duration = detect_duration(source)

    saturation = detect_saturation(source)
    if saturation is None:
        print("  Could not measure the saturation; treating it as colour.")
        is_bw = False
    else:
        is_bw = detect_bw(saturation, config.bw_saturation_threshold)
    film_grain = config.film_grain_bw if is_bw else config.film_grain
    if film_grain > 0 and not nvenc_supports_film_grain():
        # The encode leaves the option out, so the report must say 0.
        film_grain = 0

    timestamps = detect_scene_timestamps(source, config.scene_threshold)
    timestamps.append(total_duration)
    timestamps = sorted(set(timestamps))
    timestamps = merge_short_scenes(
        timestamps, config.min_scene_sec, total_duration,
    )
    timestamps = split_long_scenes(timestamps, config.max_scene_sec)

    scenes = split_source(source, timestamps, work_dir)

    return SourceAnalysis(
        orientation=orientation,
        fps=fps,
        keyframe_interval=keyframe_interval,
        is_bw=is_bw,
        film_grain=film_grain,
        has_audio=has_audio,
        scenes=scenes,
    )
