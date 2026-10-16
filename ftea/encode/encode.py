"""Final encode at selected CQs and concatenation.

_build_encode_cmd is the one encode command: the CQ probes, the final
encode and the start-up trial all run it, so a probe is scored on the
same kind of stream that is published.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import PipelineConfig, Variant, nvenc_supports_film_grain
from .scene import SceneChunk, SourceAnalysis
from .tools import run_tool

if TYPE_CHECKING:  # probe imports this module for the builder
    from .probe import ProbeResult

# Every H.264 variant is encoded at this profile and level, whatever its
# resolution. master.m3u8 derives its avc1 CODECS string from these two
# (package._codec_string), so the playlist describes the real stream.
H264_PROFILE = "high"
H264_LEVEL = "4.0"
# The AV1 pixel format for each allowed AV1_BIT_DEPTH. Set explicitly,
# or a 10-bit source would stay 10-bit under an 8-bit setting.
AV1_PIX_FMT = {8: "yuv420p", 10: "p010le"}


@dataclass
class VariantOutput:
    """Output of a single concatenated variant."""

    variant: Variant
    path: Path
    file_size_kb: int
    bitrate_kbps: int


def _build_encode_cmd(
    chunk: SceneChunk,
    variant: Variant,
    cq: int,
    film_grain: int,
    keyframe_interval: int,
    config: PipelineConfig,
    output_path: Path,
) -> list[str]:
    """Build the ffmpeg encode command for a single scene chunk."""
    cmd = [
        "ffmpeg", "-y", "-i", str(chunk.path),
        "-vf", f"scale={variant.width}:{variant.height}:flags=lanczos",
    ]

    if variant.codec == "av1":
        cmd += [
            "-c:v", "av1_nvenc",
            "-preset", config.nvenc_preset,
            "-tune", config.nvenc_tune,
            "-rc", config.nvenc_rc,
            "-cq", str(cq),
            "-b:v", "0",
            "-maxrate", variant.maxrate,
            "-bufsize", variant.bufsize,
            "-multipass", config.nvenc_multipass,
            "-rc-lookahead", str(config.nvenc_rc_lookahead),
            "-g", str(keyframe_interval),
            "-keyint_min", str(keyframe_interval),
            "-pix_fmt", AV1_PIX_FMT[variant.bit_depth],
        ]
        if film_grain > 0 and nvenc_supports_film_grain():
            cmd += ["-film_grain", str(film_grain)]
    else:
        # h264_nvenc doesn't support -tune uhq; use hq instead
        h264_tune = "hq" if config.nvenc_tune == "uhq" else config.nvenc_tune
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", config.nvenc_preset,
            "-tune", h264_tune,
            "-rc", config.nvenc_rc,
            "-cq", str(cq),
            "-b:v", "0",
            "-maxrate", variant.maxrate,
            "-bufsize", variant.bufsize,
            "-profile:v", H264_PROFILE,
            "-level", H264_LEVEL,
            "-multipass", config.nvenc_multipass,
            "-rc-lookahead", str(config.nvenc_rc_lookahead),
            "-g", str(keyframe_interval),
            "-keyint_min", str(keyframe_interval),
        ]

    cmd += ["-an", str(output_path)]
    return cmd


def _build_concat_cmd(
    chunk_paths: list[Path], output_path: Path,
) -> tuple[list[str], Path]:
    """Build ffmpeg concat command. Returns (cmd, concat_list_path)."""
    concat_list = output_path.with_suffix(".concat.txt")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path),
    ]
    return cmd, concat_list


def _encode_scene(
    chunk: SceneChunk,
    variant: Variant,
    cq: int,
    film_grain: int,
    keyframe_interval: int,
    config: PipelineConfig,
    output_path: Path,
) -> None:
    """Encode a single scene chunk at the selected CQ."""
    cmd = _build_encode_cmd(
        chunk, variant, cq, film_grain,
        keyframe_interval, config, output_path,
    )
    run_tool(cmd)
    time.sleep(0.5)  # NVENC session cooldown (Windows DLL init stability)


def _write_concat_list(
    chunk_paths: list[Path], concat_list: Path,
) -> None:
    """Write the ffmpeg concat demuxer list.

    Same fix strategy as _score_vmaf in probe.py: prefer a path form
    the code controls over escaping a caller-controlled one. Each
    entry is written relative to the list's own directory rather than
    absolute. Chunks and the list both live under the clip's work_dir
    (named after the source file, apostrophe and all), so that
    apostrophe sits in the shared ancestor and cancels out of the
    relative path entirely -- it never reaches the quoted line ffmpeg
    parses. A relative path is only apostrophe-free if the surviving
    (non-shared) components are too, so any "'" still present is
    escaped using the concat demuxer's convention for a literal single
    quote inside a quoted token: close, escape, reopen -- ' becomes
    '\\''.
    """
    concat_dir = concat_list.parent
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in chunk_paths:
            rel = os.path.relpath(p, concat_dir)
            escaped = rel.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


def _concat_variant(chunk_paths: list[Path], output_path: Path) -> None:
    """Concatenate encoded scene chunks into one file."""
    cmd, concat_list = _build_concat_cmd(chunk_paths, output_path)
    _write_concat_list(chunk_paths, concat_list)
    run_tool(cmd)
    concat_list.unlink()


def encode_and_concat(
    scenes: list[SceneChunk],
    probe_results: list[ProbeResult],
    source_analysis: SourceAnalysis,
    config: PipelineConfig,
    variants: list[Variant],
    work_dir: Path,
) -> list[VariantOutput]:
    """Encode all scenes at selected CQs and concatenate per variant."""
    encoded_dir = work_dir / "encoded"
    output_dir = work_dir / "variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build CQ lookup: (scene_index, codec, resolution) -> selected_cq
    cq_map = {}
    for pr in probe_results:
        key = (pr.scene_index, pr.variant.codec, pr.variant.resolution)
        cq_map[key] = pr.selected_cq

    outputs = []
    for variant in variants:
        variant_dir = encoded_dir / f"{variant.codec}_{variant.resolution}p"
        variant_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        for scene in scenes:
            cq = cq_map[(scene.index, variant.codec, variant.resolution)]
            chunk_out = variant_dir / f"chunk_{scene.index:03d}.mp4"
            _encode_scene(
                chunk=scene, variant=variant, cq=cq,
                film_grain=source_analysis.film_grain,
                keyframe_interval=source_analysis.keyframe_interval,
                config=config, output_path=chunk_out,
            )
            chunk_paths.append(chunk_out)

        # Concatenate
        concat_out = output_dir / f"{variant.codec}_{variant.resolution}p.mp4"
        _concat_variant(chunk_paths, concat_out)

        # Get file metrics
        size_bytes = concat_out.stat().st_size
        total_duration = sum(s.duration for s in scenes)

        # Try ffprobe first, fall back to file_size / duration
        result = run_tool([
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=bit_rate",
            "-of", "csv=p=0",
            str(concat_out),
        ], check=False)
        try:
            bitrate = int(result.stdout.strip() or "0") // 1000
        except ValueError:  # ffprobe prints N/A when it cannot tell
            bitrate = 0
        if bitrate == 0 and total_duration > 0:
            bitrate = int(size_bytes * 8 / total_duration / 1000)

        outputs.append(VariantOutput(
            variant=variant,
            path=concat_out,
            file_size_kb=size_bytes // 1024,
            bitrate_kbps=bitrate,
        ))

    return outputs
