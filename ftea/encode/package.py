"""HLS packaging with muxed audio, a thumbnail and the master playlist.

master.m3u8 is written last, after the thumbnail, every variant's
segments and the bundled player files.

Every file ffmpeg writes here is named without a folder, and ffmpeg
runs inside the folder it writes to; only the inputs are full paths.
That is what works on both systems. The hls muxer puts the playlist's
folder in front of the init segment's name whenever the playlist path
contains a '/', as every Linux path does, so a full init path there
became a doubled path that does not exist; with a Windows path it
found no '/' and wrote a bare name into the current folder instead.
The segment name and a jpg thumbnail's name are also read as
templates in which a '%' starts a frame number, so a '%' in the clip's
folder name could turn the folder into a different one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, Variant
from .encode import H264_LEVEL, H264_PROFILE, VariantOutput
from .scene import SourceAnalysis
from .tools import run_tool

# Files from the ftea/player/ folder copied into every clip's output
# folder, so the folder can be served, hosted and watched on its own:
# the page, the hls.js build it loads, and hls.js's license, which must
# travel with every copy of hls.js.
PLAYER_FILES: tuple[str, ...] = (
    "player.html", "hls.min.js", "hls.js-LICENSE.txt",
)


# The URI attribute of a variant playlist's EXT-X-MAP tag.
MAP_URI = re.compile(r'URI="([^"]*)"')
# profile_idc of each H.264 profile name ffmpeg's -profile:v takes.
AVC_PROFILE_IDC = {"baseline": 0x42, "main": 0x4D, "high": 0x64}
# AV1 level (seq_level_idx, Main tier) written for each resolution.
AV1_LEVELS = {"1080": "08M", "720": "05M", "360": "01M"}


def _codec_string(variant: Variant) -> str:
    """The HLS CODECS value for a variant, as it is encoded.

    H.264 is avc1.PPCCLL, built from the profile and level the encode
    command passes. AV1 stops after the bit depth: the optional fields
    after it come all together or not at all, and the colour fields
    among them are not known, because the encode does not set them.
    """
    if variant.codec == "av1":
        level = AV1_LEVELS[variant.resolution]
        return f"av01.0.{level}.{variant.bit_depth:02d}"
    profile_idc = AVC_PROFILE_IDC[H264_PROFILE]
    level_idc = round(float(H264_LEVEL) * 10)
    return f"avc1.{profile_idc:02x}00{level_idc:02x}"


def _build_thumbnail_cmd(source: Path, output: Path, fmt: str) -> list[str]:
    """Build ffmpeg command to extract first frame as thumbnail.

    Run it inside the output's folder: the output is named without one.
    """
    cmd = [
        "ffmpeg", "-y", "-ss", "0", "-i", str(Path(source).absolute()),
        "-vframes", "1",
    ]
    if fmt == "webp":
        cmd += ["-c:v", "libwebp", "-quality", "80"]
    else:
        cmd += ["-q:v", "20"]
    cmd.append(Path(output).name)
    return cmd


def _build_audio_cmd(
    source: Path, output: Path, config: PipelineConfig,
) -> list[str]:
    """Build ffmpeg command to extract audio as AAC."""
    return [
        "ffmpeg", "-y", "-i", str(source),
        "-vn", "-c:a", "aac",
        "-b:a", config.audio_bitrate,
        "-ar", str(config.audio_sample_rate),
        str(output),
    ]


def _build_segment_cmd(
    variant_file: Path,
    variant: Variant,
    config: PipelineConfig,
    audio_file: Optional[Path] = None,
) -> list[str]:
    """Build ffmpeg HLS segmentation command with optional muxed audio.

    Run it inside the variant's folder (see the module docstring): the
    playlist, init segment and segments are named without a folder.
    """
    variant_key = f"{variant.codec}_{variant.resolution}p"

    cmd = ["ffmpeg", "-y", "-i", str(Path(variant_file).absolute())]
    if audio_file is not None:
        cmd += ["-i", str(Path(audio_file).absolute())]
        cmd += ["-c:v", "copy", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "copy", "-an"]

    cmd += [
        "-f", "hls",
        "-hls_time", str(config.hls_segment_sec),
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", f"init_{variant_key}.mp4",
        "-hls_segment_filename", f"{variant_key}_%03d.m4s",
        "-hls_playlist_type", "vod",
        "-hls_list_size", "0",
        "stream.m3u8",
    ]
    return cmd


def _write_master_playlist(
    variant_outputs: list[VariantOutput],
    has_audio: bool,
    output_dir: Path,
) -> Path:
    """Write master.m3u8 with muxed audio CODECS."""
    master_path = output_dir / "master.m3u8"

    lines = ["#EXTM3U", "#EXT-X-VERSION:7", ""]

    for vo in variant_outputs:
        v = vo.variant
        variant_key = f"{v.codec}_{v.resolution}p"
        video_codecs = _codec_string(v)
        maxrate_int = int(v.maxrate.replace("k", "")) * 1000

        codecs = f"{video_codecs},mp4a.40.2" if has_audio else video_codecs

        inf = f"#EXT-X-STREAM-INF:BANDWIDTH={maxrate_int}"
        inf += f",RESOLUTION={v.width}x{v.height}"
        inf += f',CODECS="{codecs}"'

        lines.append(inf)
        lines.append(f"{variant_key}/stream.m3u8")

    master_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n",
    )
    return master_path


@dataclass
class PackageOutput:
    """Result of HLS packaging."""

    master_m3u8: Path
    thumbnail: Path
    hls_dir: Path


def _extract_thumbnail(source: Path, output: Path, fmt: str) -> None:
    """Extract first frame as thumbnail."""
    run_tool(_build_thumbnail_cmd(source, output, fmt), cwd=output.parent)


def _extract_audio(
    source: Path, output_dir: Path, config: PipelineConfig,
) -> Path:
    """Extract audio to a single AAC file."""
    audio_path = output_dir / "audio_extracted.m4a"
    run_tool(_build_audio_cmd(source, audio_path, config))
    return audio_path


def _relativize_init_uri(playlist: Path, init_path: Path) -> None:
    """Point a variant playlist's EXT-X-MAP at its init segment by name.

    ffmpeg writes the init name it was given into the EXT-X-MAP tag.
    That is a bare name, so this is a guard: a player loading the folder
    from a web server needs a name relative to the playlist, and a full
    path there would play only on this machine. The playlist is read and
    written as UTF-8. A path in it contains the clip's folder name, and
    decoding that with the Windows code page would garble a non-ASCII
    name, so the replace below would miss.

    Raises RuntimeError unless every EXT-X-MAP line then names the init
    file alone: ffmpeg writing the path in some other form must fail the
    clip, not ship a playlist that only plays on this machine.
    """
    text = playlist.read_text(encoding="utf-8")
    for written in (str(init_path).replace("\\", "/"), str(init_path)):
        text = text.replace(written, init_path.name)
    uris = [
        MAP_URI.search(line) for line in text.splitlines()
        if line.startswith("#EXT-X-MAP:")
    ]
    if not uris or any(
        uri is None or uri.group(1) != init_path.name for uri in uris
    ):
        raise RuntimeError(
            f"{playlist.parent.name}/{playlist.name}: EXT-X-MAP does not "
            f"name {init_path.name} alone, so it would not play once "
            "published"
        )
    playlist.write_text(text, encoding="utf-8", newline="\n")


def _segment_variant(
    variant_file: Path,
    variant: Variant,
    output_dir: Path,
    config: PipelineConfig,
    audio_file: Optional[Path] = None,
) -> Path:
    """Segment a concatenated variant into HLS with muxed audio."""
    variant_key = f"{variant.codec}_{variant.resolution}p"
    variant_dir = output_dir / variant_key
    variant_dir.mkdir(parents=True, exist_ok=True)

    run_tool(
        _build_segment_cmd(variant_file, variant, config, audio_file),
        cwd=variant_dir,
    )

    playlist = variant_dir / "stream.m3u8"
    _relativize_init_uri(playlist, variant_dir / f"init_{variant_key}.mp4")
    return playlist


def copy_player_files(
    output_dir: Path,
    names: Optional[Sequence[str]] = None,
    *,
    source_dir: Optional[Path] = None,
) -> list[Path]:
    """Copy the bundled player files into one clip's output folder.

    `names` defaults to PLAYER_FILES and `source_dir` to the player
    folder inside the installed ftea package.
    """
    wanted = PLAYER_FILES if names is None else tuple(names)
    copied: list[Path] = []
    for name in wanted:
        target = output_dir / name
        target.write_bytes(read_player_file(name, source_dir=source_dir))
        copied.append(target)
    return copied


def player_dir():
    """The player folder inside the installed ftea package."""
    return resources.files("ftea").joinpath("player")


def read_player_file(name: str, *, source_dir: Optional[Path] = None) -> bytes:
    """One bundled player file, from `source_dir` or player_dir().

    FileNotFoundError, naming the file, when it is not there.
    """
    source = (player_dir() if source_dir is None else source_dir).joinpath(
        name,
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"bundled player file {name} is missing from the ftea package"
        )
    return source.read_bytes()


def package_hls(
    variant_outputs: list[VariantOutput],
    source: Path,
    source_analysis: SourceAnalysis,
    config: PipelineConfig,
    output_dir: Path,
) -> PackageOutput:
    """Package all variants into HLS with muxed audio and thumbnail."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thumbnail
    thumb_path = output_dir / f"thumbnail.{config.thumb_format}"
    _extract_thumbnail(source, thumb_path, config.thumb_format)

    # Extract audio once (muxed into each variant during segmentation)
    audio_file = None
    if source_analysis.has_audio:
        audio_file = _extract_audio(source, output_dir, config)

    # Segment each variant (with muxed audio)
    for vo in variant_outputs:
        _segment_variant(vo.path, vo.variant, output_dir, config, audio_file)

    # Clean up extracted audio (already muxed into segments)
    if audio_file is not None and audio_file.exists():
        audio_file.unlink()

    # Bundled player files, before the master playlist, so a folder that
    # has master.m3u8 also has everything needed to watch it.
    copy_player_files(output_dir)

    # Master playlist, last
    master = _write_master_playlist(
        variant_outputs, source_analysis.has_audio, output_dir,
    )

    return PackageOutput(
        master_m3u8=master,
        thumbnail=thumb_path,
        hls_dir=output_dir,
    )
