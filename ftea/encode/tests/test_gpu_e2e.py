"""End to end on a real NVENC GPU. Opt in with: python -m pytest -m gpu

Each test makes its own short clip with ffmpeg (nothing is committed),
encodes it with reframing off, and checks the package a phone would
load. Skipped unless -m gpu is given and the machine passes the
startup checks.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from ftea import paths
from ftea.encode import preflight
from ftea.encode.config import PipelineConfig
from ftea.encode.main import run

pytestmark = pytest.mark.gpu
VARIANTS = (
    "av1_1080p", "av1_720p", "av1_360p",
    "h264_1080p", "h264_720p", "h264_360p",
)


@pytest.fixture
def nvenc_machine(gpu_run):
    problems = preflight.check_toolchain(PipelineConfig())
    if problems:
        pytest.skip("not an NVENC machine: " + "; ".join(problems))
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    if " libx264 " not in listing:
        pytest.skip("ffmpeg has no libx264 to make the test clips with")


def _make_clip(path, size: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True, check=True,
    )


def _assert_playable_package(folder, resolution: str) -> dict:
    master = (folder / "master.m3u8").read_text(encoding="utf-8")
    assert master.count("#EXT-X-STREAM-INF") == 6
    assert f"RESOLUTION={resolution}" in master
    for variant in VARIANTS:
        playlist = (folder / variant / "stream.m3u8").read_text(
            encoding="utf-8",
        )
        assert f'#EXT-X-MAP:URI="init_{variant}.mp4"' in playlist
        assert (folder / variant / f"init_{variant}.mp4").is_file()
        assert list((folder / variant).glob("*.m4s"))
    assert (folder / "thumbnail.webp").is_file()
    assert (folder / "report.md").is_file()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["status"] in ("passed", "warning")
    return report


def test_should_package_a_landscape_clip_with_a_non_ascii_name(
    nvenc_machine,
):
    clip = paths.input_dir() / "Café 東京.mp4"
    _make_clip(clip, "1280x720")

    result = run(clip, reframe=False)

    assert result.exit_code == 0
    report = _assert_playable_package(
        paths.output_dir() / "Café 東京", "1920x1080",
    )
    assert report["source"] == "Café 東京.mp4"
    assert "conform" not in report


def test_should_pad_a_square_clip_onto_a_vertical_frame(nvenc_machine):
    clip = paths.input_dir() / "square.mp4"
    _make_clip(clip, "720x720")

    result = run(clip, reframe=False)

    assert result.exit_code == 0
    report = _assert_playable_package(
        paths.output_dir() / "square", "1080x1920",
    )
    assert report["orientation"] == "portrait"
    assert report["conform"]["frame"] == "1080x1920"


def test_should_package_a_clip_whose_name_has_a_percent_sign(nvenc_machine):
    """The clip's name becomes its work and output folders, and ffmpeg
    reads a '%' in an output template as the start of a frame number."""
    clip = paths.input_dir() / "100% Pure %d.mp4"
    _make_clip(clip, "1280x720")

    result = run(clip, reframe=False)

    assert result.exit_code == 0
    _assert_playable_package(
        paths.output_dir() / "100% Pure %d", "1920x1080",
    )
