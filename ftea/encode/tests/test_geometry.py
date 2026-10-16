"""Tests for ftea.encode.geometry."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ftea.encode import geometry
from ftea.encode.geometry import (
    Geometry,
    build_conform_cmd,
    conform,
    conform_entry,
    conform_filter,
    needs_conform,
    normalize_rotation,
    probe_geometry,
    target_frame,
)
from ftea.encode.tools import ToolError


def _ffprobe_returning(monkeypatch, payload, seen=None):
    """Make geometry.run_tool answer like ffprobe with `payload`."""
    def fake(cmd, **kwargs):
        if seen is not None:
            seen.append(cmd)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess(cmd, 0, text, "")

    monkeypatch.setattr(geometry, "run_tool", fake)


class TestNormalizeRotation:
    @pytest.mark.parametrize("raw, expected", [
        (None, 0), (0, 0), (90, 90), (-90, 270), ("-90", 270),
        (180.4, 180), (270, 270), (450, 90), (-180, 180), ("sideways", 0),
    ])
    def test_should_round_to_a_quarter_turn(self, raw, expected):
        assert normalize_rotation(raw) == expected


class TestGeometry:
    @pytest.mark.parametrize("rotation", [90, 270])
    def test_should_swap_width_and_height_for_a_quarter_turn(
        self, rotation,
    ):
        shape = Geometry(width=1920, height=1080, rotation=rotation)
        assert (shape.display_width, shape.display_height) == (1080, 1920)
        assert shape.label == "1080x1920"

    @pytest.mark.parametrize("rotation", [0, 180])
    def test_should_keep_the_coded_size_otherwise(self, rotation):
        shape = Geometry(width=1920, height=1080, rotation=rotation)
        assert (shape.display_width, shape.display_height) == (1920, 1080)


class TestProbeGeometry:
    def test_should_read_rotation_from_the_display_matrix(self, monkeypatch):
        _ffprobe_returning(monkeypatch, {"streams": [{
            "width": 1920, "height": 1080,
            "side_data_list": [
                {"side_data_type": "Display Matrix", "rotation": -90},
            ],
        }]})
        assert probe_geometry(Path("phone.mp4")) == Geometry(1920, 1080, 270)

    def test_should_fall_back_to_the_rotate_tag(self, monkeypatch):
        _ffprobe_returning(monkeypatch, {"streams": [{
            "width": 1920, "height": 1080, "tags": {"rotate": "90"},
        }]})
        assert probe_geometry(Path("phone.mov")).rotation == 90

    def test_should_report_no_rotation_for_a_plain_clip(self, monkeypatch):
        _ffprobe_returning(
            monkeypatch, {"streams": [{"width": 1280, "height": 720}]},
        )
        assert probe_geometry(Path("clip.mp4")) == Geometry(1280, 720, 0)

    def test_should_ask_for_the_first_video_stream_as_json(
        self, monkeypatch,
    ):
        seen = []
        _ffprobe_returning(
            monkeypatch, {"streams": [{"width": 2, "height": 2}]}, seen,
        )
        probe_geometry(Path("clip.mp4"))
        cmd = seen[0]
        assert cmd[0] == "ffprobe"
        assert cmd[cmd.index("-select_streams") + 1] == "v:0"
        assert "-show_streams" in cmd
        assert cmd[cmd.index("-of") + 1] == "json"
        assert cmd[-1] == "clip.mp4"

    @pytest.mark.parametrize("payload, message", [
        ({"streams": []}, "no video stream"),
        ({}, "no video stream"),
        ({"streams": [{"width": 0, "height": 1080}]}, "no frame size"),
        ({"streams": [{"codec_type": "video"}]}, "no frame size"),
        ("not json", "unreadable output"),
    ])
    def test_should_raise_a_readable_error_for_odd_output(
        self, monkeypatch, payload, message,
    ):
        _ffprobe_returning(monkeypatch, payload)
        with pytest.raises(ToolError, match=message):
            probe_geometry(Path("odd.mp4"))


class TestTargetFrame:
    @pytest.mark.parametrize("shape, frame", [
        (Geometry(1920, 1080), (1920, 1080)),
        (Geometry(1080, 1920), (1080, 1920)),
        (Geometry(1080, 1080), (1080, 1920)),
        (Geometry(1920, 1080, 90), (1080, 1920)),
        (Geometry(1440, 1080), (1920, 1080)),
    ])
    def test_should_pick_the_encode_frame(self, shape, frame):
        assert target_frame(shape) == frame


class TestNeedsConform:
    @pytest.mark.parametrize("shape, expected", [
        (Geometry(1920, 1080), False),
        (Geometry(1080, 1920), False),
        (Geometry(1920, 1080, 270), False),  # rotated phone clip, 9:16
        (Geometry(1920, 1088), False),       # 0.7% off: within tolerance
        (Geometry(1080, 1080), True),        # square
        (Geometry(1080, 1350), True),        # 4:5
        (Geometry(1080, 2340), True),        # tall screen recording
        (Geometry(1440, 1080), True),        # 4:3 landscape
    ])
    def test_should_conform_only_shapes_off_the_ladder(self, shape, expected):
        assert needs_conform(shape) is expected


class TestConformCommand:
    def test_should_fit_and_pad_without_stretching(self):
        assert conform_filter(1080, 1920) == (
            "scale=1080:1920:force_original_aspect_ratio=decrease"
            ":force_divisible_by=2,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )

    def test_should_keep_the_sound_and_encode_near_losslessly(self):
        cmd = build_conform_cmd(Path("in.mp4"), Path("out.mp4"), (1080, 1920))
        assert cmd[:4] == ["ffmpeg", "-y", "-i", "in.mp4"]
        assert cmd[4:8] == ["-map", "0:v:0", "-map", "0:a:0?"]
        assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
        assert cmd[cmd.index("-cq") + 1] == "10"
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[-1] == "out.mp4"

    def test_should_describe_the_conform_for_the_report(self):
        assert conform_entry(Geometry(1080, 1080)) == {
            "applied": True, "source": "1080x1080", "rotation": 0,
            "frame": "1080x1920",
        }


def _ffmpeg_with_libx264() -> bool:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return " libx264 " in listing


@pytest.mark.ffmpeg
def test_should_pad_a_square_clip_onto_a_vertical_frame(tmp_path):
    if not _ffmpeg_with_libx264():
        pytest.skip("ffmpeg and ffprobe with libx264 are not on PATH")
    square = tmp_path / "square.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=320x320:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(square),
        ],
        capture_output=True, check=True,
    )

    padded = conform(
        square, tmp_path / "w", probe_geometry(square),
        video_args=(
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p",
        ),
    )

    assert probe_geometry(padded) == Geometry(1080, 1920, 0)
