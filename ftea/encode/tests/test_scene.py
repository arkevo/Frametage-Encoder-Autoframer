"""Tests for ftea.encode.scene."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path

import pytest

from ftea.encode import scene
from ftea.encode.config import PipelineConfig
from ftea.encode.geometry import Geometry
from ftea.encode.tools import ToolError


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class TestOrientation:
    @pytest.mark.parametrize(("size", "expected"), [
        ((1920, 1080), "landscape"), ((1080, 1920), "portrait"),
        ((1080, 1080), "portrait"),
    ])
    def test_should_call_a_square_frame_portrait_like_the_conform_step(
        self, size, expected,
    ):
        """geometry.target_frame puts a square clip on the vertical frame;
        one rule for both keeps a square from getting landscape variants.
        """
        assert scene.detect_orientation(*size) == expected


class TestDisplayDimensions:
    def test_should_return_the_display_size_of_a_rotated_phone_clip(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            scene, "probe_geometry", lambda source: Geometry(1920, 1080, 270),
        )
        assert scene.detect_dimensions(Path("phone.mp4")) == (1080, 1920)

    def test_should_treat_a_rotated_phone_clip_as_portrait(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            scene, "probe_geometry", lambda source: Geometry(1920, 1080, 90),
        )
        monkeypatch.setattr(scene, "detect_fps", lambda source: 30.0)
        monkeypatch.setattr(scene, "detect_has_audio", lambda source: True)
        monkeypatch.setattr(scene, "detect_duration", lambda source: 4.0)
        monkeypatch.setattr(scene, "detect_saturation", lambda source: 40.0)
        monkeypatch.setattr(
            scene, "detect_scene_timestamps", lambda source, threshold: [0.0],
        )
        monkeypatch.setattr(
            scene, "split_source", lambda source, timestamps, work_dir: [],
        )

        analysis = scene.analyze_and_split(
            Path("phone.mp4"), PipelineConfig(), tmp_path,
        )

        assert analysis.orientation == "portrait"
        assert analysis.keyframe_interval == 60


class TestToolCalls:
    def test_should_read_scene_cuts_from_ffmpeg_output(self, monkeypatch):
        stderr = (
            "[Parsed_showinfo_1] n:0 pts:60 pts_time:2.5 pos:1\n"
            "[Parsed_showinfo_1] n:1 pts:174 pts_time:7.25 pos:2\n"
        )
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stderr=stderr),
        )
        assert scene.detect_scene_timestamps(Path("c.mp4"), 0.3) == [
            0.0, 2.5, 7.25,
        ]

    @pytest.mark.parametrize(("stdout", "expected"), [
        ("24/1\n", 24.0),
        ("30000/1001\n", 29.97),
        # A clip with rotation side data: ffprobe's csv writer adds an
        # empty field for the side-data section after the rate.
        ("24/1,\n", 24.0),
    ])
    def test_should_read_the_frame_rate_from_ffprobe_output(
        self, monkeypatch, stdout, expected,
    ):
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stdout=stdout),
        )
        assert scene.detect_fps(Path("c.mp4")) == expected

    def test_should_report_audio_only_when_ffprobe_finds_a_stream(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stdout="audio\n"),
        )
        assert scene.detect_has_audio(Path("c.mp4")) is True
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stdout=""),
        )
        assert scene.detect_has_audio(Path("c.mp4")) is False

    def test_should_average_saturation_from_ffmpeg_output(self, monkeypatch):
        stderr = (
            "lavfi.signalstats.SATAVG=10\nlavfi.signalstats.SATAVG=20\n"
        )
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stderr=stderr),
        )
        assert scene.detect_saturation(Path("c.mp4")) == 15.0

    def test_should_say_unknown_when_ffmpeg_prints_no_saturation(
        self, monkeypatch,
    ):
        """No samples is not a saturation of 0, which would read as B&W."""
        monkeypatch.setattr(
            scene, "run_tool", lambda cmd, **kw: _completed(stderr="done\n"),
        )
        assert scene.detect_saturation(Path("c.mp4")) is None

    @pytest.mark.parametrize("probe", [
        lambda: scene.detect_has_audio(Path("c.mp4")),
        lambda: scene.detect_saturation(Path("c.mp4")),
        lambda: scene.detect_scene_timestamps(Path("c.mp4"), 0.3),
    ], ids=["audio", "saturation", "scene cuts"])
    def test_should_fail_the_clip_when_the_tool_fails(
        self, monkeypatch, probe,
    ):
        """A failed run used to read as no audio (a silent package), a
        black-and-white clip (film grain added) or a single scene."""
        monkeypatch.setattr(
            scene, "run_tool", _failing_tool("moov atom not found"),
        )
        with pytest.raises(ToolError, match="moov atom not found"):
            probe()


def _failing_tool(stderr: str):
    """A fake run_tool for a program that exits 1, honouring check=."""
    def fake(cmd, *, check=True, **kwargs):
        if check:
            raise ToolError(f"{cmd[0]} failed (exit 1): {stderr}")
        return _completed(stderr=stderr, returncode=1)

    return fake


def _analysed(monkeypatch, tmp_path, *, saturation, grain_supported=True,
              config=None):
    """analyze_and_split on a 4 s clip, with every tool call faked."""
    monkeypatch.setattr(
        scene, "probe_geometry", lambda source: Geometry(1080, 1920),
    )
    monkeypatch.setattr(scene, "detect_fps", lambda source: 24.0)
    monkeypatch.setattr(scene, "detect_has_audio", lambda source: True)
    monkeypatch.setattr(scene, "detect_duration", lambda source: 4.0)
    monkeypatch.setattr(scene, "detect_saturation", lambda source: saturation)
    monkeypatch.setattr(
        scene, "detect_scene_timestamps", lambda source, threshold: [0.0],
    )
    monkeypatch.setattr(
        scene, "split_source", lambda source, timestamps, work_dir: [],
    )
    monkeypatch.setattr(
        scene, "nvenc_supports_film_grain", lambda: grain_supported,
    )
    return scene.analyze_and_split(
        Path("c.mp4"), config or PipelineConfig(), tmp_path,
    )


class TestFilmGrain:
    def test_should_give_a_black_and_white_clip_the_bw_grain(
        self, monkeypatch, tmp_path,
    ):
        analysis = _analysed(monkeypatch, tmp_path, saturation=3.0)
        assert (analysis.is_bw, analysis.film_grain) == (True, 8)

    def test_should_give_a_colour_clip_the_colour_grain(
        self, monkeypatch, tmp_path,
    ):
        analysis = _analysed(
            monkeypatch, tmp_path, saturation=40.0,
            config=PipelineConfig(film_grain=4),
        )
        assert (analysis.is_bw, analysis.film_grain) == (False, 4)

    def test_should_record_no_grain_when_the_encoder_cannot_add_it(
        self, monkeypatch, tmp_path,
    ):
        """The report used to say 8 for grain that was never encoded."""
        analysis = _analysed(
            monkeypatch, tmp_path, saturation=3.0, grain_supported=False,
        )
        assert (analysis.is_bw, analysis.film_grain) == (True, 0)

    def test_should_treat_an_unknown_saturation_as_colour(
        self, monkeypatch, tmp_path, capsys,
    ):
        analysis = _analysed(monkeypatch, tmp_path, saturation=None)
        assert (analysis.is_bw, analysis.film_grain) == (False, 0)
        assert "Could not measure the saturation" in capsys.readouterr().out


def _segmenter_writing(count: int, calls=None):
    """A fake run_tool that cuts `count` chunks, as ffmpeg's segmenter:
    the output pattern is read from the folder ffmpeg runs in."""
    def fake(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        folder = Path(kwargs.get("cwd") or ".")
        for index in range(count):
            name = str(cmd[-1]).replace("%03d", f"{index:03d}")
            (folder / name).write_bytes(b"new")
        return _completed()

    return fake


class TestSplitSource:
    @pytest.fixture(autouse=True)
    def _one_second_chunks(self, monkeypatch):
        """split_source probes every chunk it cut; tests that time
        chunks set their own lengths."""
        monkeypatch.setattr(scene, "detect_duration", lambda path: 1.0)

    def test_should_return_only_the_chunks_this_split_cut(
        self, monkeypatch, tmp_path,
    ):
        """An earlier failed attempt leaves its chunks in work/<clip>/.

        Run 1 cut four scenes and failed; run 2 (a new export of the same
        name, or new scene settings) cuts two. The two old chunks must
        never be encoded onto the new clip.
        """
        scenes = tmp_path / "scenes"
        scenes.mkdir()
        for index in range(4):
            (scenes / f"chunk_{index:03d}.mp4").write_bytes(b"old")
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(2))
        lengths = {"chunk_000.mp4": 4.0, "chunk_001.mp4": 5.0}
        monkeypatch.setattr(
            scene, "detect_duration", lambda path: lengths[Path(path).name],
        )

        chunks = scene.split_source(Path("c.mp4"), [0.0, 4.0, 9.0], tmp_path)

        assert [
            (c.index, c.path.name, c.start_time, c.duration) for c in chunks
        ] == [(0, "chunk_000.mp4", 0.0, 4.0), (1, "chunk_001.mp4", 4.0, 5.0)]
        assert sorted(p.name for p in scenes.iterdir()) == [
            "chunk_000.mp4", "chunk_001.mp4",
        ]

    def test_should_accept_fewer_chunks_than_cut_points(
        self, monkeypatch, tmp_path,
    ):
        """Stream copy cuts only at keyframes, so two cut points inside
        one keyframe interval give one chunk fewer. That is not an error.
        """
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(2))

        chunks = scene.split_source(
            Path("c.mp4"), [0.0, 2.0, 2.5, 9.0], tmp_path,
        )

        assert [c.path.name for c in chunks] == [
            "chunk_000.mp4", "chunk_001.mp4",
        ]

    def test_should_record_the_times_of_the_chunks_ffmpeg_cut(
        self, monkeypatch, tmp_path,
    ):
        """Stream copy snaps every cut to the next keyframe. The chunks
        must carry what was cut, not the plan: twelve planned scenes cut
        as five ten-second chunks are five chunks summing to the clip.
        """
        lengths = [10.417, 10.417, 10.417, 10.416, 8.333]
        monkeypatch.setattr(
            scene, "run_tool", _segmenter_writing(len(lengths)),
        )
        by_name = {f"chunk_{i:03d}.mp4": d for i, d in enumerate(lengths)}
        monkeypatch.setattr(
            scene, "detect_duration", lambda path: by_name[Path(path).name],
        )
        planned = [0.0, 4.46, 9.42, 16.0, 23.62, 31.25, 33.08, 36.62,
                   40.79, 42.12, 44.67, 47.67, 50.0]

        chunks = scene.split_source(Path("c.mp4"), planned, tmp_path)

        assert [c.duration for c in chunks] == lengths
        assert [c.start_time for c in chunks] == [
            0.0, 10.417, 20.834, 31.251, 41.667,
        ]
        assert round(sum(c.duration for c in chunks), 3) == 50.0

    def test_should_time_a_single_chunk_from_its_file(
        self, monkeypatch, tmp_path,
    ):
        """One keyframe in the whole clip: many cut points, one chunk."""
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(1))
        monkeypatch.setattr(scene, "detect_duration", lambda path: 20.0)

        chunks = scene.split_source(
            Path("c.mp4"), [0.0, 4.46, 9.42, 16.0, 20.0], tmp_path,
        )

        assert [(c.start_time, c.duration) for c in chunks] == [(0.0, 20.0)]

    @pytest.mark.parametrize(("seconds", "written"), [
        (226 / 24, "9.416"), (107 / 24, "4.458"), (2.5, "2.500"),
        (1001 / 30000 * 300, "10.010"),
    ])
    def test_should_write_a_cut_time_at_or_before_its_frame(
        self, seconds, written,
    ):
        assert scene.cut_time(seconds) == written
        assert float(written) <= seconds + 1e-9

    def test_should_give_the_segmenter_cut_times_rounded_down(
        self, monkeypatch, tmp_path,
    ):
        """Rounded to nearest, 9.41667 would be written 9.417, after the
        keyframe on that frame, and stream copy would skip to the next."""
        calls = []
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(2, calls))

        scene.split_source(Path("c.mp4"), [0.0, 226 / 24, 20.0], tmp_path)

        cmd = calls[0][0]
        assert cmd[cmd.index("-segment_times") + 1] == "9.416"

    def test_should_fail_when_ffmpeg_cuts_no_chunk(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(0))

        with pytest.raises(RuntimeError, match="no scene chunks"):
            scene.split_source(Path("c.mp4"), [0.0, 4.0, 9.0], tmp_path)

    def test_should_cut_into_the_scenes_folder_of_the_work_folder(
        self, monkeypatch, tmp_path,
    ):
        calls = []
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(2, calls))

        scene.split_source(Path("c.mp4"), [0.0, 4.0, 9.0], tmp_path)

        cmd, kwargs = calls[0]
        assert cmd[cmd.index("-segment_times") + 1] == "4.000"
        assert kwargs["cwd"] == tmp_path / "scenes"

    def test_should_name_the_chunk_pattern_without_a_folder(
        self, monkeypatch, tmp_path,
    ):
        """The segment muxer reads its whole output path as a template, in
        which a '%' must start a frame number: a work folder named after
        100% Pure.mp4 would make ffmpeg refuse the pattern."""
        calls = []
        monkeypatch.setattr(scene, "run_tool", _segmenter_writing(2, calls))
        monkeypatch.chdir(tmp_path)

        scene.split_source(Path("c.mp4"), [0.0, 4.0, 9.0], tmp_path / "w")

        cmd, _ = calls[0]
        assert cmd[-1] == "chunk_%03d.mp4"
        assert cmd[cmd.index("-i") + 1] == str(tmp_path / "c.mp4")


class TestSceneBoundaries:
    def test_should_merge_scenes_shorter_than_the_minimum(self):
        assert scene.merge_short_scenes(
            [0.0, 0.4, 5.0, 10.0], 1.0, 10.0,
        ) == [0.0, 5.0, 10.0]

    def test_should_merge_a_short_last_scene_into_the_one_before(self):
        assert scene.merge_short_scenes(
            [0.0, 5.0, 9.5, 10.0], 1.0, 10.0,
        ) == [0.0, 5.0, 10.0]

    def test_should_keep_a_clip_shorter_than_the_minimum_whole(self):
        assert scene.merge_short_scenes(
            [0.0, 0.3, 0.6, 0.8], 1.0, 0.8,
        ) == [0.0, 0.8]

    def test_should_split_scenes_longer_than_the_maximum(self):
        assert scene.split_long_scenes([0.0, 25.0], 10.0) == [
            0.0, 8.333, 16.667, 25.0,
        ]


class TestReframeStageContract:
    def test_should_keep_the_signature_the_reframe_stage_imports(self):
        params = inspect.signature(scene.detect_scene_timestamps).parameters
        assert list(params) == ["source", "threshold"]


def _ffmpeg_with_libx264() -> bool:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return " libx264 " in listing


@pytest.mark.ffmpeg
class TestSplitSourceRealFfmpeg:
    @pytest.mark.parametrize("timestamps", [
        [0.0, 1.0, 2.0, 3.0], [0.0, 3.0],
    ], ids=["segments", "one chunk"])
    @pytest.mark.parametrize("clip_name", ["clip", "100% Pure", "it's %d"])
    def test_should_cut_chunks_in_a_folder_named_after_any_clip(
        self, monkeypatch, tmp_path, clip_name, timestamps,
    ):
        if not _ffmpeg_with_libx264():
            pytest.skip("ffmpeg and ffprobe with libx264 are not on PATH")
        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "testsrc=duration=3:size=64x64:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10",
                str(source),
            ],
            capture_output=True, check=True,
        )
        elsewhere = tmp_path / "current folder"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        chunks = scene.split_source(
            source, timestamps, tmp_path / "work" / clip_name,
        )

        assert len(chunks) == len(timestamps) - 1
        assert all(c.path.stat().st_size > 0 for c in chunks)
        assert all(c.path.parent.parent.name == clip_name for c in chunks)
        assert list(elsewhere.iterdir()) == []
