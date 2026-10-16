"""Tests for ftea.encode.package."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ftea.encode import package
from ftea.encode.config import PipelineConfig
from ftea.encode.encode import VariantOutput, _build_encode_cmd
from ftea.encode.package import (
    PLAYER_FILES,
    _codec_string,
    _relativize_init_uri,
    _write_master_playlist,
    copy_player_files,
)
from ftea.encode.scene import SceneChunk, SourceAnalysis


def _outputs(orientation: str = "portrait") -> list[VariantOutput]:
    return [
        VariantOutput(
            variant=v, path=Path(f"{v.codec}_{v.resolution}p.mp4"),
            file_size_kb=1, bitrate_kbps=1,
        )
        for v in PipelineConfig().build_variants(orientation)
    ]


class TestCodecStrings:
    """The strings master.m3u8 advertises, pinned so that changing them
    is deliberate. A player chooses a variant by these, so each must
    describe the stream as it is actually encoded.
    """

    def test_should_keep_the_current_codec_strings(self):
        assert [_codec_string(o.variant) for o in _outputs()] == [
            "av01.0.08M.10", "av01.0.05M.10", "av01.0.01M.10",
            "avc1.640028", "avc1.640028", "avc1.640028",
        ]

    def test_should_advertise_the_h264_profile_and_level_encoded(
        self, tmp_path,
    ):
        """Every H.264 variant is encoded at one profile and level; the
        CODECS string used to claim Main 3.1 and Baseline 2.1 for 720p
        and 360p."""
        profile_idc = {"baseline": 0x42, "main": 0x4D, "high": 0x64}
        for output in _outputs():
            variant = output.variant
            if variant.codec != "h264":
                continue
            cmd = _build_encode_cmd(
                chunk=SceneChunk(
                    index=0, path=tmp_path / "c.mp4", start_time=0.0,
                    duration=1.0,
                ),
                variant=variant, cq=25, film_grain=0, keyframe_interval=48,
                config=PipelineConfig(), output_path=tmp_path / "o.mp4",
            )
            profile = cmd[cmd.index("-profile:v") + 1]
            level = round(float(cmd[cmd.index("-level") + 1]) * 10)
            assert _codec_string(variant) == (
                f"avc1.{profile_idc[profile]:02x}00{level:02x}"
            )

    def test_should_advertise_the_av1_bit_depth_encoded(self):
        variants = PipelineConfig(av1_bit_depth=8).build_variants("portrait")
        assert _codec_string(variants[0]) == "av01.0.08M.08"


class TestMasterPlaylist:
    def test_should_list_six_variants_with_muxed_audio(self, tmp_path):
        text = _write_master_playlist(
            _outputs(), True, tmp_path,
        ).read_text(encoding="utf-8")

        assert text.count("#EXT-X-STREAM-INF") == 6
        assert (
            "#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1080x1920,"
            'CODECS="av01.0.08M.10,mp4a.40.2"'
        ) in text
        assert "av1_1080p/stream.m3u8" in text

    def test_should_leave_out_the_audio_codec_without_audio(self, tmp_path):
        text = _write_master_playlist(
            _outputs(), False, tmp_path,
        ).read_text(encoding="utf-8")
        assert "mp4a" not in text

    def test_should_write_lf_line_endings(self, tmp_path):
        master = _write_master_playlist(_outputs(), True, tmp_path)
        assert b"\r\n" not in master.read_bytes()


class TestRelativizeInitUri:
    @pytest.mark.parametrize("slashes", ["native", "forward"])
    def test_should_point_the_map_at_the_file_in_a_non_ascii_folder(
        self, tmp_path, slashes,
    ):
        variant_dir = tmp_path / "Café 東京" / "av1_1080p"
        variant_dir.mkdir(parents=True)
        init_path = variant_dir / "init_av1_1080p.mp4"
        written = (
            str(init_path) if slashes == "native"
            else str(init_path).replace("\\", "/")
        )
        playlist = variant_dir / "stream.m3u8"
        playlist.write_bytes((
            "#EXTM3U\n"
            f'#EXT-X-MAP:URI="{written}"\n'
            "#EXTINF:2.000000,\n"
            "av1_1080p_000.m4s\n"
        ).encode("utf-8"))

        _relativize_init_uri(playlist, init_path)

        text = playlist.read_bytes().decode("utf-8")
        assert '#EXT-X-MAP:URI="init_av1_1080p.mp4"' in text
        assert "Café" not in text
        assert "\r\n" not in text

    @pytest.mark.parametrize("map_line", [
        '#EXT-X-MAP:URI="//?/D:/out/clip/av1_1080p/init_av1_1080p.mp4"',
        '#EXT-X-MAP:URI="elsewhere/init_av1_1080p.mp4"',
        "",
    ], ids=["other spelling", "other folder", "no map"])
    def test_should_fail_when_the_map_does_not_end_up_as_the_file_name(
        self, tmp_path, map_line,
    ):
        """A local path left in EXT-X-MAP plays here and nowhere else;
        the clip must fail rather than be marked finished."""
        variant_dir = tmp_path / "av1_1080p"
        variant_dir.mkdir()
        playlist = variant_dir / "stream.m3u8"
        playlist.write_text(
            f"#EXTM3U\n{map_line}\n#EXTINF:2.000000,\nav1_1080p_000.m4s\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="EXT-X-MAP"):
            _relativize_init_uri(playlist, variant_dir / "init_av1_1080p.mp4")


class TestPlayerFiles:
    def test_should_ship_the_player_and_hls_js(self):
        assert PLAYER_FILES == (
            "player.html", "hls.min.js", "hls.js-LICENSE.txt",
        )

    def test_should_copy_the_bundled_player_by_default(self, tmp_path):
        copied = copy_player_files(tmp_path)
        assert [path.name for path in copied] == list(PLAYER_FILES)
        assert all(path.stat().st_size > 0 for path in copied)

    def test_should_copy_each_named_file(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "player.html").write_text("<html></html>", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        copied = copy_player_files(out, ["player.html"], source_dir=bundle)

        assert copied == [out / "player.html"]
        assert (out / "player.html").read_text(encoding="utf-8") == (
            "<html></html>"
        )

    def test_should_fail_loudly_when_a_named_file_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="player.html"):
            copy_player_files(
                tmp_path, ["player.html"], source_dir=tmp_path / "empty",
            )

    def test_should_read_from_the_package_by_default(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            package, "PLAYER_FILES", ("missing-on-purpose.html",),
        )
        with pytest.raises(FileNotFoundError, match="ftea package"):
            copy_player_files(tmp_path)


class TestPackageOrder:
    def test_should_write_the_master_playlist_last(
        self, monkeypatch, tmp_path,
    ):
        order = MagicMock()
        order.audio.return_value = tmp_path / "audio_extracted.m4a"
        monkeypatch.setattr(package, "_extract_thumbnail", order.thumbnail)
        monkeypatch.setattr(package, "_extract_audio", order.audio)
        monkeypatch.setattr(package, "_segment_variant", order.segment)
        monkeypatch.setattr(package, "copy_player_files", order.player)
        monkeypatch.setattr(package, "_write_master_playlist", order.master)
        analysis = SourceAnalysis(
            orientation="portrait", fps=24.0, keyframe_interval=48,
            is_bw=False, film_grain=0, has_audio=True, scenes=[],
        )

        package.package_hls(
            variant_outputs=_outputs()[:1], source=tmp_path / "c.mp4",
            source_analysis=analysis, config=PipelineConfig(),
            output_dir=tmp_path / "out",
        )

        assert [call[0] for call in order.mock_calls] == [
            "thumbnail", "audio", "segment", "player", "master",
        ]


def _variant(codec: str = "h264", resolution: str = "360"):
    return next(
        v for v in PipelineConfig().build_variants("portrait")
        if (v.codec, v.resolution) == (codec, resolution)
    )


class TestSegmentCommand:
    """FFmpeg's hls muxer writes the init segment to the playlist's folder
    plus -hls_fmp4_init_filename whenever the playlist path contains a
    '/', which every Linux path does: an absolute init name there gives
    a doubled path and packaging fails. The segment name is a printf
    template, so a '%' in any folder above it fails too. Every output is
    therefore a bare name, and ffmpeg runs inside the variant folder.
    """

    def test_should_name_every_output_without_a_folder(self, tmp_path):
        cmd = package._build_segment_cmd(
            tmp_path / "in.mp4", _variant(), PipelineConfig(),
            tmp_path / "audio.m4a",
        )

        assert cmd[cmd.index("-hls_fmp4_init_filename") + 1] == (
            "init_h264_360p.mp4"
        )
        assert cmd[cmd.index("-hls_segment_filename") + 1] == (
            "h264_360p_%03d.m4s"
        )
        assert cmd[-1] == "stream.m3u8"

    def test_should_give_the_inputs_as_absolute_paths(
        self, monkeypatch, tmp_path,
    ):
        """ffmpeg runs in the variant folder, so a relative input would
        be looked for there."""
        monkeypatch.chdir(tmp_path)
        cmd = package._build_segment_cmd(
            Path("in.mp4"), _variant(), PipelineConfig(), Path("audio.m4a"),
        )

        inputs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
        assert inputs == [
            str(tmp_path / "in.mp4"), str(tmp_path / "audio.m4a"),
        ]

    def test_should_run_ffmpeg_inside_the_variant_folder(
        self, monkeypatch, tmp_path,
    ):
        calls = []

        def fake(cmd, **kwargs):
            calls.append(kwargs)
            (Path(kwargs["cwd"]) / cmd[-1]).write_text(
                '#EXTM3U\n#EXT-X-MAP:URI="init_h264_360p.mp4"\n',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(package, "run_tool", fake)
        out = tmp_path / "100% Pure"

        playlist = package._segment_variant(
            tmp_path / "in.mp4", _variant(), out, PipelineConfig(),
        )

        assert [call["cwd"] for call in calls] == [out / "h264_360p"]
        assert playlist == out / "h264_360p" / "stream.m3u8"


def _ffmpeg_with_libx264() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return " libx264 " in listing


def _make_clip(path: Path, *, audio: bool) -> None:
    """A 3 s 64x64 clip with a keyframe every second."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=3:size=64x64:rate=10",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=duration=3", "-c:a", "aac"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10",
        *([] if audio else ["-an"]), str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


@pytest.mark.ffmpeg
class TestPackageRealFfmpeg:
    """Package a short clip with the real ffmpeg, on whatever system the
    tests run on: Linux once failed every clip here, and a bare init
    name without the right working folder once landed in the current
    folder on Windows."""

    @pytest.mark.parametrize("clip_name", [
        "clip", "100% Pure", "Café 東京 it's %d",
    ])
    def test_should_write_a_complete_package_in_the_clip_folder(
        self, monkeypatch, tmp_path, clip_name,
    ):
        if not _ffmpeg_with_libx264():
            pytest.skip("ffmpeg with libx264 is not on PATH")
        media = tmp_path / "media"
        media.mkdir()
        source, video = media / "source.mp4", media / "h264_360p.mp4"
        _make_clip(source, audio=True)
        _make_clip(video, audio=False)
        elsewhere = tmp_path / "current folder"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        out = (tmp_path / "output" / clip_name).absolute()
        analysis = SourceAnalysis(
            orientation="portrait", fps=10.0, keyframe_interval=20,
            is_bw=False, film_grain=0, has_audio=True, scenes=[],
        )

        package.package_hls(
            variant_outputs=[VariantOutput(
                variant=_variant(), path=video, file_size_kb=1,
                bitrate_kbps=1,
            )],
            source=source, source_analysis=analysis,
            config=dataclasses.replace(PipelineConfig(), thumb_format="jpg"),
            output_dir=out,
        )

        variant_dir = out / "h264_360p"
        text = (variant_dir / "stream.m3u8").read_text(encoding="utf-8")
        named = [
            line for line in text.splitlines()
            if line and not line.startswith("#")
        ]
        assert (variant_dir / "init_h264_360p.mp4").stat().st_size > 0
        assert '#EXT-X-MAP:URI="init_h264_360p.mp4"' in text
        assert named and all((variant_dir / n).is_file() for n in named)
        assert (out / "master.m3u8").is_file()
        assert (out / "thumbnail.jpg").stat().st_size > 0
        assert list(elsewhere.iterdir()) == []
