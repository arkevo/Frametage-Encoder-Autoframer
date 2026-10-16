"""Tests for ftea.encode.encode: concat lists and variant metrics."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ftea.encode import encode as encode_mod
from ftea.encode.config import PipelineConfig
from ftea.encode.encode import _concat_variant, _write_concat_list
from ftea.encode.probe import ProbeResult
from ftea.encode.scene import SceneChunk, SourceAnalysis
from ftea.encode.tools import ToolError


def _ffmpeg_available() -> bool:
    """ffmpeg on PATH with libx264, which the test clips are made with."""
    if shutil.which("ffmpeg") is None:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return " libx264 " in listing


def _make_test_clip(path: Path) -> None:
    """Generate a tiny 1s testsrc clip for concat regression tests."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an", str(path),
        ],
        capture_output=True, check=True,
    )


@pytest.mark.ffmpeg
class TestConcatVariantRealFfmpeg:
    """Regression tests that run the real ffmpeg concat demuxer."""

    def test_should_concat_successfully_when_dirname_has_apostrophe(
        self, tmp_path,
    ):
        if not _ffmpeg_available():
            pytest.skip("ffmpeg not on PATH")

        # Mirrors the production layout: chunks live under
        # work_dir/encoded/<variant>/, the concat list + output live
        # under work_dir/variants/. work_dir's name carries the
        # apostrophe, exactly like the batch failing in production.
        work_dir = tmp_path / "It's dir"
        chunk_dir = work_dir / "encoded" / "av1_1080p"
        chunk_dir.mkdir(parents=True)
        variant_dir = work_dir / "variants"
        variant_dir.mkdir(parents=True)

        chunk0 = chunk_dir / "chunk_000.mp4"
        chunk1 = chunk_dir / "chunk_001.mp4"
        _make_test_clip(chunk0)
        _make_test_clip(chunk1)

        output_path = variant_dir / "av1_1080p.mp4"

        _concat_variant([chunk0, chunk1], output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_should_concat_successfully_when_no_shared_apostrophe_ancestor(
        self, tmp_path,
    ):
        """Escape path, not cancellation: chunks and the concat list
        do NOT share an apostrophe-bearing ancestor directory, so
        os.path.relpath keeps a literal "'" in the written path and
        the ' -> '\\'' escaping must fire for ffmpeg to parse the
        quoted concat-list line. Against the old `file '{p}'`
        implementation (no escaping), ffmpeg's concat demuxer treats
        the embedded apostrophe as closing the quoted token early and
        fails to parse the remainder of the line as part of the same
        path, so the chunk isn't found and concat fails.
        """
        if not _ffmpeg_available():
            pytest.skip("ffmpeg not on PATH")

        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()
        chunk_dir = tmp_path / "It's chunks"
        chunk_dir.mkdir()

        chunk0 = chunk_dir / "chunk_000.mp4"
        chunk1 = chunk_dir / "chunk_001.mp4"
        _make_test_clip(chunk0)
        _make_test_clip(chunk1)

        output_path = variant_dir / "av1_1080p.mp4"

        _concat_variant([chunk0, chunk1], output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_should_concat_successfully_when_filename_has_apostrophe(
        self, tmp_path,
    ):
        """Escape path triggered purely by an apostrophe in the chunk
        FILENAME, with no divergent-ancestor directory structure at
        all -- chunks and the concat list share every parent
        directory, and none of those shared directories carry a "'".
        The apostrophe survives into the relative path solely because
        it's part of the filename component itself, which the
        cancellation strategy (shared-ancestor apostrophe) can never
        remove.
        """
        if not _ffmpeg_available():
            pytest.skip("ffmpeg not on PATH")

        work_dir = tmp_path / "clean_dir"
        chunk_dir = work_dir / "encoded" / "av1_1080p"
        chunk_dir.mkdir(parents=True)
        variant_dir = work_dir / "variants"
        variant_dir.mkdir(parents=True)

        chunk0 = chunk_dir / "chunk's_000.mp4"
        chunk1 = chunk_dir / "chunk's_001.mp4"
        _make_test_clip(chunk0)
        _make_test_clip(chunk1)

        output_path = variant_dir / "av1_1080p.mp4"

        _concat_variant([chunk0, chunk1], output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_should_leave_list_file_when_concat_fails(self, tmp_path):
        if not _ffmpeg_available():
            pytest.skip("ffmpeg not on PATH")

        # Chunk path does not exist, so ffmpeg's concat exits non-zero
        # and run_tool raises before concat_list.unlink() runs.
        chunk_dir = tmp_path / "It's chunks"
        chunk_dir.mkdir()
        missing_chunk = chunk_dir / "missing.mp4"
        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()
        output_path = variant_dir / "av1_1080p.mp4"

        with pytest.raises(ToolError):
            _concat_variant([missing_chunk], output_path)

        concat_list = output_path.with_suffix(".concat.txt")
        assert concat_list.exists()


class TestWriteConcatList:
    """Unit tests on the concat list content, no ffmpeg required."""

    def test_should_cancel_apostrophe_when_common_ancestor_has_one(
        self, tmp_path,
    ):
        # Same layout as production: the apostrophe lives in work_dir,
        # an ancestor shared by both the list and the chunk, so the
        # relative form written should not contain it at all.
        work_dir = tmp_path / "It's Movie"
        chunk_dir = work_dir / "encoded" / "av1_1080p"
        chunk_dir.mkdir(parents=True)
        variant_dir = work_dir / "variants"
        variant_dir.mkdir(parents=True)
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.touch()
        concat_list = variant_dir / "av1_1080p.concat.txt"

        _write_concat_list([chunk_path], concat_list)

        content = concat_list.read_text()
        rel = os.path.relpath(chunk_path, variant_dir)
        assert "'" not in rel
        assert content == f"file '{rel}'\n"

    def test_should_escape_apostrophe_when_relative_path_retains_one(
        self, tmp_path,
    ):
        # Divergent directory names (no shared apostrophe-bearing
        # ancestor) so the relative path still contains a "'" and the
        # escaping path must kick in.
        concat_dir = tmp_path / "variants"
        concat_dir.mkdir()
        chunk_dir = tmp_path / "It's chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.touch()
        concat_list = concat_dir / "out.concat.txt"

        _write_concat_list([chunk_path], concat_list)

        content = concat_list.read_text()
        rel = os.path.relpath(chunk_path, concat_dir)
        assert "'" in rel  # sanity: this scenario keeps an apostrophe
        expected = rel.replace("'", "'\\''")
        assert content == f"file '{expected}'\n"

    def test_should_preserve_spaces_when_path_contains_spaces(
        self, tmp_path,
    ):
        concat_dir = tmp_path / "variants"
        concat_dir.mkdir()
        chunk_dir = tmp_path / "My Chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk 000.mp4"
        chunk_path.touch()
        concat_list = concat_dir / "out.concat.txt"

        _write_concat_list([chunk_path], concat_list)

        content = concat_list.read_text()
        rel = os.path.relpath(chunk_path, concat_dir)
        assert content == f"file '{rel}'\n"

    def test_should_write_one_line_per_chunk_when_multiple_chunks(
        self, tmp_path,
    ):
        concat_dir = tmp_path / "variants"
        concat_dir.mkdir()
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk0 = chunk_dir / "chunk_000.mp4"
        chunk1 = chunk_dir / "chunk_001.mp4"
        chunk0.touch()
        chunk1.touch()
        concat_list = concat_dir / "out.concat.txt"

        _write_concat_list([chunk0, chunk1], concat_list)

        lines = concat_list.read_text().splitlines()
        assert len(lines) == 2


class TestVariantMetrics:
    def test_should_fall_back_to_size_over_duration_when_ffprobe_says_na(
        self, monkeypatch, tmp_path,
    ):
        """ffprobe prints N/A for a bitrate it cannot read.

        That must not fail a clip whose encode succeeded.
        """
        def fake_concat(chunk_paths, output_path):
            output_path.write_bytes(b"x" * 250_000)

        monkeypatch.setattr(encode_mod, "_encode_scene", lambda **kw: None)
        monkeypatch.setattr(encode_mod, "_concat_variant", fake_concat)
        monkeypatch.setattr(
            encode_mod, "run_tool",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "N/A\n", ""),
        )
        config = PipelineConfig()
        variant = config.build_variants("portrait")[0]
        scene = SceneChunk(
            index=0, path=tmp_path / "c.mp4", start_time=0.0, duration=2.0,
        )
        analysis = SourceAnalysis(
            orientation="portrait", fps=24.0, keyframe_interval=48,
            is_bw=False, film_grain=0, has_audio=False, scenes=[scene],
        )
        probes = [ProbeResult(
            scene_index=0, variant=variant, cq_low=26, cq_high=34,
            vmaf_low=95.0, vmaf_high=93.0, selected_cq=34,
            selected_vmaf=93.0,
        )]

        outputs = encode_mod.encode_and_concat(
            scenes=[scene], probe_results=probes, source_analysis=analysis,
            config=config, variants=[variant], work_dir=tmp_path / "w",
        )

        # 250,000 bytes * 8 bits / 2 s / 1000 = 1000 kbps
        assert outputs[0].bitrate_kbps == 1000
        assert outputs[0].file_size_kb == 244


def _variant(codec: str, config: PipelineConfig, res: str = "1080"):
    return next(
        v for v in config.build_variants("portrait")
        if (v.codec, v.resolution) == (codec, res)
    )


def _encode_cmd(tmp_path, codec="av1", film_grain=0, config=None):
    config = config or PipelineConfig()
    return encode_mod._build_encode_cmd(
        chunk=SceneChunk(
            index=0, path=tmp_path / "chunk_000.mp4", start_time=0.0,
            duration=2.0,
        ),
        variant=_variant(codec, config), cq=30, film_grain=film_grain,
        keyframe_interval=48, config=config,
        output_path=tmp_path / "out.mp4",
    )


class TestBuildEncodeCmd:
    def test_should_add_film_grain_when_the_encoder_supports_it(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            encode_mod, "nvenc_supports_film_grain", lambda: True,
        )
        cmd = _encode_cmd(tmp_path, film_grain=8)
        assert cmd[cmd.index("-film_grain") + 1] == "8"

    def test_should_leave_film_grain_out_when_the_encoder_lacks_it(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            encode_mod, "nvenc_supports_film_grain", lambda: False,
        )
        assert "-film_grain" not in _encode_cmd(tmp_path, film_grain=8)

    @pytest.mark.parametrize(("codec", "grain"), [("av1", 0), ("h264", 8)])
    def test_should_add_no_film_grain_without_a_grain_or_to_h264(
        self, monkeypatch, tmp_path, codec, grain,
    ):
        monkeypatch.setattr(
            encode_mod, "nvenc_supports_film_grain", lambda: True,
        )
        assert "-film_grain" not in _encode_cmd(
            tmp_path, codec=codec, film_grain=grain,
        )

    @pytest.mark.parametrize(("depth", "pix_fmt"), [
        (10, "p010le"), (8, "yuv420p"),
    ])
    def test_should_encode_av1_at_the_configured_bit_depth(
        self, tmp_path, depth, pix_fmt,
    ):
        """Without an explicit format an 8-bit setting kept a 10-bit
        source at 10 bits while master.m3u8 said 8."""
        cmd = _encode_cmd(tmp_path, config=PipelineConfig(av1_bit_depth=depth))
        assert cmd[cmd.index("-pix_fmt") + 1] == pix_fmt


class TestEncodeAndConcat:
    def test_should_encode_each_scene_at_its_own_cq(
        self, monkeypatch, tmp_path,
    ):
        config = PipelineConfig()
        variants = [_variant("av1", config), _variant("h264", config)]
        scenes = [
            SceneChunk(
                index=i, path=tmp_path / f"c{i}.mp4", start_time=2.0 * i,
                duration=2.0,
            )
            for i in range(2)
        ]
        chosen = {
            (0, "av1"): 31, (1, "av1"): 27, (0, "h264"): 24, (1, "h264"): 29,
        }
        probes = [
            ProbeResult(
                scene_index=s.index, variant=v, cq_low=0, cq_high=0,
                vmaf_low=0.0, vmaf_high=0.0,
                selected_cq=chosen[(s.index, v.codec)], selected_vmaf=93.0,
            )
            for s in scenes for v in variants
        ]
        encoded, joined = [], []
        monkeypatch.setattr(
            encode_mod, "_encode_scene",
            lambda **kw: encoded.append((
                kw["variant"].codec, kw["chunk"].index, kw["cq"],
                kw["output_path"].parent.name, kw["output_path"].name,
            )),
        )

        def fake_concat(chunk_paths, output_path):
            joined.append((
                output_path.name,
                [f"{p.parent.name}/{p.name}" for p in chunk_paths],
            ))
            output_path.write_bytes(b"x" * 1000)

        monkeypatch.setattr(encode_mod, "_concat_variant", fake_concat)
        monkeypatch.setattr(
            encode_mod, "run_tool",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "8000\n", ""),
        )
        analysis = SourceAnalysis(
            orientation="portrait", fps=24.0, keyframe_interval=48,
            is_bw=False, film_grain=0, has_audio=False, scenes=scenes,
        )

        encode_mod.encode_and_concat(
            scenes=scenes, probe_results=probes, source_analysis=analysis,
            config=config, variants=variants, work_dir=tmp_path / "w",
        )

        assert encoded == [
            ("av1", 0, 31, "av1_1080p", "chunk_000.mp4"),
            ("av1", 1, 27, "av1_1080p", "chunk_001.mp4"),
            ("h264", 0, 24, "h264_1080p", "chunk_000.mp4"),
            ("h264", 1, 29, "h264_1080p", "chunk_001.mp4"),
        ]
        assert joined == [
            ("av1_1080p.mp4",
             ["av1_1080p/chunk_000.mp4", "av1_1080p/chunk_001.mp4"]),
            ("h264_1080p.mp4",
             ["h264_1080p/chunk_000.mp4", "h264_1080p/chunk_001.mp4"]),
        ]
