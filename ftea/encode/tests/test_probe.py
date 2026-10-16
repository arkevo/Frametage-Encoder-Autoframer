"""Tests for ftea.encode.probe: VMAF scoring and CQ selection."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from ftea import paths
from ftea.encode import encode as encode_mod
from ftea.encode import probe
from ftea.encode.config import PipelineConfig
from ftea.encode.probe import _score_vmaf, select_cq
from ftea.encode.scene import SceneChunk

PAYLOAD = {"pooled_metrics": {"vmaf": {"mean": 93.5}}}


def _log_name(cmd) -> str:
    lavfi = cmd[cmd.index("-lavfi") + 1]
    return lavfi.split("log_path=", 1)[1]


def _ffmpeg_writing(payload, calls=None):
    """A fake run_tool that writes the libvmaf log ffmpeg would write."""
    def fake(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        if payload is not None:
            log = Path(kwargs["cwd"]) / _log_name(cmd)
            log.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake


def _clip(folder: Path, name: str = "s0_h264_1080_cq29.mp4") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"")
    return path


class TestScoreVmaf:
    def test_should_return_the_mean_and_remove_the_log(
        self, monkeypatch, tmp_path,
    ):
        calls = []
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(PAYLOAD, calls))

        score = _score_vmaf(_clip(tmp_path), _clip(tmp_path, "ref.mp4"))

        assert score == 93.5
        cmd, kwargs = calls[0]
        assert not (Path(kwargs["cwd"]) / _log_name(cmd)).exists()

    def test_should_pass_libvmaf_a_bare_file_name(
        self, monkeypatch, tmp_path,
    ):
        """No folder, drive colon or apostrophe may reach the option.

        ffmpeg's filter parser treats ':' as a separator and "'" as a
        quote with no reliable escape, and a path relative to the current
        folder cannot be built at all across Windows drives.
        """
        calls = []
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(PAYLOAD, calls))
        clip_dir = tmp_path / "It's a test clip part 1" / "probes"

        _score_vmaf(_clip(clip_dir), _clip(tmp_path, "ref.mp4"))

        cmd, kwargs = calls[0]
        assert re.fullmatch(r"[0-9a-f]{16}\.json", _log_name(cmd))
        assert Path(kwargs["cwd"]) == paths.work_dir() / ".vmaf"

    def test_should_not_depend_on_the_current_folder(
        self, monkeypatch, tmp_path,
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(PAYLOAD))

        score = _score_vmaf(_clip(tmp_path / "w"), _clip(tmp_path, "r.mp4"))

        assert score == 93.5

    def test_should_give_ffmpeg_absolute_input_paths(
        self, monkeypatch, tmp_path,
    ):
        calls = []
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(PAYLOAD, calls))
        monkeypatch.chdir(tmp_path)
        _clip(tmp_path / "w")
        _clip(tmp_path, "ref.mp4")

        _score_vmaf(Path("w/s0_h264_1080_cq29.mp4"), Path("ref.mp4"))

        cmd, _ = calls[0]
        inputs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-i"]
        assert len(inputs) == 2
        assert all(Path(p).is_absolute() for p in inputs)

    def test_should_raise_naming_the_clip_when_no_log_is_written(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(None))
        encoded = _clip(tmp_path)

        with pytest.raises(RuntimeError, match=encoded.name):
            _score_vmaf(encoded, _clip(tmp_path, "ref.mp4"))

    def test_should_ignore_a_stale_log_when_ffmpeg_writes_nothing(
        self, monkeypatch, tmp_path,
    ):
        """A killed run can leave a log behind at exactly this name.

        It must never be read as this run's score.
        """
        encoded = _clip(tmp_path)
        token = hashlib.sha1(str(encoded).encode("utf-8")).hexdigest()[:16]
        log_dir = paths.work_dir() / ".vmaf"
        log_dir.mkdir(parents=True)
        (log_dir / f"{token}.json").write_text(
            json.dumps({"pooled_metrics": {"vmaf": {"mean": 12.3}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(None))

        with pytest.raises(RuntimeError, match=encoded.name):
            _score_vmaf(encoded, _clip(tmp_path, "ref.mp4"))

    def test_should_use_a_different_log_for_each_probe(
        self, monkeypatch, tmp_path,
    ):
        calls = []
        monkeypatch.setattr(probe, "run_tool", _ffmpeg_writing(PAYLOAD, calls))
        reference = _clip(tmp_path, "ref.mp4")

        _score_vmaf(_clip(tmp_path, "s0.mp4"), reference)
        _score_vmaf(_clip(tmp_path, "s1.mp4"), reference)

        assert _log_name(calls[0][0]) != _log_name(calls[1][0])


class TestSelectCq:
    def test_should_prefer_the_cheaper_probe_when_both_meet_the_target(
        self,
    ):
        assert select_cq(26, 96.0, 34, 93.5, 93) == 34

    def test_should_take_the_only_probe_that_meets_the_target(self):
        assert select_cq(26, 94.0, 34, 90.0, 93) == 26

    def test_should_take_the_better_probe_when_neither_meets_it(self):
        assert select_cq(26, 91.0, 34, 88.0, 93) == 26


def _scene(folder: Path, index: int) -> SceneChunk:
    return SceneChunk(
        index=index, path=folder / f"chunk_{index:03d}.mp4",
        start_time=4.0 * index, duration=4.0,
    )


class TestEncodeProbe:
    def test_should_run_the_final_encode_command(self, monkeypatch, tmp_path):
        """A probe made with other options than the final encode is
        scored on a different stream, so the CQ it picks is wrong."""
        monkeypatch.setattr(
            encode_mod, "nvenc_supports_film_grain", lambda: True,
        )
        ran = []
        monkeypatch.setattr(
            probe, "run_tool", lambda cmd, **kw: ran.append(cmd),
        )
        config = PipelineConfig()
        for variant in config.build_variants("portrait"):
            kwargs = dict(
                chunk=_scene(tmp_path, 0), variant=variant, cq=30,
                film_grain=8, keyframe_interval=48, config=config,
                output_path=tmp_path / f"{variant.codec}.mp4",
            )
            probe._encode_probe(**kwargs)
            assert ran[-1] == encode_mod._build_encode_cmd(**kwargs)

    def test_should_take_its_command_from_the_encode_builder(
        self, monkeypatch, tmp_path,
    ):
        ran = []
        monkeypatch.setattr(
            probe, "run_tool", lambda cmd, **kw: ran.append(cmd),
        )
        monkeypatch.setattr(
            encode_mod, "_build_encode_cmd", lambda *a, **kw: ["built"],
        )
        config = PipelineConfig()

        probe._encode_probe(
            chunk=_scene(tmp_path, 0),
            variant=config.build_variants("portrait")[0], cq=30,
            film_grain=0, keyframe_interval=48, config=config,
            output_path=tmp_path / "p.mp4",
        )

        assert ran == [["built"]]


class TestRunProbing:
    def test_should_pair_each_probe_with_its_own_score(
        self, monkeypatch, tmp_path,
    ):
        """Two scenes x two variants, every probe scored differently, so
        a swapped or misplaced score changes the chosen CQ."""
        config = PipelineConfig()
        variants = [
            v for v in config.build_variants("portrait")
            if v.resolution == "1080"
        ]
        scores = {
            # Scene 0: both probes meet 93, so the cheaper (higher CQ) wins.
            "s0_av1_1080_cq26": 97.0, "s0_av1_1080_cq34": 94.0,
            "s0_h264_1080_cq21": 96.0, "s0_h264_1080_cq29": 93.5,
            # Scene 1: only the lower CQ meets 93.
            "s1_av1_1080_cq26": 95.0, "s1_av1_1080_cq34": 90.0,
            "s1_h264_1080_cq21": 93.2, "s1_h264_1080_cq29": 89.0,
        }
        made = []
        monkeypatch.setattr(
            probe, "_encode_probe",
            lambda **kw: made.append(kw["output_path"].stem),
        )
        monkeypatch.setattr(
            probe, "_score_vmaf", lambda encoded, ref: scores[encoded.stem],
        )
        monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

        results = probe.run_probing(
            scenes=[_scene(tmp_path, 0), _scene(tmp_path, 1)], config=config,
            variants=variants, film_grain=0, keyframe_interval=48,
            work_dir=tmp_path,
        )

        assert [
            (r.scene_index, r.variant.codec, r.cq_low, r.cq_high,
             r.vmaf_low, r.vmaf_high, r.selected_cq, r.selected_vmaf)
            for r in results
        ] == [
            (0, "av1", 26, 34, 97.0, 94.0, 34, 94.0),
            (0, "h264", 21, 29, 96.0, 93.5, 29, 93.5),
            (1, "av1", 26, 34, 95.0, 90.0, 26, 95.0),
            (1, "h264", 21, 29, 93.2, 89.0, 21, 93.2),
        ]
        assert sorted(made) == sorted(scores)
