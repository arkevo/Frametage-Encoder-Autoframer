"""Tests for ftea.encode.report module."""

import dataclasses
import json
from pathlib import Path

import pytest

import ftea.encode.report as report_module
from ftea.encode.config import Variant
from ftea.encode.probe import ProbeResult
from ftea.encode.encode import VariantOutput
from ftea.encode.scene import SceneChunk, SourceAnalysis
from ftea.encode.report import (
    ClipReport,
    generate_clip_report,
    generate_markdown_report,
    save_clip_report,
    generate_batch_report,
    save_batch_report,
)


# --- Fixtures ---

def _make_source_analysis(
    orientation="landscape",
    fps=24.0,
    is_bw=False,
    film_grain=0,
    has_audio=True,
    scene_count=2,
    scene_durations=None,
) -> SourceAnalysis:
    if scene_durations is None:
        scene_durations = [5.0] * scene_count
    scenes = [
        SceneChunk(
            index=i,
            path=Path(f"/tmp/scene_{i}.mp4"),
            start_time=sum(scene_durations[:i]),
            duration=scene_durations[i],
        )
        for i in range(scene_count)
    ]
    return SourceAnalysis(
        orientation=orientation,
        fps=fps,
        keyframe_interval=round(fps * 2),
        is_bw=is_bw,
        film_grain=film_grain,
        has_audio=has_audio,
        scenes=scenes,
    )


def _make_variant(codec="av1", resolution="1080", bit_depth=10) -> Variant:
    return Variant(
        codec=codec,
        resolution=resolution,
        width=1920,
        height=1080,
        start_cq=30,
        maxrate="3000k",
        bufsize="6000k",
        vmaf_target=93,
        bit_depth=bit_depth,
    )


def _make_probe_results(
    scene_count=2,
    variants=None,
    vmaf_values=None,
) -> list[ProbeResult]:
    if variants is None:
        variants = [
            _make_variant("av1", "1080", 10),
            _make_variant("h264", "1080", 8),
        ]
    if vmaf_values is None:
        vmaf_values = [95.0] * scene_count
    results = []
    for scene_idx in range(scene_count):
        vmaf = vmaf_values[scene_idx]
        for v in variants:
            results.append(ProbeResult(
                scene_index=scene_idx,
                variant=v,
                cq_low=28,
                cq_high=32,
                vmaf_low=vmaf - 1,
                vmaf_high=vmaf,
                selected_cq=30,
                selected_vmaf=vmaf,
            ))
    return results


def _make_variant_outputs(variants=None) -> list[VariantOutput]:
    if variants is None:
        variants = [
            _make_variant("av1", "1080", 10),
            _make_variant("h264", "1080", 8),
        ]
    return [
        VariantOutput(
            variant=v,
            path=Path(f"/tmp/{v.codec}_{v.resolution}p.mp4"),
            file_size_kb=500,
            bitrate_kbps=1500,
        )
        for v in variants
    ]


# --- Tests: generate_clip_report ---

class TestGenerateClipReport:
    def test_should_return_passed_when_all_vmaf_above_floor(self):
        analysis = _make_source_analysis()
        probes = _make_probe_results(vmaf_values=[95.0, 95.0])
        outputs = _make_variant_outputs()

        report = generate_clip_report(
            video_id="test-video",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=42.5,
            vmaf_floor=91,
        )

        assert report.status == "passed"
        assert report.video_id == "test-video"
        assert report.encode_time_sec == 42.5
        assert report.scene_count == 2
        assert report.error is None

    def test_should_return_warning_when_vmaf_below_floor(self):
        analysis = _make_source_analysis()
        probes = _make_probe_results(vmaf_values=[89.0, 89.0])
        outputs = _make_variant_outputs()

        report = generate_clip_report(
            video_id="low-vmaf",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=30.0,
            vmaf_floor=91,
        )

        assert report.status == "warning"

    def test_should_use_custom_vmaf_floor(self):
        """A VMAF of 89 is 'passed' when floor is 88."""
        analysis = _make_source_analysis()
        probes = _make_probe_results(vmaf_values=[89.0, 89.0])
        outputs = _make_variant_outputs()

        report = generate_clip_report(
            video_id="custom-floor",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=30.0,
            vmaf_floor=88,
        )

        assert report.status == "passed"

    def test_should_return_failed_when_error_provided(self):
        analysis = _make_source_analysis()

        report = generate_clip_report(
            video_id="error-video",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=[],
            variant_outputs=[],
            encode_time_sec=5.0,
            vmaf_floor=91,
            error="Timeout exceeded",
        )

        assert report.status == "failed"
        assert report.error == "Timeout exceeded"

    def test_should_build_per_scene_variant_data(self):
        analysis = _make_source_analysis(scene_count=2)
        variants = [_make_variant("av1", "1080", 10)]
        probes = _make_probe_results(scene_count=2, variants=variants)
        outputs = _make_variant_outputs(variants=variants)

        report = generate_clip_report(
            video_id="scenes-test",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=20.0,
            vmaf_floor=91,
        )

        assert len(report.scenes) == 2
        assert "av1_1080p" in report.scenes[0]["variants"]

    def test_should_compute_duration_weighted_vmaf_mean(self):
        """Short scene (0.5s) at VMAF 88, long scene (9.5s) at VMAF 96.
        Arithmetic mean = 92.0, duration-weighted = ~95.6.
        """
        analysis = _make_source_analysis(
            scene_count=2,
            scene_durations=[0.5, 9.5],
        )
        variants = [_make_variant("av1", "1080", 10)]
        probes = _make_probe_results(
            scene_count=2, variants=variants, vmaf_values=[88.0, 96.0],
        )
        outputs = _make_variant_outputs(variants=variants)

        report = generate_clip_report(
            video_id="weighted-test",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=20.0,
            vmaf_floor=91,
        )

        av1_variant = report.variants[0]
        # Duration-weighted: (88*0.5 + 96*9.5) / (0.5+9.5) = 95.6
        assert av1_variant["vmaf_mean"] == 95.6
        assert av1_variant["vmaf_min"] == 88.0

    def test_should_fall_back_to_arithmetic_when_no_durations(self):
        """If scene durations can't be matched, fall back to arithmetic."""
        analysis = _make_source_analysis(scene_count=2)
        variants = [_make_variant("av1", "1080", 10)]
        probes = _make_probe_results(
            scene_count=2, variants=variants, vmaf_values=[95.0, 95.0],
        )
        outputs = _make_variant_outputs(variants=variants)

        report = generate_clip_report(
            video_id="equal-test",
            source=Path("/input/test.mp4"),
            source_analysis=analysis,
            probe_results=probes,
            variant_outputs=outputs,
            encode_time_sec=20.0,
            vmaf_floor=91,
        )

        av1_variant = report.variants[0]
        assert av1_variant["vmaf_mean"] == 95.0

    def test_should_include_source_metadata(self):
        analysis = _make_source_analysis(
            orientation="portrait", fps=30.0, is_bw=True,
            film_grain=8, has_audio=False,
        )

        report = generate_clip_report(
            video_id="meta-test",
            source=Path("/input/portrait.mp4"),
            source_analysis=analysis,
            probe_results=[],
            variant_outputs=[],
            encode_time_sec=10.0,
            vmaf_floor=91,
        )

        assert report.orientation == "portrait"
        assert report.fps == 30.0
        assert report.is_bw is True
        assert report.film_grain == 8
        assert report.has_audio is False
        assert report.source == "portrait.mp4"


# --- Tests: save_clip_report ---

class TestSaveClipReport:
    def test_should_write_json_to_path(self, tmp_path):
        report = ClipReport(
            video_id="save-test",
            source="/input/test.mp4",
            orientation="landscape",
            fps=24.0,
            is_bw=False,
            film_grain=0,
            has_audio=True,
            scene_count=1,
            duration_s=5.0,
            encode_time_sec=10.0,
            scenes=[],
            variants=[],
            status="passed",
        )
        out = tmp_path / "reports" / "save-test.json"
        save_clip_report(report, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["video_id"] == "save-test"
        assert data["status"] == "passed"

    def test_should_create_parent_dirs(self, tmp_path):
        report = ClipReport(
            video_id="dirs-test",
            source="/input/test.mp4",
            orientation="landscape",
            fps=24.0,
            is_bw=False,
            film_grain=0,
            has_audio=True,
            scene_count=0,
            duration_s=0.0,
            encode_time_sec=0.0,
            scenes=[],
            variants=[],
            status="passed",
        )
        deep = tmp_path / "a" / "b" / "c" / "report.json"
        save_clip_report(report, deep)
        assert deep.exists()


# --- Tests: generate_batch_report ---

class TestGenerateBatchReport:
    def test_should_summarize_passed_and_failed(self):
        passed = ClipReport(
            video_id="ok", source="/a.mp4", orientation="landscape",
            fps=24.0, is_bw=False, film_grain=0, has_audio=True,
            scene_count=1, duration_s=12.0, encode_time_sec=30.0,
            scenes=[], variants=[{"vmaf_mean": 95.0}], status="passed",
        )
        failed = ClipReport(
            video_id="bad", source="/b.mp4", orientation="landscape",
            fps=24.0, is_bw=False, film_grain=0, has_audio=True,
            scene_count=1, duration_s=8.0, encode_time_sec=5.0,
            scenes=[], variants=[], status="failed", error="Timeout",
        )

        batch = generate_batch_report(
            clip_reports=[passed, failed],
            started_at="2026-03-06T00:00:00Z",
            completed_at="2026-03-06T00:01:00Z",
            wall_time_sec=60.0,
            retry_summary={},
        )

        assert batch["total"] == 2
        assert batch["encoded"] == 1
        assert batch["failed"] == 1
        assert batch["wall_time_sec"] == 60.0

    def test_should_include_per_clip_data(self):
        report = ClipReport(
            video_id="clip1", source="/a.mp4", orientation="landscape",
            fps=24.0, is_bw=False, film_grain=0, has_audio=True,
            scene_count=1, duration_s=15.0, encode_time_sec=20.0,
            scenes=[], variants=[{"vmaf_mean": 93.0}], status="passed",
        )

        batch = generate_batch_report(
            clip_reports=[report],
            started_at="2026-03-06T00:00:00Z",
            completed_at="2026-03-06T00:00:20Z",
            wall_time_sec=20.0,
            retry_summary={},
        )

        assert len(batch["clips"]) == 1
        assert batch["clips"][0]["video_id"] == "clip1"
        assert batch["clips"][0]["status"] == "passed"

    def test_should_handle_empty_clips_list(self):
        batch = generate_batch_report(
            clip_reports=[],
            started_at="2026-03-06T00:00:00Z",
            completed_at="2026-03-06T00:00:00Z",
            wall_time_sec=0.0,
            retry_summary={},
        )

        assert batch["total"] == 0
        assert batch["encoded"] == 0
        assert batch["avg_per_clip_sec"] == 0


# --- Tests: save_batch_report ---

class TestSaveBatchReport:
    def test_should_write_batch_json(self, tmp_path):
        report = {"total": 5, "encoded": 4, "failed": 1}
        out = tmp_path / "batch_report.json"
        save_batch_report(report, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total"] == 5


# --- Tests: the additive reframe field ---

_REFRAME_APPLIED = {
    "applied": True,
    "reason": "landscape",
    "elapsed_sec": 41.2,
    "source": {
        "width": 1920, "height": 1080, "rotation": 0,
        "display_width": 1920, "display_height": 1080,
        "orientation": "landscape", "fps": 25.0, "duration_s": 12.5,
    },
}

_REFRAME_SKIPPED = {
    "applied": False,
    "reason": "already_vertical",
    "elapsed_sec": 0.0,
    "source": {
        "width": 1080, "height": 1920, "rotation": 0,
        "display_width": 1080, "display_height": 1920,
        "orientation": "portrait", "fps": 25.0, "duration_s": 12.5,
    },
}


_REFRAME_FAILED = {
    "applied": False,
    "outcome": "failed",
    "reason": "landscape",
    "elapsed_sec": 12.0,
    "source": dict(_REFRAME_APPLIED["source"]),
}


def _clip(video_id="c", status="passed", reframe=None, error=None):
    return ClipReport(
        video_id=video_id, source=f"/input/{video_id}.mp4",
        orientation="landscape", fps=24.0, is_bw=False, film_grain=0,
        has_audio=True, scene_count=1, duration_s=10.0,
        encode_time_sec=1.0, scenes=[], variants=[],
        status=status, error=error, reframe=reframe,
    )


class TestClipReportReframeField:
    def test_should_default_to_none(self):
        assert _clip().reframe is None

    def test_should_carry_the_entry_when_supplied(self):
        assert _clip(reframe=_REFRAME_APPLIED).reframe == _REFRAME_APPLIED

    def test_should_be_the_last_field_so_positional_build_still_works(self):
        """main.py's failure branch builds this positionally-compatible;
        adding the field anywhere but last would break that call."""
        names = list(ClipReport.__dataclass_fields__)
        assert names[-5:] == [
            "error", "reframe", "conform", "source_size", "source_digest",
        ]


class TestGenerateClipReportReframe:
    def test_should_default_reframe_to_none(self):
        report = generate_clip_report(
            video_id="x", source=Path("/a.mp4"),
            source_analysis=_make_source_analysis(),
            probe_results=[], variant_outputs=[],
            encode_time_sec=1.0, vmaf_floor=91,
        )
        assert report.reframe is None

    def test_should_pass_the_entry_through(self):
        report = generate_clip_report(
            video_id="x", source=Path("/a.mp4"),
            source_analysis=_make_source_analysis(),
            probe_results=[], variant_outputs=[],
            encode_time_sec=1.0, vmaf_floor=91,
            reframe=_REFRAME_APPLIED,
        )
        assert report.reframe == _REFRAME_APPLIED

    def test_should_keep_status_passed_for_a_skip(self):
        """A reframe skip is not a clip failure — it still encodes."""
        report = generate_clip_report(
            video_id="x", source=Path("/a.mp4"),
            source_analysis=_make_source_analysis(),
            probe_results=[], variant_outputs=[],
            encode_time_sec=1.0, vmaf_floor=91,
            reframe=_REFRAME_SKIPPED,
        )
        assert report.status == "passed"


class TestSaveClipReportReframeKey:
    def test_should_omit_the_key_entirely_when_stage_did_not_run(
        self, tmp_path,
    ):
        """Flag off => per-clip report JSON is byte-identical to before."""
        out = tmp_path / "r.json"
        save_clip_report(_clip(), out)

        assert "reframe" not in json.loads(out.read_text())

    def test_should_include_the_key_when_stage_ran(self, tmp_path):
        out = tmp_path / "r.json"
        save_clip_report(_clip(reframe=_REFRAME_APPLIED), out)

        data = json.loads(out.read_text())
        assert data["reframe"]["applied"] is True
        assert data["reframe"]["source"]["display_height"] == 1080

    def test_should_keep_every_other_key_unchanged(self, tmp_path):
        out = tmp_path / "r.json"
        save_clip_report(_clip(), out)

        assert set(json.loads(out.read_text())) == {
            "video_id", "source", "orientation", "fps", "is_bw",
            "film_grain", "has_audio", "scene_count", "duration_s",
            "encode_time_sec", "scenes", "variants", "status", "error",
            "source_size", "source_digest",
        }


class TestBatchReportReframeRollup:
    def test_should_omit_the_rollup_when_no_clip_ran_the_stage(self):
        batch = generate_batch_report(
            clip_reports=[_clip("a"), _clip("b")],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )
        assert "reframe" not in batch

    def test_should_count_applied_skipped_and_failed(self):
        batch = generate_batch_report(
            clip_reports=[
                _clip("a", reframe=_REFRAME_APPLIED),
                _clip("b", reframe=_REFRAME_SKIPPED),
                _clip("c", reframe=_REFRAME_SKIPPED),
                _clip("d", status="failed", error="reframe failed",
                      reframe=_REFRAME_FAILED),
                _clip("e"),
            ],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )

        assert batch["reframe"] == {"applied": 1, "skipped": 2, "failed": 1}

    def test_should_count_what_the_stage_did_not_how_the_clip_ended(self):
        """A clip that reframed and then failed in the encode is a
        reframe that worked; the clip failure is in the totals."""
        batch = generate_batch_report(
            clip_reports=[
                _clip("a", status="failed", error="NVENC error",
                      reframe=_REFRAME_APPLIED),
                _clip("b", status="failed", error="NVENC error",
                      reframe=_REFRAME_SKIPPED),
            ],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )

        assert batch["reframe"] == {"applied": 1, "skipped": 1, "failed": 0}
        assert batch["failed"] == 2

    def test_should_trust_the_stage_outcome_over_the_applied_flag(self):
        entry = dict(_REFRAME_SKIPPED, outcome="failed")
        batch = generate_batch_report(
            clip_reports=[
                _clip("a", status="failed", error="x", reframe=entry),
            ],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )
        assert batch["reframe"] == {"applied": 0, "skipped": 0, "failed": 1}

    def test_should_not_change_encoded_or_failed_counts(self):
        batch = generate_batch_report(
            clip_reports=[
                _clip("a", reframe=_REFRAME_SKIPPED),
                _clip("b", reframe=_REFRAME_SKIPPED),
            ],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )

        assert batch["total"] == 2
        assert batch["encoded"] == 2
        assert batch["failed"] == 0


# --- Tests: what the published output folder says ---

class TestReportFilesForUsers:
    def test_should_record_only_the_source_file_name(self):
        report = generate_clip_report(
            video_id="x", source=Path("/videos/private folder/clip one.mp4"),
            source_analysis=_make_source_analysis(), probe_results=[],
            variant_outputs=[], encode_time_sec=1.0, vmaf_floor=91,
        )
        assert report.source == "clip one.mp4"

    def test_should_save_conform_only_when_it_was_applied(self, tmp_path):
        out = tmp_path / "r.json"
        save_clip_report(_clip(), out)
        assert "conform" not in json.loads(out.read_text(encoding="utf-8"))

        entry = {"applied": True, "source": "1080x1080", "rotation": 0,
                 "frame": "1080x1920"}
        save_clip_report(dataclasses.replace(_clip(), conform=entry), out)
        assert json.loads(out.read_text(encoding="utf-8"))["conform"] == entry

    def test_should_count_conformed_clips_in_the_batch(self):
        batch = generate_batch_report(
            clip_reports=[
                _clip("a"),
                dataclasses.replace(_clip("b"), conform={"applied": True}),
            ],
            started_at="t0", completed_at="t1", wall_time_sec=1.0,
            retry_summary={},
        )
        assert batch["conform"] == {"applied": 1}

    def test_should_carry_no_gate_check(self):
        batch = generate_batch_report([_clip()], "t0", "t1", 1.0, {})
        assert "gate_check" not in batch
        assert batch["not_attempted"] == []

    def test_should_count_clips_a_halted_batch_never_reached(self):
        batch = generate_batch_report(
            [_clip("a", status="failed")], "t0", "t1", 1.0, {},
            not_attempted=("b", "c"),
        )
        assert batch["not_attempted"] == ["b", "c"]
        assert (batch["total"], batch["failed"]) == (3, 1)

    def test_should_leave_the_old_report_whole_when_the_write_fails(
        self, tmp_path, monkeypatch,
    ):
        out = tmp_path / "report.json"
        save_clip_report(_clip("old"), out)

        def refuse(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(report_module.os, "replace", refuse)
        with pytest.raises(OSError, match="disk full"):
            save_clip_report(_clip("new"), out)

        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["video_id"] == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["report.json"]

    def test_should_write_markdown_with_source_and_error(self, tmp_path):
        out = tmp_path / "report.md"
        failed = _clip(
            "c", status="failed",
            error="ToolError: ffmpeg failed (exit 1): a | b",
        )

        generate_markdown_report(failed, out)

        text = out.read_text(encoding="utf-8")
        assert "| Source | /input/c.mp4 |" in text
        assert "| Error | ToolError: ffmpeg failed (exit 1): a \\| b |" in text

    def test_should_write_markdown_for_any_clip_name(self, tmp_path):
        out = tmp_path / "report.md"
        generate_markdown_report(_clip("Café 東京"), out)
        assert "# Encode Report: Café 東京" in out.read_text(encoding="utf-8")
