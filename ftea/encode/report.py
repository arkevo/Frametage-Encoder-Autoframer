"""Quality reporting — per-clip and batch reports."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .probe import ProbeResult
from .encode import VariantOutput
from .scene import SourceAnalysis


@dataclass
class ClipReport:
    """Quality report for a single clip."""

    video_id: str
    source: str
    orientation: str
    fps: float
    is_bw: bool
    film_grain: int
    has_audio: bool
    scene_count: int
    duration_s: float
    encode_time_sec: float
    scenes: list[dict]
    variants: list[dict]
    status: str
    error: Optional[str] = None
    # Auto-reframe outcome, or None when the stage did not run. Built by
    # ftea.reframe.probe.reframe_report_entry(), or REFRAME_FAILED_ENTRY in
    # ftea.encode.main when the stage failed without describing the clip.
    reframe: Optional[dict] = None
    # How a clip whose shape was neither 16:9 nor 9:16 was padded onto
    # the encode frame (ftea.encode.geometry.conform_entry), or None.
    conform: Optional[dict] = None
    # The source fingerprint resume compares, so a different video with
    # the same name is never mistaken for a finished one: the file's size,
    # and a SHA-256 of its size plus its first and last MiB. None in a
    # failure report, and in reports made before fingerprints existed.
    source_size: Optional[int] = None
    source_digest: Optional[str] = None


def generate_clip_report(
    video_id: str,
    source: Path,
    source_analysis: SourceAnalysis,
    probe_results: list[ProbeResult],
    variant_outputs: list[VariantOutput],
    encode_time_sec: float,
    vmaf_floor: int,
    error: Optional[str] = None,
    reframe: Optional[dict] = None,
    conform: Optional[dict] = None,
) -> ClipReport:
    """Generate per-clip quality report.

    Builds per-scene CQ+VMAF data for each variant, computes variant-level
    summary stats (duration-weighted mean, min, std VMAF), and checks the
    VMAF floor from config.

    Only the source's file name is recorded, never its folder. The output
    folder, report included, is meant to be published as it is, and a
    local path often contains the user's account name.

    `reframe` carries the auto-reframe stage's outcome for this clip, or
    None when the stage did not run. A reframe SKIP does not change the
    status: an already-vertical clip is still encoded and packaged, so it
    genuinely passed. `conform` works the same way for the padding step.
    """
    # Build scene duration lookup
    scene_durations: dict[int, float] = {
        s.index: s.duration for s in source_analysis.scenes
    }

    # Build per-scene data
    scenes_by_index: dict[int, dict] = {}

    for pr in probe_results:
        if pr.scene_index not in scenes_by_index:
            scenes_by_index[pr.scene_index] = {
                "index": pr.scene_index,
                "variants": {},
            }
        variant_key = f"{pr.variant.codec}_{pr.variant.resolution}p"
        scenes_by_index[pr.scene_index]["variants"][variant_key] = {
            "cq_low": pr.cq_low,
            "vmaf_low": round(pr.vmaf_low, 2),
            "cq_high": pr.cq_high,
            "vmaf_high": round(pr.vmaf_high, 2),
            "selected_cq": pr.selected_cq,
            "selected_vmaf": round(pr.selected_vmaf, 2),
        }

    scenes_data = [scenes_by_index[i] for i in sorted(scenes_by_index)]

    # Add duration from source_analysis scenes
    for sd in scenes_data:
        if sd["index"] in scene_durations:
            sd["duration_sec"] = round(scene_durations[sd["index"]], 2)

    # Build per-variant summary with duration-weighted VMAF
    variants_data = []
    for vo in variant_outputs:
        variant_probes = [
            pr for pr in probe_results
            if pr.variant.codec == vo.variant.codec
            and pr.variant.resolution == vo.variant.resolution
        ]
        if not variant_probes:
            variants_data.append({
                "codec": vo.variant.codec,
                "resolution": f"{vo.variant.resolution}p",
                "bit_depth": vo.variant.bit_depth,
                "vmaf_mean": 0,
                "vmaf_min": 0,
                "vmaf_std": 0,
                "bitrate_kbps": vo.bitrate_kbps,
                "file_size_kb": vo.file_size_kb,
            })
            continue

        vmafs = [pr.selected_vmaf for pr in variant_probes]
        vmaf_min = min(vmafs)

        # Duration-weighted mean
        weights = [
            scene_durations.get(pr.scene_index, 1.0)
            for pr in variant_probes
        ]
        total_weight = sum(weights)
        if total_weight > 0:
            vmaf_mean = sum(
                v * w for v, w in zip(vmafs, weights)
            ) / total_weight
        else:
            vmaf_mean = sum(vmafs) / len(vmafs)

        # Std deviation (unweighted — close enough for reporting)
        vmaf_std = (
            (sum((v - vmaf_mean) ** 2 for v in vmafs) / len(vmafs)) ** 0.5
        )

        variants_data.append({
            "codec": vo.variant.codec,
            "resolution": f"{vo.variant.resolution}p",
            "bit_depth": vo.variant.bit_depth,
            "vmaf_mean": round(vmaf_mean, 2),
            "vmaf_min": round(vmaf_min, 2),
            "vmaf_std": round(vmaf_std, 2),
            "bitrate_kbps": vo.bitrate_kbps,
            "file_size_kb": vo.file_size_kb,
        })

    status = "failed" if error else "passed"
    if not error:
        for vd in variants_data:
            if vd["vmaf_min"] < vmaf_floor:
                status = "warning"
                break

    return ClipReport(
        video_id=video_id,
        source=source.name,
        orientation=source_analysis.orientation,
        fps=source_analysis.fps,
        is_bw=source_analysis.is_bw,
        film_grain=source_analysis.film_grain,
        has_audio=source_analysis.has_audio,
        scene_count=len(source_analysis.scenes),
        duration_s=round(sum(s.duration for s in source_analysis.scenes), 1),
        encode_time_sec=round(encode_time_sec, 1),
        scenes=scenes_data,
        variants=variants_data,
        status=status,
        error=error,
        reframe=reframe,
        conform=conform,
    )


def save_clip_report(report: ClipReport, path: Path) -> None:
    """Save clip report to JSON file, replacing any old one in one step.

    The JSON is written to <name>.tmp and then moved over the real name
    with os.replace, so a reader sees the old report or the new one,
    never half of one. The `reframe` and `conform` keys are omitted
    entirely when that step did not run, so their presence alone says the
    clip went through it.
    """
    data = asdict(report)
    for optional in ("reframe", "conform"):
        if data.get(optional) is None:
            data.pop(optional, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def generate_batch_report(
    clip_reports: list[ClipReport],
    started_at: str,
    completed_at: str,
    wall_time_sec: float,
    retry_summary: dict,
    not_attempted: Sequence[str] = (),
) -> dict:
    """Generate batch summary report."""
    passed = [r for r in clip_reports if r.status in ("passed", "warning")]
    failed = [r for r in clip_reports if r.status == "failed"]

    avg_time = (
        sum(r.encode_time_sec for r in passed) / len(passed)
        if passed else 0
    )

    batch = {
        "started_at": started_at,
        "completed_at": completed_at,
        "wall_time_sec": round(wall_time_sec, 1),
        "avg_per_clip_sec": round(avg_time, 1),
        # Files that could not be used (unreadable, an unusable name),
        # clips set aside (too long, a name clash) and clips never
        # reached after a BATCH_ON_ERROR=stop halt have no report, but
        # they are part of the batch the user asked for.
        "total": len(clip_reports) + len(not_attempted),
        "encoded": len(passed),
        "failed": len(failed),
        "not_attempted": list(not_attempted),
        "retry_summary": retry_summary,
        "clips": [
            {
                "video_id": r.video_id,
                "status": r.status,
                "vmaf_mean": (
                    round(
                        sum(v["vmaf_mean"] for v in r.variants)
                        / len(r.variants), 2,
                    )
                    if r.variants else None
                ),
                "encode_sec": r.encode_time_sec,
                "error": r.error,
            }
            for r in clip_reports
        ],
    }

    # Present only when the stage ran on some clip, and counted from what
    # the stage itself did. A clip that reframed and then failed in the
    # encode counts as applied here and as failed in the totals above.
    # The encoded/failed counts above are deliberately NOT affected: a
    # reframe skip is a normal pass.
    reframed = [_reframe_outcome(r.reframe) for r in clip_reports
                if r.reframe is not None]
    if reframed:
        batch["reframe"] = {
            "applied": reframed.count("reframed"),
            "skipped": reframed.count("skipped"),
            "failed": reframed.count("failed"),
        }

    conformed = [r for r in clip_reports if r.conform is not None]
    if conformed:
        batch["conform"] = {"applied": len(conformed)}

    return batch


def _reframe_outcome(entry: dict) -> str:
    """The stage's own outcome for a clip: reframed, skipped or failed.

    An entry without one comes from before the field existed, and is
    read from its applied flag.
    """
    outcome = entry.get("outcome")
    if outcome is None:
        return "reframed" if entry.get("applied") else "skipped"
    return outcome


def save_batch_report(report: dict, path: Path) -> None:
    """Save batch report to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def generate_markdown_report(report: ClipReport, path: Path) -> None:
    """Generate a human-readable markdown report for one clip.

    Written next to report.json in every clip's output folder.
    """
    lines = [
        f"# Encode Report: {report.video_id}",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Source | {_md_cell(report.source)} |",
        f"| Orientation | {report.orientation} |",
        f"| Frame Rate | {report.fps} fps |",
        f"| B&W | {report.is_bw} |",
        f"| Film Grain | {report.film_grain} |",
        f"| Scenes | {report.scene_count} |",
        f"| Audio | {report.has_audio} |",
        f"| Encode Time | {report.encode_time_sec}s |",
        f"| Status | {report.status} |",
    ]
    if report.error:
        lines.append(f"| Error | {_md_cell(report.error)} |")
    lines += [
        "",
        "## Variant Summary",
        "",
        "| Variant | Mean VMAF | Min VMAF | Std Dev"
        " | Bitrate | Size |",
        "|---------|-----------|----------|--------"
        "-|---------|------|",
    ]

    for v in report.variants:
        size_mb = v["file_size_kb"] / 1024
        lines.append(
            f"| {v['codec'].upper()} {v['resolution']}"
            f" ({v['bit_depth']}-bit)"
            f" | {v['vmaf_mean']}"
            f" | {v['vmaf_min']}"
            f" | {v['vmaf_std']}"
            f" | {v['bitrate_kbps']} kbps"
            f" | {size_mb:.1f} MB |"
        )

    # Per-scene tables, grouped by codec
    codecs = sorted({v["codec"] for v in report.variants})
    for codec in codecs:
        codec_variants = [
            v for v in report.variants if v["codec"] == codec
        ]
        resolutions = [v["resolution"] for v in codec_variants]

        lines += [
            "",
            f"## Per-Scene: {codec.upper()}",
            "",
        ]

        header = "| Scene | Duration |"
        separator = "|-------|----------|"
        for res in resolutions:
            header += f" {res} CQ | {res} VMAF |"
            separator += "--------|----------|"
        lines += [header, separator]

        for scene in report.scenes:
            row = f"| {scene['index']} | {scene.get('duration_sec', '?')}s |"
            for res in resolutions:
                key = f"{codec}_{res}"
                vdata = scene["variants"].get(key, {})
                cq = vdata.get("selected_cq", "?")
                vmaf = vdata.get("selected_vmaf", "?")
                row += f" {cq} | {vmaf} |"
            lines.append(row)

    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _md_cell(text: str) -> str:
    """Make text safe inside one markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")
