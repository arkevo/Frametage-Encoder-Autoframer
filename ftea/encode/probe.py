"""Per-scene CQ probing engine with parallel VMAF scoring."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from . import encode
from .config import PipelineConfig, Variant
from .scene import SceneChunk
from .tools import run_tool


@dataclass
class ProbeResult:
    """Result of CQ probing for one scene × variant."""

    scene_index: int
    variant: Variant
    cq_low: int
    cq_high: int
    vmaf_low: float
    vmaf_high: float
    selected_cq: int
    selected_vmaf: float


def select_cq(
    cq_low: int, vmaf_low: float,
    cq_high: int, vmaf_high: float,
    vmaf_target: int,
) -> int:
    """Select the best CQ from two probes.

    Rules:
    1. Prefer CQ that produces VMAF >= target
    2. Among those above target, pick closest (saves bits)
    3. If both below target, pick lower CQ (higher quality)
    """
    low_above = vmaf_low >= vmaf_target
    high_above = vmaf_high >= vmaf_target

    if low_above and high_above:
        if abs(vmaf_high - vmaf_target) <= abs(vmaf_low - vmaf_target):
            return cq_high
        return cq_low
    elif low_above:
        return cq_low
    elif high_above:
        return cq_high
    else:
        return cq_low


def _encode_probe(
    chunk: SceneChunk,
    variant: Variant,
    cq: int,
    film_grain: int,
    keyframe_interval: int,
    config: PipelineConfig,
    output_path: Path,
) -> Path:
    """Encode a single probe with NVENC.

    The command is the final encode's own (encode._build_encode_cmd), so
    the probe is scored on the stream the chosen CQ will produce: same
    preset, tune, rate control, multipass, lookahead, caps, bit depth
    and film grain.
    """
    run_tool(encode._build_encode_cmd(
        chunk, variant, cq, film_grain, keyframe_interval, config,
        output_path,
    ))
    return output_path


def _score_vmaf(encoded: Path, reference: Path) -> float:
    """Score VMAF for an encoded probe against its source chunk.

    Both are scaled to 1920x1080 so every score uses the same reference
    size. ffmpeg's filter-option parser treats "'" as a quote with no
    reliable escape and ":" as an option separator, so the libvmaf
    log_path must contain neither. It is therefore a bare file name made
    from a hash of the probe's path, and ffmpeg runs inside the log
    folder. A path built from the source name could carry an apostrophe,
    an absolute Windows path always carries a drive colon, and a path
    relative to the current folder cannot be built across drives.

    The name is the same on every run for the same probe path, so a run
    that was killed can leave a log behind at exactly this name. It is
    deleted before ffmpeg starts, so the check below can only ever see a
    log that this call wrote.
    """
    log_dir = paths.work_dir() / ".vmaf"
    log_dir.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha1(str(encoded).encode("utf-8")).hexdigest()[:16]
    log_name = f"{token}.json"
    vmaf_log = log_dir / log_name
    vmaf_log.unlink(missing_ok=True)

    run_tool(
        [
            "ffmpeg", "-y",
            "-i", str(Path(encoded).resolve()),
            "-i", str(Path(reference).resolve()),
            "-lavfi",
            "[0:v]scale=1920:1080:flags=bicubic[main];"
            "[1:v]scale=1920:1080:flags=bicubic[ref];"
            "[main][ref]libvmaf=model=version=vmaf_v0.6.1"
            f":log_fmt=json:log_path={log_name}",
            "-f", "null", "-",
        ],
        cwd=log_dir,
    )

    if not vmaf_log.exists():
        raise RuntimeError(f"libvmaf wrote no log for {encoded.name}")
    data = json.loads(vmaf_log.read_text(encoding="utf-8"))
    vmaf_log.unlink()
    return data["pooled_metrics"]["vmaf"]["mean"]


def _vmaf_worker(args: tuple) -> tuple[str, float]:
    """Thread-pool task that scores one probe.

    Accepts a tuple for ThreadPoolExecutor compatibility.
    Args: (probe_key, encoded_path, reference_path)
    Returns: (probe_key, vmaf_score)
    """
    probe_key, encoded_path, reference_path = args
    score = _score_vmaf(Path(encoded_path), Path(reference_path))
    return probe_key, score


def run_probing(
    scenes: list[SceneChunk],
    config: PipelineConfig,
    variants: list[Variant],
    film_grain: int,
    keyframe_interval: int,
    work_dir: Path,
) -> list[ProbeResult]:
    """Run CQ probing for all scenes x variants.

    Phase 1: Encode probe_low and probe_high for every scene x variant
             (sequential -- GPU is fast, ~0.3s each)
    Phase 2: Score all probes with parallel VMAF workers (ThreadPoolExecutor)
    Phase 3: Select best CQ per scene x variant using select_cq()

    Uses ThreadPoolExecutor instead of ProcessPoolExecutor because the heavy
    work is done by subprocess (ffmpeg), so threads give equivalent parallelism
    without Windows spawn-mode issues (pickling, module re-import deadlocks).
    """
    probes_dir = work_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    probe_jobs: list[dict] = []

    for scene in scenes:
        for variant in variants:
            cq_low = max(0, variant.start_cq - config.cq_bracket)
            cq_high = variant.start_cq + config.cq_bracket

            key_low = (
                f"s{scene.index}_{variant.codec}"
                f"_{variant.resolution}_cq{cq_low}"
            )
            key_high = (
                f"s{scene.index}_{variant.codec}"
                f"_{variant.resolution}_cq{cq_high}"
            )
            path_low = probes_dir / f"{key_low}.mp4"
            path_high = probes_dir / f"{key_high}.mp4"

            probe_jobs.append({
                "scene": scene, "variant": variant,
                "cq": cq_low, "key": key_low, "path": path_low,
                "label": "low", "cq_low": cq_low, "cq_high": cq_high,
                "key_low": key_low, "key_high": key_high,
            })
            probe_jobs.append({
                "scene": scene, "variant": variant,
                "cq": cq_high, "key": key_high, "path": path_high,
                "label": "high",
            })

    # Phase 1: Encode probes sequentially
    # Small delay between NVENC sessions to avoid driver DLL init failures on Windows
    print(f"  Encoding {len(probe_jobs)} probes...")
    for job in probe_jobs:
        _encode_probe(
            chunk=job["scene"],
            variant=job["variant"],
            cq=job["cq"],
            film_grain=film_grain,
            keyframe_interval=keyframe_interval,
            config=config,
            output_path=job["path"],
        )
        time.sleep(0.5)

    # Phase 2: Score VMAF in parallel
    print(f"  Scoring VMAF with {config.vmaf_workers} workers...")
    vmaf_args = [
        (job["key"], str(job["path"]), str(job["scene"].path))
        for job in probe_jobs
    ]

    vmaf_scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=config.vmaf_workers) as pool:
        futures = {
            pool.submit(_vmaf_worker, args): args[0]
            for args in vmaf_args
        }
        for future in as_completed(futures):
            key, score = future.result()
            vmaf_scores[key] = score

    # Phase 3: Select CQ per scene x variant
    # Reuse cq/key values stored in "low" jobs to avoid recomputing
    results = []
    for job in probe_jobs:
        if job["label"] != "low":
            continue

        cq_low = job["cq_low"]
        cq_high = job["cq_high"]
        vmaf_low = vmaf_scores[job["key_low"]]
        vmaf_high = vmaf_scores[job["key_high"]]

        selected = select_cq(
            cq_low, vmaf_low, cq_high, vmaf_high,
            job["variant"].vmaf_target,
        )
        selected_vmaf = (
            vmaf_low if selected == cq_low else vmaf_high
        )

        results.append(ProbeResult(
            scene_index=job["scene"].index,
            variant=job["variant"],
            cq_low=cq_low, cq_high=cq_high,
            vmaf_low=vmaf_low, vmaf_high=vmaf_high,
            selected_cq=selected, selected_vmaf=selected_vmaf,
        ))

    return results
