"""Startup checks: can this machine run the encoder at all?

run() calls check_toolchain once, after pre-scan finds work and before
any clip is touched, so a missing program or an unsuitable GPU produces
one clear message instead of the same failure on every clip, three
times over (the first attempt and two retries).

The bare one-frame trials tell a GPU or driver problem apart from a card
that cannot encode AV1. Then a short sample is encoded with the exact
options a clip uses, so an FFmpeg build or driver too old for one of
them (the uhq tune, say) is caught here, naming the setting.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import encode
from .config import ConfigValidationError, PipelineConfig
from .scene import SceneChunk
from .tools import ToolError, run_tool, stderr_tail

GUIDE = "docs/PROCESS-GUIDE.md"
REQUIRED_PROGRAMS = ("ffmpeg", "ffprobe")
# Order matters: when H.264 cannot start, the GPU or its driver is the
# problem, and the other trials would only repeat it.
REQUIRED_ENCODERS = ("h264_nvenc", "av1_nvenc", "hevc_nvenc")
REQUIRED_FILTER = "libvmaf"
# Software AV1 decoders, either of which will do. VMAF scoring reads the
# av1_nvenc probes back, and FFmpeg's own av1 decoder works only with
# hardware acceleration, which ffmpeg does not pick on its own.
AV1_DECODERS = ("libdav1d", "libaom-av1")
# The encoder each THUMB_FORMAT needs that not every FFmpeg build has.
THUMB_ENCODERS = {"webp": "libwebp"}
# NVENC refuses very small frames; 256x256 is above every encoder's
# minimum size.
TRIAL_INPUT = "color=c=black:s=256x256:r=24:d=0.2"
LIST_TIMEOUT_SEC = 30
TRIAL_TIMEOUT_SEC = 60
# The real-options trials encode this 1 s sample, written to a temp
# folder first. lavfi and the FFV1 encoder are part of every FFmpeg build.
SAMPLE_INPUT = "testsrc2=size=1920x1080:rate=24:duration=1"
SAMPLE_NAME = "trial_sample.mkv"
SAMPLE_KEYFRAME_INTERVAL = 48
# (codec, resolution) of the variants tried with their real options.
REAL_TRIALS = (("av1", "1080"), ("h264", "1080"))
# The lines ffmpeg prints when it does not know an option or a value
# ("Unable to parse option" and "Option not found" come with them).
OPTION_ERRORS = (
    re.compile(r"Error setting option (\S+) to value"),
    re.compile(r"Unrecognized option '-?([^']+)'"),
)
# The config.env setting behind each encode option a user can change.
SETTING_FOR_OPTION = {
    "preset": "NVENC_PRESET",
    "tune": "NVENC_TUNE",
    "rc": "NVENC_RC",
    "multipass": "NVENC_MULTIPASS",
    "rc-lookahead": "NVENC_RC_LOOKAHEAD",
    "pix_fmt": "AV1_BIT_DEPTH",
}
# A value to suggest instead, where one is known to work on older builds.
FALLBACK_VALUE = {"NVENC_TUNE": "hq"}


def check_toolchain(config: PipelineConfig) -> list[str]:
    """Problems that stop the encoder from running; empty means ready."""
    missing = _missing_programs()
    if missing:
        return missing
    try:
        return (
            _missing_capabilities(config)
            or _failed_trials()
            or _failed_real_trials(config)
        )
    except ToolError as exc:
        return [f"could not query ffmpeg: {exc}"]
    except ConfigValidationError as exc:
        return [f"config.env: {exc}"]


def _missing_programs() -> list[str]:
    return [
        f"{name} is not on PATH. Install an FFmpeg build with NVENC and "
        f"libvmaf (see {GUIDE})."
        for name in REQUIRED_PROGRAMS
        if shutil.which(name) is None
    ]


def _listed(name: str, listing: str) -> bool:
    """Whether `ffmpeg -encoders` or `-filters` output lists `name`."""
    pattern = rf"(?m)^\s*\S+\s+{re.escape(name)}\s"
    return re.search(pattern, listing) is not None


def _listing(kind: str) -> str:
    """The output of `ffmpeg -encoders`, `-decoders` or `-filters`."""
    return run_tool(
        ["ffmpeg", "-hide_banner", f"-{kind}"], timeout=LIST_TIMEOUT_SEC,
    ).stdout


def _missing_capabilities(config: PipelineConfig) -> list[str]:
    """Every encoder, decoder and filter a clip will need, checked before
    any GPU time is spent: packaging and scoring come late in a clip."""
    encoders = _listing("encoders")
    filters = _listing("filters")
    problems = [
        f"this FFmpeg build has no {name} encoder. ftea needs an FFmpeg "
        f"built with NVIDIA NVENC (see {GUIDE})."
        for name in REQUIRED_ENCODERS
        if not _listed(name, encoders)
    ]
    if not _listed(REQUIRED_FILTER, filters):
        problems.append(
            "this FFmpeg build has no libvmaf filter, which ftea uses to "
            f"measure quality. Install a build that includes it (see {GUIDE})."
        )
    if not any(_listed(name, _listing("decoders")) for name in AV1_DECODERS):
        problems.append(
            "this FFmpeg build has no software AV1 decoder (libdav1d or "
            "libaom-av1), which ftea needs to measure the quality of the "
            f"AV1 variants. Install a build that includes one (see {GUIDE})."
        )
    thumb_encoder = THUMB_ENCODERS.get(config.thumb_format)
    if thumb_encoder and not _listed(thumb_encoder, encoders):
        problems.append(
            f"this FFmpeg build has no {thumb_encoder} encoder for "
            f"THUMB_FORMAT={config.thumb_format}. Install a build that "
            "includes it, or set THUMB_FORMAT=jpg in config.env."
        )
    return problems


def _failed_trials() -> list[str]:
    problems: list[str] = []
    for name in REQUIRED_ENCODERS:
        result = run_tool(
            [
                "ffmpeg", "-hide_banner", "-v", "error",
                "-f", "lavfi", "-i", TRIAL_INPUT,
                "-frames:v", "1", "-c:v", name, "-f", "null", "-",
            ],
            timeout=TRIAL_TIMEOUT_SEC,
            check=False,
        )
        if result.returncode == 0:
            continue
        problems.append(_trial_problem(name, stderr_tail(result.stderr)))
        if name == "h264_nvenc":
            break
    return problems


def _trial_problem(name: str, detail: str) -> str:
    if name == "h264_nvenc":
        return (
            f"the GPU could not start an NVENC encode ({detail}). Check "
            "that an NVIDIA GPU and a current driver are installed: "
            "`nvidia-smi` should list the GPU."
        )
    if name == "av1_nvenc":
        return (
            f"the GPU could not start an AV1 encode ({detail}). AV1 "
            "hardware encoding needs an NVIDIA RTX 40-series GPU or newer."
        )
    return f"the GPU could not start a {name} encode ({detail})."


def _write_sample(folder: Path) -> Path:
    """A 1 s 1080p test pattern for the real-options trials."""
    sample = folder / SAMPLE_NAME
    run_tool(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", SAMPLE_INPUT, "-c:v", "ffv1", str(sample),
        ],
        timeout=TRIAL_TIMEOUT_SEC,
    )
    return sample


def _failed_real_trials(config: PipelineConfig) -> list[str]:
    """Encode the sample with the exact options a clip would use.

    The command comes from the encoder's own builder, so the preset, the
    tune, multipass, lookahead, 10-bit AV1 and the H.264 profile and
    level are all exercised as configured.
    """
    variants = {
        (v.codec, v.resolution): v
        for v in config.build_variants("landscape")
    }
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ftea-trial-") as name:
        folder = Path(name)
        chunk = SceneChunk(
            index=0, path=_write_sample(folder), start_time=0.0,
            duration=1.0,
        )
        for key in REAL_TRIALS:
            variant = variants[key]
            cmd = encode._build_encode_cmd(
                chunk=chunk, variant=variant, cq=variant.start_cq,
                film_grain=0, keyframe_interval=SAMPLE_KEYFRAME_INTERVAL,
                config=config,
                output_path=folder / f"trial_{variant.codec}.mp4",
            )
            result = run_tool(cmd, timeout=TRIAL_TIMEOUT_SEC, check=False)
            if result.returncode != 0:
                problems.append(
                    _real_trial_problem(cmd, result.stderr, config),
                )
    return problems


def _rejected_option(stderr: Optional[str]) -> Optional[str]:
    """The option ffmpeg says it does not know, if it says so."""
    for pattern in OPTION_ERRORS:
        found = pattern.search(stderr or "")
        if found:
            return found.group(1)
    return None


def _real_trial_problem(
    cmd: list[str], stderr: Optional[str], config: PipelineConfig,
) -> str:
    encoder = cmd[cmd.index("-c:v") + 1]
    option = _rejected_option(stderr)
    setting = SETTING_FOR_OPTION.get(option or "")
    if setting is not None:
        value = getattr(config, setting.lower())
        fallback = FALLBACK_VALUE.get(setting)
        advice = (
            f"set {setting}={fallback}" if fallback
            else f"change {setting} in config.env"
        )
        return (
            f"this FFmpeg does not know {setting}={value}; install a newer "
            f"build or {advice}."
        )
    if option is not None:
        return (
            f"this FFmpeg does not accept the -{option} option ftea uses "
            f"with {encoder}; install a newer FFmpeg build (see {GUIDE})."
        )
    return (
        f"{encoder} starts, but not with ftea's encode settings "
        f"({stderr_tail(stderr)}). Install a newer FFmpeg build and "
        f"driver (see {GUIDE})."
    )
