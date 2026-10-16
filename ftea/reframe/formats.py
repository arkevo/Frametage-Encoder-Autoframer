"""The encoders, containers, presets and devices the reframe stage uses.

settings.py checks REFRAME_CODEC, REFRAME_CONTAINER, REFRAME_PRESET and
REFRAME_DEVICE against these when config.env loads, so a bad value
stops a clip before the shot, title and detection passes rather than
after them. render.py checks the codec and container again just before
ffmpeg runs.

Standard library only, and it imports nothing else from the package, so
settings.py can use it without loading the render stage.
"""

import re
from typing import Dict, Optional, Tuple

NVENC_CODECS = frozenset({"hevc_nvenc", "h264_nvenc", "av1_nvenc"})
LOSSLESS_CODECS = frozenset({"ffv1"})

# Codec -> the containers it may be muxed into. ffmpeg 8.1 will
# in fact mux ffv1 into mp4, but almost nothing else reads it, so the
# pairing is rejected with a message naming the fix rather than silently
# producing a file the operator cannot open. Narrowed to the pairings
# the render actually produces and tests -- mov/nut/avi are not promised
# until something exercises them.
CONTAINERS: Dict[str, Tuple[str, ...]] = {
    "hevc_nvenc": ("mp4", "mkv"),
    "h264_nvenc": ("mp4", "mkv"),
    "av1_nvenc": ("mp4", "mkv"),
    "ffv1": ("mkv",),
}

# Every value ffmpeg's NVENC encoders accept for -preset: p1 (fastest)
# to p7 (best), plus the older names ffmpeg still maps onto them.
NVENC_PRESETS = frozenset({
    "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "default", "slow", "medium", "fast", "hp", "hq", "bd",
    "ll", "llhq", "llhp", "lossless", "losslesshp",
})

# What torch.device accepts for this stage: the CPU, the default GPU,
# or a numbered GPU.
_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")


def output_format_problem(codec: str, container: str) -> Optional[str]:
    """Why a codec/container pair cannot be rendered, or None if it can.

    Both values are expected lower-cased, with no leading dot.
    """
    allowed = CONTAINERS.get(codec)
    if allowed is None:
        supported = ", ".join(sorted(CONTAINERS))
        return (
            f"unsupported codec '{codec}' for REFRAME_CODEC; "
            f"supported: {supported}"
        )
    if container in allowed:
        return None
    remedy = ""
    if codec in LOSSLESS_CODECS:
        remedy = " -- set REFRAME_CONTAINER=mkv"
    return (
        f"{codec} cannot be muxed into {container}; allowed "
        f"containers for REFRAME_CONTAINER: {', '.join(allowed)}{remedy}"
    )


def preset_problem(codec: str, preset: str) -> Optional[str]:
    """Why a preset cannot be used with a codec, or None if it can.

    Only NVENC codecs take a preset; FFV1 never receives one.
    """
    if codec not in NVENC_CODECS or preset in NVENC_PRESETS:
        return None
    return (
        f"REFRAME_PRESET must be p1 to p7 (or another NVENC preset name) "
        f"for {codec}, got '{preset}'"
    )


def is_device(value: str) -> bool:
    """True for cpu, cuda, or cuda:N."""
    return bool(_DEVICE.match(value))
