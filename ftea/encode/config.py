"""Encoder settings: config.env parsing, validation and the variant ladder.

Parses a config.env file (KEY=value lines) into a typed PipelineConfig
and builds the six encode variants (AV1 and H.264 at 1080p, 720p and
360p) with orientation-aware dimensions.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from .tools import run_tool

BATCH_ON_ERROR_CHOICES = ("skip", "stop")
# The thumbnail formats the bundled player and ftea preview look for.
THUMB_FORMATS = ("webp", "jpg")
ORIENTATIONS = ("portrait", "landscape")

# Only keys with this prefix may be overridden from the environment, so
# a stray exported variable such as FILM_GRAIN can never change how the
# encoder behaves. That covers every REFRAME* key, the stage's included.
REFRAME_ENV_PREFIX = "REFRAME"

# The keys the reframe stage reads (ftea/reframe/settings.py), which
# checks their values. unknown_keys is the one place that reports a key
# nobody reads, the stage's misspellings included; the stage itself
# ignores keys it does not know. The encoder never imports the stage,
# so the list lives here, and a test keeps it equal to the keys the
# stage's code reads.
REFRAME_STAGE_KEYS = (
    "REFRAME_MODELS_DIR", "REFRAME_PERSON_MODEL", "REFRAME_FACE_MODEL",
    "REFRAME_STRIDE", "REFRAME_BATCH_SIZE", "REFRAME_DEVICE",
    "REFRAME_MIN_CONFIDENCE_PERSON", "REFRAME_MIN_CONFIDENCE_FACE",
    "REFRAME_DETECT_LONG_SIDE",
    "REFRAME_FACE_WEIGHT", "REFRAME_HYSTERESIS_MARGIN",
    "REFRAME_HYSTERESIS_SEC", "REFRAME_CENTRALITY_WEIGHT",
    "REFRAME_CONFIDENCE_WEIGHT", "REFRAME_PERSISTENCE_WEIGHT",
    "REFRAME_IOU_MATCH_MIN", "REFRAME_TRACK_GAP_SEC",
    "REFRAME_DEAD_ZONE", "REFRAME_DEAD_ZONE_RELEASE",
    "REFRAME_SMOOTHING_HZ", "REFRAME_MAX_PAN_FRAC_PER_SEC",
    "REFRAME_ASPECT_W", "REFRAME_ASPECT_H", "REFRAME_CODEC", "REFRAME_CQ",
    "REFRAME_PRESET", "REFRAME_CONTAINER", "REFRAME_FFMPEG",
    "REFRAME_SCENE_THRESHOLD", "REFRAME_MIN_SHOT_SEC",
    "REFRAME_TITLE_ENABLED", "REFRAME_TITLE_MODEL",
    "REFRAME_TITLE_LONG_SIDE", "REFRAME_TITLE_PROBE_FRAMES",
    "REFRAME_TITLE_MIN_HEIGHT", "REFRAME_TITLE_MIN_LINES",
    "REFRAME_TITLE_CENTRE_TOL", "REFRAME_TITLE_MAX_SKEW",
    "REFRAME_TITLE_EDGE_MARGIN",
)

# Every config.env key the encoder reads. A test checks this list
# against the keys load_config actually reads.
ENCODER_KEYS = (
    "NVENC_PRESET", "NVENC_TUNE", "NVENC_RC", "NVENC_MULTIPASS",
    "NVENC_RC_LOOKAHEAD", "CQ_BRACKET",
    "AV1_START_CQ_1080", "AV1_START_CQ_720", "AV1_START_CQ_360",
    "H264_START_CQ_1080", "H264_START_CQ_720", "H264_START_CQ_360",
    "VMAF_TARGET_1080", "VMAF_TARGET_720", "VMAF_TARGET_360",
    "VMAF_FLOOR", "VMAF_WORKERS",
    "AV1_MAXRATE_1080", "AV1_MAXRATE_720", "AV1_MAXRATE_360",
    "H264_MAXRATE_1080", "H264_MAXRATE_720", "H264_MAXRATE_360",
    "BUFSIZE_MULTIPLIER", "AV1_BIT_DEPTH", "H264_BIT_DEPTH",
    "FILM_GRAIN", "FILM_GRAIN_BW", "BW_SATURATION_THRESHOLD",
    "SCENE_THRESHOLD", "MAX_SCENE_SEC", "MIN_SCENE_SEC",
    "HLS_SEGMENT_SEC", "THUMB_FORMAT", "AUDIO_BITRATE",
    "AUDIO_SAMPLE_RATE", "BATCH_ON_ERROR", "CLIP_TIMEOUT",
    "CLIP_TIMEOUT_PER_SEC", "MAX_CLIP_SEC",
    "MAX_KEPT_FAILURES", "RETRY_MAX", "RETRY_VMAF_RELAXATION",
    "RETRY_TIMEOUT_MULTIPLIER", "REFRAME", "REFRAME_PYTHON",
    "REFRAME_TIMEOUT", "REFRAME_KEEP_CROPS", "REFRAME_CROPS_DIR",
)


@functools.cache
def nvenc_supports_film_grain() -> bool:
    """Whether this ffmpeg's av1_nvenc accepts the -film_grain option."""
    result = run_tool(
        ["ffmpeg", "-hide_banner", "-h", "encoder=av1_nvenc"], check=False,
    )
    return "film_grain" in result.stdout


class ConfigValidationError(Exception):
    """Raised when config.env holds an invalid value."""


@dataclass(frozen=True)
class Variant:
    """A single encode variant (codec x resolution)."""

    codec: str
    resolution: str
    width: int
    height: int
    start_cq: int
    maxrate: str
    bufsize: str
    vmaf_target: int
    bit_depth: int


@dataclass
class PipelineConfig:
    """Every encoder setting, with the defaults config.env.example ships."""

    # NVENC
    nvenc_preset: str = "p7"
    nvenc_tune: str = "uhq"
    nvenc_rc: str = "vbr"
    nvenc_multipass: str = "fullres"
    nvenc_rc_lookahead: int = 32

    # Probe
    cq_bracket: int = 4

    # Starting CQs per resolution
    av1_start_cq: dict = field(
        default_factory=lambda: {"1080": 30, "720": 33, "360": 38}
    )
    h264_start_cq: dict = field(
        default_factory=lambda: {"1080": 25, "720": 28, "360": 32}
    )

    # VMAF
    vmaf_targets: dict = field(
        default_factory=lambda: {"1080": 93, "720": 93, "360": 88}
    )
    vmaf_floor: int = 91
    vmaf_workers: int = 8

    # Bitrate caps
    av1_maxrates: dict = field(
        default_factory=lambda: {"1080": "3000k", "720": "1500k", "360": "400k"}
    )
    h264_maxrates: dict = field(
        default_factory=lambda: {"1080": "4000k", "720": "2500k", "360": "800k"}
    )
    bufsize_multiplier: int = 2

    # Bit depth
    av1_bit_depth: int = 10
    h264_bit_depth: int = 8

    # Film grain
    film_grain: int = 0
    film_grain_bw: int = 8
    bw_saturation_threshold: int = 10

    # Scene detection
    scene_threshold: float = 0.3
    max_scene_sec: float = 10.0
    min_scene_sec: float = 1.0

    # HLS
    hls_segment_sec: int = 2

    # Thumbnail
    thumb_format: str = "webp"

    # Audio
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 44100

    # Batch
    batch_on_error: str = "skip"
    # A clip's encode limit: clip_timeout + clip_timeout_per_sec x its
    # length in seconds.
    clip_timeout: int = 300
    clip_timeout_per_sec: int = 30
    # Longer videos are skipped at pre-scan; 0 means no limit.
    max_clip_sec: int = 600
    max_kept_failures: int = 5

    # Retry
    retry_max: int = 2
    retry_vmaf_relaxation: int = 2
    retry_timeout_multiplier: int = 2

    # Auto-reframe stage (its own virtualenv; on by default)
    reframe_enabled: bool = True
    reframe_python: str = ""
    reframe_timeout: int = 600
    reframe_keep_crops: bool = True
    reframe_crops_dir: str = ""

    def build_variants(self, orientation: str) -> list[Variant]:
        """Build 6 variants (2 codecs x 3 resolutions) for the orientation.

        `orientation` is "portrait" or "landscape"; anything else raises
        ValueError rather than quietly getting landscape frames.
        """
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"unknown orientation {orientation!r}: expected "
                f"{' or '.join(ORIENTATIONS)}"
            )
        variants: list[Variant] = []
        for codec in ("av1", "h264"):
            for res in ("1080", "720", "360"):
                w, h = (
                    _portrait_dims(res)
                    if orientation == "portrait"
                    else _landscape_dims(res)
                )
                start_cq_map = (
                    self.av1_start_cq if codec == "av1" else self.h264_start_cq
                )
                maxrate_map = (
                    self.av1_maxrates if codec == "av1" else self.h264_maxrates
                )
                bit_depth = (
                    self.av1_bit_depth if codec == "av1"
                    else self.h264_bit_depth
                )
                maxrate = maxrate_map[res]
                _check_maxrate(_maxrate_key(codec, res), maxrate)
                bufsize = f"{int(maxrate[:-1]) * self.bufsize_multiplier}k"

                variants.append(
                    Variant(
                        codec=codec,
                        resolution=res,
                        width=w,
                        height=h,
                        start_cq=start_cq_map[res],
                        maxrate=maxrate,
                        bufsize=bufsize,
                        vmaf_target=self.vmaf_targets[res],
                        bit_depth=bit_depth,
                    )
                )
        return variants


def _maxrate_key(codec: str, res: str) -> str:
    """The config.env key of one bitrate cap, such as H264_MAXRATE_360."""
    return f"{codec.upper()}_MAXRATE_{res}"


def _check_maxrate(key: str, maxrate: str) -> None:
    """A cap is a whole number of kbit/s above 0, written as <number>k."""
    digits = maxrate[:-1]
    if not (maxrate.endswith("k") and digits.isdigit() and int(digits) > 0):
        raise ConfigValidationError(
            f"{key} must be '<number>k' above 0, such as 800k, "
            f"got '{maxrate}'"
        )


def _portrait_dims(res: str) -> tuple[int, int]:
    return {"1080": (1080, 1920), "720": (720, 1280), "360": (360, 640)}[res]


def _landscape_dims(res: str) -> tuple[int, int]:
    return {"1080": (1920, 1080), "720": (1280, 720), "360": (640, 360)}[res]


def _parse_env(path: Path, *, missing_ok: bool = False) -> dict[str, str]:
    """Parse a config.env file into a flat key-value dict.

    Read as UTF-8. A byte-order mark (Windows Notepad adds one) is
    dropped, and Windows line endings are fine. Any other encoding
    raises ConfigValidationError, so every reader of config.env (the
    install check included) can say so instead of crashing.
    """
    if missing_ok and not path.is_file():
        return {}
    result: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ConfigValidationError(
            f"{path} is not UTF-8 text. Save it as UTF-8 and try again."
        ) from None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _get_str(values: Mapping[str, str], key: str, default: str) -> str:
    return values.get(key, "").strip() or default


def _get_int(values: Mapping[str, str], key: str, default: int) -> int:
    value = values.get(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigValidationError(
            f"{key} must be an integer, got '{value}'"
        ) from None


def _get_float(values: Mapping[str, str], key: str, default: float) -> float:
    value = values.get(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigValidationError(
            f"{key} must be a number, got '{value}'"
        ) from None


TRUE_WORDS = ("true", "yes", "on", "1")
FALSE_WORDS = ("false", "no", "off", "0")


def _get_bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    """A true/false switch. Anything else is refused, not read as false:
    REFRAME=on or a typo must never quietly turn reframing off."""
    raw = values.get(key, "").strip()
    value = raw.lower()
    if not value:
        return default
    if value in TRUE_WORDS:
        return True
    if value in FALSE_WORDS:
        return False
    raise ConfigValidationError(
        f"{key} must be true or false (or yes/no, on/off, 1/0), got '{raw}'"
    )


def _reframe_env_overlay(env: Optional[Mapping[str, str]]) -> dict[str, str]:
    """The REFRAME* overrides in an environment mapping.

    Empty values are dropped, so an exported-but-empty variable does not
    silently defeat a real config.env setting.
    """
    if not env:
        return {}
    return {
        key: value
        for key, value in env.items()
        if key.startswith(REFRAME_ENV_PREFIX) and str(value).strip()
    }


def unknown_keys(config_path: Path) -> list[str]:
    """Settings in config.env that nothing reads, such as a typo.

    Known means read by the encoder (ENCODER_KEYS) or by the reframe
    stage (REFRAME_STAGE_KEYS). A misspelt REFRAME_ key is reported like
    any other: nothing else would notice it.
    """
    known = set(ENCODER_KEYS) | set(REFRAME_STAGE_KEYS)
    return sorted(
        key for key in _parse_env(config_path, missing_ok=True)
        if key not in known
    )


def load_config(
    config_path: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
    missing_ok: bool = False,
) -> PipelineConfig:
    """Load and validate the encoder settings from a config.env file.

    Pass env (normally os.environ) to let REFRAME* keys, and only those,
    be overridden from the environment. With missing_ok, an absent file
    means every setting takes its default.

    Raises FileNotFoundError when the file is absent and missing_ok is
    false, and ConfigValidationError when a value is out of range.
    """
    values = _parse_env(config_path, missing_ok=missing_ok)
    values.update(_reframe_env_overlay(env))
    d = PipelineConfig()

    config = PipelineConfig(
        nvenc_preset=_get_str(values, "NVENC_PRESET", d.nvenc_preset),
        nvenc_tune=_get_str(values, "NVENC_TUNE", d.nvenc_tune),
        nvenc_rc=_get_str(values, "NVENC_RC", d.nvenc_rc),
        nvenc_multipass=_get_str(
            values, "NVENC_MULTIPASS", d.nvenc_multipass,
        ),
        nvenc_rc_lookahead=_get_int(
            values, "NVENC_RC_LOOKAHEAD", d.nvenc_rc_lookahead,
        ),
        cq_bracket=_get_int(values, "CQ_BRACKET", d.cq_bracket),
        av1_start_cq={
            "1080": _get_int(
                values, "AV1_START_CQ_1080", d.av1_start_cq["1080"],
            ),
            "720": _get_int(values, "AV1_START_CQ_720", d.av1_start_cq["720"]),
            "360": _get_int(values, "AV1_START_CQ_360", d.av1_start_cq["360"]),
        },
        h264_start_cq={
            "1080": _get_int(
                values, "H264_START_CQ_1080", d.h264_start_cq["1080"],
            ),
            "720": _get_int(
                values, "H264_START_CQ_720", d.h264_start_cq["720"],
            ),
            "360": _get_int(
                values, "H264_START_CQ_360", d.h264_start_cq["360"],
            ),
        },
        vmaf_targets={
            "1080": _get_int(
                values, "VMAF_TARGET_1080", d.vmaf_targets["1080"],
            ),
            "720": _get_int(values, "VMAF_TARGET_720", d.vmaf_targets["720"]),
            "360": _get_int(values, "VMAF_TARGET_360", d.vmaf_targets["360"]),
        },
        vmaf_floor=_get_int(values, "VMAF_FLOOR", d.vmaf_floor),
        vmaf_workers=_get_int(values, "VMAF_WORKERS", d.vmaf_workers),
        av1_maxrates={
            "1080": _get_str(
                values, "AV1_MAXRATE_1080", d.av1_maxrates["1080"],
            ),
            "720": _get_str(values, "AV1_MAXRATE_720", d.av1_maxrates["720"]),
            "360": _get_str(values, "AV1_MAXRATE_360", d.av1_maxrates["360"]),
        },
        h264_maxrates={
            "1080": _get_str(
                values, "H264_MAXRATE_1080", d.h264_maxrates["1080"],
            ),
            "720": _get_str(
                values, "H264_MAXRATE_720", d.h264_maxrates["720"],
            ),
            "360": _get_str(
                values, "H264_MAXRATE_360", d.h264_maxrates["360"],
            ),
        },
        bufsize_multiplier=_get_int(
            values, "BUFSIZE_MULTIPLIER", d.bufsize_multiplier,
        ),
        av1_bit_depth=_get_int(values, "AV1_BIT_DEPTH", d.av1_bit_depth),
        h264_bit_depth=_get_int(values, "H264_BIT_DEPTH", d.h264_bit_depth),
        film_grain=_get_int(values, "FILM_GRAIN", d.film_grain),
        film_grain_bw=_get_int(values, "FILM_GRAIN_BW", d.film_grain_bw),
        bw_saturation_threshold=_get_int(
            values, "BW_SATURATION_THRESHOLD", d.bw_saturation_threshold,
        ),
        scene_threshold=_get_float(
            values, "SCENE_THRESHOLD", d.scene_threshold,
        ),
        max_scene_sec=_get_float(values, "MAX_SCENE_SEC", d.max_scene_sec),
        min_scene_sec=_get_float(values, "MIN_SCENE_SEC", d.min_scene_sec),
        hls_segment_sec=_get_int(
            values, "HLS_SEGMENT_SEC", d.hls_segment_sec,
        ),
        thumb_format=_get_str(
            values, "THUMB_FORMAT", d.thumb_format,
        ).lower(),
        audio_bitrate=_get_str(values, "AUDIO_BITRATE", d.audio_bitrate),
        audio_sample_rate=_get_int(
            values, "AUDIO_SAMPLE_RATE", d.audio_sample_rate,
        ),
        batch_on_error=_get_str(
            values, "BATCH_ON_ERROR", d.batch_on_error,
        ).lower(),
        clip_timeout=_get_int(values, "CLIP_TIMEOUT", d.clip_timeout),
        clip_timeout_per_sec=_get_int(
            values, "CLIP_TIMEOUT_PER_SEC", d.clip_timeout_per_sec,
        ),
        max_clip_sec=_get_int(values, "MAX_CLIP_SEC", d.max_clip_sec),
        max_kept_failures=_get_int(
            values, "MAX_KEPT_FAILURES", d.max_kept_failures,
        ),
        retry_max=_get_int(values, "RETRY_MAX", d.retry_max),
        retry_vmaf_relaxation=_get_int(
            values, "RETRY_VMAF_RELAXATION", d.retry_vmaf_relaxation,
        ),
        retry_timeout_multiplier=_get_int(
            values, "RETRY_TIMEOUT_MULTIPLIER", d.retry_timeout_multiplier,
        ),
        reframe_enabled=_get_bool(values, "REFRAME", d.reframe_enabled),
        reframe_python=_get_str(values, "REFRAME_PYTHON", d.reframe_python),
        reframe_timeout=_get_int(
            values, "REFRAME_TIMEOUT", d.reframe_timeout,
        ),
        reframe_keep_crops=_get_bool(
            values, "REFRAME_KEEP_CROPS", d.reframe_keep_crops,
        ),
        reframe_crops_dir=_get_str(
            values, "REFRAME_CROPS_DIR", d.reframe_crops_dir,
        ),
    )

    _validate(config)
    return config


def load_reframe_launch_config(
    config_path: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
    missing_ok: bool = False,
) -> PipelineConfig:
    """Only the settings `ftea reframe` launches the stage with.

    REFRAME_PYTHON and REFRAME_TIMEOUT are read and checked; every other
    field keeps its default. An encoder setting `ftea reframe` never uses,
    such as a bad VMAF target, must not stop it. The stage reads its own
    REFRAME_* settings itself. Raises like load_config.
    """
    values = _parse_env(config_path, missing_ok=missing_ok)
    values.update(_reframe_env_overlay(env))
    d = PipelineConfig()
    config = replace(
        d,
        reframe_python=_get_str(values, "REFRAME_PYTHON", d.reframe_python),
        reframe_timeout=_get_int(
            values, "REFRAME_TIMEOUT", d.reframe_timeout,
        ),
    )
    _validate_reframe_timeout(config)
    return config


def _validate_reframe_timeout(config: PipelineConfig) -> None:
    if config.reframe_timeout < 30:
        raise ConfigValidationError("REFRAME_TIMEOUT must be >= 30")


def _validate(config: PipelineConfig) -> None:
    """Validate config values are within acceptable ranges.

    The retry passes are built from this config with dataclasses.replace
    and are not validated again, so every rule that keeps a retry usable
    (its time limit and its relaxed targets) is checked here.
    """
    for res, target in config.vmaf_targets.items():
        if not (50 <= target <= 100):
            raise ConfigValidationError(
                f"VMAF target for {res}p must be 50-100, got {target}"
            )
    if not (50 <= config.vmaf_floor <= 100):
        raise ConfigValidationError(
            f"VMAF floor must be 50-100, got {config.vmaf_floor}"
        )
    if config.vmaf_workers < 1:
        raise ConfigValidationError("VMAF_WORKERS must be >= 1")
    _validate_encode(config)
    _validate_scenes(config)
    _validate_batch(config)


def _validate_encode(config: PipelineConfig) -> None:
    """Probe bracket, bitrate caps and bit depths."""
    if config.cq_bracket < 1:
        # At 0 both probes are the same file, scored twice at once.
        raise ConfigValidationError(
            f"CQ_BRACKET must be >= 1, got {config.cq_bracket}"
        )
    for codec, maxrates in (
        ("av1", config.av1_maxrates), ("h264", config.h264_maxrates),
    ):
        for res, maxrate in maxrates.items():
            _check_maxrate(_maxrate_key(codec, res), maxrate)
    if config.bufsize_multiplier < 1:
        raise ConfigValidationError(
            f"BUFSIZE_MULTIPLIER must be >= 1, got {config.bufsize_multiplier}"
        )
    if config.av1_bit_depth not in (8, 10):
        raise ConfigValidationError(
            f"AV1_BIT_DEPTH must be 8 or 10, got {config.av1_bit_depth}"
        )
    if config.h264_bit_depth != 8:
        raise ConfigValidationError(
            "H264_BIT_DEPTH must be 8: the H.264 variants are encoded in "
            f"8-bit High profile, got {config.h264_bit_depth}"
        )
    if config.thumb_format not in THUMB_FORMATS:
        raise ConfigValidationError(
            "THUMB_FORMAT must be webp or jpg, the formats the bundled "
            f"player shows, got '{config.thumb_format}'"
        )


def _validate_scenes(config: PipelineConfig) -> None:
    """Scene lengths and the HLS segment length."""
    if config.max_scene_sec <= 0:
        raise ConfigValidationError(
            f"MAX_SCENE_SEC must be above 0, got {config.max_scene_sec}"
        )
    if not (0 <= config.min_scene_sec <= config.max_scene_sec):
        raise ConfigValidationError(
            "MIN_SCENE_SEC must be between 0 and MAX_SCENE_SEC "
            f"({config.max_scene_sec}), got {config.min_scene_sec}"
        )
    if config.hls_segment_sec < 1:
        raise ConfigValidationError("HLS_SEGMENT_SEC must be >= 1")
    if not (0.0 < config.scene_threshold < 1.0):
        raise ConfigValidationError("SCENE_THRESHOLD must be between 0 and 1")


def _validate_batch(config: PipelineConfig) -> None:
    """Time limits, batch behaviour and the retry passes."""
    if config.clip_timeout < 30:
        raise ConfigValidationError("CLIP_TIMEOUT must be >= 30")
    if config.clip_timeout_per_sec < 0:
        raise ConfigValidationError("CLIP_TIMEOUT_PER_SEC must be >= 0")
    if config.max_clip_sec < 0:
        raise ConfigValidationError("MAX_CLIP_SEC must be >= 0")
    _validate_reframe_timeout(config)
    if config.batch_on_error not in BATCH_ON_ERROR_CHOICES:
        raise ConfigValidationError(
            "BATCH_ON_ERROR must be skip or stop, got "
            f"'{config.batch_on_error}'"
        )
    if config.retry_max < 0:
        raise ConfigValidationError("RETRY_MAX must be >= 0")
    if config.max_kept_failures < 0:
        raise ConfigValidationError("MAX_KEPT_FAILURES must be >= 0")
    if config.retry_timeout_multiplier < 1:
        # At 0 every retry would start with no time left.
        raise ConfigValidationError(
            "RETRY_TIMEOUT_MULTIPLIER must be >= 1, got "
            f"{config.retry_timeout_multiplier}"
        )
    most = min(config.vmaf_targets.values()) - 50
    if not (0 <= config.retry_vmaf_relaxation <= most):
        raise ConfigValidationError(
            f"RETRY_VMAF_RELAXATION must be between 0 and {most}: it "
            "lowers every VMAF target on the later retries, and no target "
            f"may drop below 50, got {config.retry_vmaf_relaxation}"
        )
