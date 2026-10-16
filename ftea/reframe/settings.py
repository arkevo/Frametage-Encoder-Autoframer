"""Typed access to the REFRAME_* tuning keys in config.env.

This is the stage-side half of the REFRAME_* namespace. The encoder-side
half (REFRAME, REFRAME_PYTHON, REFRAME_TIMEOUT, REFRAME_KEEP_CROPS,
REFRAME_CROPS_DIR) lives in ftea/encode/config.py and is deliberately
disjoint from this one: the encoder runs this stage as a subprocess and
never imports this module, so the two halves cannot disagree because
they never parse the same key. The encoder does keep a list of this
module's keys (REFRAME_STAGE_KEYS) so it can report a misspelt one;
this module ignores keys it does not read. Both halves read a switch
the same way: true/yes/on/1 or false/no/off/0, anything else an error.

Path-valued keys (REFRAME_MODELS_DIR and the three model-file keys)
follow the repository path contract in ftea.paths: blank means the
default, `~` expands, and a relative value is anchored at the data root,
never the working directory. REFRAME_FFMPEG is not resolved, because a
bare command name must still be found on PATH.

Standard library only -- this module must stay importable without torch
so its tests run on any machine.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .. import paths
from .errors import ReframeError
from .formats import is_device, output_format_problem, preset_problem

# Only keys with this prefix may be overridden from the environment, for
# the same bounded-blast-radius reason as ftea/encode/config.py.
REFRAME_ENV_PREFIX = "REFRAME"


@dataclass(frozen=True)
class ReframeSettings:
    """Every tuning knob the reframe stages read, already typed."""

    models_dir: Path
    person_model: str
    face_model: str
    stride: int
    batch_size: int
    device: str
    face_weight: float
    hysteresis_margin: float
    hysteresis_sec: float
    dead_zone: float
    smoothing_hz: float
    max_pan_frac_per_sec: float
    aspect_w: int
    aspect_h: int
    codec: str
    cq: int
    preset: str
    container: str
    # --- detection (detect.py). Replaces the single
    # REFRAME_MIN_CONFIDENCE, which could not work: person scores reach
    # 0.9856 while face scores never exceed 0.7462 on our test footage,
    # so one floor cannot serve both, and 0.8 is what separates a person
    # from a curtain.
    min_confidence_person: float
    min_confidence_face: float
    # 0 == native display resolution. Measure recall at the new scale
    # before changing this default: no reliable recall-vs-scale numbers
    # exist yet.
    detect_long_side: int
    # --- subject selection (subject.py). Appended at the END of
    # the field list so every existing positional construction stays
    # valid. All dimensionless fractions, never pixels.
    centrality_weight: float
    confidence_weight: float
    persistence_weight: float
    iou_match_min: float
    track_gap_sec: float
    # --- camera path (camera.py). How close to centred the
    # subject must get before the latched camera disengages, as a
    # fraction of the dead-zone half-width.
    dead_zone_release: float
    # --- rendering (render.py). Absolute path to the
    # ffmpeg binary; empty resolves "ffmpeg" from PATH. ffprobe is
    # derived from this by name (render.ffprobe_for), so there is
    # deliberately no REFRAME_FFPROBE key. The render and the debug
    # video use it; shot detection (shots.py) and the input gate
    # (probe.py) always run ffmpeg and ffprobe from PATH, the same
    # binaries the encoder's own scene detection runs.
    ffmpeg: str
    # --- shot detection (shots.py). pipeline.run_stages cannot thread a
    # configured shot threshold it cannot read, so the two shot keys
    # live here too. Defaults mirror shots.DEFAULT_SCENE_THRESHOLD /
    # DEFAULT_MIN_SHOT_SEC.
    scene_threshold: float
    min_shot_sec: float
    # --- title detection (text.py). Title-card detection and the
    # letterbox treatment. Appended at the END of the field list so
    # every existing positional construction stays valid. All
    # dimensionless fractions or counts, never pixels -- except
    # title_long_side, which is the detector's input long side in px
    # and follows REFRAME_DETECT_LONG_SIDE's precedent.
    title_enabled: bool
    title_model: str
    title_long_side: int
    title_probe_frames: int
    title_min_height: float
    title_min_lines: int
    title_centre_tol: float
    title_max_skew: float
    title_edge_margin: float


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a config.env file, matching encode/config.py's grammar exactly.

    A missing file yields an empty mapping rather than an error: every
    REFRAME_* key has a usable default, and the reframe stage must not
    fail merely because it was invoked outside a configured checkout.
    A byte-order mark (Windows Notepad adds one) is dropped, as the
    encoder drops it. Any other encoding raises ReframeError, as the
    encoder refuses it.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ReframeError(
            f"{path} is not UTF-8 text. Save it as UTF-8 and try again."
        ) from None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _overlay(env: Optional[Mapping[str, str]]) -> dict[str, str]:
    """Extract REFRAME_* overrides from an environment mapping."""
    if not env:
        return {}
    return {
        key: value
        for key, value in env.items()
        if key.startswith(REFRAME_ENV_PREFIX) and str(value).strip()
    }


def _get_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ReframeError(f"{key} must be an integer, got '{raw}'") from None


def _get_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ReframeError(f"{key} must be a number, got '{raw}'") from None


def _get_str(env: dict[str, str], key: str, default: str) -> str:
    """A text value; blank means the default, as for the number keys."""
    raw = env.get(key, "").strip()
    return raw or default


# The same words ftea/encode/config.py accepts for REFRAME and
# REFRAME_KEEP_CROPS, so every switch in config.env reads alike.
TRUE_WORDS = ("true", "yes", "on", "1")
FALSE_WORDS = ("false", "no", "off", "0")


def _get_bool(env: dict[str, str], key: str, default: bool) -> bool:
    """A true/false switch. Anything else is refused, not read as false.

    REFRAME_TITLE_ENABLED defaults to true, so reading a typo such as
    'ture' as false would quietly turn title detection off.
    """
    raw = env.get(key, "").strip()
    value = raw.lower()
    if not value:
        return default
    if value in TRUE_WORDS:
        return True
    if value in FALSE_WORDS:
        return False
    raise ReframeError(
        f"{key} must be true or false (or yes/no, on/off, 1/0), got '{raw}'"
    )


def _resolve(values: Mapping[str, str], key: str) -> Optional[Path]:
    """paths.resolve_setting_path, with the key named in its error.

    On Windows a value such as Git Bash's /c/... is refused rather than
    silently read as a different folder; the message says which key.
    """
    try:
        return paths.resolve_setting_path(values.get(key, ""))
    except ValueError as exc:
        raise ReframeError(f"{key}: {exc}") from None


def _path_value(values: Mapping[str, str], key: str) -> str:
    """A path-valued model key, resolved by the repository path contract.

    Blank stays "" so the caller's default file applies. Otherwise
    `~` expands and a relative value is anchored at the data root, never
    the working directory, and the result is an absolute path string.
    """
    resolved = _resolve(values, key)
    return "" if resolved is None else str(resolved)


def _models_dir(values: Mapping[str, str]) -> Path:
    """REFRAME_MODELS_DIR, or <data root>/models when it is blank."""
    resolved = _resolve(values, "REFRAME_MODELS_DIR")
    return paths.models_dir() if resolved is None else resolved


def _require(condition: bool, key: str, rule: str, value: object) -> None:
    """Raise a ReframeError naming the key, its rule, and the bad value."""
    if not condition:
        raise ReframeError(f"{key} must be {rule}, got {value!r}")


def _check_render_format(codec: str, container: str, preset: str) -> None:
    """The render's codec, container and preset checks, run at load.

    render.py makes the codec/container check again before ffmpeg runs;
    making it here too means a bad value fails before detection.
    """
    problem = (
        output_format_problem(codec.lower(), container.lower())
        or preset_problem(codec.lower(), preset)
    )
    if problem is not None:
        raise ReframeError(problem)


def load_settings(
    config_path: Path, *, env: Optional[Mapping[str, str]] = None,
) -> ReframeSettings:
    """Read the REFRAME_* stage keys from config.env, then from `env`.

    Pass `env` (normally os.environ) to let REFRAME_* keys — and only
    those — override the file. Raises ReframeError with the offending key
    name for any value outside its documented range, including a codec,
    container, preset or device the stage cannot use, so a bad value
    stops the clip before any detection work.
    """
    values = _parse_env_file(Path(config_path))
    values.update(_overlay(env))

    stride = _get_int(values, "REFRAME_STRIDE", 3)
    batch_size = _get_int(values, "REFRAME_BATCH_SIZE", 16)
    face_weight = _get_float(values, "REFRAME_FACE_WEIGHT", 2.0)
    hysteresis_margin = _get_float(values, "REFRAME_HYSTERESIS_MARGIN", 0.2)
    hysteresis_sec = _get_float(values, "REFRAME_HYSTERESIS_SEC", 1.0)
    dead_zone = _get_float(values, "REFRAME_DEAD_ZONE", 0.9)
    smoothing_hz = _get_float(values, "REFRAME_SMOOTHING_HZ", 0.6)
    max_pan = _get_float(values, "REFRAME_MAX_PAN_FRAC_PER_SEC", 0.20)
    aspect_w = _get_int(values, "REFRAME_ASPECT_W", 9)
    aspect_h = _get_int(values, "REFRAME_ASPECT_H", 16)
    cq = _get_int(values, "REFRAME_CQ", 10)
    codec = _get_str(values, "REFRAME_CODEC", "hevc_nvenc")
    container = _get_str(values, "REFRAME_CONTAINER", "mp4")
    preset = _get_str(values, "REFRAME_PRESET", "p7").lower()
    device = _get_str(values, "REFRAME_DEVICE", "cuda").lower()

    if "REFRAME_MIN_CONFIDENCE" in values:
        print(
            "  reframe: REFRAME_MIN_CONFIDENCE was removed; use "
            "REFRAME_MIN_CONFIDENCE_PERSON (0.8) and "
            "REFRAME_MIN_CONFIDENCE_FACE (0.35)",
            file=sys.stderr,
        )

    person_floor = _get_float(
        values, "REFRAME_MIN_CONFIDENCE_PERSON", 0.8)
    face_floor = _get_float(values, "REFRAME_MIN_CONFIDENCE_FACE", 0.35)
    detect_long_side = _get_int(values, "REFRAME_DETECT_LONG_SIDE", 0)

    centrality_weight = _get_float(
        values, "REFRAME_CENTRALITY_WEIGHT", 0.5)
    confidence_weight = _get_float(
        values, "REFRAME_CONFIDENCE_WEIGHT", 0.3)
    persistence_weight = _get_float(
        values, "REFRAME_PERSISTENCE_WEIGHT", 0.5)
    iou_match_min = _get_float(values, "REFRAME_IOU_MATCH_MIN", 0.3)
    track_gap_sec = _get_float(values, "REFRAME_TRACK_GAP_SEC", 0.25)
    dead_zone_release = _get_float(
        values, "REFRAME_DEAD_ZONE_RELEASE", 0.6)
    scene_threshold = _get_float(values, "REFRAME_SCENE_THRESHOLD", 0.3)
    min_shot_sec = _get_float(values, "REFRAME_MIN_SHOT_SEC", 0.4)

    title_enabled = _get_bool(values, "REFRAME_TITLE_ENABLED", True)
    title_long_side = _get_int(values, "REFRAME_TITLE_LONG_SIDE", 640)
    title_probe_frames = _get_int(
        values, "REFRAME_TITLE_PROBE_FRAMES", 5)
    title_min_height = _get_float(
        values, "REFRAME_TITLE_MIN_HEIGHT", 0.04)
    title_min_lines = _get_int(values, "REFRAME_TITLE_MIN_LINES", 2)
    title_centre_tol = _get_float(
        values, "REFRAME_TITLE_CENTRE_TOL", 0.05)
    title_max_skew = _get_float(values, "REFRAME_TITLE_MAX_SKEW", 10.0)
    title_edge_margin = _get_float(
        values, "REFRAME_TITLE_EDGE_MARGIN", 0.01)

    _require(stride >= 1, "REFRAME_STRIDE", ">= 1", stride)
    _require(batch_size >= 1, "REFRAME_BATCH_SIZE", ">= 1", batch_size)
    _require(face_weight >= 1.0, "REFRAME_FACE_WEIGHT", ">= 1.0", face_weight)
    _require(
        0.0 <= hysteresis_margin < 1.0,
        "REFRAME_HYSTERESIS_MARGIN", "in [0, 1)", hysteresis_margin,
    )
    _require(
        hysteresis_sec >= 0.0,
        "REFRAME_HYSTERESIS_SEC", ">= 0", hysteresis_sec,
    )
    _require(
        0.0 < dead_zone < 1.0,
        "REFRAME_DEAD_ZONE", "between 0 and 1 exclusive", dead_zone,
    )
    _require(smoothing_hz > 0.0, "REFRAME_SMOOTHING_HZ", "> 0", smoothing_hz)
    _require(
        0.0 < max_pan <= 1.0,
        "REFRAME_MAX_PAN_FRAC_PER_SEC", "in (0, 1] (fraction of frame width)",
        max_pan,
    )
    _require(aspect_w >= 1, "REFRAME_ASPECT_W", ">= 1", aspect_w)
    _require(aspect_h >= 1, "REFRAME_ASPECT_H", ">= 1", aspect_h)
    _require(0 <= cq <= 51, "REFRAME_CQ", "in [0, 51]", cq)
    _require(
        not container.startswith("."),
        "REFRAME_CONTAINER", "an extension with no leading dot", container,
    )
    _check_render_format(codec, container, preset)
    _require(
        is_device(device),
        "REFRAME_DEVICE", "cpu, cuda or cuda:N", device,
    )
    _require(
        0.0 < person_floor < 1.0,
        "REFRAME_MIN_CONFIDENCE_PERSON", "between 0 and 1 exclusive",
        person_floor,
    )
    _require(
        0.0 < face_floor < 1.0,
        "REFRAME_MIN_CONFIDENCE_FACE", "between 0 and 1 exclusive",
        face_floor,
    )
    _require(
        detect_long_side == 0 or detect_long_side >= 320,
        "REFRAME_DETECT_LONG_SIDE", "0 (native) or >= 320",
        detect_long_side,
    )
    _require(
        0.0 <= centrality_weight < 1.0,
        "REFRAME_CENTRALITY_WEIGHT", "in [0, 1)", centrality_weight,
    )
    _require(
        0.0 <= confidence_weight < 1.0,
        "REFRAME_CONFIDENCE_WEIGHT", "in [0, 1)", confidence_weight,
    )
    _require(
        0.0 <= persistence_weight < 1.0,
        "REFRAME_PERSISTENCE_WEIGHT", "in [0, 1)", persistence_weight,
    )
    _require(
        0.0 < iou_match_min <= 1.0,
        "REFRAME_IOU_MATCH_MIN", "in (0, 1]", iou_match_min,
    )
    _require(
        track_gap_sec >= 0.0,
        "REFRAME_TRACK_GAP_SEC", ">= 0 (seconds)", track_gap_sec,
    )
    _require(
        0.0 <= dead_zone_release < 1.0,
        "REFRAME_DEAD_ZONE_RELEASE", "in [0, 1)", dead_zone_release,
    )
    _require(
        0.0 < scene_threshold <= 1.0,
        "REFRAME_SCENE_THRESHOLD", "in (0, 1]", scene_threshold,
    )
    _require(
        min_shot_sec >= 0.0,
        "REFRAME_MIN_SHOT_SEC", ">= 0 (seconds)", min_shot_sec,
    )
    _require(
        title_long_side >= 320,
        "REFRAME_TITLE_LONG_SIDE", ">= 320", title_long_side,
    )
    _require(
        title_probe_frames >= 1,
        "REFRAME_TITLE_PROBE_FRAMES", ">= 1", title_probe_frames,
    )
    _require(
        0.0 < title_min_height < 1.0,
        "REFRAME_TITLE_MIN_HEIGHT", "in (0, 1) (fraction of frame "
        "height)", title_min_height,
    )
    _require(
        title_min_lines >= 2,
        "REFRAME_TITLE_MIN_LINES", ">= 2", title_min_lines,
    )
    _require(
        0.0 < title_centre_tol < 1.0,
        "REFRAME_TITLE_CENTRE_TOL", "in (0, 1) (fraction of frame "
        "width)", title_centre_tol,
    )
    _require(
        0.0 <= title_max_skew <= 45.0,
        "REFRAME_TITLE_MAX_SKEW", "in [0, 45] (degrees)",
        title_max_skew,
    )
    # 0.25 each side would demand half the frame be clear space, which
    # makes every title letterbox. The cap is a guard against a typo,
    # not a tuning range.
    _require(
        0.0 <= title_edge_margin <= 0.25,
        "REFRAME_TITLE_EDGE_MARGIN", "in [0, 0.25] (fraction of frame "
        "width)", title_edge_margin,
    )

    return ReframeSettings(
        models_dir=_models_dir(values),
        person_model=_path_value(values, "REFRAME_PERSON_MODEL"),
        face_model=_path_value(values, "REFRAME_FACE_MODEL"),
        stride=stride,
        batch_size=batch_size,
        device=device,
        face_weight=face_weight,
        hysteresis_margin=hysteresis_margin,
        hysteresis_sec=hysteresis_sec,
        dead_zone=dead_zone,
        smoothing_hz=smoothing_hz,
        max_pan_frac_per_sec=max_pan,
        aspect_w=aspect_w,
        aspect_h=aspect_h,
        codec=codec,
        cq=cq,
        preset=preset,
        container=container,
        min_confidence_person=person_floor,
        min_confidence_face=face_floor,
        detect_long_side=detect_long_side,
        centrality_weight=centrality_weight,
        confidence_weight=confidence_weight,
        persistence_weight=persistence_weight,
        iou_match_min=iou_match_min,
        track_gap_sec=track_gap_sec,
        dead_zone_release=dead_zone_release,
        ffmpeg=values.get("REFRAME_FFMPEG", "").strip(),
        scene_threshold=scene_threshold,
        min_shot_sec=min_shot_sec,
        title_enabled=title_enabled,
        title_model=_path_value(values, "REFRAME_TITLE_MODEL"),
        title_long_side=title_long_side,
        title_probe_frames=title_probe_frames,
        title_min_height=title_min_height,
        title_min_lines=title_min_lines,
        title_centre_tol=title_centre_tol,
        title_max_skew=title_max_skew,
        title_edge_margin=title_edge_margin,
    )
