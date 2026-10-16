"""crops.json + a single-pass ffmpeg dynamic crop to the 9:16 intermediate.

This module is a serializer and a subprocess driver, and nothing else. It
performs NO motion arithmetic: camera.py owns 100% of the trajectory and
of the crop geometry, and this module must never
recompute a crop rectangle, interpolate, or smooth anything.

The render is ONE ffmpeg pass whose crop filter is re-aimed every frame by
a sendcmd command file:

    [0:v:0]setpts=PTS-STARTPTS,sendcmd=f=crop.sendcmd,crop=W:H:X0:0[v]

The command file is written as the constant name `crop.sendcmd` next to
the output and ffmpeg is spawned with cwd set to that directory, so the
filtergraph carries only a bare filename. That is not tidiness: ffmpeg's
lavfi option parser treats "'" as a quoting character with NO reliable
escape (the encoder's probe hits the same problem), and the per-clip work
directory is named from the clip's own filename, which often contains
apostrophes and spaces ("It's a Test Clip part 04.mp4"). A constant,
ASCII, quote-free, space-free filename is the only value that is provably
safe.

Both artifacts are written to a `.partial` sibling and renamed only after
the rendered file has been probed and found correct, so the encoder can
never pick up a half-written file.

Every string this module produces is ASCII by construction, so it needs
no stdout reconfigure block.
"""

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .camera import (
    SC2_THRESHOLD, CropTrack, crop_changes, crop_rects, crop_track_to_json,
    dead_zone_report, rect_changes,
)
from .errors import ReframeError
from .formats import (
    CONTAINERS, LOSSLESS_CODECS, NVENC_CODECS, output_format_problem,
)
from .probe import SourceMeta
from .settings import ReframeSettings
from .shots import Shot, shots_to_json


class RenderError(ReframeError):
    """Every failure this module can produce."""


CROPS_SCHEMA_VERSION = 2
SENDCMD_NAME = "crop.sendcmd"
SENDCMD_TIME_FORMAT = "{:.6f}"
FRAME_TOLERANCE = 2

# Container -> audio codecs it can carry via `-c:a copy`. A
# container absent here (mkv) accepts any audio codec passthrough.
AUDIO_CONTAINERS: Dict[str, Tuple[str, ...]] = {
    "mp4": ("aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"),
    "mov": ("aac", "mp3", "alac", "pcm_s16le", "pcm_s24le"),
}

# Archival FFV1: intra-only (-g 1) so any frame is seekable, sliced for
# parallelism, with per-slice CRCs so corruption is detectable.
FFV1_ARGS: Tuple[str, ...] = (
    "-level", "3", "-g", "1", "-slices", "24", "-slicecrc", "1",
)

RunnerFn = Callable[..., str]


def _partial(path: Path) -> Path:
    """The sibling temp name for an artifact.

    `vertical.mp4` becomes `vertical.partial.mp4`. Keeps the real
    extension (rather than appending `.partial` after it) so ffmpeg can
    infer the muxer from the filename.

    Byte-identical to ftea/reframe/main.py's helper. It is duplicated
    rather than imported because main.py imports pipeline.py which imports
    this module, so importing back would re-enter a half-initialised
    main. test_render.py asserts the two agree.
    """
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def failed_crops_path(crops_path: Path) -> Path:
    """Where a failed render leaves its crop track: crops.failed.json."""
    return crops_path.with_name(
        f"{crops_path.stem}.failed{crops_path.suffix}")


def ffmpeg_binary(settings: ReframeSettings) -> str:
    """The ffmpeg to spawn: REFRAME_FFMPEG, or `ffmpeg` from PATH."""
    return settings.ffmpeg.strip() or "ffmpeg"


def ffprobe_for(ffmpeg: str) -> str:
    """The ffprobe that ships beside a given ffmpeg.

    Derived rather than configured so the two binaries can never come
    from different builds. There is deliberately no REFRAME_FFPROBE.
    """
    cleaned = ffmpeg.strip()
    if not cleaned:
        return "ffprobe"
    # Split on the raw string rather than through pathlib: Path.with_name
    # re-renders the whole path in the HOST flavour, so a POSIX path
    # handed to a Windows run comes back with backslashes. Keeping the
    # caller's separators byte-for-byte is the point.
    cut = max(cleaned.rfind("/"), cleaned.rfind("\\"))
    directory, name = cleaned[:cut + 1], cleaned[cut + 1:]
    if not name.lower().startswith("ffmpeg"):
        raise RenderError(
            f"REFRAME_FFMPEG must name an ffmpeg binary, got '{name}'"
        )
    return directory + "ffprobe" + name[len("ffmpeg"):]


def normalise_format(settings: ReframeSettings) -> Tuple[str, str]:
    """(codec, container), lower-cased, stripped, dot-free."""
    codec = settings.codec.strip().lower()
    container = settings.container.strip().lower().lstrip(".")
    return codec, container


def validate_output_format(codec: str, container: str) -> None:
    """Reject a codec/container pairing BEFORE ffmpeg is spawned.

    settings.load_settings makes the same check when config.env loads;
    this one covers settings built any other way.
    """
    problem = output_format_problem(codec, container)
    if problem is not None:
        raise RenderError(problem)


def validate_audio_passthrough(
    source_audio_codec: Optional[str], container: str,
) -> None:
    """Reject an audio codec/container pairing BEFORE ffmpeg is spawned.

    `-c:a copy` fails outright when the source's audio codec has no
    mapping into the target container -- PCM and FLAC masters are
    reachable inputs (mov/mkv/avi are accepted by the input gate). A
    container absent from AUDIO_CONTAINERS (mkv) accepts anything, and a
    source with no audio stream at all (None) has nothing to reject.
    """
    if source_audio_codec is None:
        return
    allowed = AUDIO_CONTAINERS.get(container)
    if allowed is not None and source_audio_codec not in allowed:
        raise RenderError(
            f"source audio '{source_audio_codec}' cannot be muxed into "
            f"{container} with -c:a copy -- set REFRAME_CONTAINER=mkv"
        )


def sendcmd_lines(track: CropTrack) -> Tuple[str, ...]:
    """The ffmpeg sendcmd command list for one clip.

    Dispatches on whether the clip has a canvas. A clip with no
    letterbox shot goes through _sendcmd_lines_x, which is the function
    this module has always had, textually unchanged -- so its stream is
    byte-identical to a build without this feature BY CONSTRUCTION, not
    because a test happens to agree.
    """
    if track.canvas is None:
        return _sendcmd_lines_x(track)
    return _sendcmd_lines_rect(track)


# The order sendcmd receives a rectangle's components in. The verified
# script is `crop w, crop h, crop x, crop y` and it rendered exit 0 with
# a constant 608x1080 output; there is no measurement saying another
# order is safe, so this one is not a style choice.
RECT_ORDER: Tuple[str, ...] = ("w", "h", "x", "y")


def _sendcmd_lines_rect(track: CropTrack) -> Tuple[str, ...]:
    """The command list for a clip that letterboxes at least one shot.

    One line per frame where ANY component of the crop rectangle
    changes, carrying only the components that actually moved --
    camera.rect_changes is a lossless run-length read of the same data,
    and sendcmd's parse cost is super-linear.

    Same half-a-frame-early convention as the x-only emitter, and the
    same FLOAT comparison guard: fixed-point decimal strings only order
    correctly while their integer parts are the same width, so a string
    compare reads "10.020833" as <= "9.979167" and reports a collision
    on every clip longer than ten seconds.
    """
    fps = track.fps
    if fps <= 0:
        raise RenderError(f"fps must be positive, got {fps}")

    lines: List[str] = []
    previous: Optional[str] = None
    for frame, delta in rect_changes(track):
        stamp = SENDCMD_TIME_FORMAT.format(max(0.0, (frame - 0.5) / fps))
        if previous is not None and float(stamp) <= float(previous):
            raise RenderError(
                f"sendcmd timestamps collide at frame {frame} "
                f"({stamp} <= {previous}); fps {fps} is too high for "
                f"{SENDCMD_TIME_FORMAT} resolution"
            )
        previous = stamp
        parts = ", ".join(
            f"crop {key} {delta[key]}"
            for key in RECT_ORDER if key in delta)
        lines.append(f"{stamp} {parts};")
    if not lines:
        raise RenderError("rect_changes produced no commands")
    return tuple(lines)


def _sendcmd_lines_x(track: CropTrack) -> Tuple[str, ...]:
    """The ffmpeg sendcmd command list for one clip's crop trajectory.

    One line per CHANGE in the crop's left edge, not one per frame:
    camera.crop_changes is a lossless run-length read of the same data,
    and sendcmd's parse cost is
    super-linear -- 106 us per command at 36 000 commands versus 10 us at
    1 000.

    Each command is timestamped HALF A FRAME BEFORE the frame it belongs
    to. sendcmd fires a command on the first frame whose timestamp is >=
    the command's start time, so a command placed even 0.45 of a frame
    late applies one frame late for the whole clip -- a silent
    desynchronisation between crops.json and the pixels. Half a frame
    early is the widest symmetric margin available.
    """
    fps = track.fps
    if fps <= 0:
        raise RenderError(f"fps must be positive, got {fps}")

    lines: List[str] = []
    previous: Optional[str] = None
    for frame, value in crop_changes(track):
        stamp = SENDCMD_TIME_FORMAT.format(max(0.0, (frame - 0.5) / fps))
        # Compared as NUMBERS, not as strings. Fixed-point decimal
        # strings only order correctly while the integer parts are the
        # same width, so a string compare reads "10.020833" as <=
        # "9.979167" and reports a collision on every clip longer than
        # ten seconds. Equality of the formatted values -- the genuine
        # collision this guards -- survives the float compare intact.
        if previous is not None and float(stamp) <= float(previous):
            raise RenderError(
                f"sendcmd timestamps collide at frame {frame} "
                f"({stamp} <= {previous}); fps {fps} is too high for "
                f"{SENDCMD_TIME_FORMAT} resolution"
            )
        previous = stamp
        lines.append(f"{stamp} crop x {value};")
    if not lines:
        raise RenderError("crop_changes produced no commands")
    return tuple(lines)


def write_sendcmd(track: CropTrack, path: Path) -> int:
    """Write the command file and return the number of command lines.

    ASCII, LF endings, a `#` header for whoever reads it later. Written
    with encoding="ascii" so a non-ASCII byte is a loud error here rather
    than a mojibake filter argument at render time.
    """
    lines = sendcmd_lines(track)
    canvas = ""
    if track.canvas is not None:
        canvas = (f", canvas {track.canvas.width}x{track.canvas.height}"
                  f" pad_y {track.canvas.pad_y}")
    header = (
        f"# ftea/reframe/render.py: {len(lines)} commands, "
        f"{track.total_frames} frames, {track.fps:.3f} fps, "
        f"{track.window.crop_w}x{track.window.crop_h}{canvas}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join((header,) + lines) + "\n"
    try:
        path.write_text(body, encoding="ascii", newline="\n")
    except (OSError, UnicodeEncodeError) as exc:
        raise RenderError(f"could not write {path.name}: {exc}") from exc
    return len(lines)


def filtergraph(track: CropTrack, *,
                script_name: str = SENDCMD_NAME) -> str:
    """The one filter chain the render pass uses.

    Two shapes, chosen ONCE per clip:

    * No canvas -- the graph used without title detection, unchanged.
      This is the common path and it must stay byte-identical.
    * A canvas -- the padded graph. `pad` is NOT runtime-commandable
      while `crop`'s w/h/x/y and `scale`'s w/h ARE, so the padding
      is static and applied once, before the commandable crop, and a
      static scale normalises both states back to crop_w x crop_h.

    `[0:v:0]` and not `[0:v]`: a source carrying an attached picture has
    two video streams, and `[0:v]` is ambiguous there.

    The crop is seeded with frame 0's rectangle so the graph is already
    correct before the first sendcmd command fires.

    `script_name` is a BARE filename and ffmpeg is spawned with cwd set to
    the directory holding it. An apostrophe anywhere in a lavfi option
    value is unescapable, and the per-clip work directory is named from
    the clip's filename -- so the only safe value is one we choose.
    """
    window = track.window
    if track.canvas is None:
        return (
            f"[0:v:0]setpts=PTS-STARTPTS,sendcmd=f={script_name},"
            f"crop={window.crop_w}:{window.crop_h}:"
            f"{track.flatten()[0]}:{window.y}[v]"
        )
    canvas = track.canvas
    first_w, first_h, first_x, first_y = crop_rects(track)[0]
    return (
        f"[0:v:0]setpts=PTS-STARTPTS,"
        f"pad={canvas.width}:{canvas.height}:"
        f"{canvas.pad_x}:{canvas.pad_y}:black,"
        f"sendcmd=f={script_name},"
        f"crop={first_w}:{first_h}:{first_x}:{first_y},"
        f"scale={window.crop_w}:{window.crop_h}[v]"
    )


def video_args(codec: str, settings: ReframeSettings) -> List[str]:
    """The encoder flags for one codec family.

    NVENC uses constqp rather than `vbr -cq`: this intermediate has no
    bitrate ceiling and wants none, because the downstream encoder's
    per-scene VMAF probing has to see clean source material. FFV1 ignores
    cq and preset entirely.
    """
    if codec in LOSSLESS_CODECS:
        return ["-c:v", codec, *FFV1_ARGS]
    if codec in NVENC_CODECS:
        return [
            "-c:v", codec,
            "-preset", settings.preset,
            "-rc", "constqp",
            "-qp", str(settings.cq),
        ]
    supported = ", ".join(sorted(CONTAINERS))
    raise RenderError(
        f"unsupported codec '{codec}' for REFRAME_CODEC; "
        f"supported: {supported}"
    )


# One shot start costs at most 11 characters of command line; 2000 keeps
# the whole argv far below Windows' 32,767-character limit.
MAX_FORCED_KEYFRAMES = 2000


def keyframe_args(
    codec: str, start_frames: Sequence[int], fps: float,
) -> List[str]:
    """Force a keyframe where each shot starts, so the encoder's stream
    copy split can cut on the shot changes instead of the codec's own
    GOP (about ten seconds for hevc_nvenc). -forced-idr makes them IDR
    frames, which is what the segment muxer cuts at.
    """
    if codec in LOSSLESS_CODECS:
        return []
    # Rounded DOWN to the millisecond, as ftea.encode.scene.cut_time
    # does (the reframe environment does not import the encoder): a
    # time just after the frame puts the keyframe on the next one.
    times = [f"{math.floor(frame / fps * 1000 + 1e-6) / 1000:.3f}"
             for frame in sorted(set(start_frames)) if frame > 0]
    if not times:
        return []
    if len(times) > MAX_FORCED_KEYFRAMES:
        print(f"  {len(times)} shots: too many to force keyframes at; "
              "the encoder will cut on the codec's own keyframes.",
              file=sys.stderr)
        return []
    return ["-force_key_frames", ",".join(times), "-forced-idr", "1"]


def build_command(
    source: Path,
    out_partial: Path,
    track: CropTrack,
    settings: ReframeSettings,
    *,
    ffmpeg: str = "ffmpeg",
    script_name: str = SENDCMD_NAME,
) -> List[str]:
    """The complete ffmpeg argv for the single render pass.

    Audio is COPIED, never dropped: the encoder hands the
    intermediate to analyze_and_split and package_hls, so a video-only
    intermediate would make has_audio False and lose the audio silently.
    """
    codec, _container = normalise_format(settings)
    # ABSOLUTE, always. ffmpeg is spawned with cwd set to the output
    # directory, so a relative --source from a direct command-line run
    # would be looked up under the work directory and fail with "No
    # such file or directory". (The encoder already passes absolute
    # paths.) Only the filtergraph's script name stays relative, and
    # that one is a constant this module chose.
    source = Path(source).resolve()
    out_partial = Path(out_partial).resolve()
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(source),
        "-filter_complex", filtergraph(track, script_name=script_name),
        "-map", "[v]", "-map", "0:a?", "-sn", "-dn",
        *video_args(codec, settings),
        *keyframe_args(
            codec, [shot.start_frame for shot in track.shots], track.fps,
        ),
        "-c:a", "copy",
        "-fps_mode", "passthrough",
        str(out_partial),
    ]


@dataclass(frozen=True)
class RenderedInfo:
    """What one ffprobe call says about a rendered intermediate."""

    width: int
    height: int
    frames: int
    codec: str
    pix_fmt: str
    has_audio: bool
    rotation: Optional[float]
    video_start_s: float
    audio_start_s: Optional[float]


def _tail(text: Optional[str], lines: int = 3) -> str:
    """The last few non-blank lines of a stderr blob, on one line."""
    if not text:
        return "no stderr"
    kept = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(kept[-lines:]) or "no stderr"


def _default_runner(argv: Sequence[str], *,
                    cwd: Optional[Path] = None) -> str:
    """Run a binary, return stdout, convert every failure to RenderError.

    Each module raises exactly one exception type across its public
    surface. Spawn failure, non-zero exit and a
    decoding error are all converted here, at the boundary.
    """
    try:
        done = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise RenderError(f"binary not found: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RenderError(
            f"{Path(argv[0]).name} failed (exit {exc.returncode}): "
            f"{_tail(exc.stderr)}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise RenderError(
            f"{Path(argv[0]).name} produced undecodable output: {exc}"
        ) from exc
    return done.stdout or ""


def parse_probe_json(text: str) -> RenderedInfo:
    """Turn one `ffprobe -of json` payload into a RenderedInfo."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RenderError(
            f"ffprobe returned no parseable JSON: {exc}"
        ) from exc

    streams = data.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video"), None,
    )
    if video is None:
        raise RenderError("rendered file has no video stream")

    rotation: Optional[float] = None
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = float(side["rotation"])
            break

    try:
        frames = int(video.get("nb_read_packets") or 0)
    except (TypeError, ValueError):
        frames = 0

    audio = next(
        (s for s in streams if s.get("codec_type") == "audio"), None,
    )
    audio_start_s: Optional[float] = None
    if audio is not None and audio.get("start_time") is not None:
        try:
            audio_start_s = float(audio["start_time"])
        except (TypeError, ValueError):
            audio_start_s = None

    return RenderedInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frames=frames,
        codec=str(video.get("codec_name") or ""),
        pix_fmt=str(video.get("pix_fmt") or ""),
        has_audio=audio is not None,
        rotation=rotation,
        video_start_s=float(video.get("start_time") or 0.0),
        audio_start_s=audio_start_s,
    )


def probe_rendered(path: Path, *, ffprobe: str = "ffprobe",
                   runner: Optional[RunnerFn] = None) -> RenderedInfo:
    """One ffprobe call: geometry, packet count, rotation, audio and
    the per-stream start time.

    -count_packets rather than -count_frames: one video packet is one
    frame here, and counting packets does not decode.
    """
    run = runner or _default_runner
    argv = [
        ffprobe, "-v", "error", "-count_packets",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,"
        "nb_read_packets,pix_fmt,start_time:stream_side_data=rotation",
        "-of", "json", str(path),
    ]
    return parse_probe_json(run(argv))


def verify_output(info: RenderedInfo, track: CropTrack,
                  meta: SourceMeta) -> None:
    """Five checks, run BEFORE either artifact is renamed into place."""
    window = track.window
    if (info.width, info.height) != (window.crop_w, window.crop_h):
        raise RenderError(
            f"rendered geometry {info.width}x{info.height} does not "
            f"match the planned crop {window.crop_w}x{window.crop_h}"
        )

    drift = info.frames - track.total_frames
    if abs(drift) > FRAME_TOLERANCE:
        raise RenderError(
            f"rendered frame count {info.frames} differs from the "
            f"planned {track.total_frames} by {drift} "
            f"(tolerance {FRAME_TOLERANCE})"
        )

    if info.rotation is not None:
        raise RenderError(
            f"rendered file carries a rotation of {info.rotation} "
            "degrees; the intermediate must be in display space"
        )

    if meta.has_audio and not info.has_audio:
        raise RenderError(
            "render produced a silent intermediate (source has audio)"
        )

    if meta.has_audio and info.audio_start_s is not None:
        delta = abs(info.video_start_s - info.audio_start_s)
        tolerance = 1.0 / track.fps
        if delta > tolerance:
            raise RenderError(
                f"rendered A/V start offset {delta:.3f}s exceeds one "
                f"frame ({tolerance:.3f}s); the source's video and "
                "audio do not start together and setpts moved only "
                "the video"
            )


def crops_document(
    meta: SourceMeta,
    track: CropTrack,
    shots: Sequence[Shot],
    out_path: Path,
    settings: ReframeSettings,
    *,
    frames: Optional[int] = None,
) -> dict:
    """The crops.json envelope.

    camera.crop_track_to_json is nested verbatim under "track" -- it
    already carries, per shot and per FRAME, the crop's left edge `x`,
    the zero-order-held subject centre `subject_x`, `observed` and
    `inside`. Those subject centres are what makes the dead-zone metric
    (subject inside the dead zone in >= 95% of observed samples)
    measurable at all; the crop rectangle alone cannot express it.

    The same numbers are ALSO hoisted to a top-level block so a reader
    does not have to know which nested key the metric lives under. That
    block's key is "success_criterion_2", a name kept for compatibility
    with existing crops.json readers. Both come from the same pure
    function of the same CropTrack, so they cannot disagree.

    `frames` is what ffprobe counted in the rendered file. It is kept
    separate from track.total_frames on purpose: they may legitimately
    differ by up to FRAME_TOLERANCE, and a consumer indexing the
    per-frame arrays needs to know which number is which.

    Schema 2 adds, all additive: a `"treatment"` of "crop" or
    "letterbox" on every shot inside "track", an optional `"title"`
    beside it carrying the detected card, and an optional top-level
    `"canvas"` present only when some shot letterboxes. A reader has to
    be able to tell "no treatment key because this file predates the
    feature" from "no treatment key because something broke", which is
    what the version bump buys; nothing in schema 1 moved or changed
    meaning.
    """
    codec, container = normalise_format(settings)
    report = dead_zone_report(track)
    counted = track.total_frames if frames is None else frames
    return {
        "schema": CROPS_SCHEMA_VERSION,
        "generator": "ftea/reframe/render.py",
        "source": {
            "name": meta.path.name,
            "width": meta.width,
            "height": meta.height,
            "rotation": meta.rotation,
            "display_width": meta.display_width,
            "display_height": meta.display_height,
            "orientation": meta.orientation,
            "fps": meta.fps,
            "duration_s": meta.duration_s,
            "has_audio": meta.has_audio,
        },
        "output": {
            "name": out_path.name,
            "codec": codec,
            "container": container,
            "cq": settings.cq,
            "preset": settings.preset,
            "width": track.window.crop_w,
            "height": track.window.crop_h,
            "frames": counted,
            "planned_frames": track.total_frames,
            "frame_drift": counted - track.total_frames,
            "video_args": video_args(codec, settings),
        },
        "aspect": {"w": settings.aspect_w, "h": settings.aspect_h},
        **({"canvas": {
            "width": track.canvas.width,
            "height": track.canvas.height,
            "pad_x": track.canvas.pad_x,
            "pad_y": track.canvas.pad_y,
        }} if track.canvas is not None else {}),
        "success_criterion_2": {
            "threshold": SC2_THRESHOLD,
            "fraction": report.fraction,
            "passes": report.passes,
            "measurable": report.measurable,
            "observed_samples": report.observed_samples,
            "inside_samples": report.inside_samples,
            "clamped_samples": report.clamped_samples,
            "switch_samples": report.switch_samples,
        },
        "shots": shots_to_json(list(shots)),
        "track": crop_track_to_json(track),
    }


def write_crops(document: dict, path: Path) -> None:
    """Write crops.json: ASCII, 2-space indent, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = json.dumps(document, indent=2, ensure_ascii=True)
        path.write_text(body + "\n", encoding="ascii", newline="\n")
    except (OSError, TypeError, ValueError) as exc:
        raise RenderError(
            f"could not write {path.name}: {exc}"
        ) from exc


@dataclass(frozen=True)
class RenderResult:
    """Everything run_stages and its callers need afterwards."""

    output: Path
    crops: Path
    script: Path
    codec: str
    container: str
    width: int
    height: int
    planned_frames: int
    frames: int
    commands: int
    elapsed_s: float
    command: Tuple[str, ...]

    @property
    def drift(self) -> int:
        """frames - planned_frames. Zero when the counts match."""
        return self.frames - self.planned_frames


def _probe_source_audio_codec(
    source: Path, *, ffprobe: str, runner: RunnerFn,
) -> Optional[str]:
    """The source's first audio stream's codec name, or None if absent.

    One extra ffprobe call, issued only when meta.has_audio is true, so
    validate_audio_passthrough can reject a PCM/FLAC master into mp4
    before ffmpeg is spawned rather than via a raw mux error.
    """
    argv = [
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,codec_type",
        "-of", "json", str(source),
    ]
    try:
        payload = json.loads(runner(argv))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RenderError(
            f"ffprobe returned no parseable JSON for the source's "
            f"audio stream: {exc}"
        ) from exc
    # codec_type is requested so a payload carrying more than the
    # selected stream cannot be misread as audio. When ffprobe omits it,
    # -select_streams a:0 has already narrowed the list to audio.
    for stream in payload.get("streams", []):
        if stream.get("codec_type") in (None, "audio"):
            return stream.get("codec_name")
    return None


def render_vertical(
    source: Path,
    meta: SourceMeta,
    track: CropTrack,
    shots: Sequence[Shot],
    out_path: Path,
    crops_path: Path,
    settings: ReframeSettings,
    *,
    runner: Optional[RunnerFn] = None,
) -> RenderResult:
    """Render the 9:16 intermediate and write the crop track.

    Order of operations is the contract:
      validate -> write the script -> render -> probe -> verify ->
      write crops.json -> rename crops -> rename the video.

    The video's arrival at its final name is the commit marker, so it is
    renamed LAST: ftea/reframe/main.py checks both files exist and
    the encoder's _retain_crops copies crops.json out using the
    payload's own path, so the video being final must imply the crop
    track already is.
    """
    run = runner or _default_runner
    codec, container = normalise_format(settings)
    validate_output_format(codec, container)

    ffmpeg = ffmpeg_binary(settings)
    ffprobe = ffprobe_for(settings.ffmpeg)

    if meta.has_audio:
        source_codec = _probe_source_audio_codec(
            source, ffprobe=ffprobe, runner=run)
        validate_audio_passthrough(source_codec, container)

    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    out_partial = _partial(out_path)
    crops_partial = _partial(crops_path)
    for stale in (out_partial, crops_partial):
        if stale.exists():
            stale.unlink()

    script = work_dir / SENDCMD_NAME
    commands = write_sendcmd(track, script)

    failed_path = failed_crops_path(crops_path)
    try:
        argv = build_command(source, out_partial, track, settings,
                             ffmpeg=ffmpeg, script_name=SENDCMD_NAME)
        started = time.time()
        run(argv, cwd=work_dir)
        elapsed = round(time.time() - started, 3)

        if not out_partial.exists() or out_partial.stat().st_size == 0:
            raise RenderError(
                f"ffmpeg exited 0 but wrote nothing to {out_partial.name}"
            )

        info = probe_rendered(out_partial, ffprobe=ffprobe, runner=run)
        verify_output(info, track, meta)
    except RenderError as exc:
        # The trajectory is known good (write_sendcmd succeeded
        # above), so it is not lost along with the retained work dir.
        # A full disk can break this write too; the render's own reason
        # is what the report needs, so it is kept either way.
        try:
            write_crops(
                crops_document(meta, track, shots, out_path, settings,
                               frames=None),
                failed_path)
            note = f"crop track written to {failed_path.name}"
        except RenderError as salvage:
            note = f"crop track NOT saved: {salvage}"
        raise RenderError(f"{exc} ({note})") from exc

    document = crops_document(meta, track, shots, out_path, settings,
                              frames=info.frames)
    write_crops(document, crops_partial)

    crops_partial.replace(crops_path)
    out_partial.replace(out_path)

    return RenderResult(
        output=out_path, crops=crops_path, script=script,
        codec=codec, container=container,
        width=info.width, height=info.height,
        planned_frames=track.total_frames, frames=info.frames,
        commands=commands, elapsed_s=elapsed, command=tuple(argv),
    )


def render_summary(result: RenderResult) -> str:
    """One ASCII progress line for run_stages to print."""
    drift = ""
    if result.frames != result.planned_frames:
        drift = f" (planned {result.planned_frames})"
    return (
        f"rendered {result.output.name} {result.width}x{result.height} "
        f"{result.codec} {result.frames} frames{drift}, "
        f"{result.commands} crop commands, {result.elapsed_s:.1f}s"
    )
