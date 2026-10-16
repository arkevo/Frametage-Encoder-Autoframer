"""Side-by-side debug video for one reframed clip.

Left pane: the ORIGINAL landscape clip with the crop window drawn as a
rectangle that moves exactly the way crops.json says it moved. Right
pane: the rendered 9:16 intermediate. One ffmpeg pass, no intermediate
file, no Python frame loop, no new dependency.

The rectangle is driven by `sendcmd` feeding `drawbox x`, which is the
same mechanism render.py uses to drive `crop x`, with the same
half-a-frame-early time convention, measured frame-exact at 24 fps.
Everything the graph needs comes out of crops.json: this tool never
re-plans a trajectory and never recomputes the crop geometry, so if
crops.json and the rendered file ever disagree, the debug video SHOWS
the disagreement instead of hiding it behind a re-derivation.

There is deliberately no text overlay. drawtext needs fontfile=<path>
inside the filtergraph string, and an absolute Windows path inside a
lavfi graph breaks the graph's option parser (the encoder's probe works
around the same trap). The frame number is on the player's timeline.

ASCII only, so no stdout-reconfigure block is needed here.
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .camera import TREATMENT_LETTERBOX, even_round
from .errors import ReframeError
from .render import RenderError, ffprobe_for


class DebugVideoError(ReframeError):
    """The crop track, the inputs or ffmpeg would not produce a video."""


class BinaryNotFoundError(DebugVideoError):
    """ffmpeg or ffprobe could not be started at all.

    Its own type so the codec fallback can tell "this codec failed"
    from "there is no ffmpeg", which no other codec would fix.
    """


DEBUG_SENDCMD_NAME = "debug.sendcmd"
SENDCMD_TIME_FORMAT = "{:.6f}"
DEFAULT_HEIGHT = 720
DEFAULT_BOX_PX = 4
DEFAULT_COLOR = "red"
DEFAULT_CODEC = "h264_nvenc"
FALLBACK_CODEC = "libx264"
DEFAULT_QP = 23
STILL_FRACTIONS = (0.1, 0.5, 0.9)
# WebP quality for committed evidence stills. 92 keeps the crop
# rectangle crisp at roughly a fifth of PNG's size.
STILL_QUALITY = 92

RunnerFn = Callable[..., str]


@dataclass(frozen=True)
class DebugSpec:
    """Everything one debug render needs, all of it read from crops.json."""

    source: Path
    vertical: Path
    out_path: Path
    fps: float
    display_w: int
    display_h: int
    crop_w: int
    crop_h: int
    y: int
    x: Tuple[int, ...]
    height: int = DEFAULT_HEIGHT
    thickness: int = DEFAULT_BOX_PX
    color: str = DEFAULT_COLOR
    # --- title detection. One (x, y, w, h) per frame in SOURCE space, so
    # a letterboxed shot draws the whole frame instead of a 608px box
    # over a picture that was rendered full-width. Appended at the END
    # with a default so existing constructions stay valid; `x` stays
    # because the pane arithmetic and flatten_x's consumers still use
    # it.
    rects: Tuple[Tuple[int, int, int, int], ...] = ()

    @property
    def left_w(self) -> int:
        return even_up(self.display_w * self.height / self.display_h)

    @property
    def right_w(self) -> int:
        return even_up(self.crop_w * self.height / self.crop_h)

    @property
    def out_w(self) -> int:
        return self.left_w + self.right_w


@dataclass(frozen=True)
class DebugResult:
    """What one debug render produced, as measured by ffprobe."""

    out_path: Path
    script: Path
    width: int
    height: int
    frames: int
    commands: int
    elapsed_s: float
    codec: str
    command: Tuple[str, ...]


def even_up(value: float) -> int:
    """The pane width ffmpeg's `scale=-2:H` actually produces.

    MEASURED against ffmpeg 8.1 rather than assumed: scaling 2560x1080
    to height 720 gives 1706, not the 1708 a round-UP rule predicts, and
    scaling 1920x816 gives 1694, not 1696 -- while 2560x1440 scales from
    an exact 405.0 to 406, which banker's rounding would have made 404.
    The rule is round-HALF-UP to the nearest even, and camera.even_round
    already implements exactly that, so this delegates rather than
    keeping a second copy that can drift.

    Getting this wrong is not cosmetic: render_debug_video checks the
    stacked geometry against the spec, so a 2px disagreement on an
    ultrawide source fails the render.
    """
    return even_round(float(value))


def flatten_x(crops: dict) -> Tuple[int, ...]:
    """The per-frame crop left edge, shots concatenated in order."""
    track = crops.get("track")
    if not isinstance(track, dict):
        raise DebugVideoError("crops.json has no 'track' object")
    shots = track.get("shots") or []
    if not shots:
        raise DebugVideoError("crops.json track has no shots")
    flat: List[int] = []
    for shot in shots:
        flat.extend(int(v) for v in shot.get("x", ()))
    total = int(track.get("total_frames", -1))
    if len(flat) != total:
        raise DebugVideoError(
            f"crop track has {len(flat)} x values but total_frames is {total}"
        )
    return tuple(flat)


def rects_from_crops(
    crops: dict,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """One (x, y, w, h) per FRAME, in SOURCE coordinates.

    Source coordinates, not canvas ones: the left pane shows the
    ORIGINAL landscape clip, so a letterboxed shot is drawn as the
    whole frame -- which is exactly what the renderer kept. Reading
    `treatment` rather than inferring it from the geometry keeps this
    tool a reader: it re-plans nothing, so a disagreement between
    crops.json and the pixels is visible instead of papered over.
    """
    track = crops.get("track")
    if not isinstance(track, dict):
        raise DebugVideoError("crops.json has no 'track' object")
    frame_width = int(track.get("frame_width", 0))
    frame_height = int(track.get("frame_height", 0))
    crop_w = int(track.get("crop_w", 0))
    crop_h = int(track.get("crop_h", 0))
    y = int(track.get("y", 0))
    rects: List[Tuple[int, int, int, int]] = []
    for shot in track.get("shots") or []:
        if shot.get("treatment") == TREATMENT_LETTERBOX:
            rects.extend([(0, 0, frame_width, frame_height)]
                         * len(shot.get("x") or ()))
        else:
            rects.extend((int(value), y, crop_w, crop_h)
                         for value in shot.get("x") or ())
    total = int(track.get("total_frames", -1))
    if len(rects) != total:
        raise DebugVideoError(
            f"crop track has {len(rects)} rectangles but total_frames "
            f"is {total}")
    return tuple(rects)


# The order sendcmd receives a rectangle's components in, matching
# render.RECT_ORDER so the two command streams read the same way.
RECT_ORDER: Tuple[str, ...] = ("w", "h", "x", "y")


def rect_changes(
    rects: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, dict]]:
    """(frame, {changed component: value}). Lossless; frame 0 is full.

    Only the components that actually MOVED are emitted, so a clip with
    no letterbox shot emits exactly one `drawbox x` per change and no
    w/h/y noise -- the same stream this module produced before title
    detection existed.
    """
    if not rects:
        raise DebugVideoError("cannot compress an empty rect series")
    first = rects[0]
    changes: List[Tuple[int, dict]] = [
        (0, {"x": int(first[0]), "y": int(first[1]),
             "w": int(first[2]), "h": int(first[3])})]
    index_of = {"x": 0, "y": 1, "w": 2, "h": 3}
    for frame in range(1, len(rects)):
        delta = {
            key: int(rects[frame][index])
            for key, index in index_of.items()
            if rects[frame][index] != rects[frame - 1][index]
        }
        if delta:
            changes.append((frame, delta))
    return changes


def x_changes(x: Sequence[int]) -> List[Tuple[int, int]]:
    """Run-length compress the x series. changes[0] is always (0, x[0])."""
    if not x:
        raise DebugVideoError("cannot compress an empty x series")
    changes = [(0, int(x[0]))]
    for frame in range(1, len(x)):
        if int(x[frame]) != changes[-1][1]:
            changes.append((frame, int(x[frame])))
    return changes


def box_thickness(
    display_h: int, height: int, target_px: int = DEFAULT_BOX_PX,
) -> int:
    """Source-space thickness that lands at ~target_px after the scale.

    A fixed pixel thickness would vanish at 4K (12 source px become 4 on
    a 720-tall pane) and dominate at 720p. Expressing it as a ratio of
    the two heights follows the stage's rule of computing from geometry
    rather than branching on resolution: nothing here is a per-tier
    constant.
    """
    return max(2, int(round(target_px * display_h / float(height))))


def spec_from_crops(
    crops: dict, source: Path, vertical: Path, out_path: Path, *,
    height: int = DEFAULT_HEIGHT, color: str = DEFAULT_COLOR,
    target_px: int = DEFAULT_BOX_PX,
) -> DebugSpec:
    """Build a DebugSpec from a crops.json document."""
    track = crops.get("track")
    if not isinstance(track, dict):
        raise DebugVideoError("crops.json has no 'track' object")
    meta = crops.get("source") or {}
    fps = float(track.get("fps") or meta.get("fps") or 0.0)
    if fps <= 0.0:
        raise DebugVideoError(f"crops.json has a non-positive fps ({fps})")
    display_w = int(meta.get("display_width") or track.get("frame_width") or 0)
    display_h = int(
        meta.get("display_height") or track.get("frame_height") or 0)
    crop_w = int(track.get("crop_w", 0))
    crop_h = int(track.get("crop_h", 0))
    if not 0 < crop_w <= display_w:
        raise DebugVideoError(
            f"crop_w {crop_w} is not inside a {display_w}px-wide frame"
        )
    if not 0 < crop_h <= display_h:
        raise DebugVideoError(
            f"crop_h {crop_h} is not inside a {display_h}px-tall frame"
        )
    if height < 2:
        raise DebugVideoError(f"pane height {height} is too small")
    return DebugSpec(
        source=source, vertical=vertical, out_path=out_path, fps=fps,
        display_w=display_w, display_h=display_h,
        crop_w=crop_w, crop_h=crop_h, y=int(track.get("y", 0)),
        x=flatten_x(crops), height=height,
        thickness=box_thickness(display_h, height, target_px), color=color,
        rects=rects_from_crops(crops),
    )


def _box_is_static(spec: DebugSpec) -> bool:
    """True when only `x` ever moves, which is the common path.

    A clip with no letterbox shot has a constant y/w/h, so the x-only
    stream below is not merely equivalent to the rect stream -- it is
    the stream this module emitted before title detection, unchanged.
    """
    if not spec.rects:
        return True
    _x0, y0, w0, h0 = spec.rects[0]
    return all((y, w, h) == (y0, w0, h0)
               for _x, y, w, h in spec.rects)


def sendcmd_lines(spec: DebugSpec) -> Tuple[str, ...]:
    """One header comment, then the drawbox commands.

    T = max(0.0, (frame - 0.5) / fps). A command fires on the first frame
    whose PTS >= T, so aiming half a frame EARLY is frame-exact while
    half a frame late lags by one frame for the rest of the clip. This is
    render.py's convention, measured again here for drawbox.

    Dispatches on whether the box ever changes shape. A
    clip with no letterbox shot takes the x-only path, byte-identical to
    what this module always emitted; a clip that letterboxes commands
    w/h/x/y, all four of which were confirmed runtime-commandable on
    ffmpeg 8.1.
    """
    header = f"# debug_video: {len(spec.x)} frames @ {spec.fps:g} fps"
    if _box_is_static(spec):
        # Driven by the RECTANGLE's x, not by spec.x. On a clip with no
        # letterbox shot the two are identical by construction, so this
        # stream is byte-identical to the one before title detection --
        # but on a clip
        # that letterboxes END TO END the box shape never changes and
        # this path is still taken, and there the box sits at x=0 while
        # spec.x holds the crop trajectory. Using spec.x there would
        # command the box back to the crop window and undo the seed.
        lines = [header]
        for frame, value in x_changes(
                [rect[0] for rect in spec.rects] if spec.rects
                else list(spec.x)):
            seconds = max(0.0, (frame - 0.5) / spec.fps)
            stamp = SENDCMD_TIME_FORMAT.format(seconds)
            lines.append(f"{stamp} drawbox x {int(value)};")
        return tuple(lines)

    lines = [header]
    previous: Optional[str] = None
    for frame, delta in rect_changes(spec.rects):
        seconds = max(0.0, (frame - 0.5) / spec.fps)
        stamp = SENDCMD_TIME_FORMAT.format(seconds)
        # Compared as NUMBERS, never as strings: "10.020833" <=
        # "9.979167" is true as a string compare and killed every clip
        # over ten seconds until a test caught it.
        if previous is not None and float(stamp) <= float(previous):
            raise DebugVideoError(
                f"sendcmd timestamps collide at frame {frame} "
                f"({stamp} <= {previous})")
        previous = stamp
        parts = ", ".join(
            f"drawbox {key} {delta[key]}"
            for key in RECT_ORDER if key in delta)
        lines.append(f"{stamp} {parts};")
    return tuple(lines)


def write_sendcmd(spec: DebugSpec, path: Path) -> int:
    """Write the script as ASCII with LF endings. Returns command count."""
    lines = sendcmd_lines(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            "\n".join(lines) + "\n", encoding="ascii", newline="\n")
    except (OSError, UnicodeEncodeError) as exc:
        raise DebugVideoError(f"could not write {path.name}: {exc}") from None
    return len(lines) - 1


def filtergraph(
    spec: DebugSpec, *, script_name: str = DEBUG_SENDCMD_NAME,
) -> str:
    """The whole graph. Contains a bare script NAME and no path at all.

    The static drawbox is seeded with frame 0's RECTANGLE, exactly as
    render.filtergraph seeds its crop, so the overlay is already right
    before the first sendcmd command fires.
    """
    if spec.rects:
        first_x, first_y, first_w, first_h = spec.rects[0]
    else:
        first_x = spec.x[0] if spec.x else 0
        first_y, first_w, first_h = spec.y, spec.crop_w, spec.crop_h
    return (
        f"[0:v:0]setpts=PTS-STARTPTS,"
        f"sendcmd=f={script_name},"
        f"drawbox=x={first_x}:y={first_y}:w={first_w}:h={first_h}"
        f":color={spec.color}:t={spec.thickness},"
        f"scale=-2:{spec.height}[l];"
        f"[1:v:0]setpts=PTS-STARTPTS,scale=-2:{spec.height}[r];"
        f"[l][r]hstack=inputs=2:shortest=1[v]"
    )


def build_command(
    spec: DebugSpec, *, ffmpeg: str = "ffmpeg",
    script_name: str = DEBUG_SENDCMD_NAME, codec: str = DEFAULT_CODEC,
    qp: int = DEFAULT_QP,
) -> List[str]:
    """The exact ffmpeg argv. Run it with cwd = out_path.parent."""
    video: List[str] = (
        ["-c:v", codec, "-preset", "p5", "-rc", "constqp", "-qp", str(qp)]
        if codec.endswith("nvenc")
        else ["-c:v", codec, "-crf", str(qp), "-preset", "veryfast"]
    )
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(spec.source),
        "-i", str(spec.vertical),
        "-filter_complex", filtergraph(spec, script_name=script_name),
        "-map", "[v]", "-an", "-sn", "-dn",
        *video,
        "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough",
        str(spec.out_path),
    ]


def _run(argv: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    """Spawn a tool and return stdout. Raises DebugVideoError, only."""
    try:
        result = subprocess.run(
            [str(a) for a in argv], cwd=None if cwd is None else str(cwd),
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        raise BinaryNotFoundError(f"{argv[0]} not found on PATH") from None
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-1:] or [""]
        raise DebugVideoError(f"{argv[0]} failed: {tail[0]}") from None
    return result.stdout


def _ffprobe_for(ffmpeg: str) -> str:
    """The ffprobe beside a given ffmpeg, found as render.ffprobe_for
    finds it: any letter case, the caller's own path separators.

    A binary not named ffmpeg* falls back to ffprobe from PATH rather
    than raising, because the debug video never fails a clip.
    """
    try:
        return ffprobe_for(ffmpeg)
    except RenderError:
        return "ffprobe"


def probe_debug(
    path: Path, *, ffprobe: str = "ffprobe",
    runner: Optional[RunnerFn] = None,
) -> Tuple[int, int, int]:
    """(width, height, frames) of a rendered debug video."""
    run = runner or _run
    raw = run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets",
        "-of", "json", str(path),
    ])
    try:
        stream = json.loads(raw)["streams"][0]
    except (TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise DebugVideoError(f"ffprobe gave no usable JSON: {exc}") from None
    return (
        int(stream.get("width", 0)),
        int(stream.get("height", 0)),
        int(stream.get("nb_read_packets") or 0),
    )


def render_debug_video(
    spec: DebugSpec, *, ffmpeg: Optional[str] = None,
    runner: Optional[RunnerFn] = None, codec: Optional[str] = None,
    require_inputs: bool = False,
) -> DebugResult:
    """Render the side-by-side video. One ffmpeg pass, one ffprobe.

    Order of operations: check the inputs -> mkdir -> write the sendcmd
    script -> ffmpeg with cwd = out_path.parent (the script is referenced
    by BARE NAME, never by path) -> assert the file is non-empty ->
    ffprobe -> assert the geometry is what the spec planned.

    NVENC is tried first because the debug video is large and this runs
    per clip; a failure is logged on stderr and falls back to libx264
    once, and the codec actually used is reported. If libx264 fails too,
    the error names both failures. A missing ffmpeg is not retried.

    There is no partial-rename dance here: unlike render.py's
    intermediate, nothing downstream consumes this file, so a
    half-written debug video costs a re-run and nothing else.
    """
    if require_inputs:
        for path in (spec.source, spec.vertical):
            if not Path(path).is_file():
                raise DebugVideoError(f"input {Path(path).name} is absent")

    binary = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    script = spec.out_path.parent / DEBUG_SENDCMD_NAME
    commands = write_sendcmd(spec, script)

    run = runner or _run
    chosen = codec or DEFAULT_CODEC
    command = build_command(spec, ffmpeg=binary, codec=chosen)
    started = time.time()
    try:
        run(command, cwd=spec.out_path.parent)
    except BinaryNotFoundError:
        raise
    except DebugVideoError as first:
        if chosen == FALLBACK_CODEC:
            raise
        failed = chosen
        print(f"  reframe: debug video: {failed} failed ({first}); "
              f"retrying with {FALLBACK_CODEC}", file=sys.stderr)
        chosen = FALLBACK_CODEC
        command = build_command(spec, ffmpeg=binary, codec=chosen)
        try:
            run(command, cwd=spec.out_path.parent)
        except DebugVideoError as second:
            raise DebugVideoError(
                f"{second} ({failed} had failed first: {first})"
            ) from first
    elapsed = round(time.time() - started, 3)

    if not spec.out_path.is_file() or spec.out_path.stat().st_size == 0:
        raise DebugVideoError(f"{spec.out_path.name} is empty after ffmpeg")

    width, height, frames = probe_debug(
        spec.out_path, ffprobe=_ffprobe_for(binary), runner=runner)
    if (width, height) != (spec.out_w, spec.height):
        raise DebugVideoError(
            f"debug video is {width}x{height}, expected "
            f"{spec.out_w}x{spec.height}"
        )

    return DebugResult(
        out_path=spec.out_path, script=script, width=width, height=height,
        frames=frames, commands=commands, elapsed_s=elapsed, codec=chosen,
        command=tuple(command),
    )


def extract_stills(
    result: DebugResult, out_dir: Path, *,
    fractions: Sequence[float] = STILL_FRACTIONS,
    fps: float = 24.0, ffmpeg: Optional[str] = None,
    runner: Optional[RunnerFn] = None,
) -> List[Path]:
    """Pull a few WebP stills out of the debug video.

    Handy for a README image or a bug report: the videos are large, the
    stills are small. WebP, not PNG: at quality 92 a still is about a
    fifth of the PNG size, with the crop rectangle and the underlying
    texture both still plainly legible.
    """
    binary = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    run = runner or _run
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fraction in fractions:
        frame = int(max(0, min(result.frames - 1, result.frames * fraction)))
        target = out_dir / f"{result.out_path.stem}_f{frame:05d}.webp"
        run([
            binary, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{frame / float(fps):.3f}", "-i", str(result.out_path),
            "-frames:v", "1", "-quality", str(STILL_QUALITY), str(target),
        ])
        written.append(target)
    return written


_USAGE = (
    "Usage: python -m ftea.reframe.debug_video --crops <crops.json> "
    "--source <original video> --out <mp4> [--vertical P] [--height N] "
    "[--stills DIR]"
)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI. 0 = ok, 1 = error, 2 = an input is missing.

    --source is required: crops.json records only the original's file
    name, and neither `ftea run` nor `ftea reframe` keeps the original
    beside it. --vertical defaults to the file crops.json names, in the
    folder crops.json is in, which is where `ftea reframe` leaves both.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    options = {"--crops": None, "--out": None, "--source": None,
               "--vertical": None, "--height": None, "--stills": None}
    while args:
        flag = args.pop(0)
        if flag not in options or not args:
            print(_USAGE, file=sys.stderr)
            return 1
        options[flag] = args.pop(0)
    if not all(options[key] for key in ("--crops", "--out", "--source")):
        print(_USAGE, file=sys.stderr)
        return 1

    crops_path = Path(options["--crops"])
    if not crops_path.is_file():
        print(f"  debug: no crops.json at {crops_path}", file=sys.stderr)
        return 2
    try:
        crops = json.loads(crops_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"  debug: malformed crops.json: {exc}", file=sys.stderr)
        return 1

    source = Path(options["--source"])
    vertical = Path(options["--vertical"]) if options["--vertical"] else (
        crops_path.parent / crops["output"]["name"])
    if not source.is_file() or not vertical.is_file():
        print(
            f"  debug: need both {source.name} and {vertical.name}",
            file=sys.stderr,
        )
        return 2

    try:
        spec = spec_from_crops(
            crops, source, vertical, Path(options["--out"]),
            height=int(options["--height"] or DEFAULT_HEIGHT),
        )
        result = render_debug_video(spec, require_inputs=True)
        stills = (
            extract_stills(result, Path(options["--stills"]), fps=spec.fps)
            if options["--stills"] else []
        )
    except DebugVideoError as exc:
        print(f"  debug: {exc}", file=sys.stderr)
        return 1

    print(f"  debug: {result.out_path.name} {result.width}x{result.height} "
          f"{result.frames}f {result.commands} commands "
          f"{result.elapsed_s:.1f}s {result.codec}", file=sys.stderr)
    print(json.dumps({
        "out": str(result.out_path), "script": str(result.script),
        "width": result.width, "height": result.height,
        "frames": result.frames, "commands": result.commands,
        "elapsed_s": result.elapsed_s, "codec": result.codec,
        "stills": [str(p) for p in stills],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
