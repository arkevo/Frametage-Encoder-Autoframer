"""Command-line entry point for `ftea`.

    ftea run [video-or-folder] [--reframe | --no-reframe] [--force]
    ftea preview [clip] [--port N] [--local-only]
    ftea reframe <video> [--debug-video] [--out FOLDER]

Each command imports its own module only when it runs, so `ftea
--version` stays instant and `ftea run` never loads the preview server.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, paths
from .console import shell_arg, tolerate_unprintable_text

if TYPE_CHECKING:
    from .vertical import VerticalResult

# COMMANDS, USAGE, the parser and the dispatch all come from _REGISTRY at
# the end of this module, the one list of commands.
WATCH_HINT_LIMIT = 5
# Problems any command can meet before doing anything: no checkout, or
# an FTEA_HOME that names a folder that does not exist.
_SETUP_ERRORS = (paths.RepoRootError, paths.DataRootError)


def _port(text: str) -> int:
    """A port number written in ASCII digits only: int() alone would also
    take "8_765", " 80 " and digits of other scripts."""
    value = int(text) if text.isascii() and text.isdigit() else -1
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a port number (0-65535)"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftea",
        description=(
            "Reframe landscape video to vertical 9:16, encode it as a "
            "six-variant HLS package, and watch it on a phone."
        ),
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"ftea {__version__}",
    )
    commands = parser.add_subparsers(
        dest="command", metavar="{" + ",".join(COMMANDS) + "}",
    )
    for add, _ in _REGISTRY.values():
        add(commands)
    return parser


def _add_run(commands) -> None:
    run = commands.add_parser(
        "run", help="reframe and encode videos",
        description=(
            "Reframe each landscape video and encode every new video in a "
            "file or folder. With no argument, the input folder is used."
        ),
    )
    run.add_argument(
        "input", nargs="?",
        help="a video file or a folder of videos (default: the input folder)",
    )
    choice = run.add_mutually_exclusive_group()
    choice.add_argument(
        "--reframe", dest="reframe", action="store_const", const=True,
        help="reframe landscape videos even if REFRAME=false in config.env",
    )
    choice.add_argument(
        "--no-reframe", dest="reframe", action="store_const", const=False,
        help="do not reframe in this run",
    )
    run.add_argument(
        "--force", action="store_true",
        help="encode videos again even when their output is complete",
    )


def _add_preview(commands) -> None:
    preview = commands.add_parser(
        "preview", help="watch a finished clip in a browser or on a phone",
        description=(
            "Serve one finished clip folder so this computer and phones on "
            "the same network can play it."
        ),
    )
    preview.add_argument(
        "clip", nargs="?",
        help=(
            "a clip folder, its master.m3u8, or a clip name in the output "
            "folder (optional when there is only one clip)"
        ),
    )
    preview.add_argument(
        "--port", type=_port,
        help=(
            "port to listen on (default: 8765, or the next free port; "
            "0 picks any free port)"
        ),
    )
    preview.add_argument(
        "--local-only", action="store_true",
        help="let only this computer connect",
    )


def _add_reframe(commands) -> None:
    reframe = commands.add_parser(
        "reframe", help="make a vertical video without encoding it",
        description=(
            "Run only the reframe stage on one video and write the "
            "vertical MP4. A video that is not landscape is left alone: "
            "nothing is written and ftea reframe exits 0."
        ),
    )
    reframe.add_argument("video", help="a landscape video file")
    reframe.add_argument(
        "--debug-video", action="store_true",
        help=(
            "also write debug.mp4: the original with the crop window, "
            "beside the result"
        ),
    )
    reframe.add_argument(
        "--out",
        help=(
            "a new or empty folder for the results, or this video's "
            "earlier results (default: output/<name>-vertical)"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit status.

    0 ok (including `ftea preview` stopped with Ctrl+C), 1 failed,
    2 usage error, 130 when `ftea run` or `ftea reframe` is interrupted.
    """
    tolerate_unprintable_text()
    words = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(words)
    except SystemExit as exc:
        return _status(exc.code)
    if args.command is None:
        print(USAGE, file=sys.stderr)
        print("ftea: choose a command: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    _, handler = _REGISTRY[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def _status(code: object) -> int:
    """argparse exits with 0 (help, version) or 2 (usage error); None is
    read as 0, and anything else that is not a number as a usage
    error."""
    if code is None:
        return 0
    return code if isinstance(code, int) else 2


def _fail(problem: object) -> int:
    print(f"Error: {problem}", file=sys.stderr)
    return 1


def _run(args: argparse.Namespace) -> int:
    from .encode.main import PipelineError, run

    source = None if args.input is None else Path(args.input)
    try:
        result = run(source, reframe=args.reframe, force=args.force)
    except (PipelineError, *_SETUP_ERRORS) as exc:
        return _fail(exc)
    _print_watch_hint(result.outputs)
    return result.exit_code


def _print_watch_hint(outputs: Sequence[Path]) -> None:
    if not outputs:
        return
    print()
    print("Watch it on this computer or a phone:")
    for folder in outputs[:WATCH_HINT_LIMIT]:
        print(f"  ftea preview {shell_arg(folder.name)}")
    hidden = len(outputs) - WATCH_HINT_LIMIT
    if hidden > 0:
        print(f"  ...and {hidden} more in {outputs[0].parent}")


def _preview(args: argparse.Namespace) -> int:
    """A named clip can be previewed without a usable output folder (a
    path works anywhere), but a name that is not found says why the
    output folder was not searched."""
    from .preview import PreviewError, resolve_clip_dir, serve

    setup_error = None
    try:
        output = paths.output_dir()
    except _SETUP_ERRORS as exc:
        if args.clip is None:
            return _fail(exc)
        output, setup_error = None, exc
    try:
        clip = resolve_clip_dir(args.clip, output_dir=output)
    except PreviewError as exc:
        if setup_error is None:
            return _fail(exc)
        return _fail(
            f"{exc}\nThe output folder could not be searched: {setup_error}"
        )
    try:
        return serve(clip, port=args.port, local_only=args.local_only)
    except PreviewError as exc:
        return _fail(exc)


def _reframe(args: argparse.Namespace) -> int:
    from .encode.main import PipelineError
    from .vertical import VerticalError, make_vertical

    video = Path(args.video)
    try:
        result = make_vertical(
            video, out_dir=None if args.out is None else Path(args.out),
            debug_video=args.debug_video,
        )
    except (VerticalError, PipelineError, *_SETUP_ERRORS, OSError) as exc:
        return _fail(exc)
    _print_vertical(result, video)
    return result.exit_code


def _print_vertical(result: VerticalResult, video: Path) -> None:
    from .vertical import PARTIAL_DIR

    if result.status == "skipped":
        print(
            f"Nothing to reframe: {video.name} is not a landscape video "
            f"({result.reason or 'no reason given'}). Nothing was written."
        )
        return
    if result.status == "failed":
        print(f"Reframe failed: {result.error}", file=sys.stderr)
        print(
            f"Its partial files are in {result.out_dir / PARTIAL_DIR}.",
            file=sys.stderr,
        )
        print(
            "If it keeps failing, run `bash install.sh` again; it ends by "
            "checking the reframe environment.", file=sys.stderr,
        )
        return
    print(f"Vertical video: {result.vertical}")
    if result.crops is not None:
        print(f"Crop track:     {result.crops}")
    else:
        print("No crop track was written for this video.")
    if result.debug is not None:
        print(
            f"Debug video:    {result.debug} (the original with the crop "
            "window, beside the result)"
        )
    if result.debug_error:
        print(
            f"The debug video failed: {result.debug_error}. The vertical "
            "video is fine.", file=sys.stderr,
        )


# Every command: how its arguments are added, and what runs it.
_REGISTRY: dict[str, tuple[
    Callable[[argparse._SubParsersAction], None],
    Callable[[argparse.Namespace], int],
]] = {
    "run": (_add_run, _run),
    "preview": (_add_preview, _preview),
    "reframe": (_add_reframe, _reframe),
}
COMMANDS = tuple(_REGISTRY)
USAGE = "usage: ftea {" + ",".join(COMMANDS) + "} ... | ftea --version"
