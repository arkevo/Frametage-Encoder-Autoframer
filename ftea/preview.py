"""`ftea preview`: serve one clip's output folder to this computer and the LAN.

The folder is served read-only, without directory listings, with an
explicit table of content types, and with caching turned off, so a phone
on the same network can open player.html and play the package. A folder
made before the player existed gets the bundled player files instead.
The server never changes the working folder, holds no file open
between requests, and reads each file of up to MAX_BUFFERED_BYTES into
memory and closes it before sending its body, so the clip folder is held
only for the disk read and not for the whole Wi-Fi transfer: `ftea run
--force` can replace the clip even while a phone is playing it. A client
that goes away mid-download (every seek and every variant switch does)
is normal and is not printed.

Standard library only.
"""

from __future__ import annotations

import io
import os
import platform
import re
import socket
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, TextIO
from urllib.parse import unquote, urlsplit

from .console import shell_arg

DEFAULT_PORT = 8765
# How many ports from the default are tried when it is taken. A port the
# user chose with --port is tried alone.
PORT_ATTEMPTS = 10
PLAYER_PAGE = "player.html"
MASTER_PLAYLIST = "master.m3u8"
INFO_HINT = "?info=1"
# Files up to this size are read into memory and closed before their
# body is sent, so a download in flight never holds the clip folder open
# (Windows cannot rename a folder while a file in it is open). A
# package's segments are a few MiB; anything larger is streamed.
MAX_BUFFERED_BYTES = 64 * 1024 * 1024

# Content types, by extension. Explicit on purpose: on Windows, Python's
# mimetypes module also reads the registry, which other software can
# change, and a browser refuses a playlist or segment served with the
# wrong type. Anything not listed is served as FALLBACK_TYPE.
MIME_TYPES: Mapping[str, str] = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".m4s": "video/iso.segment",
    ".mp4": "video/mp4",
    ".ts": "video/mp2t",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
FALLBACK_TYPE = "application/octet-stream"

# Addresses never offered as a LAN URL: loopback, the self-assigned
# link-local range a machine falls back to without DHCP, and 0.x.
EXCLUDED_PREFIXES = ("127.", "169.254.", "0.")

# A documentation-only address (TEST-NET-1). Connecting a UDP socket to
# it sends nothing; it only asks the system which local address would
# carry the traffic, which is the address of the adapter holding the
# default route rather than a virtual adapter.
_ROUTE_PROBE = ("192.0.2.1", 9)

# Browsers ask for these on their own; a 404 for them is not a problem.
# The bundled hls.min.js names a source map that is not shipped, which a
# browser asks for while its developer tools are open.
QUIET_MISSING = frozenset({
    "/favicon.ico", "/apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png", "/hls.min.js.map",
})
_SEGMENT = re.compile(r"^/([A-Za-z0-9]+_\d+p)/[^/]+\.(?:m4s|ts)$")
# Checked in order: an Android user agent also says "Linux". iPadOS
# Safari sends a Mac user agent by default, so "Macintosh" may be an iPad.
DEVICE_HINTS = (
    ("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"),
    ("Windows", "Windows"), ("Macintosh", "Mac or iPad"), ("Linux", "Linux"),
)


class PreviewError(RuntimeError):
    """Nothing to preview where the user pointed, or no port to use."""


def content_type(path: str) -> str:
    """The Content-Type for a file name, from MIME_TYPES only."""
    return MIME_TYPES.get(os.path.splitext(path)[1].lower(), FALLBACK_TYPE)


def variant_of(route: str) -> Optional[str]:
    """The variant folder a segment request belongs to, if it is one."""
    match = _SEGMENT.match(route)
    return match.group(1) if match else None


def device_name(agent: str) -> str:
    for needle, name in DEVICE_HINTS:
        if needle in agent:
            return name
    return "unknown device"


# --- Addresses ---------------------------------------------------------------

def _routed_address() -> Optional[str]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(_ROUTE_PROBE)
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def _hostname_addresses() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return [info[4][0] for info in infos]


def lan_addresses(
    routed: Optional[Callable[[], Optional[str]]] = None,
    others: Optional[Callable[[], list[str]]] = None,
) -> list[str]:
    """IPv4 addresses a phone might reach, the routed one first."""
    first = (routed or _routed_address)()
    found: list[str] = []
    for address in [first, *(others or _hostname_addresses)()]:
        if (address and not address.startswith(EXCLUDED_PREFIXES)
                and address not in found):
            found.append(address)
    return found


# --- Which folder to serve -----------------------------------------------------

def clip_folders(folder: Path) -> list[Path]:
    """Sub-folders of `folder` that hold a finished package."""
    if not folder.is_dir():
        return []
    return sorted(
        (
            path for path in folder.iterdir()
            if path.is_dir() and not path.name.startswith(".")
            and (path / MASTER_PLAYLIST).is_file()
        ),
        key=lambda path: path.name.casefold(),
    )


def resolve_clip_dir(
    argument: Optional[str], *, output_dir: Optional[Path],
) -> Path:
    """The clip folder to serve.

    `argument` is a clip folder, its master.m3u8, or a clip name inside
    output_dir; a relative path is read from the working folder first.
    Without an argument, the only finished clip in output_dir is used.
    """
    if argument is None:
        return _only_clip(output_dir)
    given = Path(argument).expanduser()
    if given.name.lower() == MASTER_PLAYLIST and given.is_file():
        return given.parent.resolve()
    candidates = [given]
    if output_dir is not None and not given.is_absolute():
        candidates.append(output_dir / given)
    for candidate in candidates:
        if (candidate / MASTER_PLAYLIST).is_file():
            return candidate.resolve()
    for candidate in candidates:
        if candidate.is_dir():
            raise PreviewError(_not_a_clip(candidate))
    raise PreviewError(
        f"{argument} was not found. Name a finished clip folder, for "
        "example: ftea preview <clip name>"
    )


def _only_clip(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        raise PreviewError(
            "name the clip folder to preview, for example: "
            "ftea preview output/<clip name>"
        )
    clips = clip_folders(output_dir)
    if len(clips) == 1:
        return clips[0].resolve()
    if not clips:
        raise PreviewError(
            f"{output_dir} has no finished clips yet. Run `ftea run` first."
        )
    lines = [f"  ftea preview {shell_arg(c.name)}" for c in clips]
    raise PreviewError("choose a clip to preview:\n" + "\n".join(lines))


def _not_a_clip(folder: Path) -> str:
    clips = clip_folders(folder)
    if not clips:
        return (
            f"{folder} has no {MASTER_PLAYLIST}, so it is not a finished "
            "clip folder."
        )
    lines = [f"  ftea preview {shell_arg(str(c))}" for c in clips]
    return f"{folder} holds clips; choose one:\n" + "\n".join(lines)


# --- What happens on the network ----------------------------------------------

class ActivityLog:
    """Prints what each device does, without one line per request."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._lock = threading.Lock()
        self._variants: dict[str, str] = {}
        self._range_clients: set[str] = set()

    def record(
        self, client: str, route: str, status: int,
        byte_range: Optional[str] = None, agent: str = "",
    ) -> None:
        with self._lock:
            lines = self._lines(client, route, status, byte_range, agent)
            for line in lines:
                print(line, file=self._out, flush=True)

    def note(self, line: str) -> None:
        """Print one line that is not about a request."""
        with self._lock:
            print(line, file=self._out, flush=True)

    def _lines(self, client, route, status, byte_range, agent) -> list[str]:
        lines: list[str] = []
        if client not in self._variants:
            self._variants[client] = ""
            lines.append(f"{client} connected ({device_name(agent)})")
        if status >= 400 and route not in QUIET_MISSING:
            lines.append(f"{client} {status} {route}")
        variant = variant_of(route)
        if variant and status < 400 and variant != self._variants[client]:
            self._variants[client] = variant
            lines.append(f"{client} playing {variant}")
        if byte_range and client not in self._range_clients:
            self._range_clients.add(client)
            lines.append(
                f"{client} asked for a byte range ({byte_range}); this "
                "server always sends whole files"
            )
        return lines


class PreviewHandler(SimpleHTTPRequestHandler):
    """Serves one folder by the rules in the module docstring."""

    server_version = "ftea-preview"
    sys_version = ""

    def guess_type(self, path) -> str:
        return content_type(str(path))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "Folders are not listed")
        return None

    def send_head(self):
        parts = urlsplit(self.path)
        route = parts.path
        if route == "/":
            query = f"?{parts.query}" if parts.query else ""
            self.send_response(302)
            self.send_header("Location", f"/{PLAYER_PAGE}{query}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        bundled = self.server.fallbacks.get(unquote(route.lstrip("/")))
        if bundled is not None:
            self.send_response(200)
            self.send_header("Content-Type", content_type(route))
            self.send_header("Content-Length", str(len(bundled)))
            self.end_headers()
            return io.BytesIO(bundled)
        head = super().send_head()
        if head is None or self.command != "GET":
            return head
        return self._in_memory(head)

    def _in_memory(self, handle):
        """Read a small file and close it before its body is sent."""
        if os.fstat(handle.fileno()).st_size > MAX_BUFFERED_BYTES:
            return handle
        with handle:
            return io.BytesIO(handle.read())

    def log_request(self, code="-", size="-") -> None:
        headers = getattr(self, "headers", None)
        self.server.activity.record(
            self.client_address[0],
            urlsplit(getattr(self, "path", "")).path,
            int(code) if isinstance(code, int) else 0,
            headers.get("Range") if headers is not None else None,
            headers.get("User-Agent", "") if headers is not None else "",
        )

    def log_message(self, format, *args) -> None:
        """Silenced: ActivityLog reports what matters."""


class PreviewServer(ThreadingHTTPServer):
    """One clip folder, its bundled-player fallbacks and its log."""

    daemon_threads = True
    # On Windows, SO_REUSEADDR lets a second server bind a port that is
    # already in use, so a busy port would look free. Elsewhere it only
    # allows a quick restart after Ctrl+C.
    allow_reuse_address = os.name != "nt"

    def __init__(
        self, address: tuple[str, int], directory: Path, *,
        fallbacks: Mapping[str, bytes], activity: ActivityLog,
    ) -> None:
        self.fallbacks = dict(fallbacks)
        self.activity = activity
        handler = partial(PreviewHandler, directory=str(directory))
        super().__init__(address, handler)

    def server_bind(self) -> None:
        """On Windows, take the port exclusively. Without that, a bind to
        0.0.0.0 succeeds on a port another program already holds on
        127.0.0.1, and localhost then reaches that program; another
        program could also bind 127.0.0.1 on top of this server."""
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    def handle_error(self, request, client_address) -> None:
        """A client that goes away mid-download is normal: hls.js and
        Safari cancel a download on every seek and every variant switch,
        so that prints nothing. Any other error prints one activity line
        and keeps its traceback on stderr, so a real bug is still seen."""
        error = sys.exc_info()[1]
        if isinstance(error, ConnectionError):
            return
        self.activity.note(
            f"{client_address[0]} server error: "
            f"{type(error).__name__}: {error}"
        )
        super().handle_error(request, client_address)


def open_server(
    directory: Path, *, port: int, bind: str, attempts: int,
    fallbacks: Mapping[str, bytes], activity: ActivityLog,
) -> PreviewServer:
    """Listen on `port`, or on the next free one within `attempts`."""
    last = min(port + max(1, attempts), 65536) - 1
    error: Optional[OSError] = None
    for candidate in range(port, last + 1):
        try:
            return PreviewServer(
                (bind, candidate), directory,
                fallbacks=fallbacks, activity=activity,
            )
        except OSError as exc:
            error = exc
    tried = str(port) if last == port else f"{port}-{last}"
    raise PreviewError(
        f"could not listen on port {tried} ({error}). Choose a free port "
        "with --port."
    )


def bundled_fallbacks(clip_dir: Path) -> dict[str, bytes]:
    """The bundled player files this clip folder does not have."""
    from .encode.package import PLAYER_FILES, read_player_file

    missing = [
        name for name in PLAYER_FILES if not (clip_dir / name).is_file()
    ]
    try:
        return {name: read_player_file(name) for name in missing}
    except FileNotFoundError as exc:
        raise PreviewError(
            f"{exc}, and {clip_dir} has no copy of its own"
        ) from None


def firewall_program() -> str:
    """The program Windows Firewall asks about. A venv's python.exe
    relaunches the base interpreter, which is the process that listens."""
    return getattr(sys, "_base_executable", "") or sys.executable


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def banner(
    clip_dir: Path, port: int, addresses: Sequence[str], *,
    local_only: bool, bundled: Sequence[str] = (),
    program: Optional[str] = None, wsl: bool = False,
    windows: Optional[bool] = None,
) -> list[str]:
    """What serve() prints first. `bundled` names the player files served
    from the ftea package because the folder lacks them."""
    windows = os.name == "nt" if windows is None else windows
    page = f":{port}/{PLAYER_PAGE}"
    lines = [f"Serving {clip_dir}", f"  On this computer: http://localhost{page}"]
    if local_only:
        lines.append("  Only this computer can open it (--local-only).")
    elif addresses:
        lines.append(
            "  On a phone or tablet on the same Wi-Fi: "
            f"http://{addresses[0]}{page}"
        )
        lines += [
            f"  If that does not open, try: http://{a}{page}"
            for a in addresses[1:]
        ]
    else:
        lines.append(
            "  No network address was found, so only this computer can "
            "open it."
        )
    if wsl and not local_only:
        lines.append(
            "This is WSL: a phone usually cannot reach this address. Run "
            "ftea preview from Windows, or turn on WSL's mirrored "
            "networking."
        )
    lines.append(
        f"Add {INFO_HINT} to the address to see which player and variant "
        "are playing."
    )
    if PLAYER_PAGE in bundled:
        lines.append(
            "This folder has no player of its own, so the bundled player "
            "is served."
        )
    elif bundled:
        lines.append(f"Serving the bundled copy of: {', '.join(bundled)}.")
    if not local_only:
        lines.append(
            "Anyone on this network can open these addresses while ftea "
            "preview runs."
        )
    if windows and not local_only:
        lines += _firewall_lines(program or firewall_program())
    lines.append("Press Ctrl+C to stop.")
    return lines


def _firewall_lines(program: str) -> list[str]:
    return [
        f"On Windows, if a firewall window appears for {program}, allow "
        "access on Private networks.",
        "If no window appears and a phone cannot connect, Windows Firewall "
        "may already be blocking it. Open Windows Security > Firewall & "
        "network protection > Allow an app through firewall, and allow "
        f"{program} on Private networks.",
    ]


def serve(
    clip_dir: Path, *, port: Optional[int] = None, local_only: bool = False,
    out: Optional[TextIO] = None,
    ready: Optional[Callable[[PreviewServer], None]] = None,
    addresses: Optional[Callable[[], list[str]]] = None,
) -> int:
    """Serve clip_dir until Ctrl+C; returns the exit status (0).

    With no port, DEFAULT_PORT is tried and then the next ones, up to
    PORT_ATTEMPTS in all. A port given is tried alone, and 0 takes any
    free port. --local-only listens on 127.0.0.1, otherwise on every
    address.
    """
    out = sys.stdout if out is None else out
    fallbacks = bundled_fallbacks(clip_dir)
    server = open_server(
        clip_dir, port=DEFAULT_PORT if port is None else port,
        bind="127.0.0.1" if local_only else "0.0.0.0",
        attempts=PORT_ATTEMPTS if port is None else 1,
        fallbacks=fallbacks, activity=ActivityLog(out),
    )
    try:
        found = [] if local_only else (addresses or lan_addresses)()
        for line in banner(
            clip_dir, server.server_address[1], found,
            local_only=local_only, bundled=tuple(fallbacks), wsl=_is_wsl(),
        ):
            print(line, file=out, flush=True)
        if ready is not None:
            ready(server)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.", file=out, flush=True)
    finally:
        server.server_close()
    return 0
