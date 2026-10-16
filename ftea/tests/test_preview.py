"""Tests for ftea.preview, the LAN preview server.

Every server here listens on 127.0.0.1 only. A LAN bind on Windows can
open a firewall dialog on the desktop, which a test must never do; the
few tests that bind 0.0.0.0 never listen on it (_no_listen).
"""

from __future__ import annotations

import contextlib
import http.client
import io
import os
import socket
import struct
import threading
import time
from pathlib import Path

from urllib.parse import quote, urlsplit

import pytest

from ftea import console, preview
from ftea.encode import package
from ftea.encode.package import PLAYER_FILES

BUNDLE = Path(preview.__file__).resolve().parent / "player"


def _clip(root: Path, name: str = "Café 東京 clip", *, player=True) -> Path:
    """A clip folder shaped like the encoder's output, with tiny files."""
    clip = root / name
    variant = clip / "av1_1080p"
    variant.mkdir(parents=True)
    (clip / "master.m3u8").write_text(
        "#EXTM3U\n#EXT-X-VERSION:7\n", encoding="utf-8",
    )
    (variant / "stream.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    (variant / "init_av1_1080p.mp4").write_bytes(b"init")
    (variant / "av1_1080p_000.m4s").write_bytes(b"segment")
    (clip / "thumbnail.webp").write_bytes(b"webp")
    if player:
        for name in PLAYER_FILES:
            (clip / name).write_bytes(b"own " + name.encode("ascii"))
    return clip


@contextlib.contextmanager
def _serving(clip: Path, **options):
    """Run serve() on a free localhost port in a thread."""
    out = io.StringIO()
    started = threading.Event()
    holder = {}

    def ready(server):
        holder["server"] = server
        started.set()

    kwargs = {
        "port": 0, "local_only": True, "out": out, "ready": ready,
        "addresses": lambda: [],
    }
    kwargs.update(options)
    thread = threading.Thread(
        target=preview.serve, args=(clip,), kwargs=kwargs, daemon=True,
    )
    thread.start()
    assert started.wait(10), "the preview server did not start"
    server = holder["server"]
    try:
        yield server.server_address[1], out
    finally:
        server.shutdown()
        thread.join(10)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _stopped_at_once(monkeypatch):
    """serve_forever raises Ctrl+C at once, so serve() returns."""
    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(preview.PreviewServer, "serve_forever", interrupted)


def _no_listen(monkeypatch):
    """Bind without listening. A test may then bind 0.0.0.0, because a
    socket that never listens cannot open a firewall dialog."""
    monkeypatch.setattr(
        preview.PreviewServer, "server_activate", lambda self: None,
    )


def _get(port, path, method="GET", headers=None):
    """(status, headers, body). The headers object ignores case, because
    http.server writes its own Content-type with a lower-case t."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        return response.status, response.headers, body
    finally:
        connection.close()


def _follow(port, path, hops=3):
    """(status, body), following redirects the way a browser would."""
    for _ in range(hops):
        status, headers, body = _get(port, path)
        if status not in (301, 302) or not headers["Location"]:
            return status, body
        path = urlsplit(headers["Location"]).path or "/"
    return status, body


class TestContentTypes:
    @pytest.mark.parametrize("name, expected", [
        *sorted(("clip" + ext, kind) for ext, kind in preview.MIME_TYPES.items()),
        ("CLIP.M3U8", "application/vnd.apple.mpegurl"),
        ("notes.xyz", "application/octet-stream"),
    ])
    def test_should_use_only_the_explicit_table(self, name, expected):
        assert preview.content_type(name) == expected

    def test_should_cover_every_file_the_package_holds(self):
        for ext in (".m3u8", ".m4s", ".mp4", ".html", ".js", ".webp",
                    ".jpg", ".json", ".md", ".txt"):
            assert ext in preview.MIME_TYPES


class TestServing:
    @pytest.mark.parametrize("path, expected", [
        ("/master.m3u8", "application/vnd.apple.mpegurl"),
        ("/av1_1080p/stream.m3u8", "application/vnd.apple.mpegurl"),
        ("/av1_1080p/av1_1080p_000.m4s", "video/iso.segment"),
        ("/av1_1080p/init_av1_1080p.mp4", "video/mp4"),
        ("/thumbnail.webp", "image/webp"),
        ("/player.html", "text/html; charset=utf-8"),
        ("/hls.min.js", "text/javascript; charset=utf-8"),
    ])
    def test_should_serve_the_package_with_explicit_types(
        self, tmp_path, path, expected,
    ):
        with _serving(_clip(tmp_path)) as (port, _):
            status, headers, _ = _get(port, path)
        assert status == 200
        assert headers["Content-Type"] == expected

    def test_should_turn_caching_off(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, _):
            _, headers, _ = _get(port, "/master.m3u8")
        assert headers["Cache-Control"] == "no-store"

    def test_should_send_the_root_to_the_player(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, _):
            status, headers, _ = _get(port, "/")
        assert status == 302
        assert headers["Location"] == "/player.html"

    def test_should_keep_the_query_when_sending_the_root_to_the_player(
        self, tmp_path,
    ):
        with _serving(_clip(tmp_path)) as (port, _):
            status, headers, _ = _get(port, "/?info=1&player=native")
        assert status == 302
        assert headers["Location"] == "/player.html?info=1&player=native"

    def test_should_not_list_folders(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, _):
            status, _, body = _get(port, "/av1_1080p/")
        assert status == 404
        assert b"av1_1080p_000.m4s" not in body

    @pytest.mark.parametrize(
        "path", ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"],
    )
    def test_should_not_serve_files_outside_the_folder(self, tmp_path, path):
        (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
        with _serving(_clip(tmp_path)) as (port, _):
            status, _, body = _get(port, path)
        assert status == 404
        assert b"private" not in body

    @pytest.mark.parametrize("path", [
        "/..%5csecret.txt", "/%2e%2e%5csecret.txt", "/..\\secret.txt",
        "/%2e%2e%5c%2e%2e%5csecret.txt",
    ])
    def test_should_not_serve_files_outside_the_folder_by_backslash(
        self, tmp_path, path,
    ):
        """Windows reads a backslash as a separator too."""
        (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
        with _serving(_clip(tmp_path)) as (port, _):
            status, body = _follow(port, path)
        assert status != 200
        assert b"private" not in body

    def test_should_not_serve_a_file_by_its_drive_path(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("private", encoding="utf-8")
        with _serving(_clip(tmp_path)) as (port, _):
            for form in (str(secret), secret.as_posix()):
                status, body = _follow(port, "/" + quote(form))
                assert status != 200
                assert b"private" not in body

    def test_should_answer_head_requests_without_a_body(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, _):
            status, headers, body = _get(port, "/master.m3u8", "HEAD")
        assert status == 200
        assert headers["Content-Type"] == "application/vnd.apple.mpegurl"
        assert body == b""

    def test_should_serve_the_folders_own_player(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, out):
            _, _, body = _get(port, "/player.html")
        assert body == b"own player.html"
        assert "bundled player" not in out.getvalue()

    def test_should_serve_the_bundled_player_when_the_folder_has_none(
        self, tmp_path,
    ):
        with _serving(_clip(tmp_path, player=False)) as (port, out):
            for name in PLAYER_FILES:
                status, headers, body = _get(port, "/" + name)
                assert status == 200
                assert body == (BUNDLE / name).read_bytes()
                assert headers["Content-Type"] == preview.content_type(name)
                assert headers["Cache-Control"] == "no-store"
            status, headers, body = _get(port, "/player.html", "HEAD")
        assert (status, body) == (200, b"")
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Content-Length"] == str(
            len((BUNDLE / "player.html").read_bytes()),
        )
        assert "bundled player is served" in out.getvalue()

    def test_should_name_only_the_bundled_files_it_serves(self, tmp_path):
        clip = _clip(tmp_path)
        (clip / "hls.js-LICENSE.txt").unlink()
        with _serving(clip) as (port, out):
            _, _, body = _get(port, "/hls.js-LICENSE.txt")
            assert body == (BUNDLE / "hls.js-LICENSE.txt").read_bytes()
            assert _get(port, "/player.html")[2] == b"own player.html"
        assert "no player of its own" not in out.getvalue()
        assert "bundled copy of: hls.js-LICENSE.txt" in out.getvalue()

    def test_should_say_which_bundled_player_file_is_missing(
        self, tmp_path, monkeypatch,
    ):
        partial = tmp_path / "bundle"
        partial.mkdir()
        (partial / "player.html").write_bytes(b"page")
        monkeypatch.setattr(package, "player_dir", lambda: partial)
        clip = _clip(tmp_path / "clips", player=False)
        with pytest.raises(preview.PreviewError, match="hls.min.js"):
            preview.serve(clip, port=0, local_only=True, out=io.StringIO())

    def test_should_stream_a_file_too_large_to_buffer(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(preview, "MAX_BUFFERED_BYTES", 4)
        clip = _clip(tmp_path)
        data = bytes(range(256)) * 16
        (clip / "big.mp4").write_bytes(data)
        with _serving(clip) as (port, _):
            status, headers, body = _get(port, "/big.mp4")
        assert status == 200
        assert body == data
        assert headers["Content-Length"] == str(len(data))

    def test_should_not_print_a_traceback_for_a_malformed_request(
        self, tmp_path, capfd,
    ):
        with _serving(_clip(tmp_path)) as (port, out):
            for raw in (b"GARBAGE\r\n\r\n", b"GET / HTTP/9.9\r\n\r\n"):
                with socket.create_connection(
                    ("127.0.0.1", port), 10,
                ) as connection:
                    connection.sendall(raw)
                    while connection.recv(4096):
                        pass
        text = out.getvalue()
        assert "127.0.0.1 400" in text
        assert "127.0.0.1 505" in text
        assert "server error" not in text
        assert "Traceback" not in capfd.readouterr().err

    def test_should_leave_the_working_folder_alone_and_the_clip_folder_free(
        self, tmp_path,
    ):
        """On Windows a folder that is open or is a process's working
        folder cannot be renamed, and `ftea run --force` renames it."""
        clip = _clip(tmp_path)
        before = os.getcwd()
        with _serving(clip) as (port, _):
            assert _get(port, "/master.m3u8")[0] == 200
            assert os.getcwd() == before
            clip.rename(tmp_path / "renamed while serving")
        assert (tmp_path / "renamed while serving" / "master.m3u8").is_file()

    def test_should_free_the_clip_folder_while_a_download_is_in_flight(
        self, tmp_path,
    ):
        """A phone fetches segments back to back, so `ftea run --force`
        meets a download in flight. On Linux the rename succeeds either
        way; on Windows it fails while the server holds the file open."""
        clip = _clip(tmp_path)
        big = clip / "av1_1080p" / "av1_1080p_001.m4s"
        big.write_bytes(b"\0" * (16 * 1024 * 1024))
        with _serving(clip) as (port, _):
            with socket.create_connection(("127.0.0.1", port), 10) as sock:
                sock.sendall(
                    b"GET /av1_1080p/av1_1080p_001.m4s HTTP/1.1\r\n"
                    b"Host: localhost\r\n\r\n"
                )
                received = 0
                while received < 65536:  # past the headers, into the body
                    chunk = sock.recv(65536)
                    assert chunk, "the server closed the connection"
                    received += len(chunk)
                clip.rename(tmp_path / "renamed mid-download")
        assert (tmp_path / "renamed mid-download" / "master.m3u8").is_file()

    def test_should_print_the_real_port(self, tmp_path):
        with _serving(_clip(tmp_path)) as (port, out):
            pass
        assert f"http://localhost:{port}/player.html" in out.getvalue()

    def test_should_stop_cleanly_on_ctrl_c(self, tmp_path, monkeypatch):
        def interrupted(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(
            preview.PreviewServer, "serve_forever", interrupted,
        )
        out = io.StringIO()
        held = {}
        status = preview.serve(
            _clip(tmp_path), port=0, local_only=True, out=out,
            ready=lambda server: held.update(port=server.server_address[1]),
            addresses=lambda: [],
        )
        assert status == 0
        assert out.getvalue().splitlines()[-1] == "Stopped."
        with socket.socket() as again:
            again.bind(("127.0.0.1", held["port"]))  # the port is free


def _linger_off() -> bytes:
    """SO_LINGER 0: closing the socket resets the connection at once,
    the way a phone abandons a download on a seek."""
    return struct.pack("HH" if os.name == "nt" else "ii", 1, 0)


class TestAbortedRequests:
    """hls.js and Safari cancel a download on every seek and every
    variant switch. That is normal and must not print a traceback."""

    def _server(self, tmp_path):
        out = io.StringIO()
        server = preview.open_server(
            tmp_path, port=0, bind="127.0.0.1", attempts=1,
            fallbacks={}, activity=preview.ActivityLog(out),
        )
        return server, out

    @pytest.mark.parametrize("error", [
        ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
    ])
    def test_should_print_nothing_when_a_client_goes_away(
        self, tmp_path, capsys, error,
    ):
        server, out = self._server(tmp_path)
        try:
            try:
                raise error("the client went away")
            except error:
                server.handle_error(None, ("10.0.0.7", 50000))
        finally:
            server.server_close()
        assert out.getvalue() == ""
        assert capsys.readouterr().err == ""

    def test_should_report_a_real_error_and_keep_its_traceback(
        self, tmp_path, capsys,
    ):
        server, out = self._server(tmp_path)
        try:
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                server.handle_error(None, ("10.0.0.7", 50000))
        finally:
            server.server_close()
        assert out.getvalue().splitlines() == [
            "10.0.0.7 server error: RuntimeError: boom",
        ]
        assert "Traceback" in capsys.readouterr().err

    def test_should_not_print_a_traceback_for_an_aborted_download(
        self, tmp_path, capfd,
    ):
        clip = _clip(tmp_path)
        (clip / "big.mp4").write_bytes(b"\0" * (16 * 1024 * 1024))
        with _serving(clip) as (port, out):
            sock = socket.create_connection(("127.0.0.1", port), 10)
            sock.sendall(
                b"GET /big.mp4 HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            assert sock.recv(4096)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _linger_off())
            sock.close()
            time.sleep(2)  # let the server meet the reset connection
        assert "Traceback" not in capfd.readouterr().err
        assert "server error" not in out.getvalue()


class TestActivityLog:
    def _log(self):
        out = io.StringIO()
        return preview.ActivityLog(out), out

    def test_should_name_a_new_device_once(self):
        log, out = self._log()
        agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 27_0 like Mac OS X)"
        log.record("10.0.0.7", "/player.html", 200, agent=agent)
        log.record("10.0.0.7", "/master.m3u8", 200, agent=agent)
        assert out.getvalue().splitlines() == ["10.0.0.7 connected (iPhone)"]

    def test_should_report_each_change_of_variant(self):
        log, out = self._log()
        for route in ("/av1_1080p/av1_1080p_000.m4s",
                      "/av1_1080p/av1_1080p_001.m4s",
                      "/h264_720p/h264_720p_002.m4s"):
            log.record("10.0.0.7", route, 200)
        assert out.getvalue().splitlines()[1:] == [
            "10.0.0.7 playing av1_1080p", "10.0.0.7 playing h264_720p",
        ]

    def test_should_report_failures_but_not_missing_icons(self):
        log, out = self._log()
        log.record("10.0.0.7", "/favicon.ico", 404)
        # hls.min.js names a source map that is not shipped; a browser
        # with its developer tools open asks for it.
        log.record("10.0.0.7", "/hls.min.js.map", 404)
        log.record("10.0.0.7", "/av1_720p/stream.m3u8", 404)
        assert out.getvalue().splitlines()[1:] == [
            "10.0.0.7 404 /av1_720p/stream.m3u8",
        ]

    def test_should_report_the_first_byte_range_request_of_each_device(
        self,
    ):
        log, out = self._log()
        log.record("10.0.0.7", "/a.mp4", 200, byte_range="bytes=0-1")
        log.record("10.0.0.7", "/a.mp4", 200, byte_range="bytes=2-3")
        log.record("10.0.0.8", "/a.mp4", 200, byte_range="bytes=0-1")
        ranges = [line for line in out.getvalue().splitlines()
                  if "byte range" in line]
        assert ranges == [
            "10.0.0.7 asked for a byte range (bytes=0-1); this server "
            "always sends whole files",
            "10.0.0.8 asked for a byte range (bytes=0-1); this server "
            "always sends whole files",
        ]

    @pytest.mark.parametrize("agent, expected", [
        ("Mozilla/5.0 (Linux; Android 15; Pixel 9)", "Android"),
        ("Mozilla/5.0 (iPad; CPU OS 27_0 like Mac OS X)", "iPad"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Windows"),
        # iPadOS Safari sends a Mac user agent by default.
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 15_0)", "Mac or iPad"),
        ("Mozilla/5.0 (X11; Linux x86_64)", "Linux"),
        ("curl/8.9.1", "unknown device"),
    ])
    def test_should_name_the_device(self, agent, expected):
        assert preview.device_name(agent) == expected

    def test_should_log_what_a_real_device_fetches(self, tmp_path):
        headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 15)"}
        with _serving(_clip(tmp_path)) as (port, out):
            _get(port, "/av1_1080p/av1_1080p_000.m4s", headers=headers)
        assert "127.0.0.1 connected (Android)" in out.getvalue()
        assert "127.0.0.1 playing av1_1080p" in out.getvalue()


class TestLanAddresses:
    def test_should_put_the_routed_address_first(self):
        found = preview.lan_addresses(
            routed=lambda: "192.168.1.20",
            others=lambda: [
                "127.0.0.1", "172.20.0.1", "192.168.1.20", "169.254.3.4",
            ],
        )
        assert found == ["192.168.1.20", "172.20.0.1"]

    def test_should_fall_back_to_the_other_addresses_without_a_route(self):
        found = preview.lan_addresses(
            routed=lambda: None, others=lambda: ["10.0.0.5"],
        )
        assert found == ["10.0.0.5"]

    def test_should_find_nothing_offline(self):
        assert preview.lan_addresses(
            routed=lambda: None, others=lambda: ["127.0.0.1"],
        ) == []

    def test_should_probe_the_route_without_failing(self):
        address = preview._routed_address()
        assert address is None or address.count(".") == 3


class TestPorts:
    """The busy port is held the way Python's own http.server holds one,
    with SO_REUSEADDR. On Windows a second server that also asks for
    SO_REUSEADDR could then bind the same port and take its traffic."""

    def _open(self, tmp_path, port, attempts):
        return preview.open_server(
            tmp_path, port=port, bind="127.0.0.1", attempts=attempts,
            fallbacks={}, activity=preview.ActivityLog(io.StringIO()),
        )

    def test_should_move_to_the_next_port_when_busy(self, tmp_path):
        with socket.socket() as taken:
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            busy = taken.getsockname()[1]
            server = self._open(tmp_path, busy, 5)
            try:
                assert server.server_address[1] != busy
            finally:
                server.server_close()

    def test_should_refuse_a_busy_port_the_user_chose(self, tmp_path):
        with socket.socket() as taken:
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            busy = taken.getsockname()[1]
            with pytest.raises(preview.PreviewError, match="--port"):
                self._open(tmp_path, busy, 1)

    def test_should_not_take_a_port_this_computer_already_serves(
        self, tmp_path, monkeypatch,
    ):
        """On Windows a 0.0.0.0 bind can sit on top of another program's
        127.0.0.1 bind, and localhost then reaches that program."""
        _no_listen(monkeypatch)
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            busy = taken.getsockname()[1]
            try:
                server = preview.open_server(
                    tmp_path, port=busy, bind="0.0.0.0", attempts=1,
                    fallbacks={}, activity=preview.ActivityLog(io.StringIO()),
                )
            except preview.PreviewError:
                return
            server.server_close()
            pytest.fail(f"bound 0.0.0.0:{busy} while 127.0.0.1 holds it")

    @pytest.mark.skipif(
        os.name != "nt", reason="SO_EXCLUSIVEADDRUSE is Windows only; "
        "elsewhere a listening socket already keeps the port",
    )
    def test_should_keep_other_programs_off_its_port(
        self, tmp_path, monkeypatch,
    ):
        """Without SO_EXCLUSIVEADDRUSE, Windows lets another program bind
        127.0.0.1 on top of this server's 0.0.0.0."""
        _no_listen(monkeypatch)
        server = preview.open_server(
            tmp_path, port=0, bind="0.0.0.0", attempts=1,
            fallbacks={}, activity=preview.ActivityLog(io.StringIO()),
        )
        try:
            port = server.server_address[1]
            with socket.socket() as other:
                other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with pytest.raises(OSError):
                    other.bind(("127.0.0.1", port))
        finally:
            server.server_close()

    def test_should_never_try_past_the_last_port(self, tmp_path, monkeypatch):
        tried = []

        def refuse(address, *args, **kwargs):
            tried.append(address[1])
            raise OSError("in use")

        monkeypatch.setattr(preview, "PreviewServer", refuse)
        with pytest.raises(preview.PreviewError, match="65533-65535"):
            self._open(tmp_path, 65533, 10)
        assert tried == [65533, 65534, 65535]


class TestServeOptions:
    """What serve() asks of open_server, recorded without a LAN bind."""

    def _record(self, monkeypatch):
        seen = []
        real = preview.open_server

        def recording(directory, *, port, bind, attempts, **kwargs):
            seen.append({"port": port, "bind": bind, "attempts": attempts})
            return real(
                directory, port=0, bind="127.0.0.1", attempts=1, **kwargs,
            )

        monkeypatch.setattr(preview, "open_server", recording)
        _stopped_at_once(monkeypatch)
        return seen

    def _serve(self, tmp_path, **options):
        asked = []
        options.setdefault("local_only", True)
        status = preview.serve(
            _clip(tmp_path), out=io.StringIO(),
            addresses=lambda: asked.append(True) or ["192.168.1.20"],
            **options,
        )
        return status, asked

    def test_should_try_the_default_port_and_the_next_ones(
        self, tmp_path, monkeypatch,
    ):
        seen = self._record(monkeypatch)
        self._serve(tmp_path)
        assert seen == [{
            "port": preview.DEFAULT_PORT, "bind": "127.0.0.1",
            "attempts": preview.PORT_ATTEMPTS,
        }]

    @pytest.mark.parametrize("port", [9000, 0])
    def test_should_try_a_chosen_port_alone(
        self, tmp_path, monkeypatch, port,
    ):
        seen = self._record(monkeypatch)
        self._serve(tmp_path, port=port)
        assert seen[0]["port"] == port
        assert seen[0]["attempts"] == 1

    def test_should_refuse_a_chosen_port_that_is_busy(
        self, tmp_path, monkeypatch,
    ):
        _stopped_at_once(monkeypatch)  # a wrong success returns, not hangs
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            busy = taken.getsockname()[1]
            with pytest.raises(preview.PreviewError, match="--port"):
                preview.serve(
                    _clip(tmp_path), port=busy, local_only=True,
                    out=io.StringIO(),
                )

    def test_should_listen_on_this_computer_only_when_local_only(
        self, tmp_path, monkeypatch,
    ):
        seen = self._record(monkeypatch)
        _, asked = self._serve(tmp_path, local_only=True)
        assert seen[0]["bind"] == "127.0.0.1"
        assert asked == []

    def test_should_listen_on_every_address_for_the_network(
        self, tmp_path, monkeypatch,
    ):
        seen = self._record(monkeypatch)
        _, asked = self._serve(tmp_path, local_only=False)
        assert seen[0]["bind"] == "0.0.0.0"
        assert asked == [True]

    def test_should_really_bind_the_loopback_address_when_local_only(
        self, tmp_path, monkeypatch,
    ):
        _stopped_at_once(monkeypatch)
        bound = []
        preview.serve(
            _clip(tmp_path), port=0, local_only=True, out=io.StringIO(),
            ready=lambda server: bound.append(server.server_address[0]),
        )
        assert bound == ["127.0.0.1"]

    def test_should_free_the_port_when_the_banner_cannot_be_printed(
        self, tmp_path,
    ):
        port = _free_port()
        console = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        # Keeping the traceback keeps serve()'s frame, and with it the
        # server, alive: only server_close() can free the port now.
        with pytest.raises(UnicodeEncodeError) as info:
            preview.serve(
                _clip(tmp_path), port=port, local_only=True, out=console,
            )
        with socket.socket() as again:
            again.bind(("127.0.0.1", port))  # the port is free
        assert info.traceback


class TestResolveClipDir:
    def test_should_take_a_clip_folder(self, tmp_path):
        clip = _clip(tmp_path)
        assert preview.resolve_clip_dir(str(clip), output_dir=None) == (
            clip.resolve()
        )

    def test_should_take_the_master_playlist(self, tmp_path):
        clip = _clip(tmp_path)
        assert preview.resolve_clip_dir(
            str(clip / "master.m3u8"), output_dir=None,
        ) == clip.resolve()

    def test_should_find_a_clip_name_in_the_output_folder(
        self, tmp_path, monkeypatch,
    ):
        output = tmp_path / "output"
        clip = _clip(output, "my clip")
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
        assert preview.resolve_clip_dir("my clip", output_dir=output) == (
            clip.resolve()
        )

    def test_should_read_a_relative_name_from_the_working_folder_first(
        self, tmp_path, monkeypatch,
    ):
        output = tmp_path / "output"
        _clip(output, "my clip")
        here = _clip(tmp_path / "here", "my clip")
        monkeypatch.chdir(tmp_path / "here")
        assert preview.resolve_clip_dir("my clip", output_dir=output) == (
            here.resolve()
        )

    def test_should_ignore_set_aside_packages_when_none_is_named(
        self, tmp_path,
    ):
        """`ftea run --force` leaves .<clip>.previous beside the clip."""
        output = tmp_path / "output"
        clip = _clip(output, "only")
        _clip(output, ".only.previous")
        assert preview.resolve_clip_dir(None, output_dir=output) == (
            clip.resolve()
        )

    def test_should_pick_the_only_clip_when_none_is_named(self, tmp_path):
        clip = _clip(tmp_path / "output", "only")
        assert preview.resolve_clip_dir(
            None, output_dir=tmp_path / "output",
        ) == clip.resolve()

    @pytest.mark.parametrize("windows, quoted", [
        (True, '"second clip"'), (False, "'second clip'"),
    ])
    def test_should_list_the_choices_when_several_clips_exist(
        self, tmp_path, monkeypatch, windows, quoted,
    ):
        monkeypatch.setattr(console, "_on_windows", lambda: windows)
        _clip(tmp_path / "output", "first")
        _clip(tmp_path / "output", "second clip")
        with pytest.raises(preview.PreviewError) as info:
            preview.resolve_clip_dir(None, output_dir=tmp_path / "output")
        lines = str(info.value).splitlines()
        assert "  ftea preview first" in lines
        assert f"  ftea preview {quoted}" in lines

    @pytest.mark.parametrize("windows", [True, False])
    def test_should_list_the_clips_inside_a_folder_of_clips(
        self, tmp_path, monkeypatch, windows,
    ):
        monkeypatch.setattr(console, "_on_windows", lambda: windows)
        clip = _clip(tmp_path / "output", "first")
        with pytest.raises(preview.PreviewError, match="choose one") as info:
            preview.resolve_clip_dir(
                str(tmp_path / "output"), output_dir=None,
            )
        expected = console.shell_arg(str(clip), windows=windows)
        assert f"  ftea preview {expected}" in str(info.value).splitlines()
        if windows:
            assert "'" not in str(info.value)

    def test_should_say_when_a_folder_holds_no_package(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(preview.PreviewError, match="not a finished clip"):
            preview.resolve_clip_dir(str(tmp_path / "empty"), output_dir=None)

    def test_should_say_when_nothing_is_found(self, tmp_path):
        with pytest.raises(preview.PreviewError, match="was not found"):
            preview.resolve_clip_dir(
                str(tmp_path / "missing"), output_dir=tmp_path,
            )

    def test_should_ask_for_a_clip_without_an_output_folder(self):
        with pytest.raises(preview.PreviewError, match="name the clip"):
            preview.resolve_clip_dir(None, output_dir=None)

    def test_should_say_when_no_clip_is_finished_yet(self, tmp_path):
        with pytest.raises(preview.PreviewError, match="ftea run"):
            preview.resolve_clip_dir(None, output_dir=tmp_path)


class TestBanner:
    def test_should_offer_the_lan_address_and_the_alternatives(self, tmp_path):
        lines = preview.banner(
            tmp_path, 8765, ["192.168.1.20", "172.20.0.1"],
            local_only=False, bundled=(),
            program="C:/Python312/python.exe", windows=True,
        )
        text = "\n".join(lines)
        assert "http://localhost:8765/player.html" in text
        assert (
            "On a phone or tablet on the same Wi-Fi: "
            "http://192.168.1.20:8765/player.html"
        ) in text
        assert "If that does not open, try: http://172.20.0.1:8765/" in text
        assert (
            "if a firewall window appears for C:/Python312/python.exe, "
            "allow access on Private networks"
        ) in text
        assert "see the guide" not in text
        assert (
            "Windows Security > Firewall & network protection > Allow an "
            "app through firewall, and allow C:/Python312/python.exe on "
            "Private networks"
        ) in text
        assert "Anyone on this network" in text
        assert "WSL" not in text
        assert lines[-1] == "Press Ctrl+C to stop."

    def test_should_mention_windows_firewall_on_windows_only(
        self, tmp_path,
    ):
        text = "\n".join(preview.banner(
            tmp_path, 8765, ["192.168.1.20"], local_only=False, bundled=(),
            program="/usr/bin/python3", windows=False,
        ))
        assert "Windows" not in text
        assert "/usr/bin/python3" not in text
        assert "Anyone on this network" in text

    def test_should_say_the_player_is_bundled_only_when_it_is(
        self, tmp_path,
    ):
        whole = "\n".join(preview.banner(
            tmp_path, 8765, [], local_only=True, bundled=PLAYER_FILES,
        ))
        assert "This folder has no player of its own" in whole
        license_only = "\n".join(preview.banner(
            tmp_path, 8765, [], local_only=True,
            bundled=("hls.js-LICENSE.txt",),
        ))
        assert "no player of its own" not in license_only
        assert (
            "Serving the bundled copy of: hls.js-LICENSE.txt"
        ) in license_only

    def test_should_offer_nothing_to_the_network_when_local_only(
        self, tmp_path,
    ):
        text = "\n".join(preview.banner(
            tmp_path, 8765, ["192.168.1.20"], local_only=True, bundled=(),
            windows=True,
        ))
        assert "192.168.1.20" not in text
        assert "Only this computer can open it" in text
        assert "firewall" not in text

    def test_should_say_so_when_no_address_was_found(self, tmp_path):
        text = "\n".join(preview.banner(
            tmp_path, 8765, [], local_only=False, bundled=(),
        ))
        assert "only this computer can open it" in text

    def test_should_warn_that_a_phone_cannot_reach_wsl(self, tmp_path):
        text = "\n".join(preview.banner(
            tmp_path, 8765, ["172.28.1.2"], local_only=False,
            bundled=(), wsl=True,
        ))
        assert (
            "This is WSL: a phone usually cannot reach this address."
        ) in text
        assert "WSL's mirrored networking" in text
        local = "\n".join(preview.banner(
            tmp_path, 8765, [], local_only=True, bundled=(), wsl=True,
        ))
        assert "WSL" not in local
