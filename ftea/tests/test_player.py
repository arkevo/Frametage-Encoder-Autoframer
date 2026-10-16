"""Tests for the bundled player: the vendored hls.js and player.html.

hls.min.js is dist/hls.min.js from the npm package hls.js, byte for
byte, and hls.js-LICENSE.txt is that package's LICENSE. Version 1.7.3 was
published to npm on 2026-09-11. It was picked as the newest 1.x release at
least 14 days old, and its GitHub release v1.7.3 was confirmed. To update
it, pick the version by the same rule from the npm registry, check the
tarball against the registry's integrity hash, copy the two files here,
and change the two pins below and the date above.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ftea.encode.package import PLAYER_FILES

REPO = Path(__file__).resolve().parents[2]
PLAYER_DIR = REPO / "ftea" / "player"
HLS_JS = PLAYER_DIR / "hls.min.js"
HLS_LICENSE = PLAYER_DIR / "hls.js-LICENSE.txt"
PLAYER = PLAYER_DIR / "player.html"

HLS_JS_VERSION = "1.7.3"
HLS_JS_SHA256 = (
    "a12e7ee1cd64a69dcdb314157e45dafcba705bfb0b1440b7935cb265d374423e"
)


class TestVendoredHlsJs:
    def test_should_be_the_pinned_npm_build(self):
        data = HLS_JS.read_bytes()
        assert hashlib.sha256(data).hexdigest() == HLS_JS_SHA256

    def test_should_report_the_pinned_version(self):
        assert HLS_JS_VERSION.encode("ascii") in HLS_JS.read_bytes()

    def test_should_hold_no_carriage_returns(self):
        """The repository stores text with LF; a CR would be rewritten
        on commit and the bytes would no longer match npm's."""
        assert b"\r" not in HLS_JS.read_bytes()

    def test_should_ship_its_apache_license(self):
        text = HLS_LICENSE.read_text(encoding="utf-8")
        assert "Apache License" in text
        assert "Version 2.0" in text


def _package_data_globs() -> list[str]:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^ftea = \[(.*)\]$", text, re.MULTILINE)
    assert match, "pyproject.toml lost the ftea package-data line"
    return re.findall(r'"([^"]+)"', match.group(1))


class TestPackaging:
    def test_should_package_the_player_files(self):
        globs = _package_data_globs()
        for name in ("player.html", HLS_JS.name, HLS_LICENSE.name):
            assert any(
                fnmatch.fnmatch(f"player/{name}", pattern)
                for pattern in globs
            ), f"player/{name} is not in package-data"


class TestPlayerFiles:
    def test_should_copy_the_page_hls_js_and_its_license(self):
        assert PLAYER_FILES == (
            "player.html", "hls.min.js", "hls.js-LICENSE.txt",
        )

    def test_should_bundle_every_listed_file(self):
        for name in PLAYER_FILES:
            assert (PLAYER_DIR / name).is_file(), name


class TestPlayerPage:
    """Static rules: player.html must work offline from its own folder."""

    def test_should_contact_no_other_site(self):
        text = PLAYER.read_text(encoding="utf-8")
        assert "http:" not in text and "https:" not in text
        assert re.findall(r'src="([^"]+)"', text) == ["hls.min.js"]

    def test_should_play_the_master_playlist_beside_it(self):
        assert '"master.m3u8"' in PLAYER.read_text(encoding="utf-8")

    def test_should_play_inline_and_without_remote_playback(self):
        text = PLAYER.read_text(encoding="utf-8")
        assert "playsinline" in text
        assert "disableRemotePlayback = true" in text

    def test_should_tell_the_viewer_when_nothing_can_play(self):
        assert "This browser cannot play this video" in PLAYER.read_text(
            encoding="utf-8",
        )


NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None, reason="node is not installed; the player logic is "
    "plain JavaScript and is checked where node exists",
)


@pytest.fixture(scope="module")
def logic(tmp_path_factory) -> Path:
    """The player's pure logic block, saved where node can require it."""
    text = PLAYER.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="ftea-logic">(.*?)</script>', text, re.DOTALL,
    )
    assert match, "player.html lost its ftea-logic block"
    path = tmp_path_factory.mktemp("player") / "logic.js"
    path.write_text(match.group(1), encoding="utf-8")
    return path


def _node(logic: Path, expression: str):
    script = (
        f"const f = require({json.dumps(str(logic))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = subprocess.run(
        [NODE, "-e", script], capture_output=True, encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
class TestPlayerLogic:
    @pytest.mark.parametrize("caps, requested, expected", [
        # iPhone Safari with Managed Media Source: both work, hls.js wins.
        ({"hlsjs": True, "native": True}, "", "hlsjs"),
        # Safari without Media Source: native HLS.
        ({"hlsjs": False, "native": True}, "", "native"),
        # Chrome and Firefox.
        ({"hlsjs": True, "native": False}, "", "hlsjs"),
        ({"hlsjs": False, "native": False}, "", "none"),
        ({"hlsjs": True, "native": True}, "native", "native"),
        ({"hlsjs": True, "native": False}, "native", "none"),
        ({"hlsjs": False, "native": True}, "hlsjs", "none"),
        ({"hlsjs": True, "native": True}, "bogus", "hlsjs"),
    ])
    def test_should_choose_the_player_path(
        self, logic, caps, requested, expected,
    ):
        expression = (
            f"f.choosePath({json.dumps(caps)}, {json.dumps(requested)})"
        )
        assert _node(logic, expression) == expected

    @pytest.mark.parametrize("search, expected", [
        ("", "master.m3u8"),
        ("?src=av1_720p/stream.m3u8", "av1_720p/stream.m3u8"),
        ("?info=1&src=h264_360p/stream.m3u8", "h264_360p/stream.m3u8"),
        ("?src=a/../../x.m3u8", "master.m3u8"),
        ("?src=https://example.org/x.m3u8", "master.m3u8"),
        ("?src=//example.org/x.m3u8", "master.m3u8"),
        ("?src=player.html", "master.m3u8"),
        ("?src=%E0%A4%A", "master.m3u8"),
    ])
    def test_should_only_play_a_playlist_in_this_folder(
        self, logic, search, expected,
    ):
        expression = f"f.playlistFrom({json.dumps(search)})"
        assert _node(logic, expression) == expected

    def test_should_read_parameters(self, logic):
        assert _node(logic, 'f.param("?info=1&player=native", "player")') == (
            "native"
        )
        assert _node(logic, 'f.param("?info=1", "player")') == ""

    @pytest.mark.parametrize("codecs, expected", [
        ("av01.0.08M.10.0.110.09", "AV1"),
        ("avc1.640028", "H.264"),
        ("", "unknown codec"),
    ])
    def test_should_name_the_codec_family(self, logic, codecs, expected):
        assert _node(logic, f"f.codecFamily({json.dumps(codecs)})") == (
            expected
        )

    def test_should_say_when_a_variant_was_skipped(self, logic):
        state = {
            "path": "hlsjs", "version": "1.0.0", "listed": 6,
            "playable": 5, "size": "1080x1920", "codecs": "avc1.640028",
        }
        text = _node(logic, f"f.describe({json.dumps(state)})")
        assert text.splitlines() == [
            "player: hls.js 1.0.0",
            "variants: 5 of 6 playable",
            "now: 1080x1920 H.264",
        ]

    def test_should_describe_the_native_path(self, logic):
        text = _node(logic, 'f.describe({"path": "native"})')
        assert text.splitlines() == [
            "player: native HLS", "variant: chosen by the browser",
        ]

    def test_should_describe_a_chosen_and_an_ignored_playlist(self, logic):
        state = {
            "path": "native", "src": "av1_720p/stream.m3u8",
            "ignored": "",
        }
        text = _node(logic, f"f.describe({json.dumps(state)})")
        assert "playlist: av1_720p/stream.m3u8" in text.splitlines()
        state = {"path": "native", "src": "master.m3u8", "ignored": "../x"}
        lines = _node(logic, f"f.describe({json.dumps(state)})").splitlines()
        assert "ignored playlist: ../x" in lines
        assert not any(line.startswith("playlist:") for line in lines)

    @pytest.mark.parametrize("search, expected", [
        ("", ""),
        ("?src=av1_720p/stream.m3u8", ""),
        ("?src=https://example.org/x.m3u8", "https://example.org/x.m3u8"),
        ("?src=./av1_720p/stream.m3u8", "./av1_720p/stream.m3u8"),
        ("?info=1&src=player.html", "player.html"),
    ])
    def test_should_name_a_playlist_it_refused(self, logic, search, expected):
        assert _node(logic, f"f.ignoredSrc({json.dumps(search)})") == expected

    @pytest.mark.parametrize("caps, requested, loaded, expected", [
        # Chrome with hls.min.js missing: not the browser's fault.
        ({"hlsjs": False, "native": False}, "", False, "hls.min.js did not"),
        # ?player=native on Chrome, where hls.js would work.
        ({"hlsjs": True, "native": False}, "native", True,
         "?player=native is not available in this browser"),
        ({"hlsjs": False, "native": True}, "hlsjs", True,
         "?player=hlsjs is not available in this browser"),
        # Nothing at all can play HLS here.
        ({"hlsjs": False, "native": False}, "", True,
         "This browser cannot play this video"),
        ({"hlsjs": False, "native": False}, "native", True,
         "This browser cannot play this video"),
    ])
    def test_should_say_why_nothing_can_play(
        self, logic, caps, requested, loaded, expected,
    ):
        expression = (
            f"f.noPlayerMessage({json.dumps(caps)}, {json.dumps(requested)}, "
            f"{json.dumps(loaded)})"
        )
        assert expected in _node(logic, expression)


# Runs the page's wiring block against a stand-in for hls.js that fires
# the manifest events in the order given, and reports what the page
# shows. hls.js 1.7.3 was measured in Chrome to deliver MANIFEST_PARSED
# to the page before MANIFEST_LOADED: it fires PARSED from inside its
# own handling of LOADED.
_WIRING_HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const params = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const elements = {};
function element(id) {
  if (!elements[id]) {
    elements[id] = {
      textContent: "", poster: "", disableRemotePlayback: false, src: "",
      canPlayType: function () { return params.canPlay; },
      addEventListener: function () {}
    };
  }
  return elements[id];
}
const warnings = [];
const calls = { loaded: [], recovered: 0, destroyed: 0 };
function FakeHls() { this.handlers = {}; this.levels = []; }
FakeHls.isSupported = function () { return params.hlsSupported; };
FakeHls.version = "9.9.9";
FakeHls.Events = {
  MANIFEST_LOADED: "manifestLoaded", MANIFEST_PARSED: "manifestParsed",
  LEVEL_SWITCHED: "levelSwitched", ERROR: "hlsError"
};
FakeHls.ErrorTypes = { MEDIA_ERROR: "mediaError", NETWORK_ERROR: "networkError" };
FakeHls.prototype.on = function (name, fn) {
  (this.handlers[name] = this.handlers[name] || []).push(fn);
};
FakeHls.prototype.fire = function (name, data) {
  (this.handlers[name] || []).forEach(function (fn) { fn(name, data); });
};
FakeHls.prototype.loadSource = function (src) {
  calls.loaded.push(src);
  for (const [name, count] of params.events) {
    const data = { levels: Array.from({ length: count }, () => ({})) };
    this.fire(FakeHls.Events[name], data);
  }
  for (const data of params.errors) {
    this.fire(FakeHls.Events.ERROR, data);
  }
};
FakeHls.prototype.attachMedia = function () {};
FakeHls.prototype.recoverMediaError = function () { calls.recovered++; };
FakeHls.prototype.destroy = function () { calls.destroyed++; };
const context = {
  window: {
    location: { search: params.search },
    Hls: params.hlsMissing ? undefined : FakeHls
  },
  document: { getElementById: element },
  console: { warn: function (text) { warnings.push(String(text)); } },
  Image: function () {}
};
vm.createContext(context);
vm.runInContext(params.logic, context);
vm.runInContext(params.wiring, context);
process.stdout.write(JSON.stringify({
  info: element("info").textContent, warnings: warnings,
  message: element("message").textContent, videoSrc: element("video").src,
  loaded: calls.loaded, recovered: calls.recovered,
  destroyed: calls.destroyed
}));
"""


def _page_with(
    tmp_path: Path, events: list, *, search: str = "?info=1",
    errors: list = (), hls_supported: bool = True, can_play: str = "",
    hls_missing: bool = False,
) -> dict:
    """Run the page's wiring. can_play is what the video element's
    canPlayType answers for HLS ("" in Chrome, "maybe" in Safari)."""
    text = PLAYER.read_text(encoding="utf-8")
    logic = re.search(
        r'<script id="ftea-logic">(.*?)</script>', text, re.DOTALL,
    )
    wiring = re.findall(r"<script>(.*?)</script>", text, re.DOTALL)
    assert logic and len(wiring) == 1, "player.html lost a script block"
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "logic": logic.group(1), "wiring": wiring[0], "events": events,
        "search": search, "errors": list(errors),
        "hlsSupported": hls_supported, "canPlay": can_play,
        "hlsMissing": hls_missing,
    }), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_WIRING_HARNESS, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(harness), str(params)], capture_output=True,
        encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
class TestPlayerWiring:
    @pytest.mark.parametrize("first, second", [
        ("MANIFEST_PARSED", "MANIFEST_LOADED"),  # hls.js 1.7.3
        ("MANIFEST_LOADED", "MANIFEST_PARSED"),
    ])
    def test_should_count_the_variants_whichever_event_comes_first(
        self, tmp_path, first, second,
    ):
        shown = _page_with(tmp_path, [[first, 6], [second, 6]])
        assert shown["info"].splitlines() == [
            "player: hls.js 9.9.9", "variants: 6 of 6 playable",
        ]
        assert shown["warnings"] == []

    @pytest.mark.parametrize("events", [
        [["MANIFEST_PARSED", 3], ["MANIFEST_LOADED", 6]],  # hls.js 1.7.3
        [["MANIFEST_LOADED", 6], ["MANIFEST_PARSED", 3]],
    ])
    def test_should_warn_once_when_the_browser_skips_variants(
        self, tmp_path, events,
    ):
        shown = _page_with(tmp_path, events)
        assert "variants: 3 of 6 playable" in shown["info"].splitlines()
        assert len(shown["warnings"]) == 1
        assert "skipped 3 variant(s)" in shown["warnings"][0]

    @pytest.mark.parametrize("search, expected", [
        ("?src=https://example.org/x.m3u8", "master.m3u8"),
        ("?src=a/../../x.m3u8", "master.m3u8"),
        ("?src=av1_720p/stream.m3u8", "av1_720p/stream.m3u8"),
        ("", "master.m3u8"),
    ])
    def test_should_load_only_a_playlist_inside_the_folder(
        self, tmp_path, search, expected,
    ):
        shown = _page_with(tmp_path, [], search=search)
        assert shown["loaded"] == [expected]
        assert shown["videoSrc"] == ""

    @pytest.mark.parametrize("search, expected", [
        ("?src=//example.org/x.m3u8", "master.m3u8"),
        ("?src=h264_360p/stream.m3u8", "h264_360p/stream.m3u8"),
    ])
    def test_should_give_native_playback_only_a_playlist_inside_the_folder(
        self, tmp_path, search, expected,
    ):
        """Safari without Media Source: the page sets video.src itself."""
        shown = _page_with(
            tmp_path, [], search=search, hls_supported=False,
            can_play="maybe",
        )
        assert shown["videoSrc"] == expected
        assert shown["loaded"] == []

    def test_should_say_which_playlist_it_ignored(self, tmp_path):
        shown = _page_with(
            tmp_path, [["MANIFEST_LOADED", 6], ["MANIFEST_PARSED", 6]],
            search="?info=1&src=https://example.org/x.m3u8",
        )
        assert shown["loaded"] == ["master.m3u8"]
        assert len(shown["warnings"]) == 1
        assert "https://example.org/x.m3u8" in shown["warnings"][0]
        assert (
            "ignored playlist: https://example.org/x.m3u8"
        ) in shown["info"].splitlines()


@needs_node
class TestPlayerErrors:
    MEDIA = {"fatal": True, "type": "mediaError", "details": "bufferAppendError"}
    NETWORK = {
        "fatal": True, "type": "networkError", "details": "manifestLoadError",
    }

    def test_should_recover_from_the_first_fatal_media_error(self, tmp_path):
        shown = _page_with(tmp_path, [], errors=[self.MEDIA])
        assert (shown["recovered"], shown["destroyed"]) == (1, 0)
        assert shown["message"] == ""

    def test_should_give_up_on_a_second_fatal_media_error(self, tmp_path):
        shown = _page_with(tmp_path, [], errors=[self.MEDIA, self.MEDIA])
        assert (shown["recovered"], shown["destroyed"]) == (1, 1)
        assert shown["message"] == "Playback failed: bufferAppendError."

    def test_should_give_up_at_once_on_a_fatal_network_error(self, tmp_path):
        shown = _page_with(tmp_path, [], errors=[self.NETWORK])
        assert (shown["recovered"], shown["destroyed"]) == (0, 1)
        assert shown["message"] == "Playback failed: manifestLoadError."

    def test_should_ignore_an_error_that_is_not_fatal(self, tmp_path):
        shown = _page_with(
            tmp_path, [], errors=[dict(self.MEDIA, fatal=False)],
        )
        assert (shown["recovered"], shown["destroyed"]) == (0, 0)
        assert shown["message"] == ""


@needs_node
class TestPlayerWithoutAPath:
    def test_should_say_when_hls_js_did_not_load(self, tmp_path):
        shown = _page_with(tmp_path, [], hls_missing=True)
        assert "hls.min.js did not load" in shown["message"]
        assert shown["loaded"] == []

    def test_should_say_when_the_requested_player_is_missing(self, tmp_path):
        shown = _page_with(tmp_path, [], search="?player=native")
        assert "?player=native is not available" in shown["message"]
        assert shown["loaded"] == []
