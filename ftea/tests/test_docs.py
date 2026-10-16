"""The documentation, checked against the code it describes.

The guide and the README quote commands, settings and their defaults,
the FFmpeg sources, the Python range, content types and the encoder's
phase headings, and the notices credit every pinned download. Each of
those also lives in the code, so a change on one side alone fails here.
Standard library only: no network, no GPU, no ffmpeg.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pytest

from ftea import cli
from ftea.encode import preflight
from ftea.encode.config import PipelineConfig
from ftea.encode.package import _codec_string
from ftea.preview import MIME_TYPES, banner
from ftea.tests.test_player import HLS_JS_VERSION

REPO = Path(__file__).resolve().parents[2]
GUIDE = "docs/PROCESS-GUIDE.md"
NOTICES = "THIRD_PARTY_NOTICES.md"
HOW = "docs/HOW-IT-WORKS.md"
# Every document the parametrized tests read. The task that writes a
# document adds it here.
DOCS: tuple[str, ...] = (
    "README.md", GUIDE, HOW, "CONTRIBUTING.md", "ROADMAP.md", NOTICES,
)
# Upper-case names a document may use that are not config.env settings:
# environment variables that ftea, install.sh or pip read.
OTHER_NAMES = frozenset({
    "FTEA_HOME", "FTEA_PYTHON", "FTEA_TEST_CLIP", "HTTPS_PROXY",
    "PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL", "PIP_NO_CACHE_DIR",
})
# Constants a contributor is told to edit, and the file each lives in.
CODE_NAMES = {
    "EXPECTED_CUDA": "ftea/reframe/verify_env.py",
    "HLS_JS_SHA256": "ftea/tests/test_player.py",
    "HLS_JS_VERSION": "ftea/tests/test_player.py",
    "PYTHON_PREFERENCE": "install.sh",
}

_FENCE = re.compile(r"^(`{3,})(\w*)\s*$")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
_SETTING = re.compile(
    r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|REFRAME)(?:=[^`]*)?`"
)
_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=")
_PIN = re.compile(r"\b(?:torch|torchvision|opencv-python)==[\w.+]+")
# A tuning-table row: the key, and its default in backticks. A row
# that reads `(empty)` leaves the second group None: an empty value.
_DEFAULT_ROW = re.compile(
    r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|\s*(?:`([^`]+)`|\(empty\))\s*\|"
)
_TYPE_ROW = re.compile(r"^\|\s*`(\.[a-z0-9]+)`\s*\|\s*`([^`]+)`\s*\|")
# A quoted message or command, which the 79-column wrap may not split.
_LONG_QUOTE = re.compile(r"`[^`]{40,}`")
_EXTERNAL = ("http://", "https://", "mailto:")


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def blocks(text: str) -> Iterator[tuple[str, list[str]]]:
    """Each fenced code block in `text`: its language and its lines."""
    fence: Optional[str] = None
    lang, body = "", []
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None:
            if match:
                fence, lang, body = match.group(1), match.group(2), []
        elif match and match.group(1) == fence and not match.group(2):
            yield lang, body
            fence = None
        else:
            body.append(line)


def prose(text: str) -> list[str]:
    """The lines of `text` outside fenced code blocks."""
    kept: list[str] = []
    fence: Optional[str] = None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None:
            if match and match.group(1) == fence and not match.group(2):
                fence = None
        else:
            kept.append(line)
    return kept


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, everything but
    letters, digits, spaces, hyphens and underscores dropped, then
    spaces turned into hyphens."""
    kept = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return kept.replace(" ", "-")


def anchors(text: str) -> set[str]:
    """Every heading anchor in `text`; a repeated heading gets -1, -2."""
    found: set[str] = set()
    counts: dict[str, int] = {}
    for line in prose(text):
        match = _HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def links(text: str) -> list[str]:
    """Every link and image target in the prose of `text`."""
    return [m.group(1) for line in prose(text) for m in _LINK.finditer(line)]


def ftea_commands(text: str) -> list[list[str]]:
    """The arguments of every `ftea` line in the bash blocks of `text`."""
    found: list[list[str]] = []
    for lang, body in blocks(text):
        for line in body if lang == "bash" else []:
            stripped = line.strip()
            if stripped == "ftea" or stripped.startswith("ftea "):
                found.append(shlex.split(stripped, comments=True)[1:])
    return found


def parse_error(args: list[str]) -> Optional[str]:
    """Why `ftea <args>` would be refused, or None when it parses."""
    try:
        cli.build_parser().parse_args(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            return f"ftea {shlex.join(args)}: exit {exc.code}"
    return None


def config_defaults() -> dict[str, str]:
    """Each key in config.env.example and its value. A commented-out key
    (`# REFRAME_FFMPEG=`) counts too, with an empty value."""
    found: dict[str, str] = {}
    for line in read("config.env.example").splitlines():
        match = re.match(r"^(?:#\s*)?([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def install_sh_value(name: str) -> str:
    match = re.search(rf"^{name}=(.*)$", read("install.sh"), re.MULTILINE)
    assert match, f"install.sh sets no {name}"
    return match.group(1)


def python_range() -> str:
    """install.sh's Python window as the docs write it: "3.10 to 3.14"."""
    versions = install_sh_value("PYTHON_PREFERENCE").strip("()").split()
    ordered = sorted(versions, key=lambda v: [int(p) for p in v.split(".")])
    return f"{ordered[0]} to {ordered[-1]}"


def ffmpeg_sources() -> list[str]:
    """The download pages install.sh names for Windows and for Linux."""
    urls = []
    for name in ("FFMPEG_SOURCE_WINDOWS", "FFMPEG_SOURCE_LINUX"):
        match = re.search(r"https://[^\s']+", install_sh_value(name))
        assert match, f"install.sh's {name} names no https address"
        urls.append(match.group(0))
    return urls


def pinned_models() -> list[dict]:
    """ftea.reframe.provision.MODELS, read without importing the module
    (importing it reconfigures the console streams)."""
    tree = ast.parse(read("ftea/reframe/provision.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODELS"
            for target in node.targets
        ):
            return [
                {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                for call in node.value.elts
            ]
    raise AssertionError("provision.py defines no MODELS")


def reframe_pins() -> list[str]:
    """Every `name==version` pin in the reframe requirements files."""
    pins = []
    for name in ("requirements-torch.txt", "requirements.txt"):
        for raw in read(f"ftea/reframe/{name}").splitlines():
            line = raw.split("#", 1)[0].strip()
            if "==" in line and not line.startswith("-"):
                pins.append(line)
    return pins


class TestHelpers:
    """The parsing below must be right before any document is judged."""

    def test_should_slug_headings_the_way_github_does(self):
        assert slug("1. Requirements") == "1-requirements"
        assert slug("Why AV1 and H.264") == "why-av1-and-h264"
        assert slug("The reframe stage's GPU test") == (
            "the-reframe-stages-gpu-test"
        )

    def test_should_number_a_repeated_heading(self):
        assert anchors("# Notes\n\n## Notes\n") == {"notes", "notes-1"}

    def test_should_skip_headings_and_links_inside_code(self):
        text = (
            "# Real\n[a](here.md)\n```text\n# not a heading\n"
            "[b](nowhere.md)\n```\n"
        )
        assert anchors(text) == {"real"}
        assert links(text) == ["here.md"]

    def test_should_read_commands_from_bash_blocks_only(self):
        text = (
            "```bash\nftea run --force   # again\ncd ftea\n```\n"
            "```text\nftea preview my-clip\n```\n"
        )
        assert ftea_commands(text) == [["run", "--force"]]

    def test_should_tell_a_refused_command_from_an_accepted_one(self):
        assert parse_error(["run", "--force"]) is None
        assert parse_error(["--version"]) is None
        assert parse_error(["run", "--no-such-flag"]) is not None

    def test_should_read_the_shipped_defaults(self):
        defaults = config_defaults()
        assert defaults["VMAF_TARGET_1080"] == "93"
        assert defaults["REFRAME_FFMPEG"] == ""

    def test_should_read_the_python_range_and_sources_from_install_sh(
        self,
    ):
        assert re.fullmatch(r"3\.\d+ to 3\.\d+", python_range())
        sources = ffmpeg_sources()
        assert len(sources) == 2
        assert all(url.startswith("https://") for url in sources)

    def test_should_read_the_pinned_downloads(self):
        models = pinned_models()
        assert {"name", "url", "sha256", "license"} <= set(models[0])
        assert any(pin.startswith("torch==") for pin in reframe_pins())

    def test_should_find_every_constant_the_docs_may_name(self):
        missing = [
            f"{name} in {path}" for name, path in CODE_NAMES.items()
            if not re.search(rf"^\s*{name}\s*=", read(path), re.MULTILINE)
        ]
        assert not missing, missing


@pytest.mark.parametrize("doc", DOCS)
def test_should_resolve_every_relative_link(doc):
    text = read(doc)
    problems = [
        f"a link is split across lines: {line!r}"
        for line in prose(text)
        if line.count("](") != len(_LINK.findall(line))
    ]
    for target in links(text):
        if target.startswith(_EXTERNAL):
            continue
        path, _, anchor = target.partition("#")
        dest = (REPO / doc).parent / path if path else REPO / doc
        if not dest.exists():
            problems.append(f"{target}: no such file")
        elif anchor and dest.suffix == ".md" and anchor not in anchors(
            dest.read_text(encoding="utf-8"),
        ):
            problems.append(f"{target}: no such heading")
    assert not problems, f"{doc}: " + "; ".join(problems)


@pytest.mark.parametrize("doc", DOCS)
def test_should_show_only_commands_the_cli_accepts(doc):
    refused = [
        problem for problem in map(parse_error, ftea_commands(read(doc)))
        if problem is not None
    ]
    assert not refused, f"{doc}: {refused}"


@pytest.mark.parametrize("doc", DOCS)
def test_should_name_only_settings_that_exist(doc):
    text = read(doc)
    named = set(_SETTING.findall(text))
    for lang, body in blocks(text):
        if lang in ("ini", "bash"):
            named.update(
                match.group(1) for match in map(_ASSIGNMENT.match, body)
                if match
            )
    known = set(config_defaults()) | OTHER_NAMES | set(CODE_NAMES)
    unknown = sorted(named - known)
    assert not unknown, f"{doc} names settings that do not exist: {unknown}"


@pytest.mark.parametrize("doc", DOCS)
def test_should_quote_only_the_current_pins(doc):
    stale = sorted(set(_PIN.findall(read(doc))) - set(reframe_pins()))
    assert not stale, f"{doc} quotes {stale}; the pins: {reframe_pins()}"


@pytest.mark.parametrize("doc", DOCS)
def test_should_wrap_prose_at_79_columns(doc):
    """Prose wraps at 79 columns. Exempt: table rows, lines with a link
    or a URL, and quoted messages or commands (_LONG_QUOTE)."""
    wide = [
        line for line in prose(read(doc))
        if len(line) > 79 and not line.lstrip().startswith("|")
        and "](" not in line and "://" not in line
        and not _LONG_QUOTE.search(line)
    ]
    assert not wide, f"{doc} has prose wider than 79 columns: {wide}"


class TestThirdPartyNotices:
    def test_should_credit_every_pinned_model(self):
        text = read(NOTICES)
        missing = [
            f"{model['name']}: {field}"
            for model in pinned_models()
            for field in ("name", "url", "license")
            if str(model[field]) not in text
        ]
        assert not missing, missing

    def test_should_credit_every_pinned_reframe_package(self):
        text = read(NOTICES)
        missing = [pin for pin in reframe_pins() if pin not in text]
        assert not missing, missing

    def test_should_credit_the_vendored_hls_js(self):
        assert f"hls.js {HLS_JS_VERSION}" in read(NOTICES)


class TestContributing:
    def test_should_document_the_commands_ci_runs(self):
        text = read("CONTRIBUTING.md")
        workflow = read(".github/workflows/tests.yml")
        for command in (
            'python -m pip install -e ".[test]"', "python -m pytest",
        ):
            assert command in workflow, f"tests.yml no longer runs {command}"
            assert command in text, command

    def test_should_state_the_no_agpl_rule(self):
        text = read("CONTRIBUTING.md")
        assert "AGPL" in text and "ultralytics" in text

    def test_should_explain_the_reframe_gpu_test(self):
        text = read("CONTRIBUTING.md")
        assert "FTEA_TEST_CLIP" in text and "pip install pytest" in text
        assert "TestGpuEndToEnd" in read(
            "ftea/reframe/tests/test_detect.py",
        )


class TestRoadmap:
    # Spec: what the first release leaves out, so the roadmap names it.
    LEFT_OUT = (
        "HEVC", "CPU", "Docker", "cloud storage", "web page",
        "shared codebase",
    )

    def test_should_list_what_this_release_leaves_out(self):
        text = read("ROADMAP.md")
        missing = [item for item in self.LEFT_OUT if item not in text]
        assert not missing, missing

    def test_should_point_feature_requests_at_the_roadmap(self):
        template = read(".github/ISSUE_TEMPLATE/feature_request.md")
        assert "ROADMAP.md" in template


_PHASE_CALL = re.compile(r"_print_phase\((\d+), (.+)\)")
_PHASE_LINE = re.compile(r"^PHASE (\d+): (.+)$")


class TestProcessGuide:
    def test_should_be_where_the_code_points(self):
        assert preflight.GUIDE == GUIDE
        assert install_sh_value("GUIDE") == f'"{GUIDE}"'
        assert (REPO / GUIDE).is_file()

    def test_should_name_the_ffmpeg_sources_install_sh_names(self):
        text = read(GUIDE)
        missing = [url for url in ffmpeg_sources() if url not in text]
        assert not missing, missing

    def test_should_state_the_python_range_install_sh_accepts(self):
        assert python_range() in read(GUIDE)

    def test_should_quote_the_shipped_defaults(self):
        defaults = config_defaults()
        rows = [
            (match.group(1), match.group(2) or "")
            for match in map(_DEFAULT_ROW.match, prose(read(GUIDE)))
            if match
        ]
        assert rows, "the guide has no settings table"
        wrong = [
            f"{key}: {value!r} in the guide, {defaults.get(key)!r} shipped"
            for key, value in rows if defaults.get(key) != value
        ]
        assert not wrong, wrong

    def test_should_print_the_phases_the_encoder_prints(self):
        printed = {
            (number, title)
            for number, rest in _PHASE_CALL.findall(
                read("ftea/encode/main.py"),
            )
            for title in re.findall(r'"([^"]+)"', rest)
        }
        quoted = {
            match.groups()
            for _, body in blocks(read(GUIDE))
            for match in map(_PHASE_LINE.match, body) if match
        }
        assert quoted, "the guide quotes no phase headings"
        assert quoted <= printed, sorted(quoted - printed)

    def test_should_list_the_content_types_the_preview_server_sends(self):
        rows = [
            match.groups() for match in map(_TYPE_ROW.match,
                                            prose(read(GUIDE)))
            if match
        ]
        assert rows, "the guide has no content-type table"
        wrong = [
            f"{ext}: {kind}" for ext, kind in rows
            if MIME_TYPES.get(ext) != kind
        ]
        assert not wrong, wrong


def code_constant(name: str) -> int:
    """A whole-number setting or constant, from config.env.example,
    install.sh or a module under ftea/, in that order."""
    shipped = config_defaults().get(name, "")
    if shipped.isdigit():
        return int(shipped)
    sources = [REPO / "install.sh", *sorted((REPO / "ftea").rglob("*.py"))]
    pattern = re.compile(rf"^{name}\s*=\s*(\d+)\b", re.MULTILINE)
    for path in sources:
        match = pattern.search(path.read_text(encoding="utf-8"))
        if match:
            return int(match.group(1))
    raise AssertionError(f"no whole-number {name} in the code")


def quoted_constants() -> list[tuple[str, str]]:
    """Each phrase that quotes a code value, and the document it is in."""
    c = code_constant
    how = "docs/HOW-IT-WORKS.md"
    low = c("AV1_START_CQ_1080") - c("CQ_BRACKET")
    high = c("AV1_START_CQ_1080") + c("CQ_BRACKET")
    minutes = f"{c('MAX_CLIP_SEC') // 60} minutes"
    floor = f"`VMAF_FLOOR` ({c('VMAF_FLOOR')})"
    return [
        (GUIDE, f"about {c('MIN_FREE_GB')} GB"),
        ("README.md", f"about {c('MIN_FREE_GB')} GB"),
        (GUIDE, f"older than version {c('MIN_DRIVER_MAJOR')}"),
        (GUIDE, f"the default port is {c('DEFAULT_PORT')}"),
        (GUIDE, f"longer than {minutes}"),
        ("README.md", f"up to {minutes}"),
        (GUIDE, floor),
        (how, floor),
        (how, f"target of {c('VMAF_TARGET_360')}"),
        (how, f"CQ {low} and CQ {high}"),
        (how, f"CQ {c('REFRAME_CQ')}"),
        (GUIDE, f"`REFRAME_TIMEOUT` | `{c('REFRAME_TIMEOUT')}`"),
        (GUIDE, f"is older than {c('MIN_DRIVER_MAJOR')}."),
        (GUIDE, f"up to {spell(WORDS, c('PORT_ATTEMPTS'))} ports"),
        (GUIDE, f"up to {spell(WORDS, c('RETRY_MAX'))} more attempts"),
        (GUIDE, f"the {spell(WORDS, c('MAX_KEPT_FAILURES'))} newest"),
        (GUIDE, f"{c('DEFAULT_HEIGHT')} pixels tall"),
        (GUIDE, f"(currently {c('REFRAME_BATCH_SIZE')})"),
        (how, f"every {spell(ORDINALS, c('REFRAME_STRIDE'))} frame"),
        (how, f"on {spell(WORDS, c('REFRAME_TITLE_PROBE_FRAMES'))} frames"),
        (how, f"minus and plus {c('CQ_BRACKET')}"),
        (how, f"longer than {c('MAX_SCENE_SEC')} seconds"),
    ] + _quoted_settings() + _quoted_install_facts()


# Small whole numbers and seconds as the documents spell them.
WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten",
}
ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
SECONDS = {1.0: "a second"}
FULL_SECONDS = {1.0: "a full second"}
TIMES = {2.0: "double"}


def spell(table: dict, value: object) -> str:
    assert value in table, f"no words for {value!r}: update the docs"
    return table[value]


def setting(name: str) -> str:
    """A setting's shipped value in config.env.example, as written."""
    value = config_defaults().get(name)
    assert value, f"config.env.example ships no {name}"
    return value


def percent(name: str) -> str:
    """A fraction setting as the documents write it: 0.2 is "20%"."""
    return f"{round(float(setting(name)) * 100)}%"


def _quoted_settings() -> list[tuple[str, str]]:
    """The reframe and quality settings HOW-IT-WORKS quotes as values."""
    how = HOW
    person = setting("REFRAME_MIN_CONFIDENCE_PERSON")
    face = setting("REFRAME_MIN_CONFIDENCE_FACE")
    margin = percent("REFRAME_HYSTERESIS_MARGIN")
    hold = spell(FULL_SECONDS, float(setting("REFRAME_HYSTERESIS_SEC")))
    merge = spell(SECONDS, float(setting("MIN_SCENE_SEC")))
    weight = spell(TIMES, float(setting("REFRAME_FACE_WEIGHT")))
    return [
        (how, f"{person} for people and {face} for faces"),
        (how, f"at least {margin} higher for {hold}"),
        (how, f"middle {percent('REFRAME_DEAD_ZONE')} of the window"),
        (how, f"at least {percent('REFRAME_TITLE_MIN_HEIGHT')} of the "
              "frame's height"),
        (how, f"faces counting {weight}"),
        (how, f"scenes shorter than {merge}"),
        (how, f"cap is `{setting('AV1_MAXRATE_1080')}`"),
    ]


def _quoted_install_facts() -> list[tuple[str, str]]:
    """The CUDA version and the environment's size, which install.sh
    and verify_env.py state and the documents repeat."""
    cuda = re.search(
        r'^EXPECTED_CUDA = "([\d.]+)"', read("ftea/reframe/verify_env.py"),
        re.MULTILINE,
    )
    size = re.search(r"about ([\d.]+ GB) on Windows", read("install.sh"))
    assert cuda and size, "verify_env.py or install.sh changed shape"
    version, gb = cuda.group(1), size.group(1)
    return [
        (GUIDE, f"PyTorch's CUDA {version} package index"),
        (GUIDE, f"copy of the CUDA {version} index"),
        (GUIDE, f"not {version}"),
        (GUIDE, f"about {gb} on Windows"),
        (HOW, f"about {gb} on Windows"),
    ]


def _flat(text: str) -> str:
    """`text` in lower case with every run of whitespace one space, so a
    phrase the hard wrap split across lines still matches."""
    return " ".join(text.split()).lower()


class TestQuotedConstants:
    def test_should_quote_the_values_the_code_holds(self):
        missing = [
            f"{doc}: {phrase!r}" for doc, phrase in quoted_constants()
            if doc in DOCS and _flat(phrase) not in _flat(read(doc))
        ]
        assert not missing, missing


# Each program message the guide quotes in section 10, as an anchor (a
# literal piece of its text, inside one string of the code) and the file
# that prints it. A new or reworded message there gets its anchor here.
# The anchor only picks the file: every literal piece of the quoted
# message must then be text that one of that file's strings prints.
MESSAGES = {
    "is not on PATH. Install an FFmpeg build": "ftea/encode/preflight.py",
    "encoder. ftea needs an FFmpeg built with NVIDIA NVENC":
        "ftea/encode/preflight.py",
    "has no libvmaf filter, which ftea uses": "ftea/encode/preflight.py",
    "has no software AV1 decoder (libdav1d or ": "ftea/encode/preflight.py",
    "encoder for THUMB_FORMAT=": "ftea/encode/preflight.py",
    "the GPU could not start an NVENC encode": "ftea/encode/preflight.py",
    "the GPU could not start an AV1 encode": "ftea/encode/preflight.py",
    "this FFmpeg does not know": "ftea/encode/preflight.py",
    "starts, but not with ftea's encode settings":
        "ftea/encode/preflight.py",
    "torch is a CPU-only build": "ftea/reframe/provision.py",
    "torch is built for CUDA": "ftea/reframe/provision.py",
    " pins ": "ftea/reframe/provision.py",
    "detection failed on a batch of": "ftea/reframe/detect.py",
    "install.sh needs bash.": "install.sh",
    "is on a Windows drive, and you are running Linux (WSL).": "install.sh",
    "was created on Windows, and this is Linux.": "install.sh",
    "no usable Python found. ftea needs a standard 64-bit Python":
        "install.sh",
    "cannot create virtual environments (its ensurepip or venv":
        "install.sh",
    "nvidia-smi not found.": "install.sh",
    "nvidia-smi found no NVIDIA GPU.": "install.sh",
    "is older than": "install.sh",
    "characters long and Windows long-path support is off": "install.sh",
    "GB free here": "install.sh",
    "not on PATH:": "install.sh",
    "pointer file instead of the": "ftea/reframe/provision.py",
    "bytes with SHA-256": "ftea/reframe/provision.py",
    "could not download": "ftea/reframe/provision.py",
    "has no working pip (an earlier run was probably interrupted":
        "install.sh",
    "torch did not pass its check": "install.sh",
    "another ftea run (process": "ftea/encode/main.py",
    "is not a known setting and is ignored.": "ftea/encode/main.py",
    "config.env: ": "ftea/encode/main.py",
    "the reframe environment check failed:": "ftea/encode/main.py",
    "timed out": "ftea/encode/main.py",
    "Close what is using": "ftea/encode/main.py",
    "SKIP (same name as": "ftea/encode/main.py",
    "holds the results of": "ftea/encode/main.py",
    "has no finished clips yet": "ftea/preview.py",
    "could not listen on port": "ftea/preview.py",
    "This browser cannot play this video.": "ftea/player/player.html",
}
_SPAN = re.compile(r"``\s?(.+?)\s?``|`([^`]+)`")


def section(text: str, number: int) -> str:
    """`text` from the `## <number>. ` heading to the next `## ` one."""
    start = text.index(f"\n## {number}. ")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def printed_strings(rel: str) -> list[str]:
    """The strings a file can print. Python is read with ast, which joins
    literals split across lines and ends a piece at each f-string field;
    install.sh and player.html are read as raw text."""
    text = read(rel)
    if not rel.endswith(".py"):
        return [text]
    return [
        node.value for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def quoted_messages(text: str) -> list[str]:
    """Each line of the `text` code blocks in `text`, and each code span
    in the first column of a "The message starts with" table."""
    found = [
        line for lang, body in blocks(text) if lang == "text"
        for line in body
    ]
    in_table = False
    for line in prose(text):
        if line.startswith("| The message starts with"):
            in_table = True
        elif in_table and line.startswith("|"):
            cell = line.split("|")[1]
            found += [a or b for a, b in _SPAN.findall(cell)]
        else:
            in_table = False
    return found


# The messages sections 3 to 5 quote in prose, by the same kind of
# anchor, and the files whose strings print section 4's sample run.
EARLIER_MESSAGES = {
    "ftea is made for short clips": "ftea/encode/main.py",
    "was made from a different": "ftea/encode/main.py",
    "another ftea run (process": "ftea/encode/main.py",
    "Nothing to reframe:": "ftea/cli.py",
}
# The sample is a real run, so it also shows the probe stage's lines and
# the reframe stage's, which the encoder passes through as it gets them.
SAMPLE_RUN = (
    "ftea/encode/main.py", "ftea/cli.py", "ftea/encode/probe.py",
    "ftea/reframe/main.py", "ftea/reframe/probe.py",
    "ftea/reframe/pipeline.py",
)
# A quoted line leaves out what varies as <a placeholder> or an
# ellipsis, and may start with the status mark a helper prints.
_GAP = re.compile("<[^<>]+>|\u2026")
_MARK = re.compile("^\\s*[\u2717\u2713!]\\s+")
# Where a printed string takes a value: an f-string field, a .format or
# % field in Python, a variable or a command in install.sh.
_PY_FIELD = re.compile(r"\{[^{}]*\}|%[-+ #0-9.]*[sdrf]")
_SH_FIELD = re.compile(r"\$\{[^}]*\}|\$\([^)]*\)|\$\w+|%[-0-9.]*[sd]")
_QUOTED = re.compile(r""""((?:[^"\\]|\\.)*)"|'([^']*)'""")
_BACKSLASH = re.compile(r'\\(["\\$`])')
# A value that needs no checking: one word, or, between two literal
# pieces, two plain words (such as "ffmpeg ffprobe").
_WORD = re.compile(r"\S*")
_PLAIN = re.compile(r"\S*|[\w.+/-]+ [\w.+/-]+")
Template = tuple[Optional[str], ...]


def _template(text: str, field: re.Pattern) -> Template:
    """`text` as literal pieces, with None where a value goes."""
    parts: list[Optional[str]] = []
    for index, literal in enumerate(field.split(text)):
        if index:
            parts.append(None)
        if literal:
            parts.append(literal)
    return tuple(parts)


@lru_cache(maxsize=None)
def templates(rel: str) -> tuple[Template, ...]:
    """Every string `rel` can print, as a template: Python's string
    literals, and the quoted strings of install.sh and player.html."""
    text = read(rel)
    if not rel.endswith(".py"):
        return tuple(
            _template(_BACKSLASH.sub(r"\1", double or single), _SH_FIELD)
            for line in text.splitlines()
            if not line.lstrip().startswith("#")
            for double, single in _QUOTED.findall(line)
        )
    found: list[Template] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.JoinedStr):
            found.append(tuple(
                v.value if isinstance(v, ast.Constant) else None
                for v in node.values
            ))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(_template(node.value, _PY_FIELD))
    return tuple(found)


def _edge(part: Optional[str], start: bool) -> str:
    """A part cut into by the edge of the text: a value, or the end
    (`start`) or the beginning of a literal piece."""
    if part is None:
        return "(.*?)"
    cuts = (
        [part[k:] for k in range(len(part))] if start
        else [part[:k] for k in range(len(part), 0, -1)]
    )
    return "(?:" + "|".join(map(re.escape, cuts)) + ")"


def _inner(parts: Template) -> str:
    return "".join(
        "(.*?)" if part is None else re.escape(part) for part in parts
    )


def _pattern(parts: Template, cut_start: bool, cut_end: bool) -> str:
    """`parts` in order, each value captured so that it can be checked;
    the text may begin inside the first part or end inside the last."""
    if len(parts) == 1 and parts[0] is not None:
        if cut_start or cut_end:
            return _edge(parts[0], cut_start)
        return re.escape(parts[0])
    head = _edge(parts[0], True) if cut_start else _inner(parts[:1])
    tail = _edge(parts[-1], False) if cut_end else _inner(parts[-1:])
    return head + _inner(parts[1:-1]) + tail


@lru_cache(maxsize=None)
def _spans(template: Template) -> tuple[tuple[re.Pattern, bool, bool], ...]:
    """A pattern for each run of two or more parts of `template`, and
    whether the run starts and ends with a value."""
    return tuple(
        (
            re.compile(_pattern(template[i:j + 1], True, True), re.DOTALL),
            template[i] is None, template[j] is None,
        )
        for i in range(len(template))
        for j in range(i + 1, len(template))
    )


@lru_cache(maxsize=None)
def _fills(template: Template, kind: str) -> tuple[re.Pattern, ...]:
    """Patterns for a value that is all of a string `template` prints,
    its beginning ("prefix"), or its end ("suffix"). A run must hold a
    letter of its own, so that a bare join such as "{a}: {b}" cannot
    stand in for the message being checked."""
    n = len(template)
    if kind == "prefix":
        runs = [(template[:j + 1], False, True) for j in range(n)]
    elif kind == "suffix":
        runs = [(template[i:], True, False) for i in range(n)]
    else:
        runs = [(template, False, False)] if n else []
    return tuple(
        re.compile(_pattern(*run), re.DOTALL) for run in runs
        if any(part and re.search("[A-Za-z]", part) for part in run[0])
    )


def _printed(piece: str, rel: str) -> bool:
    """Whether `piece` is text that one string of `rel` prints, with a
    value in each field the piece crosses."""
    for template in templates(rel):
        if any(part is not None and piece in part for part in template):
            return True
        for pattern, head, tail in _spans(template):
            match = pattern.fullmatch(piece)
            if match and _values_fit(match.groups(), head, tail, rel):
                return True
    return False


def _values_fit(
    values: tuple[str, ...], head: bool, tail: bool, rel: str,
) -> bool:
    """Each value is plain: one word, or two plain words between two
    literal pieces. One value may instead be another string `rel`
    prints, with plain values of its own: a message built from a
    message. That string is cut short only where the quoted text is: a
    value at the text's end may be the string's beginning, and a value
    at its start the string's end."""
    last = len(values) - 1
    edges = {k for k, edge in ((0, head), (last, tail)) if edge}
    odd = [
        k for k, value in enumerate(values)
        if not (_WORD if k in edges else _PLAIN).fullmatch(value)
    ]
    if not odd:
        return True
    if len(odd) > 1:
        return False
    k = odd[0]
    kind = (
        "suffix" if k == 0 and head
        else "prefix" if k == last and tail
        else "whole"
    )
    return any(
        (match := pattern.fullmatch(values[k])) is not None
        and all(_PLAIN.fullmatch(value) for value in match.groups())
        for template in templates(rel) if _may_fill(values[k], template, kind)
        for pattern in _fills(template, kind)
    )


def _may_fill(value: str, template: Template, kind: str) -> bool:
    """A cheap test that rules most strings out before any pattern is
    compiled: where the string is not cut, it starts or ends the value
    (or, cut at the other end, the value starts or ends it)."""
    if not template:
        return False
    first, last = template[0], template[-1]
    starts = first is None or value.startswith(first) or (
        kind == "prefix" and first.startswith(value)
    )
    ends = last is None or value.endswith(last) or (
        kind == "suffix" and last.endswith(value)
    )
    return (kind == "suffix" or starts) and (kind == "prefix" or ends)


def unprinted(message: str, rels: tuple[str, ...]) -> list[str]:
    """The literal pieces of a quoted `message` that none of `rels`
    prints. A piece with no letter in it is not checked."""
    pieces = [
        piece.strip() for piece in _GAP.split(_MARK.sub("", message))
        if re.search(r"[A-Za-z]", piece)
    ]
    return [
        piece for piece in pieces
        if not any(_printed(piece, rel) for rel in rels)
    ]


def spans(text: str) -> list[str]:
    """Every code span in the prose of `text`."""
    return [a or b for line in prose(text) for a, b in _SPAN.findall(line)]


class TestQuotedOutput:
    def test_should_quote_the_banner_preview_prints(self):
        samples = [
            body for lang, body in blocks(section(read(GUIDE), 6))
            if lang == "text"
        ]
        assert samples, "section 6 quotes no banner"
        printed = banner(
            Path(r"C:\ftea\output\my-clip"), 8765, ["192.168.1.20"],
            local_only=False, program="C:/Python312/python.exe",
            windows=True,
        )
        assert samples[0] == printed

    def test_should_anchor_each_message_in_the_file_that_prints_it(self):
        anchors = {**MESSAGES, **EARLIER_MESSAGES}
        missing = [
            f"{anchor!r} in {rel}" for anchor, rel in anchors.items()
            if not any(anchor in s for s in printed_strings(rel))
        ]
        assert not missing, missing

    def test_should_anchor_every_quoted_message(self):
        quoted = quoted_messages(section(read(GUIDE), 10))
        assert quoted, "section 10 quotes no messages"
        loose = [
            message for message in quoted
            if not any(anchor in message for anchor in MESSAGES)
        ]
        assert not loose, loose

    def test_should_quote_every_message_as_the_code_prints_it(self):
        wrong = []
        for message in quoted_messages(section(read(GUIDE), 10)):
            files = tuple(
                rel for anchor, rel in MESSAGES.items() if anchor in message
            )
            if not files or all(unprinted(message, (rel,)) for rel in files):
                wrong.append(f"{message!r}: {unprinted(message, files)}")
        assert not wrong, wrong

    def test_should_quote_the_messages_of_sections_3_to_5(self):
        text = read(GUIDE)
        quoted = [s for n in (3, 4, 5) for s in spans(section(text, n))]
        missing = sorted(
            anchor for anchor in EARLIER_MESSAGES
            if not any(anchor in span for span in quoted)
        )
        wrong = [
            f"{span!r}: {unprinted(span, (rel,))}"
            for anchor, rel in EARLIER_MESSAGES.items()
            for span in quoted if anchor in span and unprinted(span, (rel,))
        ]
        assert not missing and not wrong, (missing, wrong)

    def test_should_print_the_sample_run_the_guide_shows(self):
        lines = [
            line for lang, body in blocks(section(read(GUIDE), 4))
            if lang == "text" for line in body
        ]
        assert lines, "section 4 shows no sample run"
        wrong = [
            f"{line!r}: {unprinted(line, SAMPLE_RUN)}" for line in lines
            if unprinted(line, SAMPLE_RUN)
        ]
        assert not wrong, wrong


class TestHowItWorks:
    def test_should_name_the_vmaf_model_the_encoder_scores_with(self):
        model = re.search(
            r"model=version=(\w[\w.]*)", read("ftea/encode/probe.py"),
        )
        assert model, "probe.py names no VMAF model"
        assert f"`{model.group(1)}`" in read(HOW)

    def test_should_quote_the_codecs_strings_the_packager_writes(self):
        text = read(HOW)
        strings = {
            _codec_string(variant)
            for variant in PipelineConfig().build_variants("portrait")
        }
        missing = sorted(s for s in strings if f"`{s}`" not in text)
        assert not missing, missing


class TestReadme:
    LINKED = {
        "docs/PROCESS-GUIDE.md", "docs/HOW-IT-WORKS.md", "ROADMAP.md",
        "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md", "LICENSE",
    }

    def test_should_link_every_document_and_the_license(self):
        targets = {t.partition("#")[0] for t in links(read("README.md"))}
        missing = sorted(self.LINKED - targets)
        assert not missing, missing

    def test_should_state_the_python_range_install_sh_accepts(self):
        assert python_range() in read("README.md")

    def test_should_clone_the_repository_pyproject_names(self):
        """One source for the URL. pyproject is read with a regex
        because tomllib is newer than the 3.10 floor."""
        repo = re.search(
            r'^Repository\s*=\s*"([^"]+)"', read("pyproject.toml"),
            re.MULTILINE,
        )
        assert repo, "pyproject.toml names no Repository URL"
        clones = [
            line.strip().split()[2]
            for doc in DOCS
            for lang, body in blocks(read(doc)) if lang == "bash"
            for line in body if line.strip().startswith("git clone ")
        ]
        assert clones, "no document shows the clone command"
        wrong = [url for url in clones if url != repo.group(1) + ".git"]
        assert not wrong, wrong


def test_should_check_all_six_documents():
    six = (TestReadme.LINKED - {"LICENSE"}) | {"README.md"}
    assert len(DOCS) == len(six) and set(DOCS) == six


IMAGES = "docs/images"
# Published images keep their pixels only. EXIF and XMP carry camera,
# software and place details, and no text scan can read them.
_METADATA_CHUNKS = frozenset({b"EXIF", b"XMP "})
MAX_IMAGE_BYTES = 400_000


def webp_chunks(data: bytes) -> list[bytes]:
    """The chunk ids of a RIFF WebP file, in order."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a RIFF WebP file")
    ids, pos = [], 12
    while pos + 8 <= len(data):
        ids.append(data[pos:pos + 4])
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        pos += 8 + size + (size & 1)
    if pos != len(data):
        raise ValueError("a chunk runs past the end of the file")
    return ids


def doc_images() -> list[Path]:
    folder = REPO / IMAGES
    return sorted(folder.iterdir()) if folder.is_dir() else []


def _riff(*chunks: tuple[bytes, bytes]) -> bytes:
    body = b"WEBP" + b"".join(
        cid + len(data).to_bytes(4, "little") + data
        + (b"\0" if len(data) % 2 else b"")
        for cid, data in chunks
    )
    return b"RIFF" + len(body).to_bytes(4, "little") + body


class TestImages:
    def test_should_list_the_chunks_of_a_webp_file(self):
        data = _riff((b"VP8 ", b"abc"), (b"EXIF", b"xy"))
        assert webp_chunks(data) == [b"VP8 ", b"EXIF"]

    def test_should_refuse_a_file_that_is_not_webp(self):
        with pytest.raises(ValueError):
            webp_chunks(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)

    @pytest.mark.parametrize(
        "image", doc_images(), ids=lambda path: path.name,
    )
    def test_should_publish_small_webp_images_without_metadata(
        self, image,
    ):
        data = image.read_bytes()
        assert image.suffix == ".webp", f"{image.name}: use WebP"
        assert len(data) <= MAX_IMAGE_BYTES, f"{image.name}: {len(data)}"
        chunks = set(webp_chunks(data)) & _METADATA_CHUNKS
        found = sorted(chunk.decode() for chunk in chunks)
        assert not found, f"{image.name} carries {found}"
