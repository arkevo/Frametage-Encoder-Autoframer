"""Tests for ftea.reframe.provision, the install-time steps.

Nothing here touches the network or imports torch: downloads go through
a fake opener, and torch and torchvision are small stand-in modules.
"""

from __future__ import annotations

import hashlib
import http.client
import inspect
import io
import os
import re
import subprocess
import sys
import types
import urllib.error

import pytest

from ftea import paths
from ftea.reframe import provision, verify_env
from ftea.reframe.detect import face_model_path, torch_home
from ftea.reframe.text import text_model_path

PAYLOAD = b"weights " * 512
OTHER = b"another " * 100
FACE = "face_detection_yunet_2023mar.onnx"
TEXT = "text_detection_en_ppocrv3_2023may.onnx"
PERSON = "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
PINS = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}


def _model(payload=PAYLOAD, *, name="w.onnx", in_torch_hub=False,
           size=None):
    return provision.ModelFile(
        name=name,
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload) if size is None else size,
        license="MIT",
        in_torch_hub=in_torch_hub,
    )


class Server:
    """A fake opener that answers each call with the next response."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, bytes):
            return io.BytesIO(item)
        return item


class Reply(io.BytesIO):
    """A response that declares its length, as the real servers do."""

    def __init__(self, body: bytes, length: int | None = None):
        super().__init__(body)
        declared = len(body) if length is None else length
        self.headers = {"Content-Length": str(declared)}


class Dropping:
    """A response whose connection resets on the first read."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        raise ConnectionResetError("connection reset by peer")


class BreaksAfter:
    """A response that sends `head`, then fails its next read with `error`.

    http.client raises IncompleteRead when a chunked transfer is cut off,
    and a reset or timeout can arrive part-way through any transfer.
    """

    def __init__(self, head: bytes, error: BaseException):
        self.head = head
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if self.head:
            head, self.head = self.head, b""
            return head
        raise self.error


def _no_sleep(seconds):
    return None


def _by_name():
    return {model.name: model for model in provision.MODELS}


class TestManifest:
    def test_should_pin_the_three_files_the_stage_loads(self):
        assert sorted(_by_name()) == sorted([FACE, TEXT, PERSON])

    def test_should_pin_each_file_by_size_and_full_sha256(self):
        for model in provision.MODELS:
            assert re.fullmatch(r"[0-9a-f]{64}", model.sha256), model.name
            assert model.size > 0, model.name
            assert model.url.startswith("https://"), model.name

    def test_should_pin_opencv_zoo_files_to_a_commit_in_the_lfs_form(self):
        zoo = [m for m in provision.MODELS if "opencv_zoo" in m.url]

        assert len(zoo) == 2
        for model in zoo:
            assert re.search(
                r"//github\.com/opencv/opencv_zoo/raw/[0-9a-f]{40}/",
                model.url,
            ), model.url
            assert model.url.endswith("/" + model.name)

    def test_should_cache_only_the_person_detector_for_torchvision(self):
        hub = [m for m in provision.MODELS if m.in_torch_hub]

        assert [m.name for m in hub] == [PERSON]
        assert hub[0].url.rsplit("/", 1)[1] == PERSON
        assert provision.person_detector() is hub[0]

    def test_should_place_each_file_where_the_stage_loads_it(
        self, tmp_path,
    ):
        models = _by_name()

        assert provision.destination(models[FACE], tmp_path) == (
            face_model_path(tmp_path, ""))
        assert provision.destination(models[TEXT], tmp_path) == (
            text_model_path(tmp_path, ""))
        assert provision.destination(models[PERSON], tmp_path) == (
            torch_home(tmp_path) / "hub" / "checkpoints" / PERSON)

    def test_should_name_each_license_as_the_environment_check_does(self):
        source = inspect.getsource(verify_env)
        models = _by_name()

        assert f'"YuNet ({models[FACE].license})"' in source
        assert f'"PP-OCRv3 DB ({models[TEXT].license})"' in source


class TestFetch:
    def test_should_download_and_verify_a_missing_file(self, tmp_path):
        target = tmp_path / "models" / "w.onnx"

        status = provision.fetch(
            _model(), target, opener=Server(PAYLOAD), sleep=_no_sleep)

        assert status == "downloaded"
        assert target.read_bytes() == PAYLOAD
        assert sorted(p.name for p in target.parent.iterdir()) == ["w.onnx"]

    def test_should_keep_a_verified_file_without_downloading(self, tmp_path):
        target = tmp_path / "w.onnx"
        target.write_bytes(PAYLOAD)
        server = Server()

        status = provision.fetch(_model(), target, opener=server)

        assert status == "present"
        assert server.calls == []

    def test_should_replace_a_damaged_copy(self, tmp_path):
        target = tmp_path / "w.onnx"
        target.write_bytes(b"truncated")

        status = provision.fetch(
            _model(), target, opener=Server(PAYLOAD), sleep=_no_sleep)

        assert status == "replaced"
        assert target.read_bytes() == PAYLOAD

    def test_should_refuse_bytes_that_do_not_match_the_pin(self, tmp_path):
        target = tmp_path / "dl" / "w.onnx"

        with pytest.raises(provision.FetchError, match="SHA-256") as info:
            provision.fetch(_model(), target,
                            opener=Server(b"x" * len(PAYLOAD)),
                            sleep=_no_sleep)

        assert "was not used" in str(info.value)
        assert list((tmp_path / "dl").iterdir()) == []

    def test_should_explain_a_git_lfs_pointer(self, tmp_path):
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   b"oid sha256:0123\nsize 2423490\n")

        server = Server(Reply(pointer))

        with pytest.raises(provision.FetchError,
                           match="git-lfs pointer") as info:
            provision.fetch(_model(), tmp_path / "dl" / "w.onnx",
                            opener=server, sleep=_no_sleep)

        assert len(server.calls) == 1
        assert "Try again later" in str(info.value)
        assert "must use the github.com" not in str(info.value)
        assert list((tmp_path / "dl").iterdir()) == []

    def test_should_not_retry_a_complete_reply_of_the_wrong_size(
        self, tmp_path,
    ):
        server = Server(Reply(b"<html>sign in to the hotel wifi</html>"))

        with pytest.raises(provision.FetchError,
                           match="changed upstream") as info:
            provision.fetch(_model(), tmp_path / "dl" / "w.onnx",
                            opener=server, sleep=_no_sleep)

        assert len(server.calls) == 1
        assert "dropped" not in str(info.value)

    def test_should_stop_a_download_larger_than_pinned(self, tmp_path):
        model = _model(size=len(PAYLOAD) - 1)

        with pytest.raises(provision.FetchError, match="more than"):
            provision.fetch(model, tmp_path / "dl" / "w.onnx",
                            opener=Server(PAYLOAD), sleep=_no_sleep)

        assert list((tmp_path / "dl").iterdir()) == []

    def test_should_retry_a_dropped_connection(self, tmp_path):
        server = Server(urllib.error.URLError("getaddrinfo failed"), PAYLOAD)

        status = provision.fetch(_model(), tmp_path / "w.onnx",
                                 opener=server, sleep=_no_sleep)

        assert status == "downloaded"
        assert len(server.calls) == 2

    def test_should_name_the_url_when_every_attempt_fails(self, tmp_path):
        errors = [urllib.error.URLError("no route to host")
                  for _ in range(provision.ATTEMPTS)]
        server = Server(*errors)

        with pytest.raises(provision.FetchError) as info:
            provision.fetch(_model(), tmp_path / "w.onnx",
                            opener=server, sleep=_no_sleep)

        assert "https://example.invalid/w.onnx" in str(info.value)
        assert f"after {provision.ATTEMPTS} attempts" in str(info.value)
        assert len(server.calls) == provision.ATTEMPTS

    def test_should_leave_nothing_behind_when_a_transfer_breaks(
        self, tmp_path,
    ):
        server = Server(*[Dropping() for _ in range(provision.ATTEMPTS)])

        with pytest.raises(provision.FetchError, match="reset"):
            provision.fetch(_model(), tmp_path / "dl" / "w.onnx",
                            opener=server, sleep=_no_sleep)

        assert list((tmp_path / "dl").iterdir()) == []

    @pytest.mark.parametrize("declared", [True, False],
                             ids=["declared-length", "no-length"])
    def test_should_retry_when_the_body_ends_early(self, tmp_path, declared):
        head = PAYLOAD[:100]
        short = Reply(head, length=len(PAYLOAD)) if declared else head
        server = Server(short, PAYLOAD)

        status = provision.fetch(_model(), tmp_path / "w.onnx",
                                 opener=server, sleep=_no_sleep)

        assert status == "downloaded"
        assert (tmp_path / "w.onnx").read_bytes() == PAYLOAD
        assert len(server.calls) == 2

    def test_should_say_where_the_connection_dropped(self, tmp_path):
        server = Server(*[Reply(PAYLOAD[:100], length=len(PAYLOAD))
                          for _ in range(provision.ATTEMPTS)])

        with pytest.raises(provision.FetchError) as info:
            provision.fetch(_model(), tmp_path / "dl" / "w.onnx",
                            opener=server, sleep=_no_sleep)

        message = str(info.value)
        assert f"after {provision.ATTEMPTS} attempts" in message
        assert (f"the connection dropped after 100 of {len(PAYLOAD):,} "
                "bytes") in message
        assert "changed upstream" not in message
        assert list((tmp_path / "dl").iterdir()) == []

    def test_should_retry_an_incomplete_chunked_transfer(self, tmp_path):
        broken = BreaksAfter(
            PAYLOAD[:1000], http.client.IncompleteRead(b"x" * 24, 3072))
        server = Server(broken, PAYLOAD)

        status = provision.fetch(_model(), tmp_path / "w.onnx",
                                 opener=server, sleep=_no_sleep)

        assert status == "downloaded"
        assert len(server.calls) == 2

    def test_should_report_an_incomplete_read_as_a_dropped_connection(
        self, tmp_path,
    ):
        server = Server(*[
            BreaksAfter(PAYLOAD[:1000],
                        http.client.IncompleteRead(b"x" * 24, 3072))
            for _ in range(provision.ATTEMPTS)])

        with pytest.raises(provision.FetchError) as info:
            provision.fetch(_model(), tmp_path / "w.onnx",
                            opener=server, sleep=_no_sleep)

        assert (f"the connection dropped after 1,024 of {len(PAYLOAD):,} "
                "bytes") in str(info.value)

    def test_should_retry_another_http_protocol_error(self, tmp_path):
        server = Server(http.client.BadStatusLine("garbage"), PAYLOAD)

        status = provision.fetch(_model(), tmp_path / "w.onnx",
                                 opener=server, sleep=_no_sleep)

        assert status == "downloaded"
        assert len(server.calls) == 2

    def test_should_restart_the_file_when_a_retry_follows_a_partial_transfer(
        self, tmp_path,
    ):
        broken = BreaksAfter(PAYLOAD[:1000], ConnectionResetError("reset"))
        server = Server(broken, PAYLOAD)

        status = provision.fetch(_model(), tmp_path / "w.onnx",
                                 opener=server, sleep=_no_sleep)

        assert status == "downloaded"
        assert (tmp_path / "w.onnx").read_bytes() == PAYLOAD


@pytest.fixture
def two_models(monkeypatch):
    models = (_model(name="a.onnx"),
              _model(OTHER, name="b.pth", in_torch_hub=True))
    monkeypatch.setattr(provision, "MODELS", models)
    monkeypatch.setattr(provision, "RETRY_DELAY_SEC", 0)
    return models


class TestModelsCommand:
    def test_should_fetch_every_file_into_the_data_root(
        self, two_models, monkeypatch, capsys,
    ):
        monkeypatch.setattr(provision, "_open", Server(PAYLOAD, OTHER))

        assert provision.main(["models"]) == 0

        models = paths.models_dir()
        assert (models / "a.onnx").read_bytes() == PAYLOAD
        hub = torch_home(models) / "hub" / "checkpoints" / "b.pth"
        assert hub.read_bytes() == OTHER
        out = capsys.readouterr().out
        assert f"models folder: {models}" in out
        assert "✓ a.onnx: downloaded, SHA-256 verified" in out

    def test_should_follow_the_models_folder_set_in_config_env(
        self, two_models, monkeypatch,
    ):
        paths.config_path().write_text(
            "REFRAME_MODELS_DIR=weights\n", encoding="utf-8")
        monkeypatch.setattr(provision, "_open", Server(PAYLOAD, OTHER))

        assert provision.main(["models"]) == 0

        assert (paths.data_root() / "weights" / "a.onnx").is_file()

    def test_should_report_each_failed_file_and_exit_1(
        self, two_models, monkeypatch, capsys,
    ):
        monkeypatch.setattr(
            provision, "_open", Server(b"x" * len(PAYLOAD), OTHER))

        assert provision.main(["models"]) == 1

        out = capsys.readouterr().out
        assert "✗ a.onnx:" in out
        assert "✓ b.pth: downloaded" in out

    def test_should_report_an_unexpected_error_and_try_the_next_file(
        self, two_models, monkeypatch, capsys,
    ):
        monkeypatch.setattr(provision, "_open", Server(
            ValueError("unknown url type: 'htps'"), OTHER))

        assert provision.main(["models"]) == 1

        lines = capsys.readouterr().out.splitlines()
        assert "  ✗ a.onnx: ValueError: unknown url type: 'htps'" in lines
        assert any(line.startswith("  ✓ b.pth: downloaded")
                   for line in lines)

    def test_should_keep_verified_files_on_a_second_run(
        self, two_models, monkeypatch, capsys,
    ):
        monkeypatch.setattr(provision, "_open", Server(PAYLOAD, OTHER))
        assert provision.main(["models"]) == 0
        (paths.models_dir() / "a.onnx").write_bytes(b"damaged")
        capsys.readouterr()
        server = Server(PAYLOAD)
        monkeypatch.setattr(provision, "_open", server)

        assert provision.main(["models"]) == 0

        out = capsys.readouterr().out
        assert "✓ a.onnx: replaced a damaged copy, SHA-256 verified" in out
        assert "✓ b.pth: already present, SHA-256 verified" in out
        assert len(server.calls) == 1

    def test_should_report_a_target_that_is_a_folder_and_continue(
        self, two_models, monkeypatch, capsys,
    ):
        (paths.models_dir() / "a.onnx").mkdir(parents=True)
        monkeypatch.setattr(provision, "_open", Server(PAYLOAD, OTHER))

        assert provision.main(["models"]) == 1

        out = capsys.readouterr().out
        assert "✗ a.onnx: " in out
        assert "✓ b.pth: downloaded" in out

    def test_should_explain_a_config_env_that_is_not_utf8(
        self, two_models, monkeypatch, capsys,
    ):
        paths.config_path().write_text("REFRAME_STRIDE=3", encoding="utf-16")
        server = Server()
        monkeypatch.setattr(provision, "_open", server)

        assert provision.main(["models"]) == 1

        out = capsys.readouterr().out
        assert "✗ settings" in out and "not UTF-8" in out
        assert server.calls == []

    def test_should_stop_on_a_broken_config_env(
        self, two_models, monkeypatch, capsys,
    ):
        paths.config_path().write_text("REFRAME_STRIDE=0\n", encoding="utf-8")
        server = Server()
        monkeypatch.setattr(provision, "_open", server)

        assert provision.main(["models"]) == 1

        out = capsys.readouterr().out
        assert "✗ settings" in out and "REFRAME_STRIDE" in out
        assert server.calls == []


def _fake(version, cuda=None):
    return types.SimpleNamespace(
        __version__=version, version=types.SimpleNamespace(cuda=cuda))


@pytest.fixture
def fake_stack(monkeypatch):
    """Install stand-in torch and torchvision modules; returns a setter."""
    def install(torch_version="2.11.0+cu128", cuda="12.8",
                vision_version="0.26.0+cu128"):
        monkeypatch.setitem(sys.modules, "torch", _fake(torch_version, cuda))
        monkeypatch.setitem(
            sys.modules, "torchvision", _fake(vision_version))
    return install


@pytest.fixture
def in_reframe_venv(monkeypatch, tmp_path):
    """Run as install.sh does: sys.prefix is .venv-reframe, a virtualenv."""
    venv = tmp_path / "checkout" / ".venv-reframe"
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "Python312"))
    return venv


class TestTorchBuild:
    def test_should_accept_the_pinned_cuda_build(self):
        problems = provision.torch_build_problems(
            _fake("2.11.0+cu128", "12.8"), _fake("0.26.0+cu128"), PINS,
            "12.8")

        assert problems == []

    def test_should_catch_a_cpu_only_torch(self):
        problems = provision.torch_build_problems(
            _fake("2.11.0+cpu", None), _fake("0.26.0+cu128"), PINS, "12.8")

        assert problems[0] == "torch is a CPU-only build (2.11.0+cpu)"
        assert len(problems) == 2

    def test_should_catch_another_cuda_build(self):
        problems = provision.torch_build_problems(
            _fake("2.11.0+cu128", "13.0"), _fake("0.26.0+cu128"), PINS,
            "12.8")

        assert problems == ["torch is built for CUDA 13.0, not 12.8"]

    def test_should_catch_a_replaced_torch(self):
        problems = provision.torch_build_problems(
            _fake("2.10.0+cu128", "12.8"), _fake("0.26.0+cu128"), PINS,
            "12.8")

        assert problems == [
            "torch is 2.10.0+cu128, but requirements-torch.txt pins "
            "2.11.0+cu128"]

    def test_should_treat_a_missing_pin_as_a_problem(self):
        problems = provision.torch_build_problems(
            _fake("2.11.0+cu128", "12.8"), _fake("0.26.0+cu128"),
            {"torch": "2.11.0+cu128"}, "12.8")

        assert problems == [
            "requirements-torch.txt has no == pin for torchvision, so its "
            "build cannot be checked"]

    def test_should_catch_a_replaced_torchvision(self):
        problems = provision.torch_build_problems(
            _fake("2.11.0+cu128", "12.8"), _fake("0.26.0"), PINS, "12.8")

        assert problems == [
            "torchvision is 0.26.0, but requirements-torch.txt pins "
            "0.26.0+cu128"]

    def test_should_read_the_pins_install_sh_installs(self):
        assert provision.read_pins(provision.TORCH_REQUIREMENTS) == PINS

    def test_should_pass_the_pinned_build_through_main(
        self, fake_stack, capsys,
    ):
        fake_stack()

        assert provision.main(["torch-build"]) == 0
        assert "✓ torch 2.11.0+cu128 and torchvision 0.26.0+cu128" in (
            capsys.readouterr().out)

    def test_should_explain_the_fix_when_the_build_is_wrong(
        self, fake_stack, capsys, in_reframe_venv,
    ):
        fake_stack(torch_version="2.11.0+cpu", cuda=None)

        assert provision.main(["torch-build"]) == 1
        out = capsys.readouterr().out
        assert "✗ torch is a CPU-only build" in out
        assert "Delete .venv-reframe and run install.sh again: rm -rf" in out

    def test_should_print_the_delete_line_when_torch_will_not_import(
        self, monkeypatch, capsys, in_reframe_venv,
    ):
        def broken():
            raise ImportError("cannot import name '_C' from 'torch'")

        monkeypatch.setattr(provision, "_import_stack", broken)

        assert provision.main(["torch-build"]) == 1
        out = capsys.readouterr().out
        assert "✗ torch import: ImportError" in out
        assert "The installation is incomplete." in out
        assert f'rm -rf "{in_reframe_venv.as_posix()}"' in out
        assert "replaced the CUDA build" not in out

    @pytest.mark.parametrize("base", [
        "/usr", "C:/Python312", "/opt/.venv-reframe"])
    def test_should_never_name_a_python_that_is_not_a_virtualenv(
        self, monkeypatch, base,
    ):
        monkeypatch.setattr(sys, "prefix", base)
        monkeypatch.setattr(sys, "base_prefix", base)

        advice = provision.delete_and_rerun()

        assert "rm -rf" not in advice and base not in advice
        assert "Delete the .venv-reframe folder" in advice
        assert "run install.sh again" in advice

    def test_should_never_name_another_virtualenv(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "Python312"))

        advice = provision.delete_and_rerun()

        assert "rm -rf" not in advice
        assert "Delete the .venv-reframe folder" in advice

    def test_should_not_blame_the_redistributable_off_windows(
        self, in_reframe_venv,
    ):
        error = OSError("libcudart.so.12: cannot open shared object file")

        hint = provision.import_failure_hint(error, platform="linux")

        assert "Visual C++" not in hint
        assert hint.startswith("The installation is incomplete.")

    def test_should_name_the_redistributable_for_a_windows_dll_error(self):
        error = OSError("[WinError 126] The specified module could not be "
                        "found. Error loading c10.dll")

        hint = provision.import_failure_hint(error, platform="win32")

        assert "Visual C++ Redistributable" in hint
        assert "usual" in hint and "rm -rf" not in hint


@pytest.fixture
def fake_weights(monkeypatch):
    """Install a stand-in torchvision whose person-detector URL is set."""
    def install(url):
        weights = types.SimpleNamespace(
            COCO_V1=types.SimpleNamespace(url=url))
        detection = types.ModuleType("torchvision.models.detection")
        detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights = weights
        for name in ("torchvision", "torchvision.models"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(
            sys.modules, "torchvision.models.detection", detection)
    return install


class TestTorchvisionUrl:
    def test_should_pass_when_torchvision_expects_the_pinned_file(
        self, fake_weights, capsys,
    ):
        fake_weights(provision.person_detector().url)

        assert provision.main(["torchvision-url"]) == 0
        assert "✓ torchvision will load the verified" in (
            capsys.readouterr().out)

    def test_should_fail_when_torchvision_will_not_import(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setitem(sys.modules, "torchvision.models.detection",
                            None)

        assert provision.main(["torchvision-url"]) == 1
        assert "✗ torchvision import: " in capsys.readouterr().out

    def test_should_fail_when_torchvision_expects_another_file(
        self, fake_weights, capsys,
    ):
        fake_weights("https://download.pytorch.org/models/other-0000.pth")

        assert provision.main(["torchvision-url"]) == 1
        assert "other-0000.pth" in capsys.readouterr().out


class TestMain:
    @pytest.mark.parametrize(
        "argv", [[], ["bogus"], ["models", "extra"]], ids=repr)
    def test_should_reject_a_missing_or_unknown_command(self, argv, capsys):
        assert provision.main(argv) == 2
        assert "usage" in capsys.readouterr().err

    def test_should_import_with_the_standard_library_only(self, tmp_path):
        """install.sh runs `models` and the FFmpeg check with the base
        Python, before any virtual environment exists. -S hides
        site-packages, so a third-party import anywhere below these
        modules fails here instead of on a fresh machine."""
        code = (
            "import sys, ftea.reframe.provision, ftea.install_checks; "
            "print(sorted(m for m in ('numpy', 'cv2', 'torch', "
            "'torchvision') if m in sys.modules))"
        )
        env = {k: v for k, v in os.environ.items() if k != paths.ENV_HOME}
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=tmp_path, env=paths.reframe_subprocess_env(env),
            capture_output=True, text=True, encoding="utf-8",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]"

    def test_should_run_as_a_module_with_the_launch_rule(self):
        result = subprocess.run(
            [sys.executable, "-m", "ftea.reframe.provision"],
            cwd=paths.source_root(),
            env=paths.reframe_subprocess_env(os.environ),
            capture_output=True, text=True, encoding="utf-8",
        )

        assert result.returncode == 2
        assert "usage" in result.stderr
