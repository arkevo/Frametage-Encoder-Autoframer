"""Tests for ftea.reframe.verify_env -- the reframe environment check.

torch, torchvision and cv2 are never imported here: the checks that need
them take the module as an argument, and these tests hand in a small
stand-in with only the attributes each check reads.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ftea import paths
from ftea.reframe import verify_env


@pytest.fixture(autouse=True)
def _fresh_counter(monkeypatch):
    monkeypatch.setattr(verify_env, "MISSING", 0)


class _Tensor:
    """Just enough of a tensor for check_kernel's one expression."""

    def __init__(self, err, fail=None, devices=None):
        self.err, self.fail = err, fail
        self.devices = [] if devices is None else devices

    def to(self, device):
        if self.fail is not None:
            raise self.fail
        self.devices.append(device)
        return self

    def cpu(self):
        return self

    def __matmul__(self, other):
        return self

    def __sub__(self, other):
        return self

    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return self.err


def _torch(*, cuda="12.8", capability=(8, 9), arch=("sm_86", "sm_120"),
           available=True, err=1e-6, kernel_fails=None, count=1, seen=None):
    """A torch stand-in; `seen` records the GPU index each call asked
    about and the device the kernel ran on."""
    seen = {} if seen is None else seen
    seen.setdefault("indexes", [])
    seen.setdefault("devices", [])

    def asked(value, index):
        seen["indexes"].append(index)
        return value

    return SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            get_device_capability=lambda index: asked(capability, index),
            get_arch_list=lambda: list(arch),
            is_available=lambda: available,
            device_count=lambda: count,
            get_device_name=lambda index: asked("Fake RTX", index),
        ),
        randn=lambda *shape: _Tensor(err, kernel_fails, seen["devices"]),
    )


class TestHasNativeKernels:
    def test_should_accept_older_minor_kernels_of_the_same_major(self):
        """An RTX 40-series card (8.9) runs a cu128 wheel's sm_86 code."""
        assert verify_env.has_native_kernels(["sm_86", "sm_120"], (8, 9))

    def test_should_accept_the_exact_device_arch(self):
        assert verify_env.has_native_kernels(["sm_86", "sm_120"], (12, 0))

    def test_should_reject_a_major_with_no_kernels(self):
        arch = ["sm_80", "sm_86", "sm_90", "sm_120"]
        assert not verify_env.has_native_kernels(arch, (7, 5))

    def test_should_reject_kernels_built_for_a_newer_minor(self):
        assert not verify_env.has_native_kernels(["sm_86"], (8, 0))

    def test_should_not_count_ptx_as_native(self):
        assert not verify_env.has_native_kernels(["compute_89"], (8, 9))


class TestTorchChecks:
    def test_should_pass_the_arch_check_for_an_ada_card(self, capsys):
        verify_env.check_arch(_torch(capability=(8, 9)))

        assert verify_env.MISSING == 0
        assert "✓ native kernels for compute capability 8.9" in \
            capsys.readouterr().out

    def test_should_fail_the_build_check_on_another_cuda_build(
        self, capsys,
    ):
        verify_env.check_cuda_build(_torch(cuda="13.0"))

        assert verify_env.MISSING == 1
        assert "install.sh" in capsys.readouterr().out

    def test_should_expect_the_cuda_build_requirements_pins(self):
        """Survives install.sh splitting requirements.txt: exactly one
        requirements*.txt beside verify_env carries the CUDA index URL."""
        here = Path(verify_env.__file__).parent
        index_builds = [
            match.group(1)
            for path in sorted(here.glob("requirements*.txt"))
            for match in re.finditer(
                r"^--index-url\s+\S+/whl/cu(\d+)\s*$",
                path.read_text(encoding="utf-8"), re.MULTILINE)
        ]

        assert len(index_builds) == 1, (
            "expected exactly one CUDA --index-url across requirements*.txt,"
            f" found {len(index_builds)}")
        assert index_builds[0] == verify_env.EXPECTED_CUDA.replace(".", "")


class TestKernelCheck:
    def test_should_pass_when_the_gpu_agrees_with_the_cpu(self, capsys):
        verify_env.check_kernel(_torch(err=1e-6))

        assert verify_env.MISSING == 0
        assert "✓ CUDA kernel vs CPU reference" in capsys.readouterr().out

    def test_should_fail_when_the_gpu_disagrees_with_the_cpu(self, capsys):
        verify_env.check_kernel(_torch(err=1e-3))

        assert verify_env.MISSING == 1
        assert "✗ CUDA kernel vs CPU reference" in capsys.readouterr().out

    def test_should_fail_with_the_reason_when_no_kernel_runs(self, capsys):
        reason = "no kernel image is available for execution on the device"
        verify_env.check_kernel(_torch(kernel_fails=RuntimeError(reason)))

        assert verify_env.MISSING == 1
        assert f"✗ CUDA kernel execution: RuntimeError: {reason}" in \
            capsys.readouterr().out


class TestTorchModuleCheck:
    def test_should_fail_when_no_cuda_device_is_visible(self, capsys):
        verify_env.check_torch_module(_torch(available=False), "cuda")

        assert verify_env.MISSING == 1
        assert "✗ torch.cuda.is_available()" in capsys.readouterr().out

    def test_should_check_the_gpu_reframe_device_names(self, capsys):
        """REFRAME_DEVICE=cuda:1 detects on the second GPU; checking the
        first would pass a machine whose second GPU cannot run it."""
        seen = {}
        verify_env.check_torch_module(_torch(count=2, seen=seen), "cuda:1")

        assert verify_env.MISSING == 0
        assert set(seen["indexes"]) == {1}
        assert seen["devices"] == ["cuda:1", "cuda:1"]

    def test_should_fail_when_the_named_gpu_does_not_exist(self, capsys):
        verify_env.check_torch_module(_torch(count=1), "cuda:1")

        assert verify_env.MISSING == 1
        out = capsys.readouterr().out
        assert "✗ REFRAME_DEVICE=cuda:1" in out and "1 GPU" in out

    def test_should_run_every_gpu_check_for_a_cuda_device(self, capsys):
        verify_env.check_torch_module(_torch(count=2), "cuda:1")

        out = capsys.readouterr().out
        assert verify_env.MISSING == 0
        assert "✓ CUDA device: Fake RTX" in out
        assert "✓ CUDA kernel vs CPU reference" in out

    def test_should_skip_the_gpu_checks_when_detection_runs_on_the_cpu(
        self, capsys,
    ):
        """REFRAME_DEVICE=cpu needs neither a CUDA build nor a visible
        GPU, so their absence must not stop the run."""
        verify_env.check_torch_module(
            _torch(cuda=None, available=False), "cpu")

        out = capsys.readouterr().out
        assert verify_env.MISSING == 0
        assert "REFRAME_DEVICE=cpu" in out
        assert "CUDA device" not in out


def _fake_torchvision(monkeypatch, loaded):
    """torch and torchvision stand-ins recording what the check loads."""
    class Model:
        def __init__(self, **kwargs):
            loaded.append(("model", kwargs))

        def load_state_dict(self, state):
            loaded.append(("state", state))

    detection = ModuleType("torchvision.models.detection")
    detection.fasterrcnn_mobilenet_v3_large_fpn = Model
    detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights = SimpleNamespace(
        COCO_V1="COCO_V1")
    models = ModuleType("torchvision.models")
    models.detection = detection
    torchvision = ModuleType("torchvision")
    torchvision.models = models
    torchvision.__version__ = "0.26.0"
    torch = ModuleType("torch")
    torch.load = lambda path, map_location=None: {"from": path}
    for name, module in (("torch", torch), ("torchvision", torchvision),
                         ("torchvision.models", models),
                         ("torchvision.models.detection", detection)):
        monkeypatch.setitem(sys.modules, name, module)


class TestPersonModelCheck:
    def test_should_fail_when_the_configured_weights_are_missing(
        self, tmp_path, capsys,
    ):
        """The stage refuses a missing REFRAME_PERSON_MODEL, so the
        once-per-run check must too, not test the default weights."""
        missing = tmp_path / "person.pth"

        verify_env.check_person_detector(str(missing))

        out = capsys.readouterr().out
        assert verify_env.MISSING == 1
        assert "REFRAME_PERSON_MODEL" in out
        assert str(missing) in out

    def test_should_load_the_configured_weights_when_they_exist(
        self, tmp_path, monkeypatch, capsys,
    ):
        weights = tmp_path / "person.pth"
        weights.write_bytes(b"x")
        loaded = []
        _fake_torchvision(monkeypatch, loaded)

        verify_env.check_person_detector(str(weights))

        assert verify_env.MISSING == 0
        assert loaded[0][1]["weights"] is None
        assert loaded[0][1]["num_classes"] == 91
        assert loaded[1] == ("state", {"from": str(weights)})
        assert "✓ person detector" in capsys.readouterr().out

    def test_should_load_the_default_weights_when_none_are_configured(
        self, monkeypatch,
    ):
        loaded = []
        _fake_torchvision(monkeypatch, loaded)

        verify_env.check_person_detector("")

        assert verify_env.MISSING == 0
        assert loaded == [("model", {"weights": "COCO_V1"})]


class TestModels:
    def test_should_look_for_models_under_the_data_root(self):
        settings = verify_env.load_settings(paths.config_path())

        face, text_model = verify_env.model_paths(settings)

        assert face == paths.models_dir() / "face_detection_yunet_2023mar.onnx"
        assert text_model == (
            paths.models_dir() / "text_detection_en_ppocrv3_2023may.onnx")

    def test_should_honour_reframe_models_dir_from_config_env(self):
        paths.config_path().write_text("REFRAME_MODELS_DIR=weights\n")
        settings = verify_env.load_settings(paths.config_path())

        face, _ = verify_env.model_paths(settings)

        assert face.parent == paths.data_root() / "weights"

    def test_should_fail_with_an_install_hint_when_weights_are_missing(
        self, tmp_path, capsys,
    ):
        found = verify_env.check_model_files(
            tmp_path / "face.onnx", tmp_path / "text.onnx")

        assert found == (False, False)
        assert verify_env.MISSING == 2
        assert capsys.readouterr().out.count("run install.sh") == 2

    def test_should_pass_when_both_weight_files_exist(self, tmp_path):
        face, text_model = tmp_path / "face.onnx", tmp_path / "text.onnx"
        face.write_bytes(b"x")
        text_model.write_bytes(b"y")

        assert verify_env.check_model_files(face, text_model) == (True, True)
        assert verify_env.MISSING == 0


class TestFfmpeg:
    def test_should_check_the_configured_ffmpeg_as_well_as_path(
        self, monkeypatch,
    ):
        found = {"ffmpeg": "/usr/bin/ffmpeg", "/opt/ff/ffmpeg": None}
        monkeypatch.setattr(verify_env.shutil, "which", found.get)

        binaries = verify_env.ffmpeg_binaries("/opt/ff/ffmpeg")

        assert binaries == [
            ("ffmpeg on PATH", "/usr/bin/ffmpeg"),
            ("REFRAME_FFMPEG (/opt/ff/ffmpeg)", None),
        ]

    def test_should_fail_every_missing_binary_without_running_it(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setattr(verify_env.shutil, "which", lambda name: None)

        def must_not_run(*args, **kwargs):
            raise AssertionError("nothing to run when no ffmpeg was found")

        monkeypatch.setattr(verify_env.subprocess, "run", must_not_run)

        verify_env.check_ffmpeg("/opt/ff/ffmpeg")

        assert verify_env.MISSING == 3
        assert capsys.readouterr().out.count("NOT FOUND") == 3

    @staticmethod
    def _fake_ffmpeg(*, session_exit=0, session_err="", sessions=None):
        """A subprocess.run stand-in for a capable ffmpeg whose one-frame
        encode exits with session_exit; `sessions` records the encoder
        of each one."""
        listings = {
            "-encoders": " V....D hevc_nvenc  NVIDIA NVENC hevc encoder\n"
                         " V....D h264_nvenc  NVIDIA NVENC H.264 encoder\n"
                         " V....D av1_nvenc   NVIDIA NVENC av1 encoder\n",
            "filter=crop": "  x                 <string>     ..FV.....T. "
                           "set the x crop area expression\n",
            "-filters": " ... sendcmd           N->N       Send commands\n",
        }

        def run(argv, **kwargs):
            for key, text in listings.items():
                if key in argv:
                    return SimpleNamespace(returncode=0, stdout=text,
                                           stderr="")
            if sessions is not None:
                sessions.append(argv[argv.index("-c:v") + 1])
            return SimpleNamespace(returncode=session_exit, stdout="",
                                   stderr=session_err)

        return run

    def test_should_pass_when_an_nvenc_session_opens(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setattr(verify_env.subprocess, "run",
                            self._fake_ffmpeg())

        verify_env._check_ffmpeg_features("ffmpeg")

        assert verify_env.MISSING == 0
        assert "✓ hevc_nvenc session opens" in capsys.readouterr().out

    def test_should_fail_with_ffmpegs_reason_when_no_nvenc_session_opens(
        self, monkeypatch, capsys,
    ):
        reason = ("[hevc_nvenc @ 0x1] Driver does not support the required "
                  "nvenc API version. Required: 13.0 Found: 12.2")
        monkeypatch.setattr(
            verify_env.subprocess, "run",
            self._fake_ffmpeg(session_exit=1, session_err=f"\n{reason}\n"))

        verify_env._check_ffmpeg_features("ffmpeg")

        out = capsys.readouterr().out
        assert verify_env.MISSING == 1
        assert f"✗ hevc_nvenc session opens: {reason}" in out

    def test_should_open_a_session_with_the_configured_codec(
        self, monkeypatch, capsys,
    ):
        """The render encodes with REFRAME_CODEC, not always hevc_nvenc."""
        sessions = []
        monkeypatch.setattr(verify_env.subprocess, "run",
                            self._fake_ffmpeg(sessions=sessions))

        verify_env._check_ffmpeg_features("ffmpeg", "av1_nvenc")

        out = capsys.readouterr().out
        assert sessions == ["av1_nvenc"]
        assert "✓ encoder av1_nvenc" in out
        assert "✓ av1_nvenc session opens" in out

    def test_should_fail_a_configured_codec_the_build_lacks(
        self, monkeypatch, capsys,
    ):
        sessions = []
        monkeypatch.setattr(verify_env.subprocess, "run",
                            self._fake_ffmpeg(sessions=sessions))

        verify_env._check_ffmpeg_features("ffmpeg", "ffv1")

        assert verify_env.MISSING == 1
        assert "✗ encoder ffv1" in capsys.readouterr().out
        assert sessions == []

    def test_should_check_the_codec_on_every_ffmpeg(self, monkeypatch):
        checked = []
        monkeypatch.setattr(verify_env.shutil, "which", lambda name: name)
        monkeypatch.setattr(
            verify_env, "_check_ffmpeg_features",
            lambda exe, codec: checked.append((exe, codec)),
        )

        verify_env.check_ffmpeg("/opt/ff/ffmpeg", codec="h264_nvenc")

        assert checked == [
            ("ffmpeg", "h264_nvenc"), ("/opt/ff/ffmpeg", "h264_nvenc"),
        ]


class TestMain:
    def test_should_point_torch_home_at_the_models_dir_before_any_check(
        self, monkeypatch,
    ):
        seen = {}
        monkeypatch.setenv("TORCH_HOME", "somewhere-else")
        monkeypatch.setattr(
            verify_env, "check_torch",
            lambda device: seen.setdefault("home", os.environ["TORCH_HOME"]))
        monkeypatch.setattr(verify_env, "check_vision",
                            lambda f, t, p: None)
        monkeypatch.setattr(verify_env, "check_ffmpeg",
                            lambda c, codec: None)

        assert verify_env.main([]) == 0
        assert seen["home"] == str(paths.models_dir() / "torch")

    def test_should_return_one_when_any_check_fails(
        self, monkeypatch, capsys,
    ):
        """The encoder's once-per-run gate reads only this exit code."""
        monkeypatch.setattr(verify_env, "check_torch", lambda device: None)
        monkeypatch.setattr(
            verify_env, "check_vision",
            lambda f, t, p: verify_env.fail("face detector", "broken"))
        monkeypatch.setattr(verify_env, "check_ffmpeg",
                            lambda c, codec: None)

        assert verify_env.main([]) == 1
        assert "=== 1 check(s) FAILED ===" in capsys.readouterr().out

    def test_should_hand_the_configured_codec_to_the_ffmpeg_check(
        self, monkeypatch,
    ):
        paths.config_path().write_text("REFRAME_CODEC=av1_nvenc", "utf-8")
        seen = {}
        monkeypatch.setattr(verify_env, "check_torch", lambda device: None)
        monkeypatch.setattr(verify_env, "check_vision",
                            lambda f, t, p: None)
        monkeypatch.setattr(
            verify_env, "check_ffmpeg",
            lambda c, codec: seen.setdefault("codec", codec))

        verify_env.main([])

        assert seen == {"codec": "av1_nvenc"}

    def test_should_hand_the_configured_device_and_person_model_on(
        self, monkeypatch, tmp_path,
    ):
        weights = tmp_path / "person.pth"
        paths.config_path().write_text(
            f"REFRAME_DEVICE=cpu\nREFRAME_PERSON_MODEL={weights}\n")
        seen = {}
        monkeypatch.setattr(verify_env, "check_torch",
                            lambda device: seen.setdefault("device", device))
        monkeypatch.setattr(
            verify_env, "check_vision",
            lambda f, t, p: seen.setdefault("person", p))
        monkeypatch.setattr(verify_env, "check_ffmpeg",
                            lambda c, codec: None)

        verify_env.main([])

        assert seen == {"device": "cpu", "person": str(weights)}

    def test_should_stop_on_an_invalid_config_before_importing_torch(
        self, monkeypatch, capsys,
    ):
        paths.config_path().write_text("REFRAME_STRIDE=0\n")

        def must_not_run():
            raise AssertionError("torch checked despite a broken config")

        monkeypatch.setattr(verify_env, "check_torch", must_not_run)

        assert verify_env.main([]) == 1
        assert "REFRAME_STRIDE" in capsys.readouterr().out

    def test_should_explain_a_config_env_that_is_not_utf8(self, capsys):
        paths.config_path().write_text("REFRAME_STRIDE=3", encoding="utf-16")

        assert verify_env.main([]) == 1
        assert "not UTF-8" in capsys.readouterr().out

    def test_should_report_a_missing_data_root_without_a_traceback(
        self, tmp_path, monkeypatch, capsys,
    ):
        """A mistyped FTEA_HOME is a failed check with the reason, the
        same as a broken config.env, not a crash at install.sh's last
        step."""
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "no-such-home"))

        assert verify_env.main([]) == 1
        assert "FTEA_HOME" in capsys.readouterr().out
