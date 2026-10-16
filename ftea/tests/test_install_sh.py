"""Tests for install.sh.

install.sh runs main only when it is executed, so these tests source it
in a fresh bash, replace the one-line functions that ask the outside
world (uname, nvidia-smi, a Python probe, df), and call its functions
directly. A few tests run the script itself.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ftea import paths

REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "install.sh"
# On Windows, System32\bash.exe (and the WindowsApps alias) start WSL,
# not Git Bash: a bare "bash" from Windows Python runs Linux.
WSL_LAUNCHERS = ("\\system32\\", "\\windowsapps\\")


def _find_bash() -> str | None:
    candidates = [shutil.which("bash")]
    if sys.platform == "win32":
        candidates.append(r"C:\Program Files\Git\usr\bin\bash.exe")
    for candidate in candidates:
        if not candidate:
            continue
        if any(part in candidate.lower() for part in WSL_LAUNCHERS):
            continue
        if Path(candidate).is_file():
            return candidate
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(
    BASH is None, reason="needs bash (Git Bash on Windows)")


@dataclass
class Ran:
    code: int
    out: str


def _env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    # Git Bash's coreutils (uname, grep, awk, sort, cygpath) sit beside
    # its bash; put them first even when pytest was started elsewhere.
    env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
    return env


def run_bash(body: str, *, env: dict[str, str] | None = None,
             cwd: Path | None = None) -> Ran:
    """Source install.sh in a fresh bash, then run `body`.

    The script goes in on stdin as bytes, so Windows never turns its
    line endings into CRLF. A failed `source` stops at once (exit 97),
    so no test can pass against a missing or broken install.sh.
    """
    script = f'source "{INSTALL_SH.as_posix()}" || exit 97\n{body}\n'
    done = subprocess.run(
        [BASH, "--noprofile", "--norc", "-s"],
        input=script.encode("utf-8"), cwd=cwd or REPO, env=_env(env),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    return Ran(done.returncode, done.stdout.decode("utf-8", "replace"))


def run_script(*args: str, cwd: Path) -> Ran:
    done = subprocess.run(
        [BASH, INSTALL_SH.as_posix(), *args], cwd=cwd, env=_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    return Ran(done.returncode, done.stdout.decode("utf-8", "replace"))


def _sub(body: str, **values: str) -> str:
    for key, value in values.items():
        body = body.replace(f"@{key}@", value)
    return body


class TestPlatform:
    @pytest.mark.parametrize("name, expected", [
        ("MINGW64_NT-10.0-22631", "windows"),
        ("MSYS_NT-10.0-22631", "windows"),
        ("Linux", "linux"),
    ])
    def test_should_recognise_git_bash_and_linux(self, name, expected):
        ran = run_bash(_sub(r'''
os_name() { echo "@NAME@"; }
detect_platform
echo "PLATFORM=$PLATFORM"
''', NAME=name))

        assert ran.code == 0
        assert f"PLATFORM={expected}" in ran.out

    @pytest.mark.parametrize("name", ["Darwin", "CYGWIN_NT-10.0"])
    def test_should_refuse_an_unsupported_system(self, name):
        ran = run_bash(_sub(r'''
os_name() { echo "@NAME@"; }
detect_platform
echo "NOT REACHED"
''', NAME=name))

        assert ran.code == 1
        assert "NOT REACHED" not in ran.out
        assert "not supported" in ran.out


class TestGuards:
    def test_should_refuse_wsl_on_a_windows_drive(self):
        ran = run_bash(r'''
PLATFORM=linux
kernel_release() { echo "6.6.87.2-microsoft-standard-WSL2"; }
CHECKOUT=/mnt/d/projects/ftea
guard_one_os
echo "NOT REACHED"
''')

        assert ran.code == 1
        assert "is on a Windows drive" in ran.out
        assert "Git Bash" in ran.out and "git clone" in ran.out

    def test_should_allow_wsl_in_the_linux_home(self, tmp_path):
        ran = run_bash(_sub(r'''
PLATFORM=linux
kernel_release() { echo "6.6.87.2-microsoft-standard-WSL2"; }
CHECKOUT='@DIR@'
guard_one_os
echo "passed"
''', DIR=tmp_path.as_posix()))

        assert ran.code == 0 and "passed" in ran.out

    def test_should_allow_a_mounted_drive_on_plain_linux(self):
        ran = run_bash(r'''
PLATFORM=linux
kernel_release() { echo "6.8.0-45-generic"; }
CHECKOUT=/mnt/d/projects/ftea
guard_one_os
echo "passed"
''')

        assert ran.code == 0 and "passed" in ran.out

    @pytest.mark.parametrize("platform, interpreter, message", [
        ("linux", ".venv-reframe/Scripts/python.exe", "created on Windows"),
        ("windows", ".venv/bin/python", "created on Linux"),
    ])
    def test_should_refuse_a_venv_built_on_the_other_system(
        self, tmp_path, platform, interpreter, message,
    ):
        found = tmp_path / interpreter
        found.parent.mkdir(parents=True)
        found.write_bytes(b"")
        ran = run_bash(_sub(r'''
PLATFORM=@PLATFORM@
kernel_release() { echo ""; }
CHECKOUT='@DIR@'
guard_one_os
echo "NOT REACHED"
''', PLATFORM=platform, DIR=tmp_path.as_posix()))

        assert ran.code == 1
        assert message in ran.out and "NOT REACHED" not in ran.out


WINDOWS_PROBE = r'''
PLATFORM=windows
have() { return 0; }
python_probe() {
  case "$*" in
@CASES@
    *) return 1 ;;
  esac
}
find_base_python
echo "rc=$?"
echo "BASE=${BASE_PY[*]}"
'''


def _windows_pythons(cases: dict[str, str]) -> Ran:
    lines = "\n".join(
        f'    "{cmd}") echo "{info}" ;;' for cmd, info in cases.items())
    return run_bash(_sub(WINDOWS_PROBE, CASES=lines))


class TestPython:
    def test_should_prefer_python_3_12(self):
        ran = _windows_pythons({
            "py -3.13": "3.13 standard win-amd64",
            "py -3.12": "3.12 standard win-amd64",
        })

        assert "rc=0" in ran.out
        assert "BASE=py -3.12" in ran.out
        assert "✓ Python: py -3.12 (3.12 standard win-amd64)" in ran.out

    def test_should_skip_unusable_interpreters(self):
        ran = _windows_pythons({
            "py -3.12": "3.12 standard win32",
            "py -3.13": "3.13 free-threaded win-amd64",
            "py -3.11": "3.11 standard win-amd64",
        })

        assert "BASE=py -3.11" in ran.out

    def test_should_list_every_rejection_and_how_to_get_python(self):
        ran = _windows_pythons({
            "py -3.12": "3.12 standard win32",
            "py -3.14": "3.14 free-threaded win-amd64",
            "python": "3.15 standard win-amd64",
        })

        assert "rc=1" in ran.out
        assert "✗ no usable Python found" in ran.out
        assert "py -3.12 is a win32 build of Python 3.12" in ran.out
        assert "py -3.14 is a free-threaded build of Python 3.14" in ran.out
        assert "python is Python 3.15; ftea needs 3.10 to 3.14" in ran.out
        assert "https://www.python.org/downloads/" in ran.out

    def test_should_list_a_python_on_path_that_did_not_run(self):
        ran = _windows_pythons({"py -3.12": "3.12 standard win32"})

        assert "rc=1" in ran.out
        assert "python is on PATH but did not run" in ran.out
        assert "python3 is on PATH but did not run" in ran.out
        # The launcher failing for a version that is not installed is
        # normal, and listing every one would bury the real reasons.
        assert "py -3.13 is on PATH" not in ran.out

    def test_should_search_versioned_names_on_linux(self):
        ran = run_bash("PLATFORM=linux\npython_candidates")

        assert ran.out.split() == [
            "python3.12", "python3.13", "python3.11", "python3.14",
            "python3.10", "python3",
        ]

    def test_should_use_ftea_python_even_with_spaces_in_its_path(self):
        ran = run_bash(r'''
PLATFORM=linux
FTEA_PYTHON="/opt/my python/bin/python3"
python_probe() {
  [ "$#" -eq 1 ] && [ "$1" = "/opt/my python/bin/python3" ] &&
    echo "3.12 standard linux-x86_64"
}
find_base_python
echo "rc=$? count=${#BASE_PY[@]} first=${BASE_PY[0]}"
''')

        assert "rc=0 count=1 first=/opt/my python/bin/python3" in ran.out

    def test_should_refuse_an_ftea_python_that_does_not_run(self):
        ran = run_bash(r'''
PLATFORM=linux
FTEA_PYTHON=/opt/gone/bin/python3
python_probe() { return 1; }
find_base_python
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert "✗ FTEA_PYTHON=/opt/gone/bin/python3 does not run" in ran.out
        assert "no arguments" not in ran.out

    def test_should_say_ftea_python_takes_no_arguments(self):
        ran = run_bash(r'''
PLATFORM=windows
FTEA_PYTHON="py -3.12"
python_probe() { return 1; }
find_base_python
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert "✗ FTEA_PYTHON=py -3.12 does not run as Python." in ran.out
        assert "one Python program, with no arguments" in ran.out

    def test_should_reject_a_python_for_another_linux_cpu(self):
        ran = run_bash(r'''
PLATFORM=linux
FTEA_PYTHON=/usr/bin/python3.12
python_probe() { echo "3.12 standard linux-aarch64"; }
find_base_python
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert ("is a linux-aarch64 build of Python 3.12; ftea needs a "
                "64-bit linux-x86_64 one") in ran.out

    def test_should_refuse_an_unusable_ftea_python(self):
        ran = run_bash(r'''
PLATFORM=linux
FTEA_PYTHON=/opt/py315/bin/python3
python_probe() { echo "3.15 standard linux-x86_64"; }
find_base_python
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert "FTEA_PYTHON=/opt/py315/bin/python3 is Python 3.15" in ran.out

    def test_should_name_the_apt_package_when_venv_is_missing(self):
        ran = run_bash(r'''
PLATFORM=linux
BASE_PY=(python3.12)
BASE_PY_INFO="3.12 standard linux-x86_64"
python_has_venv() { return 1; }
check_venv_support
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert "ensurepip or venv module is missing" in ran.out
        assert "sudo apt install python3.12-venv" in ran.out

    def test_should_not_suggest_apt_on_windows(self):
        ran = run_bash(r'''
PLATFORM=windows
BASE_PY=(py -3.12)
BASE_PY_INFO="3.12 standard win-amd64"
python_has_venv() { return 1; }
check_venv_support
echo "rc=$?"
''')

        assert "rc=1" in ran.out
        assert "ensurepip or venv module is missing" in ran.out
        assert "apt" not in ran.out
        assert "https://www.python.org/downloads/" in ran.out

    def test_should_derive_the_supported_range_from_the_preferences(self):
        assert run_bash("python_range").out.strip() == "3.10 to 3.14"


GPU = r'''
PLATFORM=linux
NVIDIA_SMI_CANDIDATES=(nvidia-smi /usr/lib/wsl/lib/nvidia-smi)
@BODY@
check_gpu
echo "rc=$? smi=$NVIDIA_SMI"
'''


class TestGpu:
    def test_should_report_the_gpu_and_its_driver(self):
        ran = run_bash(_sub(GPU, BODY=r'''
have() { return 0; }
nvidia_smi_query() { echo "NVIDIA GeForce RTX 5080, 616.64"; }
'''))

        assert "✓ GPU: NVIDIA GeForce RTX 5080, driver 616.64" in ran.out
        assert "rc=0 smi=nvidia-smi" in ran.out
        assert "  ! " not in ran.out

    @pytest.mark.parametrize("line, warned", [
        ("NVIDIA GeForce RTX 4070, 551.86", True),
        ("NVIDIA L4, 570.133.07", False),
    ])
    def test_should_only_warn_about_an_old_driver(self, line, warned):
        ran = run_bash(_sub(GPU, BODY=_sub(r'''
have() { return 0; }
nvidia_smi_query() { echo "@LINE@"; }
''', LINE=line)))

        assert "rc=0" in ran.out
        assert ("  ! NVIDIA driver" in ran.out) is warned

    def test_should_find_nvidia_smi_in_the_wsl_driver_folder(self):
        ran = run_bash(_sub(GPU, BODY=r'''
have() { [ "$1" = /usr/lib/wsl/lib/nvidia-smi ]; }
nvidia_smi_query() { echo "NVIDIA GeForce RTX 5080, 616.64"; }
'''))

        assert "rc=0 smi=/usr/lib/wsl/lib/nvidia-smi" in ran.out

    def test_should_fail_without_nvidia_smi(self):
        ran = run_bash(_sub(GPU, BODY="have() { return 1; }"))

        assert "✗ nvidia-smi not found" in ran.out
        assert "rc=1" in ran.out

    def test_should_fail_when_no_gpu_is_listed(self):
        ran = run_bash(_sub(GPU, BODY=r'''
have() { return 0; }
nvidia_smi_query() { return 0; }
'''))

        assert "✗ nvidia-smi found no NVIDIA GPU." in ran.out
        assert "rc=1" in ran.out

    def test_should_say_why_when_nvidia_smi_fails(self):
        ran = run_bash(_sub(GPU, BODY=r'''
have() { return 0; }
nvidia_smi_query() {
  echo "Failed to initialize NVML: Driver/library version mismatch"
  return 18
}
'''))

        assert "rc=1" in ran.out
        assert "✗ nvidia-smi could not list the GPUs (exit 18: Failed to " \
            "initialize NVML: Driver/library version mismatch)" in ran.out
        assert "reinstall the NVIDIA driver" in ran.out
        assert "found no NVIDIA GPU" not in ran.out

    def test_should_report_the_first_gpu_when_two_are_listed(self):
        """main runs with pipefail, and nvidia-smi can write the second
        GPU's line after a reader that wanted only one has gone."""
        ran = run_bash(_sub(GPU, BODY=r'''
set -euo pipefail
have() { return 0; }
nvidia_smi_query() {
  echo "NVIDIA GeForce RTX 5080, 616.64"
  sleep 0.3
  echo "NVIDIA GeForce RTX 4090, 616.64"
}
'''))

        assert "✓ GPU: NVIDIA GeForce RTX 5080, driver 616.64" in ran.out
        assert "rc=0" in ran.out

    def test_should_read_long_paths_under_pipefail(self):
        ran = run_bash(r'''
set -o pipefail
reg.exe() {
  echo "    LongPathsEnabled    REG_DWORD    0x1"
  sleep 0.3
  echo ""
}
if long_paths_enabled; then echo "long paths: on"; else echo "long paths: off"; fi
''')

        assert "long paths: on" in ran.out


class TestFfmpegAndDisk:
    @pytest.mark.parametrize("platform, source", [
        ("windows", "https://www.gyan.dev/ffmpeg/builds/"),
        ("linux", "https://github.com/BtbN/FFmpeg-Builds/releases"),
    ])
    def test_should_name_an_ffmpeg_build_for_the_platform(
        self, platform, source,
    ):
        ran = run_bash(_sub(r'''
PLATFORM=@PLATFORM@
have() { [ "$1" = ffmpeg ]; }
check_ffmpeg_present
echo "rc=$?"
''', PLATFORM=platform))

        assert "✗ not on PATH: ffprobe." in ran.out
        assert source in ran.out and "rc=1" in ran.out

    def test_should_pass_when_both_programs_are_on_path(self):
        ran = run_bash("have() { return 0; }\ncheck_ffmpeg_present")

        assert "✓ ffmpeg and ffprobe are on PATH" in ran.out

    @pytest.mark.parametrize("kb, expected", [
        ("5242880", "  ! 5 GB free here"),
        ("104857600", "  ✓ disk: 100 GB free"),
        ("", "  ! could not read the free disk space"),
    ])
    def test_should_only_warn_about_disk_space(self, kb, expected):
        ran = run_bash(_sub(r'''
CHECKOUT=.
free_kb() { echo "@KB@"; }
check_disk
echo "rc=$?"
''', KB=kb))

        assert expected in ran.out and "rc=0" in ran.out

    def test_should_read_df_when_names_contain_spaces(self):
        ran = run_bash(r'''
df() {
  case "$*" in
    *--output=avail*) printf '%s\n' "    Avail" "527762580" ;;
    *--output=target*) printf '%s\n' "Mounted on" "/media/me/My Passport" ;;
    *) printf '%s\n' \
         "Filesystem 1024-blocks Used Available Capacity Mounted on" \
         "/dev/sdb1 976762580 449000000 527762580 46% /media/me/My Passport" ;;
  esac
}
echo "free=[$(free_kb "/media/me/My Passport/ftea")]"
echo "mount=[$(mount_of "/media/me/My Passport/ftea")]"
''')

        assert "free=[527762580]" in ran.out
        assert "mount=[/media/me/My Passport]" in ran.out

    def test_should_remind_wsl_users_about_the_windows_drive(self):
        ran = run_bash(r'''
PLATFORM=linux
kernel_release() { echo "6.6.87.2-microsoft-standard-WSL2"; }
CHECKOUT=.
free_kb() { echo "104857600"; }
check_disk
''')

        assert "  ✓ disk: 100 GB free" in ran.out
        assert "  ! WSL counts its own virtual disk" in ran.out

    def test_should_warn_about_a_deep_checkout_without_long_paths(self):
        ran = run_bash(r'''
PLATFORM=windows
MAX_SAFE_CHECKOUT_CHARS=20
CHECKOUT=/d/projects/a/very/deep/folder/ftea
native_path() { printf '%s\n' "$1"; }
long_paths_enabled() { return 1; }
check_long_paths
echo "rc=$?"
''')

        assert "rc=0" in ran.out
        assert "  ! this folder's path is" in ran.out
        assert "C:\\ftea" in ran.out and "long paths" in ran.out

    def test_should_not_warn_when_long_paths_are_enabled(self):
        ran = run_bash(r'''
PLATFORM=windows
MAX_SAFE_CHECKOUT_CHARS=20
CHECKOUT=/d/projects/a/very/deep/folder/ftea
native_path() { printf '%s\n' "$1"; }
long_paths_enabled() { return 0; }
check_long_paths
echo "rc=$?"
''')

        assert "rc=0" in ran.out
        assert "  ! " not in ran.out

    def test_should_check_each_filesystem_the_install_writes_to(self):
        ran = run_bash(r'''
PLATFORM=linux
kernel_release() { echo ""; }
CHECKOUT=/srv/ftea
DATA_ROOT=/data/ftea
temp_dir() { echo /tmp; }
pip_cache_dir() { echo /var/cache/pip; }
existing_parent() { printf '%s\n' "$1"; }
mount_of() { case "$1" in /data/*) echo /data ;; *) echo / ;; esac; }
free_kb() { echo "104857600"; }
check_disks
echo "rc=$?"
''')

        assert "rc=0" in ran.out
        assert ran.out.count("  ✓ disk: 100 GB free") == 2
        assert "(/srv/ftea)" in ran.out and "(/data/ftea)" in ran.out

    def test_should_check_a_shared_filesystem_once(self):
        ran = run_bash(r'''
PLATFORM=linux
kernel_release() { echo ""; }
CHECKOUT=/srv/ftea
DATA_ROOT=/srv/ftea
temp_dir() { echo /tmp; }
pip_cache_dir() { echo /var/cache/pip; }
existing_parent() { printf '%s\n' "$1"; }
mount_of() { echo /; }
free_kb() { echo "104857600"; }
check_disks
echo "rc=$?"
''')

        assert "rc=0" in ran.out
        assert ran.out.count("  ✓ disk: 100 GB free") == 1


DATA_ROOT_BODY = r'''
detect_platform
CHECKOUT="$(cd -- '@CHECKOUT@' && pwd -P)"
if [ -n '@HOME@' ]; then
  if [ "$PLATFORM" = windows ]; then HOME="$(cygpath -u '@HOME@')"
  else HOME='@HOME@'; fi
fi
@SETUP@
prepare_data_root @MODE@
echo "rc=$?"
echo "DATA_ROOT=$(native_path "$DATA_ROOT")"
echo "FTEA_HOME=${FTEA_HOME-<unset>}"
'''


def _prepare(tmp_path: Path, *, ftea_home: str | None, mode: str = "create",
             cwd: Path | None = None, home: Path | None = None,
             setup: str = ""):
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop(paths.ENV_HOME, None)
    if ftea_home is not None:
        env[paths.ENV_HOME] = ftea_home
    body = _sub(DATA_ROOT_BODY, CHECKOUT=checkout.as_posix(), MODE=mode,
                HOME=home.as_posix() if home else "", SETUP=setup)
    ran = run_bash(body, env=env, cwd=cwd or tmp_path)
    values = dict(
        line.split("=", 1) for line in ran.out.splitlines()
        if re.match(r"(rc|DATA_ROOT|FTEA_HOME)=", line))
    return ran, values, checkout


class TestDataRoot:
    def test_should_use_the_checkout_when_ftea_home_is_unset(self, tmp_path):
        ran, values, checkout = _prepare(tmp_path, ftea_home=None)

        assert values["rc"] == "0"
        assert Path(values["DATA_ROOT"]).samefile(checkout)
        assert values["FTEA_HOME"] == "<unset>"
        assert (checkout / "input").is_dir()
        assert (checkout / "output").is_dir()

    def test_should_treat_a_blank_ftea_home_as_unset(self, tmp_path):
        _, values, checkout = _prepare(tmp_path, ftea_home="   ")

        assert Path(values["DATA_ROOT"]).samefile(checkout)
        assert values["FTEA_HOME"] == "<unset>"

    def test_should_create_a_relative_ftea_home_in_the_working_folder(
        self, tmp_path,
    ):
        work = tmp_path / "work"
        work.mkdir()

        _, values, _ = _prepare(tmp_path, ftea_home="data root", cwd=work)

        expected = work / "data root"
        assert (expected / "input").is_dir()
        assert Path(values["DATA_ROOT"]).samefile(expected)
        assert Path(values["FTEA_HOME"]).samefile(expected)

    def test_should_expand_a_literal_tilde(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()

        _, values, _ = _prepare(tmp_path, ftea_home="~/ftea data", home=home)

        assert Path(values["DATA_ROOT"]).samefile(home / "ftea data")

    @pytest.mark.skipif(sys.platform != "win32",
                        reason="Windows-style paths exist only on Windows")
    def test_should_accept_a_windows_style_ftea_home(self, tmp_path):
        target = tmp_path / "win style"

        _, values, _ = _prepare(tmp_path, ftea_home=str(target))

        assert target.is_dir()
        assert Path(values["FTEA_HOME"]).samefile(target)

    def test_should_create_nothing_in_check_mode(self, tmp_path):
        ran, values, checkout = _prepare(
            tmp_path, ftea_home=str(tmp_path / "later"), mode="check")

        assert values["rc"] == "0"
        assert not (tmp_path / "later").exists()
        assert not (checkout / "input").exists()
        assert "bash install.sh will create it" in ran.out

    def test_should_agree_with_the_python_data_root(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _, values, _ = _prepare(tmp_path, ftea_home="shared", cwd=work)
        probe = "from ftea import paths; print(paths.data_root())"

        for home, cwd in ((values["FTEA_HOME"], REPO), ("shared", work)):
            env = {**os.environ, paths.ENV_HOME: home}
            done = subprocess.run(
                [sys.executable, "-c", probe], cwd=cwd, env=env,
                capture_output=True, text=True, check=True)
            assert Path(done.stdout.strip()).samefile(values["DATA_ROOT"])

    def test_should_say_where_it_created_the_data_folder(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()

        ran, values, _ = _prepare(tmp_path, ftea_home="new data", cwd=work)

        assert values["rc"] == "0"
        created = [line for line in ran.out.splitlines()
                   if line.startswith("  ✓ created the data folder ")]
        assert len(created) == 1, ran.out
        shown = created[0].split("created the data folder ", 1)[1]
        assert Path(shown.split(" (FTEA_HOME", 1)[0]).samefile(
            work / "new data")
        assert Path(shown.split(" (FTEA_HOME", 1)[0]).is_absolute()

    def test_should_not_say_created_for_an_existing_folder(self, tmp_path):
        (tmp_path / "there").mkdir()

        ran, values, _ = _prepare(
            tmp_path, ftea_home=str(tmp_path / "there"))

        assert values["rc"] == "0"
        assert "created the data folder" not in ran.out

    @pytest.mark.parametrize("mode", ["create", "check"])
    def test_should_refuse_an_ftea_home_that_is_a_file(self, tmp_path, mode):
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

        ran, values, checkout = _prepare(
            tmp_path, ftea_home=str(tmp_path / "notes.txt"), mode=mode)

        assert values["rc"] == "1"
        assert "✗ FTEA_HOME" in ran.out
        assert "names a file, not a folder" in ran.out
        assert "will create it" not in ran.out
        assert not (checkout / "input").exists()

    def test_should_fail_when_ftea_home_cannot_be_created(self, tmp_path):
        (tmp_path / "a file").write_text("x", encoding="utf-8")

        ran, values, _ = _prepare(
            tmp_path, ftea_home=str(tmp_path / "a file" / "data"))

        assert values["rc"] == "1"
        assert "✗ could not create the data folder" in ran.out
        assert "✓ data folder" not in ran.out

    def test_should_fail_when_input_cannot_be_created(self, tmp_path):
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "input").write_text("not a folder", encoding="utf-8")

        ran, values, _ = _prepare(tmp_path, ftea_home=None)

        assert values["rc"] == "1"
        assert "✗ could not create input/ and output/" in ran.out
        assert "✓ data folder" not in ran.out

    def test_should_fail_when_the_data_folder_cannot_be_opened(
        self, tmp_path,
    ):
        (tmp_path / "locked").mkdir()

        ran, values, _ = _prepare(
            tmp_path, ftea_home=str(tmp_path / "locked"),
            setup="cd() { return 1; }")

        assert values["rc"] == "1"
        assert "✗ could not open the data folder" in ran.out
        assert "✓ data folder" not in ran.out

    def test_should_refuse_a_tilde_it_cannot_expand_like_python(
        self, tmp_path,
    ):
        ran, values, _ = _prepare(tmp_path, ftea_home="~other/data")

        assert values["rc"] == "1"
        assert "✗ FTEA_HOME=~other/data" in ran.out
        assert list(tmp_path.glob("~*")) == []

    def test_should_stop_on_a_mode_it_does_not_know(self, tmp_path):
        ran, _, _ = _prepare(tmp_path, ftea_home=None, mode="crate")

        assert ran.code == 1
        assert "internal error" in ran.out
        assert not (tmp_path / "checkout" / "input").exists()


CHECK_STUBS = r'''
guard_one_os() { :; }
find_base_python() {
  BASE_PY=(base-python); BASE_PY_INFO="3.12 standard x"; ok "Python stub"
}
check_venv_support() { ok "venv stub"; }
check_gpu() { ok "gpu stub"; }
check_long_paths() { :; }
check_disks() { ok "disk stub"; }
check_ffmpeg_present() { ok "ffmpeg stub"; }
run_module() { say "RUN $*"; }
# A check-mode test must never reach a real installer: if --check ever
# falls through to the install path, these stop the test loudly.
ensure_venv() { die "test reached ensure_venv"; }
pip_in() { die "test reached pip_in"; }
'''


class TestCheckMode:
    def test_should_pass_after_the_ffmpeg_test_encodes(self):
        ran = run_bash(CHECK_STUBS + "main --check")

        assert ran.code == 0
        assert "=== ftea install: machine check only ===" in ran.out
        assert "RUN base-python -m ftea.install_checks toolchain" in ran.out
        assert ran.out.rstrip().endswith("=== machine check passed ===")

    def test_should_list_every_failed_check_before_stopping(self):
        ran = run_bash(CHECK_STUBS + r'''
find_base_python() { bad "no python"; return 1; }
check_gpu() { bad "no gpu"; return 1; }
main --check
''')

        assert ran.code == 1
        assert "✗ no python" in ran.out and "✗ no gpu" in ran.out
        assert "=== 2 check(s) failed: fix the ✗ lines above" in ran.out
        assert "RUN" not in ran.out

    def test_should_count_a_failed_ffmpeg_test_as_a_failed_check(self):
        ran = run_bash(CHECK_STUBS + r'''
run_module() { say "  ✗ libvmaf missing"; return 1; }
main --check
''')

        assert ran.code == 1
        assert "=== 1 check(s) failed" in ran.out

    def test_should_not_install_when_the_data_folder_cannot_be_made(
        self, tmp_path,
    ):
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "input").write_text("not a folder", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != paths.ENV_HOME}

        ran = run_bash(CHECK_STUBS + _sub(r'''
locate_checkout() { CHECKOUT='@DIR@'; }
fetch_models() { die "test reached fetch_models"; }
main
''', DIR=checkout.as_posix()), env=env)

        assert ran.code == 1
        assert "✗ could not create input/ and output/" in ran.out
        assert "=== 1 check(s) failed" in ran.out
        assert "test reached" not in ran.out
        assert "=== ftea is installed ===" not in ran.out

    def test_should_show_help_from_any_folder(self, tmp_path):
        ran = run_script("--help", cwd=tmp_path)

        assert ran.code == 0
        assert "Usage: bash install.sh [--check]" in ran.out
        assert "3.10 to 3.14" in ran.out and "FTEA_HOME" in ran.out
        assert "no arguments" in ran.out

    @pytest.mark.parametrize("args", [["--bogus"], ["--check", "--help"]],
                             ids=repr)
    def test_should_reject_unknown_or_extra_arguments(self, tmp_path, args):
        ran = run_script(*args, cwd=tmp_path)

        assert ran.code == 2
        assert "Usage: bash install.sh" in ran.out


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True)


class TestScriptFile:
    def test_should_be_tracked_as_an_executable_file(self):
        if shutil.which("git") is None:
            pytest.skip("git is not installed")
        staged = _git("ls-files", "-s", "--", "install.sh").stdout

        assert staged.startswith("100755 "), staged or "install.sh untracked"

    def test_should_never_run_sudo(self):
        for number, line in enumerate(
                INSTALL_SH.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bsudo\b", line):
                assert re.match(r"\s*(#|say |bad |warn |die )", line), (
                    f"install.sh:{number} runs sudo: {line.strip()}")

    def test_should_ask_for_bash_when_started_with_sh(self):
        dash = shutil.which("dash")
        if dash is None:
            pytest.skip("dash (Ubuntu's sh) is not installed")
        done = subprocess.run(
            [dash, INSTALL_SH.as_posix(), "--help"], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )

        assert done.returncode == 2
        assert b"install.sh needs bash. Run it as: bash install.sh" in (
            done.stdout)


VENV_BODY = r'''
PLATFORM=linux
CHECKOUT='@DIR@'
@BODY@
ensure_venv @NAME@
echo "rc=$? VENV_PY=$VENV_PY"
'''


def _ensure(tmp_path: Path, name: str, body: str) -> Ran:
    return run_bash(_sub(VENV_BODY, DIR=tmp_path.as_posix(), NAME=name,
                         BODY=body))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class TestVenvs:
    @pytest.mark.parametrize("present, expected", [
        (("Scripts/python.exe", "bin/python"), "Scripts/python.exe"),
        (("bin/python",), "bin/python"),
    ])
    def test_should_find_the_interpreter_in_the_contract_order(
        self, tmp_path, present, expected,
    ):
        venv = tmp_path / "v"
        for relative in present:
            _touch(venv / relative)

        ran = run_bash(_sub("venv_python '@DIR@'", DIR=venv.as_posix()))

        assert ran.out.strip() == f"{venv.as_posix()}/{expected}"
        first = next(p for p in paths.venv_python_candidates(venv)
                     if p.exists())
        assert first == venv / expected

    def test_should_reuse_a_working_environment(self, tmp_path):
        _touch(tmp_path / ".venv" / "bin" / "python")

        ran = _ensure(tmp_path, ".venv", r'''
python_probe() { echo "3.12 standard linux-x86_64"; }
venv_has_pip() { return 0; }
''')

        assert "✓ .venv: reusing it (3.12 standard linux-x86_64)" in ran.out
        assert ran.out.rstrip().endswith("/.venv/bin/python")

    def test_should_stop_on_an_environment_without_python(self, tmp_path):
        (tmp_path / ".venv-reframe").mkdir()

        ran = _ensure(tmp_path, ".venv-reframe", "")

        assert ran.code == 1
        assert ".venv-reframe exists but holds no Python" in ran.out
        assert "rm -rf" in ran.out

    def test_should_stop_on_an_environment_from_an_unusable_python(
        self, tmp_path,
    ):
        _touch(tmp_path / ".venv" / "bin" / "python")

        ran = _ensure(tmp_path, ".venv", r'''
python_probe() { echo "3.9 standard linux-x86_64"; }
''')

        assert ran.code == 1
        assert ".venv was made from Python 3.9" in ran.out
        assert "rm -rf" in ran.out

    def test_should_stop_on_an_environment_without_pip(self, tmp_path):
        _touch(tmp_path / ".venv-reframe" / "bin" / "python")

        ran = _ensure(tmp_path, ".venv-reframe", r'''
python_probe() { echo "3.12 standard linux-x86_64"; }
venv_has_pip() { return 1; }
''')

        assert ran.code == 1
        assert ".venv-reframe has no working pip" in ran.out
        assert "rm -rf" in ran.out

    def test_should_create_a_missing_environment_with_the_base_python(
        self, tmp_path,
    ):
        ran = _ensure(tmp_path, ".venv", r'''
fake_python() {
  if [ "$1 $2" = "-m venv" ]; then mkdir -p "$3/bin" && : > "$3/bin/python"; fi
}
BASE_PY=(fake_python)
''')

        assert "✓ .venv: created" in ran.out
        assert (tmp_path / ".venv" / "bin" / "python").is_file()

    def test_should_stop_when_venv_creation_fails(self, tmp_path):
        ran = _ensure(tmp_path, ".venv", "BASE_PY=(false)")

        assert ran.code == 1
        assert "✗ could not create .venv with false." in ran.out
        assert "VENV_PY=" not in ran.out

    def test_should_stop_when_the_new_venv_has_no_python(self, tmp_path):
        ran = _ensure(tmp_path, ".venv-reframe", r'''
fake_python() { mkdir -p "$3"; }
BASE_PY=(fake_python)
''')

        assert ran.code == 1
        assert ".venv-reframe was created but holds no Python" in ran.out


INSTALL_STUBS = r'''
guard_one_os() { :; }
check_machine() { BASE_PY=(base-python); ok "machine stub"; }
ensure_venv() { say "VENV $1"; VENV_PY="/venvs/$1/python"; }
pip_in() { local py="$1"; shift; say "PIP $py $*"; }
run_module() { say "RUN $*"; }
'''


def _steps(out: str) -> list[str]:
    return [line for line in out.splitlines()
            if line.split(" ", 1)[0] in ("RUN", "PIP", "VENV")]


class TestInstallOrder:
    def test_should_install_in_the_load_bearing_order(self):
        ran = run_bash(INSTALL_STUBS + "main")

        reframe = "/venvs/.venv-reframe/python"
        assert ran.code == 0
        assert _steps(ran.out) == [
            "RUN base-python -m ftea.reframe.provision models",
            "VENV .venv",
            "PIP /venvs/.venv/python install --quiet -e .",
            "VENV .venv-reframe",
            f"PIP {reframe} install -r ftea/reframe/requirements-torch.txt",
            f"RUN {reframe} -m ftea.reframe.provision torch-build",
            f"PIP {reframe} install -r ftea/reframe/requirements.txt",
            f"RUN {reframe} -m ftea.reframe.provision torch-build",
            f"RUN {reframe} -m ftea.reframe.provision torchvision-url",
            f"RUN {reframe} -m ftea.reframe.verify_env",
        ]
        assert ran.out.rstrip().endswith("=== ftea is installed ===")

    def test_should_stop_before_the_rest_when_torch_is_not_cuda(self):
        ran = run_bash(INSTALL_STUBS + r'''
run_module() {
  say "RUN $*"
  case "$*" in *torch-build*) return 1 ;; esac
}
main
''')

        assert ran.code == 1
        assert "torch did not pass its check" in ran.out
        assert "requirements.txt" not in "".join(
            line for line in _steps(ran.out) if line.startswith("PIP"))

    def test_should_stop_when_torch_fails_its_check_after_step_2(self):
        ran = run_bash(INSTALL_STUBS + r'''
TORCH_CHECKS=0
run_module() {
  say "RUN $*"
  case "$*" in
    *torch-build*)
      TORCH_CHECKS=$((TORCH_CHECKS + 1))
      [ "$TORCH_CHECKS" -lt 2 ] ;;
  esac
}
main
''')

        assert ran.code == 1
        assert "torch did not pass its check after step 2" in ran.out
        steps = _steps(ran.out)
        assert sum("torch-build" in step for step in steps) == 2
        assert not any("torchvision-url" in s or "verify_env" in s
                       for s in steps)
        assert "=== ftea is installed ===" not in ran.out

    def test_should_stop_when_torchvision_would_fetch_other_weights(self):
        ran = run_bash(INSTALL_STUBS + r'''
run_module() {
  say "RUN $*"
  case "$*" in *torchvision-url*) return 1 ;; esac
}
main
''')

        assert ran.code == 1
        assert "would not load the verified person-detector weights" in (
            ran.out)
        assert not any("verify_env" in s for s in _steps(ran.out))
        assert "=== ftea is installed ===" not in ran.out

    def test_should_stop_when_the_environment_check_fails(self):
        ran = run_bash(INSTALL_STUBS + r'''
run_module() {
  say "RUN $*"
  case "$*" in *verify_env*) return 1 ;; esac
}
main
''')

        assert ran.code == 1
        assert "the reframe stage's environment check failed" in ran.out
        assert "=== next steps ===" not in ran.out
        assert "=== ftea is installed ===" not in ran.out

    def test_should_stop_when_the_models_cannot_be_verified(self):
        ran = run_bash(INSTALL_STUBS + r'''
run_module() {
  say "RUN $*"
  case "$*" in *"provision models"*) return 1 ;; esac
}
main
''')

        assert ran.code == 1
        assert "could not be downloaded and verified" in ran.out
        assert [s for s in _steps(ran.out) if s.startswith("VENV")] == []

    def test_should_create_and_download_nothing_in_check_mode(self):
        ran = run_bash(INSTALL_STUBS + "main --check")

        assert ran.code == 0
        assert _steps(ran.out) == []
        assert "=== machine check passed ===" in ran.out

    def test_should_install_nothing_after_a_failed_check(self):
        ran = run_bash(INSTALL_STUBS + r'''
check_machine() { bad "stubbed failure"; }
main
''')

        assert ran.code == 1
        assert _steps(ran.out) == []
        assert "=== 1 check(s) failed" in ran.out


class TestContract:
    def test_should_run_only_python_modules_that_exist(self):
        text = INSTALL_SH.read_text(encoding="utf-8")
        modules = set(re.findall(r"-m (ftea(?:\.\w+)+)", text))

        assert modules == {
            "ftea.install_checks", "ftea.reframe.provision",
            paths.VERIFY_ENV_MODULE,
        }
        for name in sorted(modules):
            module = importlib.import_module(name)
            assert callable(getattr(module, "main", None)), name
            assert 'if __name__ == "__main__":' in (
                inspect.getsource(module)), name

    def test_should_install_both_requirement_files_by_name(self):
        text = INSTALL_SH.read_text(encoding="utf-8")

        for name in ("requirements-torch.txt", "requirements.txt"):
            assert f"ftea/reframe/{name}" in text
            assert (REPO / "ftea" / "reframe" / name).is_file()

    def test_should_never_install_pytest_into_the_reframe_environment(self):
        assert "pytest" not in INSTALL_SH.read_text(encoding="utf-8")

    def test_should_download_only_through_the_pinned_python_step(self):
        text = INSTALL_SH.read_text(encoding="utf-8")

        assert re.search(r"\b(curl|wget)\b", text) is None

    @pytest.mark.parametrize("platform, activate", [
        ("windows", "source .venv/Scripts/activate"),
        ("linux", "source .venv/bin/activate"),
    ])
    def test_should_say_how_to_start_using_ftea(self, platform, activate):
        ran = run_bash(_sub(r'''
PLATFORM=@PLATFORM@
CHECKOUT=/srv/ftea
DATA_ROOT=/srv/ftea-data
print_next_steps
''', PLATFORM=platform))

        assert activate in ran.out
        assert "ftea run" in ran.out and "/srv/ftea-data/input" in ran.out

    @pytest.mark.parametrize("home, shown", [
        ("D:/ftea-data", True), ("", False)])
    def test_should_say_to_keep_ftea_home_set(self, home, shown):
        ran = run_bash(_sub(r'''
PLATFORM=linux
CHECKOUT=/srv/ftea
DATA_ROOT=/srv/ftea-data
if [ -n '@HOME@' ]; then export FTEA_HOME='@HOME@'; else unset FTEA_HOME; fi
print_next_steps
''', HOME=home))

        line = 'export FTEA_HOME="D:/ftea-data"'
        assert (line in ran.out) is shown
        assert ("shell profile" in ran.out) is shown
