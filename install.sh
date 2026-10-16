#!/usr/bin/env bash
# install.sh: set up ftea on Windows (Git Bash) or Linux, NVIDIA RTX GPU.
#
#   bash install.sh          check this machine, download the model
#                            weights, create the two Python environments
#                            and run the reframe stage's environment check
#   bash install.sh --check  check this machine only: nothing is created
#                            or downloaded
#   bash install.sh --help
#
# Exit code 0 when everything passed, 1 when a check or a step failed, 2
# for a usage error. It never runs sudo. It writes only inside this
# folder, the data folder (FTEA_HOME when set; a full install creates it
# if it is missing), the models folder (REFRAME_MODELS_DIR when set,
# otherwise models/ in the data folder), pip's download cache, and the
# temp folder for scratch files. Run it again at any time: finished work
# is checked and kept.
#
# Every outside query sits in a one-line function, and main runs only
# when this file is executed, so the tests can source it and replace
# those functions.

# Started with sh (dash on Ubuntu), the arrays below are a syntax error.
# This first command is plain POSIX, and dash runs top-level commands one
# at a time, so sh stops here with the fix instead.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "install.sh needs bash. Run it as: bash install.sh" >&2
  exit 2
fi

# --- Assumptions, each kept in one place and re-checked when it runs ---
# Python versions the pinned CUDA libraries publish wheels for (checked
# 2026-10-15), most preferred first: the reframe stack was verified on
# 3.12.
PYTHON_PREFERENCE=(3.12 3.13 3.11 3.14 3.10)
# Below this NVIDIA driver version install.sh warns. The real test
# encodes decide; the number only explains a failure early.
MIN_DRIVER_MAJOR=570
# The reframe environment takes about 4.5 GB on Windows and more on
# Linux, and pip's download cache about as much again the first time.
MIN_FREE_GB=12
# With Windows long-path support off (the default), pip fails part-way
# through the torch install once a file's full path reaches 260
# characters. Above this many characters in the checkout's own path,
# install.sh warns. Derived from the deepest file in .venv-reframe.
MAX_SAFE_CHECKOUT_CHARS=90
LONG_PATHS_HELP="https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation"
NVIDIA_SMI_CANDIDATES=(nvidia-smi /usr/lib/wsl/lib/nvidia-smi)
FFMPEG_SOURCE_WINDOWS='the "full" build from https://www.gyan.dev/ffmpeg/builds/'
FFMPEG_SOURCE_LINUX='a linux64-gpl build from https://github.com/BtbN/FFmpeg-Builds/releases'
GUIDE="docs/PROCESS-GUIDE.md"

# Prints "<major>.<minor> <standard|free-threaded> <platform>".
PY_PROBE='import sys, sysconfig; print("%d.%d %s %s" % (sys.version_info[0], sys.version_info[1], "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "standard", sysconfig.get_platform()))'

FAILED=0
PLATFORM=""
CHECKOUT=""
DATA_ROOT=""
BASE_PY=()
BASE_PY_INFO=""
WSL_DISK_NOTED=""
NVIDIA_SMI=""
VENV_PY=""
ENC_PY=""
RF_PY=""

# --- Output -----------------------------------------------------------
say()     { printf '%s\n' "$*"; }
ok()      { say "  ✓ $*"; }
warn()    { say "  ! $*"; }
bad()     { say "  ✗ $*"; FAILED=$((FAILED + 1)); }
die()     { say "  ✗ $*"; exit 1; }
section() { say ""; say "=== $* ==="; }

# --- Outside queries (the tests replace these) ------------------------
os_name()         { uname -s; }
kernel_release()  { cat /proc/sys/kernel/osrelease 2>/dev/null || true; }
have()            { command -v "$1" >/dev/null 2>&1; }
python_probe()    { "$@" -c "$PY_PROBE" 2>/dev/null; }
python_has_venv() { "$@" -c 'import ensurepip, venv' >/dev/null 2>&1; }
venv_has_pip()    { "$1" -m pip --version >/dev/null 2>&1; }
nvidia_smi_query() {
  "$NVIDIA_SMI" --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
}
# df prints one named column at a time, so a filesystem name or a mount
# point with spaces in it ("C:/Program Files/Git", "/media/me/My
# Passport") cannot shift the columns. GNU df, as on Linux and in Git
# Bash, supports --output.
free_kb()  { df -k --output=avail -- "$1" 2>/dev/null | awk 'NR == 2 { print $1 }'; }
mount_of() { df --output=target -- "$1" 2>/dev/null | awk 'NR == 2'; }
# grep without -q reads to the end: under pipefail, grep -q stopping at
# the first match can make reg.exe fail on its next write.
long_paths_enabled() { MSYS_NO_PATHCONV=1 reg.exe query 'HKLM\SYSTEM\CurrentControlSet\Control\FileSystem' /v LongPathsEnabled 2>/dev/null | grep 'REG_DWORD[[:space:]]*0x1' >/dev/null; }

# --- Small helpers ----------------------------------------------------
trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

native_path() {   # a path spelled the way this platform's Python reads it
  if [ "$PLATFORM" = windows ]; then
    cygpath -m "$1"
  else
    printf '%s\n' "$1"
  fi
}

python_range() {
  local sorted
  sorted="$(printf '%s\n' "${PYTHON_PREFERENCE[@]}" | sort -V)"
  printf '%s to %s' "$(head -n 1 <<<"$sorted")" "$(tail -n 1 <<<"$sorted")"
}

usage() {
  cat <<EOF
Usage: bash install.sh [--check]

Sets up ftea: checks this machine (a standard 64-bit Python $(python_range),
an NVIDIA GPU and driver, and an FFmpeg with NVENC and libvmaf), downloads
the model weights, creates .venv (the encoder) and .venv-reframe (the
reframe stage: CUDA torch, about 4.5 GB), and runs the reframe stage's
environment check. Run it again at any time; finished work is kept.

  --check   check this machine only; nothing is created or downloaded
  --help    show this help

Environment:
  FTEA_HOME     the data folder for config.env, input/, output/, work/
                and models/ (default: this folder)
  FTEA_PYTHON   the Python to build the environments from: the path to
                one Python program, no arguments, such as
                C:/Python312/python.exe or /usr/bin/python3.12 (default:
                the first of ${PYTHON_PREFERENCE[*]} that is installed)
EOF
}

# --- Where we are -----------------------------------------------------
detect_platform() {
  local name
  name="$(os_name)"
  case "$name" in
    MINGW*|MSYS*) PLATFORM=windows ;;
    Linux) PLATFORM=linux ;;
    CYGWIN*) die "Cygwin is not supported: run install.sh from Git Bash on Windows." ;;
    *) die "this system ($name) is not supported: ftea needs Windows with Git Bash, or Linux, and an NVIDIA GPU." ;;
  esac
}

is_wsl() { [ "$PLATFORM" = linux ] && kernel_release | grep -qi microsoft; }

locate_checkout() {
  CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  if [ ! -f "$CHECKOUT/pyproject.toml" ] || [ ! -f "$CHECKOUT/ftea/paths.py" ]; then
    die "install.sh must stay at the top of the ftea repository; $CHECKOUT has no pyproject.toml and ftea/paths.py."
  fi
}

# A checkout belongs to one operating system: environments built by one
# cannot be used, or safely rebuilt, by the other.
guard_one_os() {
  local venv
  if is_wsl; then
    case "$CHECKOUT" in
      /mnt/[a-zA-Z]/*)
        die "this folder ($CHECKOUT) is on a Windows drive, and you are running Linux (WSL). A checkout belongs to one operating system. Run install.sh from Git Bash on Windows, or clone the repository into your Linux home folder and run it there, for example: git clone \"$CHECKOUT\" ~/Frametage-Encoder-Autoframer" ;;
    esac
  fi
  for venv in .venv .venv-reframe; do
    if [ "$PLATFORM" = linux ] && [ -e "$CHECKOUT/$venv/Scripts/python.exe" ]; then
      die "$venv was created on Windows, and this is Linux. A checkout belongs to one operating system: clone the repository again for Linux."
    fi
    if [ "$PLATFORM" = windows ] && [ -e "$CHECKOUT/$venv/bin/python" ] &&
      [ ! -e "$CHECKOUT/$venv/Scripts/python.exe" ]; then
      die "$venv was created on Linux, and this is Windows. A checkout belongs to one operating system: clone the repository again for Windows."
    fi
  done
}

# --- The data folder --------------------------------------------------
# FTEA_HOME is read the way ftea.paths.data_root reads it: blank means
# the checkout, ~ and ~/ expand (any other ~ form is refused rather than
# guessed at), and a relative path is taken from the current folder.
# One difference is deliberate: ftea itself refuses a
# FTEA_HOME that does not exist, so a typo cannot become a new data
# root, but a full install creates it and prints the absolute path it
# created, so a typo shows up right there (--check only reports it).
# A FTEA_HOME that names a file is refused in both modes.
#
# check_machine runs this with `|| true`, which turns set -e off inside
# it, so every step that can fail is checked by hand.
prepare_data_root() {   # $1: create | check
  local raw home
  case "$1" in
    create|check) ;;
    *) die "internal error: prepare_data_root got '$1', not create or check" ;;
  esac
  raw="$(trim "${FTEA_HOME:-}")"
  if [ -z "$raw" ]; then
    unset FTEA_HOME
    DATA_ROOT="$CHECKOUT"
  else
    home="$raw"
    case "$home" in
      "~") home="$HOME" ;;
      "~/"*) home="$HOME/${home#"~/"}" ;;
      "~"*)
        bad "FTEA_HOME=$raw: only ~ and ~/ are expanded here; write the full path to the data folder instead"
        return 1 ;;
    esac
    if [ "$PLATFORM" = windows ]; then
      home="$(cygpath -u "$home")"
    fi
    case "$home" in
      /*) ;;
      *) home="$PWD/$home" ;;
    esac
    if [ -e "$home" ] && [ ! -d "$home" ]; then
      bad "FTEA_HOME=$raw names a file, not a folder ($(native_path "$home")). Point FTEA_HOME at a folder."
      return 1
    fi
    if [ ! -d "$home" ] && [ "$1" = create ]; then
      if ! mkdir -p -- "$home"; then
        bad "could not create the data folder $(native_path "$home") (FTEA_HOME)"
        return 1
      fi
      ok "created the data folder $(native_path "$home") (FTEA_HOME=$raw)"
    fi
    if [ -d "$home" ]; then
      if ! DATA_ROOT="$(cd -- "$home" && pwd -P)"; then
        bad "could not open the data folder $(native_path "$home") (FTEA_HOME)"
        return 1
      fi
    else
      DATA_ROOT="$home"
      warn "the data folder $(native_path "$home") (FTEA_HOME) does not exist yet; bash install.sh will create it"
    fi
    FTEA_HOME="$(native_path "$DATA_ROOT")"
    export FTEA_HOME
  fi
  if [ "$1" = create ] && ! mkdir -p -- "$DATA_ROOT/input" "$DATA_ROOT/output"; then
    bad "could not create input/ and output/ in the data folder $(native_path "$DATA_ROOT")"
    return 1
  fi
  ok "data folder: $(native_path "$DATA_ROOT")"
}

# --- Python -----------------------------------------------------------
python_rejection() {   # $1: probe output; prints why it is unusable
  local version build platform want
  read -r version build platform <<<"$1"
  if ! printf '%s\n' "${PYTHON_PREFERENCE[@]}" | grep -qxF -- "$version"; then
    say "Python $version; ftea needs $(python_range), the versions its CUDA libraries are published for"
    return 0
  fi
  if [ "$build" != standard ]; then
    say "a free-threaded build of Python $version, which OpenCV publishes no package for"
    return 0
  fi
  want=win-amd64
  if [ "$PLATFORM" = linux ]; then want=linux-x86_64; fi
  if [ "$platform" != "$want" ]; then
    say "a $platform build of Python $version; ftea needs a 64-bit $want one"
  fi
}

python_candidates() {
  local v
  for v in "${PYTHON_PREFERENCE[@]}"; do
    if [ "$PLATFORM" = windows ]; then say "py -$v"; else say "python$v"; fi
  done
  if [ "$PLATFORM" = windows ]; then say python; fi
  say python3
}

python_install_hint() {
  if [ "$PLATFORM" = windows ]; then
    say "Install 64-bit Python 3.12 from https://www.python.org/downloads/ (it includes the py launcher), then run install.sh again."
  else
    say "Install Python with your package manager, for example: sudo apt install python3 python3-venv"
  fi
}

find_base_python() {
  local line info why chosen
  local -a cmd rejected=()
  chosen="$(trim "${FTEA_PYTHON:-}")"
  if [ -n "$chosen" ]; then
    cmd=("$chosen")
    if ! info="$(python_probe "${cmd[@]}")"; then
      bad "FTEA_PYTHON=$chosen does not run as Python."
      case "$chosen" in
        *" "*) say "      FTEA_PYTHON must be the path to one Python program, with no arguments (not a command such as \"py -3.12\"). Spaces inside the path are fine." ;;
      esac
      return 1
    fi
    why="$(python_rejection "$info")"
    if [ -n "$why" ]; then
      bad "FTEA_PYTHON=$chosen is $why."
      return 1
    fi
    BASE_PY=("${cmd[@]}")
    BASE_PY_INFO="$info"
    ok "Python: $chosen ($info)"
    return 0
  fi
  while IFS= read -r line; do
    read -r -a cmd <<<"$line"
    have "${cmd[0]}" || continue
    if ! info="$(python_probe "${cmd[@]}")"; then
      # The py launcher fails for any version that is not installed,
      # which is normal. Anything else that is on PATH but will not run
      # (the Microsoft Store alias, a broken install) is worth naming.
      if [ "${cmd[0]}" != py ]; then
        rejected+=("$line is on PATH but did not run")
      fi
      continue
    fi
    why="$(python_rejection "$info")"
    if [ -z "$why" ]; then
      BASE_PY=("${cmd[@]}")
      BASE_PY_INFO="$info"
      ok "Python: $line ($info)"
      return 0
    fi
    rejected+=("$line is $why")
  done < <(python_candidates)
  bad "no usable Python found. ftea needs a standard 64-bit Python $(python_range)."
  for line in "${rejected[@]}"; do say "      $line"; done
  say "      $(python_install_hint)"
  return 1
}

check_venv_support() {
  local version rest
  if python_has_venv "${BASE_PY[@]}"; then
    ok "Python can create virtual environments"
    return 0
  fi
  read -r version rest <<<"$BASE_PY_INFO"
  bad "this Python cannot create virtual environments (its ensurepip or venv module is missing)."
  if [ "$PLATFORM" = linux ]; then
    say "      On Ubuntu or Debian, install the venv package, then run install.sh again:"
    say "        sudo apt install python${version}-venv"
  else
    say "      $(python_install_hint)"
  fi
  return 1
}

# --- GPU, disk, FFmpeg ------------------------------------------------
check_gpu() {
  local candidate out line name driver major status=0
  NVIDIA_SMI=""
  for candidate in "${NVIDIA_SMI_CANDIDATES[@]}"; do
    if have "$candidate"; then NVIDIA_SMI="$candidate"; break; fi
  done
  if [ -z "$NVIDIA_SMI" ]; then
    bad "nvidia-smi not found. Install the NVIDIA driver for your GPU (on Linux, the proprietary driver), then run install.sh again."
    return 1
  fi
  # Read all of the output, then take the first line: under main's
  # pipefail, `| head -n 1` fails when nvidia-smi writes a second GPU's
  # line after head has exited, and the first line would be lost.
  out="$(nvidia_smi_query)" || status=$?
  line="$(trim "${out%%$'\n'*}")"
  if [ "$status" -ne 0 ]; then
    bad "nvidia-smi could not list the GPUs (exit $status${line:+: $line})."
    say "      Update or reinstall the NVIDIA driver (or restart, if it was just updated), then run install.sh again."
    return 1
  fi
  case "$line" in
    *,*) ;;
    *) bad "nvidia-smi found no NVIDIA GPU."; return 1 ;;
  esac
  name="$(trim "${line%,*}")"
  driver="$(trim "${line##*,}")"
  ok "GPU: $name, driver $driver"
  major="${driver%%.*}"
  if [[ "$major" =~ ^[0-9]+$ ]] && [ "$major" -lt "$MIN_DRIVER_MAJOR" ]; then
    warn "NVIDIA driver $driver is older than $MIN_DRIVER_MAJOR. If the FFmpeg test below fails, install a newer driver."
  fi
  return 0
}

# Windows only, and a warning only: a deep checkout folder plus torch's
# deepest header file can pass 260 characters while pip is installing.
check_long_paths() {
  local native
  [ "$PLATFORM" = windows ] || return 0
  native="$(native_path "$CHECKOUT")"
  if [ "${#native}" -le "$MAX_SAFE_CHECKOUT_CHARS" ] || long_paths_enabled; then
    return 0
  fi
  warn "this folder's path is ${#native} characters long and Windows long-path support is off, so pip can fail part-way through the torch install on a path that is too long. Clone the repository into a shorter folder such as C:\\ftea, or turn on long paths (needs an administrator): $LONG_PATHS_HELP"
}

check_disk() {   # $1: a folder (default: the checkout)
  local folder="${1:-$CHECKOUT}" kb gb
  kb="$(free_kb "$folder")" || kb=""
  if ! [[ "$kb" =~ ^[0-9]+$ ]]; then
    warn "could not read the free disk space ($folder)"
    return 0
  fi
  gb=$((kb / 1024 / 1024))
  if [ "$gb" -lt "$MIN_FREE_GB" ]; then
    warn "$gb GB free here ($folder); the reframe environment and pip's download cache need about $MIN_FREE_GB GB the first time"
  else
    ok "disk: $gb GB free ($folder)"
  fi
  # WSL's df sees its own virtual disk, never the Windows drive that
  # holds the disk image, so the number above can look fine when it is not.
  if is_wsl && [ -z "$WSL_DISK_NOTED" ]; then
    WSL_DISK_NOTED=1
    warn "WSL counts its own virtual disk here; the Windows drive that holds it needs about $MIN_FREE_GB GB free as well"
  fi
}

temp_dir() { printf '%s\n' "${TMPDIR:-${TEMP:-/tmp}}"; }

pip_cache_dir() {   # where pip keeps the downloaded wheels
  if [ "$PLATFORM" = windows ]; then
    cygpath -u "${PIP_CACHE_DIR:-${LOCALAPPDATA:-$HOME/AppData/Local}/pip/Cache}"
  else
    printf '%s\n' "${PIP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pip}"
  fi
}

existing_parent() {   # the nearest folder that exists, for df
  local d="$1"
  while [ ! -d "$d" ] && [ "$d" != / ] && [ "$d" != . ]; do
    d="$(dirname -- "$d")"
  done
  printf '%s\n' "$d"
}

# The torch wheels pass through pip's cache and the temp folder before
# they land in the checkout, and the models (about 80 MB) go to the
# models folder, which is in the data folder unless config.env moves it,
# so each filesystem among those is measured, once per mount point.
check_disks() {
  local folder mount seen=""
  for folder in "$CHECKOUT" "$DATA_ROOT" "$(temp_dir)" "$(pip_cache_dir)"; do
    [ -n "$folder" ] || continue
    folder="$(existing_parent "$folder")"
    mount="$(mount_of "$folder")" || mount=""
    [ -n "$mount" ] || mount="$folder"
    if printf '%s\n' "$seen" | grep -qxF -- "$mount"; then continue; fi
    seen="$seen$mount"$'\n'
    check_disk "$folder"
  done
}

ffmpeg_source() {
  if [ "$PLATFORM" = windows ]; then
    say "$FFMPEG_SOURCE_WINDOWS"
  else
    say "$FFMPEG_SOURCE_LINUX"
  fi
}

check_ffmpeg_present() {
  local program
  local -a missing=()
  for program in ffmpeg ffprobe; do
    have "$program" || missing+=("$program")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    ok "ffmpeg and ffprobe are on PATH"
    return 0
  fi
  bad "not on PATH: ${missing[*]}. ftea needs an FFmpeg built with NVIDIA NVENC and libvmaf."
  say "      Get $(ffmpeg_source), put its bin folder on PATH, then run install.sh again (see $GUIDE)."
  return 1
}

# The launch rule for every Python step: run from the checkout with
# PYTHONPATH set to it and nothing else. ftea is standard-library only,
# so this works before any virtual environment exists.
run_module() {
  ( cd -- "$CHECKOUT" && PYTHONPATH="$(native_path "$CHECKOUT")" "$@" )
}

check_toolchain() {
  run_module "${BASE_PY[@]}" -m ftea.install_checks toolchain
}

check_machine() {   # $1: create | check
  local have_python=0
  section "checking this machine"
  ok "platform: $PLATFORM ($(os_name))"
  prepare_data_root "$1" || true
  if find_base_python; then
    have_python=1
    check_venv_support || true
  fi
  check_gpu || true
  check_long_paths
  check_disks
  if check_ffmpeg_present && [ "$have_python" = 1 ]; then
    check_toolchain || FAILED=$((FAILED + 1))
  fi
}

# --- Installing -------------------------------------------------------
venv_python() {   # $1: a virtual environment folder (Windows layout first)
  local relative
  for relative in Scripts/python.exe bin/python; do
    if [ -e "$1/$relative" ]; then
      say "$1/$relative"
      return 0
    fi
  done
  return 1
}

ensure_venv() {   # $1: folder name under the checkout; sets VENV_PY
  local dir="$CHECKOUT/$1" py info why
  if [ -e "$dir" ]; then
    py="$(venv_python "$dir")" ||
      die "$1 exists but holds no Python. Delete it and run install.sh again: rm -rf \"$dir\""
    info="$(python_probe "$py")" ||
      die "$1 exists but its Python does not start (was the Python it was made from removed?). Delete it and run install.sh again: rm -rf \"$dir\""
    why="$(python_rejection "$info")"
    [ -z "$why" ] ||
      die "$1 was made from $why. Delete it and run install.sh again: rm -rf \"$dir\""
    venv_has_pip "$py" ||
      die "$1 has no working pip (an earlier run was probably interrupted while creating it). Delete it and run install.sh again: rm -rf \"$dir\""
    ok "$1: reusing it ($info)"
  else
    say "  creating $1 with ${BASE_PY[*]} ..."
    ( cd -- "$CHECKOUT" && "${BASE_PY[@]}" -m venv "$1" ) ||
      die "could not create $1 with ${BASE_PY[*]}."
    py="$(venv_python "$dir")" || die "$1 was created but holds no Python."
    ok "$1: created"
  fi
  VENV_PY="$py"
}

pip_in() {   # $1: an interpreter; the rest: pip arguments, from the checkout
  local py="$1"
  shift
  ( cd -- "$CHECKOUT" && "$py" -m pip --disable-pip-version-check --no-input "$@" )
}

fetch_models() {
  section "model weights"
  run_module "${BASE_PY[@]}" -m ftea.reframe.provision models ||
    die "the model weights could not be downloaded and verified (see the ✗ lines above)."
}

install_encoder() {
  section "encoder environment (.venv)"
  ensure_venv .venv
  ENC_PY="$VENV_PY"
  pip_in "$ENC_PY" install --quiet -e . ||
    die "pip could not install ftea into .venv (see the messages above)."
  ok "ftea is installed in .venv"
}

# The order is load-bearing: CUDA torch first, everything else after,
# and the torch build checked after each step, because a later package
# can silently replace it with a build from PyPI.
install_reframe() {
  section "reframe environment (.venv-reframe)"
  say "  about 4.5 GB on Windows and more on Linux; 5 to 15 minutes the first time"
  ensure_venv .venv-reframe
  RF_PY="$VENV_PY"
  say "  step 1 of 2: torch and torchvision from the CUDA package index"
  pip_in "$RF_PY" install -r ftea/reframe/requirements-torch.txt ||
    die "pip could not install torch (see the messages above)."
  run_module "$RF_PY" -m ftea.reframe.provision torch-build ||
    die "torch did not pass its check: see the ✗ lines above."
  say "  step 2 of 2: everything else"
  pip_in "$RF_PY" install -r ftea/reframe/requirements.txt ||
    die "pip could not install the reframe stage's other packages (see the messages above)."
  run_module "$RF_PY" -m ftea.reframe.provision torch-build ||
    die "torch did not pass its check after step 2: see the ✗ lines above."
}

verify_reframe() {
  section "reframe stage environment check"
  run_module "$RF_PY" -m ftea.reframe.provision torchvision-url ||
    die "torchvision would not load the verified person-detector weights (see above)."
  run_module "$RF_PY" -m ftea.reframe.verify_env ||
    die "the reframe stage's environment check failed: fix the ✗ lines above, then run install.sh again."
}

print_next_steps() {
  local activate=".venv/bin/activate"
  if [ "$PLATFORM" = windows ]; then activate=".venv/Scripts/activate"; fi
  section "next steps"
  say "  cd \"$CHECKOUT\""
  say "  source $activate"
  if [ -n "${FTEA_HOME:-}" ]; then
    say "  export FTEA_HOME=\"$FTEA_HOME\"   # in every new shell: put it in your shell profile"
  fi
  say "  ftea run                 # encodes every video in $DATA_ROOT/input"
  say "  ftea preview <a clip folder in $DATA_ROOT/output>"
  say "  The guide: $GUIDE"
}

# --- Main -------------------------------------------------------------
main() {
  set -euo pipefail
  local mode
  case "$#:${1:-}" in
    "0:") mode=install ;;
    "1:--check") mode=check ;;
    "1:-h"|"1:--help") usage; return 0 ;;
    *) usage >&2; return 2 ;;
  esac
  if [ "$mode" = check ]; then
    say "=== ftea install: machine check only ==="
  else
    say "=== ftea install ==="
  fi
  detect_platform
  locate_checkout
  guard_one_os
  FAILED=0
  if [ "$mode" = check ]; then check_machine check; else check_machine create; fi
  if [ "$FAILED" -gt 0 ]; then
    section "$FAILED check(s) failed: fix the ✗ lines above, then run install.sh again"
    return 1
  fi
  if [ "$mode" = check ]; then
    section "machine check passed"
    return 0
  fi
  fetch_models
  install_encoder
  install_reframe
  verify_reframe
  print_next_steps
  section "ftea is installed"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
  exit $?
fi
