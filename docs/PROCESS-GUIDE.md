# ftea process guide

This guide takes you from a clean machine to a vertical clip playing on
your phone. Follow it in order the first time. After that, sections 4 to
9 are the ones you come back to.

1. [Requirements](#1-requirements)
2. [Installing](#2-installing)
3. [Preparing a video](#3-preparing-a-video)
4. [Running it](#4-running-it)
5. [Reading the output](#5-reading-the-output)
6. [Watching it on your phone](#6-watching-it-on-your-phone)
7. [Which phones get which encode](#7-which-phones-get-which-encode)
8. [Hosting the output on a website](#8-hosting-the-output-on-a-website)
9. [Tuning the settings](#9-tuning-the-settings)
10. [Troubleshooting](#10-troubleshooting)

## 1. Requirements

### GPU and driver

- An NVIDIA GeForce RTX 40-series or 50-series GPU, or another NVIDIA
  GPU whose NVENC encoder can encode AV1. `ftea` encodes every variant
  on the GPU, and NVIDIA added AV1 encoding with the RTX 40 series, so on
  an older card the install check stops with "AV1 hardware encoding needs
  an NVIDIA RTX 40-series GPU or newer."
- A current NVIDIA driver. `install.sh` warns when the driver is older
  than version 570; the test encodes it runs decide whether the driver
  actually works.
- The reframe stage runs its person detector on the GPU too. If it runs
  out of GPU memory, see
  [Out of memory while reframing](#out-of-memory-while-reframing).

### Operating system

- **Windows 10 or 11** with [Git for Windows](https://git-scm.com/download/win),
  which includes Git Bash. Run every command in this guide in Git Bash.
- **Linux (x86_64).** `install.sh` is written for Ubuntu and Debian:
  when something is missing, it names the `apt` package to install.
  `install.sh`, the test suite and a full encode ran on Ubuntu 24.04
  under WSL2, and `ftea preview` served the package it made to the same
  PC's browser. WSL2 uses the Windows NVIDIA driver, so a native Linux
  NVIDIA driver has not been verified yet.
- **WSL2** counts as Linux. Clone the repository into your Linux home
  folder, not onto a Windows drive under `/mnt/c`: `install.sh` refuses a
  checkout on a Windows drive, because a checkout belongs to one
  operating system. A phone usually cannot reach `ftea preview` running
  inside WSL; see [WSL](#wsl).

### Python

A standard 64-bit CPython from 3.10 to 3.14. The reframe stage's
packages were verified with Python 3.12, and `install.sh` prefers it when
several versions are installed. Free-threaded builds do not work,
because OpenCV publishes no package for them. The range moves when the
pinned torch version does.

- **Windows:** install Python from https://www.python.org/downloads/. It
  includes the `py` launcher, which `install.sh` uses to find each
  installed version. The reframe stage's torch also needs the Microsoft
  Visual C++ Redistributable (x64), which most Windows machines already
  have; without it, the install's reframe check fails with a message
  that links the installer (see
  [torch is a CPU-only build](#torch-is-a-cpu-only-build)).
- **Ubuntu or Debian:** the system `python3` works, but creating a
  virtual environment needs a separate package. `install.sh` names it when
  it is missing, for example `sudo apt install python3.12-venv`. Refresh
  the package lists in the same step, or `apt` may not find it:
  `sudo apt update && sudo apt install python3.12-venv`.
- To choose an interpreter yourself, set `FTEA_PYTHON` to the path of one
  Python program, for example
  `FTEA_PYTHON=C:/Python312/python.exe bash install.sh`.

### FFmpeg

`ftea` runs `ffmpeg` and `ffprobe` from your `PATH`. It needs one build
that has all of these:

- the NVIDIA encoders `h264_nvenc`, `hevc_nvenc` and `av1_nvenc`
- the `libvmaf` filter, which measures quality
- a software AV1 decoder, `libdav1d` or `libaom-av1`, which reads the AV1
  test encodes back for scoring
- the `libwebp` encoder, for the default `.webp` thumbnail (or set
  `THUMB_FORMAT=jpg`)

Builds that have them:

- **Windows:** the "full" build from https://www.gyan.dev/ffmpeg/builds/
  (the file `ffmpeg-8.1.2-full_build.7z`, in the page's 8.1.2 release
  section; not the "essentials" build). The 8.1 full build passes
  `ftea`'s checks. The page's `ffmpeg-release-full.7z` is now FFmpeg 9,
  which `ftea` has not been checked with yet. Unpack the file with File
  Explorer on recent Windows 11 or with [7-Zip](https://www.7-zip.org).
  It unpacks to a folder named after its version, such as
  `ffmpeg-8.1.2-full_build`: rename that folder to `C:\ffmpeg`, so that
  `ffmpeg.exe` is in `C:\ffmpeg\bin`. Then put that `bin` folder on your
  `PATH`: in Git Bash, add `export PATH="/c/ffmpeg/bin:$PATH"` to
  `~/.bashrc`, then open a new Git Bash window (or run
  `source ~/.bashrc`) so that it takes effect. `which ffmpeg` should then
  print `/c/ffmpeg/bin/ffmpeg`.
- **Linux:** a linux64-gpl build from
  https://github.com/BtbN/FFmpeg-Builds/releases, for example
  `ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz`. Unpack it with
  `tar xf <file>` and put its `bin` folder on your `PATH`.

You do not have to check the list by hand. `bash install.sh --check`
asks each NVIDIA encoder to encode a test frame and runs a short AV1 and
H.264 test encode with your settings, so a build that lists an encoder
but cannot use it is caught before anything is installed.

### Disk space

About 12 GB free for the first install. The reframe stage's environment
takes about 4.5 GB on Windows and about 7 GB on Linux, pip's download
cache takes about 3 to 4 GB more the first time, and the model weights
about 80 MB. Under WSL, the Windows drive that holds WSL's virtual disk
needs that space too.

Each finished clip then takes about 72 MB per minute of video for its six
variants, plus their audio tracks (the average over the clips described
in [How it works](HOW-IT-WORKS.md#about-these-numbers)).

### Time per clip

On an NVIDIA GeForce RTX 5080, 33 reframed clips with a median length of
66.9 seconds took a median of 441.3 seconds each from start to finish,
and the middle 80% of them took between 329.9 and 625.3 seconds. That is
6.75 seconds of processing per second of video; reframing took a median
of 34.0 seconds of it. These timings come from the production pipeline
`ftea` was forked from, on the GPU its operator reported. A slower GPU
takes longer. Encoding's time limit grows with the clip's length.
Reframing gets a fixed `REFRAME_TIMEOUT` per clip, so raise it for
videos of several minutes on a slower GPU. A retry multiplies every
limit by `RETRY_TIMEOUT_MULTIPLIER`.

## 2. Installing

Clone the repository and check the machine first. On Windows, run these
in Git Bash, and use a short folder such as `C:\ftea` to avoid the
[long paths](#long-paths-on-windows) problem.

```bash
cd /c                    # on Linux: cd ~
git clone https://github.com/arkevo/Frametage-Encoder-Autoframer.git ftea
cd ftea
bash install.sh --check
```

`bash install.sh --check` creates and downloads nothing. It checks the
platform, the data folder, Python, the GPU and driver, free disk space
and FFmpeg, lists every problem it finds, and ends with
`=== machine check passed ===` or
`=== N check(s) failed: fix the ✗ lines above, then run install.sh again ===`.
Each line starts with `✓` (passed), `✗` (failed) or `!` (a warning that
does not stop anything).

When the check passes, install:

```bash
bash install.sh
```

It runs the same checks, then:

1. downloads the three model weights into `models/` and checks each one
   against its pinned size and SHA-256
2. creates `.venv`, the encoder's environment, and installs `ftea` into
   it (the encoder needs nothing but Python's standard library)
3. creates `.venv-reframe`, the reframe stage's environment: the CUDA
   builds of torch and torchvision from PyTorch's CUDA 12.8 package index
   first, then OpenCV, checking after each step that torch is still the
   CUDA build
4. runs the reframe stage's environment check

The first run takes 5 to 15 minutes. You can run `bash install.sh` again
at any time: finished work is checked and kept. It never runs `sudo`,
and it never creates `config.env`.

### What a successful check looks like

On Windows, with the repository in `C:\ftea`:

```text
=== ftea install: machine check only ===

=== checking this machine ===
  ✓ platform: windows (MINGW64_NT-10.0-22631)
  ✓ data folder: C:/ftea
  ✓ Python: py -3.12 (3.12 standard win-amd64)
  ✓ Python can create virtual environments
  ✓ GPU: NVIDIA GeForce RTX 5080, driver 616.64
  ✓ disk: 65 GB free (/c/ftea)
  ✓ disk: 65 GB free (/tmp)
  ✓ ffmpeg and ffprobe are on PATH
  testing FFmpeg with real encodes on the GPU ...
    (no C:\ftea\config.env; testing the default settings)
  ✓ FFmpeg: h264_nvenc, av1_nvenc and hevc_nvenc each encoded a test frame, libvmaf is present, and AV1 and H.264 test encodes with your settings succeeded

=== machine check passed ===
```

### What a successful install ends with

```text
=== reframe stage environment check ===
  ✓ torchvision will load the verified person-detector weights
[ WARN:0@0.201] global net_impl_backend.cpp:345 cv::dnn::dnn5_v20260605::Net::Impl::setPreferableTarget Targets are not supported by the new graph engine for now
=== ftea reframe stage: environment check ===
    interpreter: C:\ftea\.venv-reframe\Scripts\python.exe
    models: C:\ftea\models

=== torch / CUDA ===
  ✓ torch: 2.11.0+cu128 (CUDA 12.8)
  ✓ CUDA device: NVIDIA GeForce RTX 5080
  ✓ native kernels for compute capability 12.0: sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
  ✓ CUDA kernel vs CPU reference: max abs err 0.00e+00

=== vision stack ===
  ✓ torchvision: 0.26.0+cu128
  ✓ cv2: 5.0.0
  ✓ person detector: torchvision faster-rcnn mobilenet_v3 (BSD-3)
  ✓ YuNet face weights: C:\ftea\models\face_detection_yunet_2023mar.onnx (232589 bytes)
  ✓ PP-OCRv3 text weights: C:\ftea\models\text_detection_en_ppocrv3_2023may.onnx (2423490 bytes)
  ✓ face detector: YuNet (MIT)
  ✓ text detector: PP-OCRv3 DB (Apache-2.0)

=== ffmpeg / NVENC / filters ===
  ✓ ffprobe on PATH: C:\ffmpeg\bin\ffprobe.EXE
  ✓ ffmpeg on PATH: C:\ffmpeg\bin\ffmpeg.EXE
  ✓ encoder hevc_nvenc
  ✓ encoder h264_nvenc
  ✓ hevc_nvenc session opens
  ✓ crop x/y runtime-commandable: dynamic crop supported
  ✓ filter sendcmd

=== all checks passed ===

=== next steps ===
  cd "/c/ftea"
  source .venv/Scripts/activate
  ftea run                 # encodes every video in /c/ftea/input
  ftea preview <a clip folder in /c/ftea/output>
  The guide: docs/PROCESS-GUIDE.md

=== ftea is installed ===
```

The `[ WARN … ]` line comes from OpenCV and does no harm; it also
appears when a run reframes.

### Activating ftea

Activate the encoder's environment in every new shell before you use
`ftea`:

```bash
source .venv/Scripts/activate     # Windows (Git Bash)
source .venv/bin/activate         # Linux
ftea --version
```

### Where things go

| What | Where | What it is |
| --- | --- | --- |
| `config.env` | data folder | Your settings. Optional; never committed |
| `input/` | data folder | The videos `ftea run` picks up by default |
| `output/` | data folder | One folder per finished clip, and `batch_report.json` |
| `work/` | data folder | Scratch space while a clip is encoded; a failed clip's folder is kept for debugging |
| `models/` | data folder | The model weights `install.sh` downloads |
| `.venv/` | checkout | The encoder's environment |
| `.venv-reframe/` | checkout | The reframe stage's environment |

The data folder is the checkout, unless you set the `FTEA_HOME`
environment variable to another folder. That folder must exist
(`install.sh` creates it), and the variable must be set in every shell
that runs `ftea`, for example in `~/.bashrc`.

### Long paths on Windows

With Windows' long-path support off, which is the default, pip can fail
part-way through installing torch once a file's full path reaches 260
characters. `install.sh` warns when the checkout's own path is too long
for that. Clone into a short folder such as `C:\ftea`, or turn long
paths on, which needs an administrator:
https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation

### Behind a package mirror

`ftea/reframe/requirements-torch.txt` names PyTorch's CUDA package index
itself, so pip's `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL` do not apply
to the torch step (proxy and cache settings such as `HTTPS_PROXY` and
`PIP_NO_CACHE_DIR` do). Behind a mirror, run `bash install.sh` once so it
creates `.venv-reframe`, install the torch step by hand from your
mirror's copy of the CUDA 12.8 index, and run `bash install.sh` again; it
finds the pinned versions installed and keeps them:

```bash
.venv-reframe/Scripts/python.exe -m pip install --index-url https://your-mirror.example/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
bash install.sh
```

On Linux, use `.venv-reframe/bin/python`.

### Updating and removing

To update, pull the latest version and run the installer again; it keeps
what is still current:

```bash
git pull
bash install.sh
```

To remove `ftea`, delete its folder, and the data folder if you set
`FTEA_HOME`. Nothing is installed anywhere else, apart from pip's
download cache.

## 3. Preparing a video

Put videos in the `input` folder, or name a file or a folder on the
command line. `ftea` reads `.mp4`, `.mov`, `.mkv` and `.avi` files with
any codec your FFmpeg can decode. A folder is read one level deep:
videos in its subfolders are not picked up.

What happens to a video depends on its shape, read after any rotation
the file records, so a phone video filmed upright counts as upright:

| Shape | What `ftea` does |
| --- | --- |
| Landscape (wider than tall) | Reframes it to 9:16, following the people in it, then encodes vertical variants |
| Vertical 9:16 | Encodes it as it is |
| Square, or vertical but not 9:16 | Pads it with black bars onto the 9:16 frame, without stretching, then encodes vertical variants |
| Landscape, with reframing off | Encodes landscape variants, padding a frame that is not 16:9 onto 16:9 |

A shot with nobody in it gets a still, centred crop. That is a normal
result, not a failure. The person detector can also report a subject in
a shot that has no people in it, and the crop then follows that instead
of staying centred; the [roadmap](../ROADMAP.md) lists this.

### Split long videos into clips

`ftea` is made for short clips. A video longer than 10 minutes
(`MAX_CLIP_SEC=600`) is skipped with
`SKIP (<length> long; ftea is made for short clips — split it first, or raise MAX_CLIP_SEC): <file>`,
and naming such a file on its own stops the run. Splitting first is
better for three reasons:

- each clip becomes its own package, which is what a viewer starts,
  stops and shares
- a clip that fails is retried or encoded again on its own, instead of
  costing you the whole video
- encoding has a per-clip time limit that grows with the clip's length
  (`CLIP_TIMEOUT` plus `CLIP_TIMEOUT_PER_SEC` for every second), so a
  short clip that hangs is noticed quickly

One way to split a video into parts of about 90 seconds without
re-encoding it (each cut lands on the next keyframe, so the parts vary
in length):

```bash
ffmpeg -i long-video.mp4 -map 0:v:0 -map "0:a:0?" -c copy -f segment -segment_time 90 -reset_timestamps 1 input/long-video-%02d.mp4
```

For the best result, cut at natural scene changes in a video editor
instead.

### Re-exporting a video under the same name

`ftea` remembers which file each finished clip was made from, by its size
and a fingerprint of its first and last megabyte. If you replace
`input/my-clip.mp4` with a new export and run again, the finished clip
is left alone, with
`SKIP: output/my-clip/ was made from a different my-clip.mp4; rename the file, or delete that folder to encode it again.`
To encode the new file, rename it, delete `output/my-clip`, or run
`ftea run --force input/my-clip.mp4`. Two files with the same name and
different extensions, such as `my-clip.mp4` and `my-clip.mov`, would
write the same folder, so the second one is skipped.

## 4. Running it

Activate the encoder's environment, then run:

```bash
source .venv/Scripts/activate          # on Linux: source .venv/bin/activate
ftea run                               # every video in the input folder
ftea run ~/Videos/my-clip.mp4          # one video, wherever it is
ftea run ~/Videos/shoot                # every video directly inside a folder
ftea run --no-reframe                  # this run only: no reframing
ftea run --force input/my-clip.mp4     # encode a finished clip again
```

Finished clips are skipped, so running again after a failure or an
interruption only does the work that is left. Only one `ftea run` can
use a data folder at a time; a second one stops with
`another ftea run (process <id>) is using <data folder>`. `--force`
keeps the old package until the new one is complete, and puts it back if
the new attempt fails.

### What each phase prints

A run has four phases. This is a real run over three clips:
`my-clip.mp4` is landscape, `my-portrait.mp4` is vertical, and
`my-rotated.mp4` is stored as landscape with a rotation that makes it
upright, as many phones record video. It is shortened where it shows
`…`: the reframe stage's progress lines for the first clip, and the
rotation in the third clip's reframe line:

```text
  No config.env at C:\ftea\config.env; using the defaults (see config.env.example).

============================================================
PHASE 1: PRE-SCAN
============================================================
  Videos: 3, already done: 0, to encode: 3
  WARNING: this FFmpeg's av1_nvenc has no -film_grain option, so FILM_GRAIN and FILM_GRAIN_BW are not applied; clip reports record grain 0.
  Auto-reframe: ON
  Checking the reframe environment...

============================================================
PHASE 2: REFRAME + ENCODE
============================================================

[1/3] my-clip.mp4
----------------------------------------
  Reframing (limit 600s)...
  reframe: no config file at C:\ftea\config.env; using the built-in defaults
  reframe: 1920x1080 landscape - reframing
  reframe: 11 shot(s)
  …
  Analyzing source...
  portrait, 24.0fps, color, 12 scenes, audio
  Probing CQ (12 scenes x 6 variants)...
  Encoding 144 probes...
  Scoring VMAF with 8 workers...
  Final encode + concat...
  HLS packaging...
  OK (381.6s)

[2/3] my-portrait.mp4
----------------------------------------
  Reframing (limit 600s)...
  reframe: no config file at C:\ftea\config.env; using the built-in defaults
  reframe: skipped: already vertical (1080x1920)
  Analyzing source...
  portrait, 24.0fps, color, 3 scenes, audio
  Probing CQ (3 scenes x 6 variants)...
  Encoding 36 probes...
  Scoring VMAF with 8 workers...
  Final encode + concat...
  HLS packaging...
  OK (120.4s)

[3/3] my-rotated.mp4
----------------------------------------
  Reframing (limit 600s)...
  reframe: no config file at C:\ftea\config.env; using the built-in defaults
  reframe: skipped: already vertical (1080x1920, …)
  Analyzing source...
  portrait, 24.0fps, color, 2 scenes, audio
  Probing CQ (2 scenes x 6 variants)...
  Encoding 24 probes...
  Scoring VMAF with 8 workers...
  Final encode + concat...
  HLS packaging...
  OK (72.4s)

============================================================
PHASE 3: RETRY
============================================================

============================================================
PHASE 4: REPORT
============================================================
  Clips encoded this run: 3 (passed 0, warning 3, failed 0)
  Wall time: 575s (9.6m)
  Batch report: C:\ftea\output\batch_report.json

Watch it on this computer or a phone:
  ftea preview my-clip
  ftea preview my-portrait
  ftea preview my-rotated
```

- **Pre-scan** finds the videos, skips the finished ones, and tests
  FFmpeg and the reframe environment once, and only when there is
  something to encode. The 8.1 builds in [FFmpeg](#ffmpeg) print the
  film-grain `WARNING` above; see `FILM_GRAIN_BW` in
  [Tuning the settings](#9-tuning-the-settings).
- **Reframe + encode** handles each clip in turn: reframing, then scene
  analysis, two test encodes per scene and variant ("Probing CQ"), the
  final encode, and packaging. A clip that is not landscape skips
  reframing, as the second and third clips do here. With reframing off,
  this phase is called `ENCODE`.
- **Retry** gives each failed clip up to two more attempts (see
  `RETRY_MAX` in [Tuning the settings](#9-tuning-the-settings)).
- **Report** writes `output/batch_report.json`, prints the totals, and
  then prints the `ftea preview` command for each clip the run
  finished, up to five.

Press Ctrl+C to stop a run; it exits with code 130, and the next run
picks up where it stopped.

### Making only a vertical video with ftea reframe

`ftea reframe` runs only the reframe stage on one video and writes a
vertical MP4, without encoding the HLS package:

```bash
ftea reframe ~/Videos/talk.mp4                   # writes output/talk-vertical/
ftea reframe ~/Videos/talk.mp4 --debug-video     # also writes debug.mp4
ftea reframe ~/Videos/talk.mp4 --out ~/Videos/talk-vertical
```

The results folder, `output/<name>-vertical/` unless you name another
with `--out`, holds:

- `vertical.mp4`: the reframed video, rendered at high quality (HEVC at
  CQ 10), not the phone package; use `ftea run` for that
- `crops.json`: the crop window for every frame
- `debug.mp4`, with `--debug-video`: the original on the left with the
  crop window drawn as a red rectangle, and the 9:16 result on the right,
  720 pixels tall. It is the quickest way to see what the camera did
- `ftea-reframe.json`: which video the folder belongs to

The folder must be new, empty, or hold that same video's earlier
results, so one folder holds one video. `ftea reframe` refuses anything
else before it starts, and names the folder. A video that is not
landscape is left alone: it prints `Nothing to reframe: …`, writes
nothing and exits 0. After a failure, the partial files stay in the
folder's `.partial` subfolder. Reframing gets `REFRAME_TIMEOUT` seconds,
twice that with `--debug-video`. `ftea run` never replaces a
`-vertical` folder made by `ftea reframe`.

### Exit codes

| Code | `ftea run` | `ftea preview` | `ftea reframe` |
| --- | --- | --- | --- |
| 0 | Every clip finished | Stopped with Ctrl+C | Done, or the video is not landscape |
| 1 | A clip failed or was not attempted, or a setup problem | Nothing to preview, or no free port | The reframe failed, the debug video failed, or a setup problem |
| 2 | The arguments were not understood | The same | The same |
| 130 | Interrupted with Ctrl+C | (not used) | Interrupted with Ctrl+C |

## 5. Reading the output

### The clip folder

```text
output/
├── batch_report.json      the summary of the last run that encoded something
└── my-clip/
    ├── master.m3u8        the playlist a player opens first
    ├── av1_1080p/         stream.m3u8, init_av1_1080p.mp4, av1_1080p_000.m4s, ...
    ├── av1_720p/  av1_360p/  h264_1080p/  h264_720p/  h264_360p/
    ├── thumbnail.webp     the first frame
    ├── player.html  hls.min.js  hls.js-LICENSE.txt
    ├── report.md  report.json
    └── crops.json         the crop window for every frame (reframed clips)
```

`master.m3u8` lists the six variants. Each variant folder holds its own
playlist and its 2-second segments, each with the audio track. The folder
is complete on its own: you can copy it anywhere and play it.

### The quality report

`report.md` is the readable version:

- a summary table: the source file name, its orientation, frame rate,
  whether it is black and white, the film grain used, the number of
  scenes, audio, the encode time and the status
- **Variant Summary**: for each variant, the mean, lowest and spread
  (standard deviation) of its scenes' VMAF scores, its average video
  bitrate and its size
- **Per-Scene: AV1** and **Per-Scene: H264**: for each scene, its length,
  and the quality setting (CQ) chosen and the VMAF reached at each size

VMAF compares the encode with its source and scores it from 0 to 100;
[How it works](HOW-IT-WORKS.md#how-quality-is-measured) explains how
`ftea` measures it, and why the 360p scores sit in the 70s.

`report.json` holds the same and more, for scripts: `video_id`,
`source` (the file name only), `orientation`, `fps`, `is_bw`,
`film_grain`, `has_audio`, `scene_count`, `duration_s`,
`encode_time_sec`, `scenes`, `variants`, `status`, `error`, the source's
size and fingerprint, and `reframe` or `conform` when they applied.
Folder names in messages are written as `<data>`, `<checkout>` and `~`,
so you can share the file.

### Why a warning status is normal

Each clip ends as `passed`, `warning` or `failed`. A clip is `warning`
when the worst scene of any variant scores below `VMAF_FLOOR` (91). That
floor applies to every variant, including the 360p ones, whose target is
88, so with the default settings almost every clip reports `warning`. It
does not mean the clip failed: the Variant Summary shows which variant
set it off.

### When a clip fails

The batch carries on (`BATCH_ON_ERROR=skip`), and the retry phase tries
the clip again. A clip that still fails has `report.json` and
`report.md` with status `failed` and the error, in `output/<clip>/`, or
in `work/<clip>/` when an earlier finished package was kept. Its work
folder stays in `work/<clip>/`, with `traceback.txt`, for debugging; the
five newest are kept (`MAX_KEPT_FAILURES`). The next `ftea run` tries
failed clips again.

## 6. Watching it on your phone

`ftea preview` serves one finished clip folder so this computer and
phones on the same network can play it in the bundled player:

```bash
ftea preview my-clip                  # a clip name in the output folder
ftea preview output/my-clip           # or a clip folder, or its master.m3u8
ftea preview my-clip --local-only     # only this computer may connect
ftea preview my-clip --port 9000      # a port of your choice
```

With no argument it previews the only finished clip in the output
folder, or lists the choices. It prints the addresses to open, for
example:

```text
Serving C:\ftea\output\my-clip
  On this computer: http://localhost:8765/player.html
  On a phone or tablet on the same Wi-Fi: http://192.168.1.20:8765/player.html
Add ?info=1 to the address to see which player and variant are playing.
Anyone on this network can open these addresses while ftea preview runs.
On Windows, if a firewall window appears for C:/Python312/python.exe, allow access on Private networks.
If no window appears and a phone cannot connect, Windows Firewall may already be blocking it. Open Windows Security > Firewall & network protection > Allow an app through firewall, and allow C:/Python312/python.exe on Private networks.
Press Ctrl+C to stop.
```

Open the phone address on a phone connected to the same Wi-Fi. While
the preview runs, it prints a line when a device first connects
(`<address> connected (iPhone)`) and a line whenever a device starts
loading a different variant (`<address> playing av1_720p`), so you can
see the player move between sizes as the connection changes. Press
Ctrl+C to stop it; it exits with code 0.

- `ftea preview` is for watching on your own network. Anyone on that
  network can open the addresses while it runs. It serves only that one
  clip folder and never lists folders. Use `--local-only` on a network
  you do not trust.
- The default port is 8765. If it is taken, `ftea preview` tries the
  next ones, up to ten ports in all, unless you chose one with `--port`.
- Many guest and public Wi-Fi networks stop devices from reaching each
  other. Use your own network.

### Windows Firewall

The first time a phone connects, Windows may show a firewall window for
the Python program the banner names. Allow access on **Private
networks**, and make sure your Wi-Fi network's profile type is
**Private** in Windows Settings.

If no window appears and the phone cannot connect, Windows is probably
already blocking that program. Pressing **Cancel** on that window once
creates a **Block** rule, and a Block rule wins over any Allow rule, so
allowing the app afterwards is not enough on its own. To remove the
Block rule (this needs an administrator):

1. Run `wf.msc` to open Windows Defender Firewall with Advanced
   Security.
2. Select **Inbound Rules** and find the rules whose program is the
   Python the banner names. A Block rule has a red icon.
3. Delete those Block rules.
4. Run `ftea preview` again, and allow access on Private networks when
   the window appears.

### WSL

Under WSL's default networking, the address `ftea preview` prints
belongs to a virtual network inside your PC, and a phone cannot reach
it; the banner says so. On the same PC,
`http://localhost:8765/player.html` still works from Windows. For a
phone, run `ftea preview` from Windows instead, or turn on WSL's
mirrored networking (Windows 11 22H2 or newer): add these lines to
`.wslconfig` in your Windows user folder, run `wsl --shutdown` from
Windows, and start WSL again.

```ini
[wsl2]
networkingMode=mirrored
```

### The player's options

Add these to the player's address; join several with `&`:

| Option | What it does |
| --- | --- |
| `?info=1` | Shows a box with the player in use (`player: hls.js <version>` or `player: native HLS`), how many variants this browser can play (`variants: 6 of 6 playable`), and the variant playing now (`now: 1080x1920 AV1`) |
| `?player=hlsjs` or `?player=native` | Forces one player, to compare them |
| `?src=h264_720p/stream.m3u8` | Plays one variant only |

A browser that cannot play AV1 shows `variants: 3 of 6 playable`: it
plays the three H.264 variants, which is normal.

## 7. Which phones get which encode

Every clip has two codecs at three sizes:

| Variants | What they are for |
| --- | --- |
| AV1, 10-bit | Phones that decode AV1 in hardware. Here AV1 needed 26% to 38% less video data than H.264 (see [How it works](HOW-IT-WORKS.md#why-av1-and-h264)) |
| H.264, 8-bit | Every phone: H.264 has played on effectively every phone of the last decade |
| 1080p, 720p, 360p | Steps the player moves between as the connection gets faster or slower |

### How the player chooses

The bundled player uses hls.js. hls.js reads the `CODECS` label of each
variant in `master.m3u8` and is designed to leave out the variants the
browser says it cannot decode; the info box (`?info=1`) shows how many
are left. Where hls.js cannot run, the player hands the playlist to the
browser's own HLS support. No device detection is involved. Which
variant a particular phone starts on has not been measured.

In `ftea`'s own end-to-end check, Chrome and Edge on a Windows PC listed
all six variants (`variants: 6 of 6 playable`) and loaded the AV1 ones.
Playback on a phone, and in Firefox, has not been observed yet.

### Phones that decode AV1 in hardware

According to the makers' own pages, checked on 2026-10-16, these phones
decode AV1: Qualcomm lists a hardware AV1 decoder in each Snapdragon
chip below, MediaTek lists AV1 among the formats each Dimensity chip
below plays or decodes, Apple announced a dedicated AV1 decoder in the
A17 Pro, and Apple lists AV1 among the video formats each later iPhone
plays. `ftea` has not yet been tested on any of them.

| Phones | Chip | Source |
| --- | --- | --- |
| iPhone 15 Pro and 15 Pro Max | A17 Pro | [Apple Newsroom](https://www.apple.com/newsroom/2023/09/apple-unveils-iphone-15-pro-and-iphone-15-pro-max/) |
| iPhone 16 and 16 Plus | A18 | [Apple](https://www.apple.com/iphone-16/specs/) |
| iPhone 16 Pro and 16 Pro Max | A18 Pro | [Apple](https://support.apple.com/en-us/121031), [Apple](https://support.apple.com/en-us/121032) |
| iPhone 17 and 17e | A19 | [Apple](https://www.apple.com/iphone-17/specs/), [Apple](https://www.apple.com/iphone-17e/specs/) |
| iPhone Air | A19 Pro | [Apple](https://www.apple.com/iphone-air/specs/) |
| iPhone 18 Pro, 18 Pro Max and iPhone Duo | A20 Pro | [Apple](https://www.apple.com/iphone-18-pro/specs/), [Apple](https://www.apple.com/iphone-duo/specs/) |
| Android phones built on these chips | Snapdragon 8 Gen 3, Snapdragon 8 Elite | [Qualcomm](https://docs.qualcomm.com/bundle/publicresource/87-71408-1_REV_C_Snapdragon_8_gen_3_Mobile_Platform_Product_Brief.pdf), [Qualcomm](https://docs.qualcomm.com/bundle/publicresource/87-83196-1_REV_B_Snapdragon_8_Elite_Mobile_Platform_Product_Brief.pdf) |
| Android phones built on these chips | MediaTek Dimensity 9300, Dimensity 9400 | [MediaTek](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9300), [MediaTek](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9400) |

According to Apple, the A17 Pro chip in the iPhone 15 Pro and 15 Pro
Max includes a dedicated AV1 decoder
([Apple Newsroom](https://www.apple.com/newsroom/2023/09/apple-unveils-iphone-15-pro-and-iphone-15-pro-max/)),
and Apple's specifications for the iPhone 15 do not list AV1
([Apple](https://support.apple.com/en-us/111831)), both checked on
2026-10-16. The iPhone 15 and 15 Plus, and older iPhones, should
therefore play the H.264 variants instead. That follows from how the
player chooses, but it has not yet been observed on such a phone. Most
recent iPhones decode HEVC in hardware, and HEVC needs less data than
H.264, so HEVC variants are on the [roadmap](../ROADMAP.md).

## 8. Hosting the output on a website

A clip folder is a complete static website. Upload the whole folder,
unchanged, to any web host or storage bucket that serves files over
HTTPS, and open `player.html` in it. Nothing runs on the server.

- **Upload everything** in the folder, keeping the variant subfolders.
  You may leave out `report.md`, `report.json` and `crops.json`: the
  player does not use them.
- **Content types.** The host must send the right `Content-Type` for the
  playlists and segments. These are the types `ftea preview` sends:

| File | Content-Type |
| --- | --- |
| `.m3u8` | `application/vnd.apple.mpegurl` |
| `.m4s` | `video/iso.segment` |
| `.mp4` | `video/mp4` |
| `.html` | `text/html; charset=utf-8` |
| `.js` | `text/javascript; charset=utf-8` |
| `.webp` | `image/webp` |
| `.jpg` | `image/jpeg` |

- **Your own page.** To play a clip in a page on another site, point an
  HLS player such as hls.js at the clip's `master.m3u8`, and have the
  host allow cross-origin requests (CORS) from that site.
- **Re-encoding.** If you encode a clip again with `--force`, upload the
  whole folder again, and clear any cache in front of it.

### An example with rclone

[rclone](https://rclone.org) copies a folder to most storage services.
With a remote you have set up as `myremote` and a bucket named
`my-bucket`, these three commands upload `output/my-clip` and set the
two content types storage services most often get wrong:

```bash
rclone copy output/my-clip myremote:my-bucket/my-clip --exclude "*.m3u8" --exclude "*.m4s"
rclone copy output/my-clip myremote:my-bucket/my-clip --include "*.m3u8" --header-upload "Content-Type: application/vnd.apple.mpegurl"
rclone copy output/my-clip myremote:my-bucket/my-clip --include "*.m4s" --header-upload "Content-Type: video/iso.segment"
```

The clip's address is then the bucket's public address followed by
`/my-clip/player.html`.

## 9. Tuning the settings

`ftea` uses the defaults in `config.env.example` until you create a
`config.env`. Copy the example into the data folder and edit the copy:

```bash
cp config.env.example config.env
```

Each setting is `KEY=value` on its own line. A key `ftea` does not know
is named in a warning when a run starts, so a typo never silently does
nothing. A setting that starts with `REFRAME` can also be set as an
environment variable for one run, which overrides `config.env`; and
`--reframe` and `--no-reframe` override `REFRAME` from either place:

```bash
REFRAME_BATCH_SIZE=8 ftea run
```

Every setting is explained in `config.env.example`. These are the ones
people change most.

### Reframing

| Setting | Default | What it does |
| --- | --- | --- |
| `REFRAME` | `true` | Reframe landscape videos to 9:16. With `false`, they are encoded in landscape |
| `REFRAME_TIMEOUT` | `600` | Seconds allowed to reframe one clip |
| `REFRAME_DEVICE` | `cuda` | Where person detection runs: `cuda`, `cuda:1` for a second GPU, or `cpu` (much slower) |
| `REFRAME_BATCH_SIZE` | `16` | Frames per detection batch. Lower it if reframing runs out of memory |
| `REFRAME_STRIDE` | `3` | Detect on every Nth frame. Higher is faster, but follows quick movement less closely |
| `REFRAME_DETECT_LONG_SIDE` | `0` | `0` detects at full size, the only setting tested for how well it finds faces. A number such as `1920` detects on smaller frames: faster, but it finds fewer faces |
| `REFRAME_FACE_WEIGHT` | `2.0` | How much more a face counts than a body when choosing whom to follow |
| `REFRAME_TITLE_ENABLED` | `true` | Detect title cards and keep them readable. `false` turns text detection off |
| `REFRAME_DEAD_ZONE` | `0.9` | How far the subject may drift inside the frame before the camera moves. Lower makes the camera livelier |
| `REFRAME_SMOOTHING_HZ` | `0.6` | How smoothly the camera follows. Higher reacts faster but looks less steady |
| `REFRAME_MAX_PAN_FRAC_PER_SEC` | `0.20` | The fastest the camera may pan |
| `REFRAME_KEEP_CROPS` | `true` | Keep the crop track, `crops.json`, in the clip folder |
| `REFRAME_MODELS_DIR` | (empty) | The model weights' folder. Empty means `models` in the data folder |
| `REFRAME_PYTHON` | (empty) | The reframe environment's Python. Empty means `.venv-reframe` in the checkout |
| `REFRAME_FFMPEG` | (empty) | A full path to the ffmpeg the reframe render should use. Empty means the one on `PATH`. `ffmpeg` and `ffprobe` must be on `PATH` either way |

### Quality and size

| Setting | Default | What it does |
| --- | --- | --- |
| `VMAF_TARGET_1080` | `93` | The VMAF target for the 1080p variants (0 to 100) |
| `VMAF_TARGET_720` | `93` | The same for 720p |
| `VMAF_TARGET_360` | `88` | The same for 360p |
| `VMAF_FLOOR` | `91` | Not a target: a clip whose worst scene in any variant scores below it is reported as `warning` |
| `AV1_START_CQ_1080` | `30` | The starting quality setting for AV1 1080p. Each scene is test-encoded at this value minus and plus `CQ_BRACKET`; a lower CQ means higher quality and more data |
| `AV1_START_CQ_720` | `33` | The same for AV1 720p |
| `AV1_START_CQ_360` | `38` | The same for AV1 360p |
| `H264_START_CQ_1080` | `25` | The same for H.264 1080p |
| `H264_START_CQ_720` | `28` | The same for H.264 720p |
| `H264_START_CQ_360` | `32` | The same for H.264 360p |
| `CQ_BRACKET` | `4` | How far either side of the starting setting the two test encodes are |
| `AV1_MAXRATE_1080` | `3000k` | The bitrate cap NVENC's rate control works to for AV1 1080p. NVENC applies it over a buffer of `BUFSIZE_MULTIPLIER` times the cap, so a clip's average can sit above it |
| `AV1_MAXRATE_720` | `1500k` | The same for AV1 720p |
| `AV1_MAXRATE_360` | `400k` | The same for AV1 360p |
| `H264_MAXRATE_1080` | `4000k` | The same for H.264 1080p |
| `H264_MAXRATE_720` | `2500k` | The same for H.264 720p |
| `H264_MAXRATE_360` | `800k` | The same for H.264 360p |
| `NVENC_PRESET` | `p7` | NVENC's speed and quality trade-off, from `p1` (fastest) to `p7` (best) |
| `FILM_GRAIN_BW` | `8` | AV1 film-grain synthesis for black-and-white clips; `FILM_GRAIN` does the same for colour clips and is off by default. Both apply only when your FFmpeg's `av1_nvenc` has a film-grain option. The 8.1 builds in [FFmpeg](#ffmpeg) have none, and a run says so when it starts |
| `HLS_SEGMENT_SEC` | `2` | Segment length in seconds |
| `THUMB_FORMAT` | `webp` | The thumbnail format, `webp` or `jpg` |
| `AUDIO_BITRATE` | `128k` | The AAC audio bitrate in every variant |

### Batches, time limits and retries

| Setting | Default | What it does |
| --- | --- | --- |
| `BATCH_ON_ERROR` | `skip` | `skip` carries on with the next clip after a failure. `stop` halts the batch at the first failure, with no retries |
| `MAX_CLIP_SEC` | `600` | Videos longer than this many seconds are skipped. `0` means no limit |
| `CLIP_TIMEOUT` | `300` | Seconds allowed to encode one clip, plus `CLIP_TIMEOUT_PER_SEC` for every second of video. Reframing has its own `REFRAME_TIMEOUT` |
| `CLIP_TIMEOUT_PER_SEC` | `30` | See `CLIP_TIMEOUT` |
| `RETRY_MAX` | `2` | How many more attempts a failed clip gets. The first retry keeps the quality targets; later ones lower them |
| `RETRY_VMAF_RELAXATION` | `2` | How far the second and later retries lower every VMAF target |
| `RETRY_TIMEOUT_MULTIPLIER` | `2` | How much longer every time limit is on a retry |
| `MAX_KEPT_FAILURES` | `5` | How many failed clips' work folders are kept for debugging |

## 10. Troubleshooting

### FFmpeg is missing NVENC, libvmaf or an AV1 decoder

`bash install.sh --check` and every `ftea run` test FFmpeg before any
work starts. These messages mean the build on your `PATH` is missing
something:

```text
ffmpeg is not on PATH. Install an FFmpeg build with NVENC and libvmaf (see docs/PROCESS-GUIDE.md).
this FFmpeg build has no av1_nvenc encoder. ftea needs an FFmpeg built with NVIDIA NVENC (see docs/PROCESS-GUIDE.md).
this FFmpeg build has no libvmaf filter, which ftea uses to measure quality. Install a build that includes it (see docs/PROCESS-GUIDE.md).
this FFmpeg build has no software AV1 decoder (libdav1d or libaom-av1), which ftea needs to measure the quality of the AV1 variants. Install a build that includes one (see docs/PROCESS-GUIDE.md).
this FFmpeg build has no libwebp encoder for THUMB_FORMAT=webp. Install a build that includes it, or set THUMB_FORMAT=jpg in config.env.
```

Install one of the builds in [FFmpeg](#ffmpeg), make sure its `bin`
folder comes first on your `PATH` (`which ffmpeg` shows which one runs),
and run `bash install.sh --check` again. To see what a build has, run
`ffmpeg -hide_banner -encoders | grep nvenc` and
`ffmpeg -hide_banner -filters | grep vmaf`.

These mean the build has the encoders, but the GPU or the driver cannot
use them:

```text
the GPU could not start an NVENC encode (<ffmpeg's message>). Check that an NVIDIA GPU and a current driver are installed: `nvidia-smi` should list the GPU.
the GPU could not start an AV1 encode (<ffmpeg's message>). AV1 hardware encoding needs an NVIDIA RTX 40-series GPU or newer.
this FFmpeg does not know NVENC_TUNE=uhq; install a newer build or set NVENC_TUNE=hq.
<encoder> starts, but not with ftea's encode settings (<ffmpeg's message>). Install a newer FFmpeg build and driver (see docs/PROCESS-GUIDE.md).
```

### torch is a CPU-only build

The reframe stage needs the CUDA build of torch from PyTorch's own
package index. On Windows, the torch on PyPI is a CPU-only build, and
any later `pip install` into `.venv-reframe` can silently replace the
CUDA build with it. `install.sh` checks the build after each install
step, and every `ftea run` that reframes checks it again. The messages:

```text
✗ torch is a CPU-only build (<version>)
✗ torch is built for CUDA <version>, not 12.8
✗ torch is <version>, but requirements-torch.txt pins <version>
```

Each is followed by a line that tells you to delete `.venv-reframe` and
run `install.sh` again. Do exactly that:

```bash
rm -rf .venv-reframe
bash install.sh
```

Never `pip install` anything into `.venv-reframe` by hand; the one
exception is pytest for the [GPU test](../CONTRIBUTING.md#the-reframe-stages-gpu-test).
On Windows, `torch import: OSError: …` usually means the Microsoft
Visual C++ Redistributable is missing; the message links the installer.

### Out of memory while reframing

```text
person detection failed on a batch of <N> frame(s): <error>; if this is a CUDA out-of-memory, lower REFRAME_BATCH_SIZE (currently 16) or set REFRAME_DEVICE=cpu to run person detection on the CPU (much slower)
face detection failed on a batch of <N> frame(s): <error>; the face detector runs on the CPU, so this is host memory: lower REFRAME_BATCH_SIZE (currently 16) or set REFRAME_DETECT_LONG_SIDE to detect on smaller frames
```

Close other programs that use the GPU, then lower the batch size in
`config.env` and run again; halve it again if it still fails:

```ini
REFRAME_BATCH_SIZE=8
```

`REFRAME_DEVICE=cpu` always works, but slowly. For a face-detection
failure, setting `REFRAME_DETECT_LONG_SIDE` to `1920` on a 4K video
detects on smaller frames, at some cost in how well faces are found.

### Install problems

| The message starts with | What to do |
| --- | --- |
| `install.sh needs bash.` | Run it as `bash install.sh`, not `sh install.sh` |
| `this folder (…) is on a Windows drive, and you are running Linux (WSL).` | Clone the repository into your Linux home folder |
| `<venv> was created on Windows, and this is Linux.` (or the reverse) | Use one checkout per operating system: clone again |
| `no usable Python found. ftea needs a standard 64-bit Python 3.10 to 3.14.` | Install a supported Python; the lines after it say why each Python found was refused |
| `this Python cannot create virtual environments (its ensurepip or venv module is missing).` | On Ubuntu or Debian, install the package the next line names, such as `python3.12-venv` |
| `nvidia-smi not found.` or `nvidia-smi found no NVIDIA GPU.` | Install or update the NVIDIA driver |
| `! NVIDIA driver <version> is older than 570.` | A warning: update the driver if the FFmpeg test fails |
| `! this folder's path is <N> characters long and Windows long-path support is off` | See [Long paths on Windows](#long-paths-on-windows) |
| `! <N> GB free here` | Free some disk space; see [Disk space](#disk-space) |
| `not on PATH: ffmpeg ffprobe.` | See [FFmpeg](#ffmpeg) |
| `<file>: the server sent a git-lfs pointer file instead of the weights` or `<file>: downloaded <N> bytes with SHA-256 …` or `<file>: could not download …` | A proxy or network problem changed or stopped the download. Fix the connection and run `bash install.sh` again |
| `.venv-reframe has no working pip (an earlier run was probably interrupted while creating it).` | Delete the folder, as the line says, and run `bash install.sh` again |
| `torch did not pass its check` | See [torch is a CPU-only build](#torch-is-a-cpu-only-build) |

### Run problems

| The message starts with | What to do |
| --- | --- |
| `another ftea run (process <id>) is using <data folder>.` | Wait for that run, or, if none is running, delete the lock file the message names |
| `WARNING: config.env setting <KEY> is not a known setting and is ignored.` | Fix the spelling of that key |
| `config.env: …` | Fix the value the message names |
| `the reframe environment check failed:` | Run `bash install.sh` again, or run with `--no-reframe` |
| a clip fails with an error that says `timed out` | Raise `CLIP_TIMEOUT_PER_SEC` or `REFRAME_TIMEOUT`, or split the video |
| `cannot set <folder> aside (…). Close what is using output/<clip>/` | Close the player or folder window that has a file in it open, then run again |
| `SKIP (same name as <file>; both would write output/<name>/)` | Rename one of the two files |
| `` SKIP: output/<name>/ holds the results of `ftea reframe` `` | Rename the video, or move that folder, to encode it |
| `` <folder> has no finished clips yet. Run `ftea run` first. `` | Encode a clip before previewing it |
| `could not listen on port …` | Choose another port with `--port`. On Windows, `WinError 10013` in the message usually means Windows has reserved those ports; `netsh interface ipv4 show excludedportrange protocol=tcp` lists the reserved ranges |
| The phone cannot open the address | Check that it is on the same Wi-Fi, see [Windows Firewall](#windows-firewall) and [WSL](#wsl), and try the other addresses the banner prints |
| The player says `This browser cannot play this video.` | Try Safari on an iPhone, or Chrome or Firefox |
