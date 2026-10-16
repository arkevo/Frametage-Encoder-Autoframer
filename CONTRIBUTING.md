# Contributing to ftea

Thank you for helping. This page covers the rules every change follows,
how to set up, and how to run the tests. For a bigger change, open an
issue first so the approach can be agreed before you write the code; the
[roadmap](ROADMAP.md) lists ideas that are already welcome.

## Rules every change follows

- **No AGPL code or models.** `ftea` is licensed under Apache-2.0, so it
  cannot ship or depend on AGPL-licensed work. That rules out
  `ultralytics` and its YOLO models, which are AGPL-3.0. Detection uses
  torchvision's Faster R-CNN (BSD-3-Clause), OpenCV Zoo's YuNet (MIT) and
  PP-OCRv3 (Apache-2.0). A new model needs a permissive license, a
  download address pinned to an exact version or commit, and its size and
  SHA-256 in `ftea/reframe/provision.py`.
- **The encoder uses only Python's standard library.** `pyproject.toml`
  keeps `dependencies = []`, and a test guards it. The reframe stage's
  packages live in `ftea/reframe/requirements-torch.txt` and
  `ftea/reframe/requirements.txt`, which `install.sh` installs into
  `.venv-reframe` in that order. Never make them a `pyproject.toml`
  extra: pip would then take a CPU-only torch from PyPI.
- **Python 3.10 is the floor.** Code must run on Python 3.10 to 3.14. A
  test parses every module with the 3.10 grammar and rejects some
  standard-library features that arrived later.
- **No media in the repository.** Tests make their own video in a
  temporary folder. `.gitignore` keeps out video, stream segments, still
  frames, audio, model weights and `config.env`. Images are allowed only
  in `docs/images/`, and only images you have the rights to.
- **LF line endings.** `.gitattributes` enforces them so that
  `install.sh` runs unchanged on Linux, and a test fails on any file with
  Windows line endings.
- **Keep the documents in step with the code.** `ftea/tests/test_docs.py`
  checks that the documents quote commands the CLI accepts, settings that
  exist with their shipped defaults, the FFmpeg sources and Python range
  in `install.sh`, the current pins and model downloads, and links that
  resolve. Change both in the same pull request.
- **One checkout per operating system.** Do not share a checkout between
  Windows and WSL: each one's virtual environments only work on the
  system that created them.

## Setting up

On Windows, use Git Bash. Clone the repository, then either run
`bash install.sh`, which creates `.venv` with `ftea` installed (see the
[process guide](docs/PROCESS-GUIDE.md#2-installing)), or, if you only
work on code and tests, create the environment yourself:

```bash
python -m venv .venv
source .venv/Scripts/activate        # on Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Running the tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

These are the two commands CI runs on every push and pull request, on
`windows-latest` and `ubuntu-latest`, with Python 3.10, 3.12 and 3.14.
The tests need no GPU and no FFmpeg: they replace every `ffmpeg` and
`ffprobe` call with a fake, and the few tests that need the real thing
skip themselves when it is missing. To run those on a machine that has
them:

```bash
python -m pytest -m ffmpeg     # needs ffmpeg and ffprobe on PATH
python -m pytest -m gpu        # needs an NVIDIA GPU and an FFmpeg with NVENC
```

### The reframe stage's GPU test

One test, `TestGpuEndToEnd` in `ftea/reframe/tests/test_detect.py`, runs
the real detectors on a real clip. It runs only inside `.venv-reframe`,
with a CUDA GPU, the downloaded weights, and the `FTEA_TEST_CLIP`
environment variable pointing at a landscape clip that shows people.
Everywhere else it skips.

`.venv-reframe` does not include pytest, so install it there by hand.
This is the one exception to the rule that nothing but the two
requirements files goes into that environment, and it is safe because
pytest brings no torch with it:

```bash
.venv-reframe/Scripts/python.exe -m pip install pytest   # Linux: .venv-reframe/bin/python
FTEA_TEST_CLIP=C:/clips/people.mp4 .venv-reframe/Scripts/python.exe -m pytest ftea/reframe/tests/test_detect.py -k TestGpuEndToEnd
```

## Commit messages and pull requests

- Write commit messages as `type(scope): summary`, for example
  `fix(preview): name the port that is in use`. Types: `feat`, `fix`,
  `docs`, `test`, `build`, `ci`, `chore`. Scopes: `encode`, `reframe`,
  `install`, `cli`, `preview`, `player`, `docs`.
- Keep one topic per pull request. Say what you changed, why, and which
  tests you ran on which system. CI must pass.
- By contributing, you agree that your contribution is licensed under the
  Apache License 2.0, as section 5 of the [license](LICENSE) describes.

## Updating a pinned dependency

- **hls.js.** Take the newest 1.x release that was published at least 14
  days ago. Check the npm package against the registry's integrity hash,
  copy its `dist/hls.min.js` byte for byte to `ftea/player/hls.min.js`,
  and its `LICENSE` to `ftea/player/hls.js-LICENSE.txt`. Then update
  `HLS_JS_VERSION`, `HLS_JS_SHA256` and the date in
  `ftea/tests/test_player.py`, and the version in
  `THIRD_PARTY_NOTICES.md`.
- **Model weights.** Change the address, size and SHA-256 together in
  `ftea/reframe/provision.py`, and the table in `THIRD_PARTY_NOTICES.md`.
- **torch and torchvision.** Change the pins and the CUDA package index
  in `ftea/reframe/requirements-torch.txt` and `EXPECTED_CUDA` in
  `ftea/reframe/verify_env.py` together. If the new wheels cover other
  Python versions, update `PYTHON_PREFERENCE` in `install.sh`; the
  documents' Python range follows it, and `test_docs.py` checks that.

## Reporting a problem

Use the issue templates. For an install problem, paste the output of
`bash install.sh --check`; for a failed encode, attach the clip's
`report.json`. Remove any folder or file names you would rather not
share.
