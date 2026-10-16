# Third-party notices

`ftea` is licensed under the [Apache License 2.0](LICENSE). This page
lists the third-party software and models it ships, downloads or relies
on, and the license of each.

## Shipped in this repository

### hls.js 1.7.3

- **Files:** `ftea/player/hls.min.js`, byte for byte the
  `dist/hls.min.js` of the npm package `hls.js@1.7.3` (published
  2026-09-11), and its license, `ftea/player/hls.js-LICENSE.txt`.
- **License:** Apache License 2.0, Copyright (c) 2017 Dailymotion. Parts
  of it are derived from videojs-contrib-hls, Copyright (c) 2013-2015
  Brightcove, also under the Apache License 2.0. The full text is in
  `hls.js-LICENSE.txt`.
- **Where it goes:** `ftea run` copies both files into every clip's
  output folder, so a clip you host carries the license with it.
- **Source:** https://github.com/video-dev/hls.js

## Documentation images

`docs/images/before-after.webp`, the picture in the README, is a single
frame, with the crop window drawn over it, from a 16:9 cut of the film
[Tears of Steel](https://mango.blender.org) (2012) by the Blender
Foundation, licensed under the Creative Commons Attribution 3.0 license
([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)).

## Downloaded by install.sh

`install.sh` downloads these three files into the models folder and
checks each one against the size and SHA-256 pinned in
`ftea/reframe/provision.py`. They are not part of this repository.

| File | What it is | License | Download |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector, from OpenCV Zoo | MIT, Copyright (c) 2020 Shiqi Yu | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| `text_detection_en_ppocrv3_2023may.onnx` | PP-OCRv3 English text detector, from OpenCV Zoo | Apache-2.0, Copyright (c) 2016 PaddlePaddle Authors | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/b32d27c8f138ffd625efc52e2d82f8b7d54dabc7/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx) |
| `fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth` | Faster R-CNN person detector weights (MobileNetV3-Large FPN, trained on COCO), from torchvision | BSD-3-Clause | [PyTorch](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth) |

The license texts are upstream:
[YuNet](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE),
[PP-OCRv3](https://github.com/opencv/opencv_zoo/blob/b32d27c8f138ffd625efc52e2d82f8b7d54dabc7/models/text_detection_ppocr/LICENSE),
[torchvision](https://github.com/pytorch/vision/blob/v0.26.0/LICENSE).
No model under the AGPL is used; [CONTRIBUTING.md](CONTRIBUTING.md)
explains why.

## Installed by install.sh into .venv-reframe

The reframe stage runs in its own Python environment, `.venv-reframe`.
`install.sh` installs these pinned packages into it from PyTorch's CUDA
12.8 package index and from PyPI:

| Package | License |
| --- | --- |
| `torch==2.11.0+cu128` | BSD-3-Clause |
| `torchvision==0.26.0+cu128` | BSD-3-Clause |
| `opencv-python==5.0.0.93` | Apache-2.0. The package bundles other libraries too, each listed in its `LICENSE-3RD-PARTY.txt` |

pip also installs what these depend on (numpy, Pillow and others), each
under its own license. The CUDA builds of torch include NVIDIA CUDA
libraries, which come under NVIDIA's own license terms. `ftea` does not
redistribute any of these packages: pip downloads them from their
publishers.

## Used, but installed by you

- **FFmpeg.** You install it yourself; see
  [the process guide](docs/PROCESS-GUIDE.md#ffmpeg). The builds the guide
  names are GPL builds. `ftea` runs `ffmpeg` and `ffprobe` as separate
  programs and does not include or link FFmpeg.
- **VMAF.** Quality is measured by the `libvmaf` filter inside your
  FFmpeg build, with its built-in `vmaf_v0.6.1` model. VMAF is Netflix's
  and is licensed under the BSD-2-Clause-Patent license:
  https://github.com/Netflix/vmaf
- **Python**, and on Windows **Git for Windows**, under their own
  licenses.

## Used by the tests only

The test suite uses pytest (MIT) and numpy (BSD-3-Clause, with parts
under other permissive licenses), installed with
`python -m pip install -e ".[test]"`. Running `ftea` does not need them.
