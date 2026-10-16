---
name: Bug report
about: Something went wrong while installing or running ftea
title: ""
labels: bug
assignees: ""
---

**What happened**
Describe the problem and what you expected instead.

**Command you ran**
```
ftea ...
```

**Output**
Paste the full terminal output. Remove any folder or file names you would
rather not share.

**Environment**
- Operating system (Windows 10/11 with Git Bash, or Linux distribution and version):
- GPU model and NVIDIA driver version (from `nvidia-smi`):
- FFmpeg build (first line of `ffmpeg -version`):
- Python version (`python --version`):
- ftea version (`ftea --version`):

**Environment check**
If the problem involves installing or reframing, paste the environment
check printed at the end of `bash install.sh`.

**Quality report**
If an encode failed or missed its quality target, attach `report.json`
from the clip's output folder. When an earlier finished package was kept,
a failed attempt's report is in the clip's folder under `work/` instead.
The report names your data folder, the checkout and your home folder as
`<data>`, `<checkout>` and `~`, but read it before you attach it and
remove anything else you would rather not share.
