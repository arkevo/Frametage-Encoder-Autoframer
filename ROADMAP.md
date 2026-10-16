# Roadmap

These are directions, not promises. If you want to work on one, open an
issue first so the approach can be agreed; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Left out of the first release

- **HEVC variants.** iPhones without an AV1 decoder are expected to play
  the H.264 variants today, though that has not yet been observed on
  such a phone. Most recent iPhones decode HEVC in hardware, and HEVC
  needs less data than H.264, so HEVC variants would serve them better.
  Adding them needs new quality and bitrate measurements before the
  README can quote any.
- **A CPU encoder fallback.** Every variant is encoded with NVIDIA NVENC
  today. A software encoder would let machines without an NVIDIA RTX GPU
  run `ftea`, much more slowly.
- **A Docker image**, with the FFmpeg build, Python and the CUDA packages
  ready-made.
- **Uploading to cloud storage** straight from `ftea run`, instead of the
  manual step in the
  [process guide](docs/PROCESS-GUIDE.md#8-hosting-the-output-on-a-website).
- **A web page for uploading videos** to a machine that runs `ftea`.
- **One shared codebase** with the production pipeline `ftea` was forked
  from. Today they are two copies, and they will drift apart.

## Reframing

- **Shots with no people in them.** The person detector can report a
  subject in a shot that has no people in it, and the crop then follows
  that instead of staying still and centred. It happened in `ftea`'s own
  end-to-end check. Telling such shots apart more reliably would fix it.

## Measuring quality

- **Scoring for phone screens.** VMAF scores every variant at 1920×1080
  with its default model, which marks the 360p and 720p variants down
  hard, so they often miss their targets, the 360p ones almost always (see
  [How it works](docs/HOW-IT-WORKS.md#how-quality-is-measured)). Scoring
  with VMAF's phone model, or at each variant's own size, and then
  revisiting the 360p and 720p targets, would make those numbers mean
  more.
- **Scoring vertical video at its own shape.** Vertical clips are scored
  on a picture squeezed to 1920×1080 today.
- **A measured fixed-bitrate comparison**, to put a number on what
  per-scene quality targets save. None has been measured, so the
  documents make no such claim.
- **Aggregating your own reports.** The README's numbers come from a
  separate tool run over the production pipeline's reports. A command
  that turns your own `report.json` files into the same tables would let
  you check them against your footage.

## Usability

- **Splitting long videos into clips** inside `ftea`, at scene changes,
  instead of by hand.
- **A firewall check in `ftea preview`** that warns when Windows Firewall
  already blocks the Python program serving the preview, instead of
  leaving the phone to time out.
- **Linux on a native NVIDIA driver**, validated end to end. So far a
  full install and encode has been verified under WSL2 only.
- **FFmpeg 9**, validated. The guide names FFmpeg 8.1 builds, which
  `ftea`'s checks pass with. BtbN's FFmpeg 9 builds for Linux need NVIDIA
  driver 610 or newer, against 570 for the 8.1 ones.
- **Playback checked on phones and in Firefox.** So far the player has
  been checked in Chrome and Edge on a Windows PC only.
