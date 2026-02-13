# Developer Guide

## Environment

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

## Common commands

```bash
python -m faceanalyze2 --help
pytest
ruff check .
```

## M1 video I/O usage

```bash
faceanalyze2 video info --video path/to/video.mp4
faceanalyze2 video frame --video path/to/video.mp4 --frame-idx 10 --out outputs/frame_0010.png
```

## Notes

- M0 CLI is a stub and does not run MediaPipe.
- M1 adds OpenCV-based video metadata probing and frame export commands.
- Keep patient/sensitive files out of git.
