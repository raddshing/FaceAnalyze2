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

## M2 landmark extraction usage

```bash
faceanalyze2 landmarks extract --video path/to/video.mp4
faceanalyze2 landmarks extract --video path/to/video.mp4 --model models/face_landmarker.task --stride 2
```

If the model file is missing, download the official Face Landmarker model:

```powershell
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models/face_landmarker.task"
```

## Notes

- M0 CLI is a stub and does not run MediaPipe.
- M1 adds OpenCV-based video metadata probing and frame export commands.
- M2 adds MediaPipe Face Landmarker VIDEO-mode extraction into `artifacts/<video_stem>/`.
- Keep patient/sensitive files out of git.
