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

## Quickstart (새 동영상)

가장 추천:

```bash
faceanalyze2 run --video "D:\\local\\sample.mp4" --task smile
```

상태 확인:

```bash
faceanalyze2 status --video "D:\\local\\sample.mp4"
```

수동 실행 순서:

```bash
faceanalyze2 landmarks extract --video "D:\\local\\sample.mp4"
faceanalyze2 segment run --video "D:\\local\\sample.mp4" --task smile
faceanalyze2 align run --video "D:\\local\\sample.mp4"
faceanalyze2 metrics run --video "D:\\local\\sample.mp4" --task smile
faceanalyze2 align viz --video "D:\\local\\sample.mp4"
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

## M3 segmentation usage

```bash
faceanalyze2 segment run --video path/to/video.mp4 --task smile
faceanalyze2 segment run --video path/to/video.mp4 --task brow --landmarks artifacts/video/landmarks.npz
```

Outputs are stored in `artifacts/<video_stem>/`:
- `signals.csv`
- `segment.json`
- `signals_plot.png`

## M4 alignment usage

```bash
faceanalyze2 align run --video path/to/video.mp4
faceanalyze2 align run --video path/to/video.mp4 --landmarks artifacts/video/landmarks.npz --segment artifacts/video/segment.json
faceanalyze2 align run --video path/to/video.mp4 --scale-z
faceanalyze2 align viz --video path/to/video.mp4 --max-frames 300 --stride 2 --n-samples 15
```

Outputs are stored in `artifacts/<video_stem>/`:
- `landmarks_aligned.npz` (`landmarks_xy_aligned` is stored in pixel coordinates)
- `alignment.json`
- `alignment_check.png`
- `trajectory_plot.png`
- `alignment_overlay.mp4` (or `alignment_overlay.avi` fallback)

`z` handling:
- default `--keep-z`: keeps original z values
- `--scale-z`: applies per-frame similarity scale to z

## M5 metrics and ROI plots usage

```bash
faceanalyze2 metrics run --video path/to/video.mp4 --task smile
faceanalyze2 metrics run --video path/to/video.mp4 --task eyeclose --rois eye,mouth --no-normalize
```

Outputs are stored in `artifacts/<video_stem>/`:
- `plots/mouth.png`, `plots/eye.png`, `plots/brow.png` (for selected rois)
- `timeseries.csv`
- `metrics.csv`
- `metrics.json`

M5 reads aligned coordinates from `landmarks_aligned.npz` key `landmarks_xy_aligned` (pixel coordinates).

## Notes

- M0 CLI is a stub and does not run MediaPipe.
- M1 adds OpenCV-based video metadata probing and frame export commands.
- M2 adds MediaPipe Face Landmarker VIDEO-mode extraction into `artifacts/<video_stem>/`.
- M3 adds robust baseline/peak segmentation from task-specific landmark signals.
- M4 adds neutral-referenced similarity alignment (translation/rotation/scale) using eye landmarks.
- M5 adds ROI left/right displacement metrics and asymmetry plots from aligned landmarks.
- Keep patient/sensitive files out of git.
