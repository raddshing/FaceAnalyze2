# FaceAnalyze2

FaceAnalyze2 is a quantitative analysis scaffold for facial palsy assessment from video.
M0 focuses on project structure and CLI plumbing only.

## Development stages

- M0: Scaffolding (CLI, package layout, docs, tests, lint setup)
- M1: Video IO and landmark provider integration
- M2: Metric computation (neutral baseline and peak frame detection)
- M3: Visualization and report output
- M4: Validation, calibration, and workflow hardening

## Quick start (stub)

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
faceanalyze2 --help
python -m faceanalyze2 --video path/to/video.mp4 --task smile
```

Current CLI behavior in M0:
- Validates that `--video` exists.
- Creates `--output-dir` if missing (default: `outputs`).
- Prints normalized runtime config summary.
