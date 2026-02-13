# FaceAnalyze2 Specification (M0)

## Project goal

FaceAnalyze2 targets quantitative assessment support for facial palsy from video.
The system should produce objective movement metrics for clinician-facing interpretation.

## Data and security rules

- Never commit patient videos/images/PII into git.
- Treat `data/`, `outputs/`, and `models/` as ignored by default.
- Use only dummy or publicly sharable files for examples/tests.

## Core analysis assumptions

- Each input video contains one prompted facial movement task.
- Task classification is out of scope.
- The pipeline should identify a neutral baseline frame and a peak response frame.
- Frame selection should rely on geometric changes (for example area/length deltas),
  not expression labels.

## M0 scope (implemented)

- Python project scaffold with `src/` package layout.
- Executable CLI (`python -m faceanalyze2` and `faceanalyze2`).
- Stub interfaces for video IO, landmark provider, metrics, and visualization.
- Basic docs, tests, and lint/test tool configuration.

## Out of scope in M0

- MediaPipe integration.
- Real video decoding and landmark extraction.
- Real metric formulas and charts.

## Future milestone sketch

- M1: Video reader + landmark provider concrete implementation.
- M2: Baseline/peak detection + task metrics.
- M3: Visual outputs and report artifacts.
- M4: Validation and deployment hardening.
