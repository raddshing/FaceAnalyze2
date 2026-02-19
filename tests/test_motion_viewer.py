from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from faceanalyze2.cli import app
from faceanalyze2.viz.motion_viewer import generate_motion_viewer

pytest.importorskip("mediapipe")

runner = CliRunner()


def _write_synthetic_artifacts(tmp_path: Path, stem: str = "sample") -> Path:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_count = 4
    n_landmarks = 478
    frame_indices = np.arange(t_count, dtype=np.int64)
    timestamps_ms = frame_indices * 33
    presence = np.ones((t_count,), dtype=bool)

    idx = np.arange(n_landmarks, dtype=np.float32)
    base = np.zeros((n_landmarks, 3), dtype=np.float32)
    base[:, 0] = 0.25 + 0.5 * ((idx % 26) / 25.0)
    base[:, 1] = 0.2 + 0.6 * ((idx // 26) / 19.0)
    base[:, 2] = 0.02 * np.sin(idx / 21.0)

    landmarks = np.zeros((t_count, n_landmarks, 3), dtype=np.float32)
    for t_idx in range(t_count):
        angle = np.deg2rad(2.0 * float(t_idx))
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        centered_xy = base[:, :2] - np.asarray([0.5, 0.5], dtype=np.float32)
        rotated_xy = (centered_xy @ rotation.T) + np.asarray(
            [0.5 + (0.01 * t_idx), 0.5 - (0.005 * t_idx)],
            dtype=np.float32,
        )

        frame = base.copy()
        frame[:, :2] = rotated_xy
        frame[:, 2] = base[:, 2] + (0.002 * t_idx)

        if t_idx == t_count - 1:
            frame[[61, 291], 0] += np.asarray([-0.02, 0.02], dtype=np.float32)
            frame[[61, 291], 1] -= 0.01
            frame[[13, 14], 1] -= 0.012

        landmarks[t_idx] = frame

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        landmarks_xyz=landmarks,
        presence=presence,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": 640, "height": 480, "frame_count": t_count}),
        encoding="utf-8",
    )
    (artifact_dir / "segment.json").write_text(
        json.dumps({"neutral_idx": 0, "peak_idx": 3, "onset_idx": 1, "offset_idx": 3}),
        encoding="utf-8",
    )
    return artifact_root


def test_generate_motion_viewer_creates_html(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)
    result = generate_motion_viewer(video_path=tmp_path / "sample.mp4", artifact_root=artifact_root)
    html_path = Path(result["html_path"])

    assert html_path.exists()
    assert html_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "MOTION_VIEWER_DATA" in html
    assert "three@0.128.0" in html
    assert "region normalize" in html
    assert "\"left_eye\"" in html


def test_generate_motion_viewer_guides_when_segment_missing(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)
    segment_path = artifact_root / "sample" / "segment.json"
    segment_path.unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        generate_motion_viewer(video_path=tmp_path / "sample.mp4", artifact_root=artifact_root)

    message = str(exc_info.value)
    assert "Segment file not found" in message
    assert "faceanalyze2 segment run --video" in message


def test_viewer_generate_cli_runs(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)

    result = runner.invoke(
        app,
        [
            "viewer",
            "generate",
            "--video",
            str(tmp_path / "sample.mp4"),
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert result.exit_code == 0
    assert "Saved 3D motion viewer to" in result.output
    assert "Open in browser..." in result.output

