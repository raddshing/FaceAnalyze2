import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from faceanalyze2.cli import app

runner = CliRunner()


def test_cli_help_runs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "FaceAnalyze2 CLI scaffold" in result.output


def test_cli_rejects_missing_video_path(tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"
    result = runner.invoke(app, ["--video", str(missing_video), "--task", "smile"])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_creates_output_dir(tmp_path: Path) -> None:
    input_video = tmp_path / "sample.mp4"
    input_video.write_bytes(b"stub")

    output_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "--video",
            str(input_video),
            "--task",
            "brow",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()
    assert '"task": "brow"' in result.output


def test_segment_command_guides_when_landmarks_are_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "segment",
            "run",
            "--video",
            str(tmp_path / "sample.mp4"),
            "--task",
            "smile",
        ],
    )
    assert result.exit_code != 0
    assert "Landmarks file not found" in result.output
    assert "faceanalyze2 landmarks extract --video" in result.output


def test_align_command_guides_when_segment_is_missing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=np.asarray([0], dtype=np.int64),
        timestamps_ms=np.asarray([0], dtype=np.int64),
        landmarks_xyz=np.zeros((1, 478, 3), dtype=np.float32),
        presence=np.asarray([True], dtype=bool),
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": 640, "height": 480, "frame_count": 1}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "align",
            "run",
            "--video",
            str(tmp_path / "sample.mp4"),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert result.exit_code != 0
    assert "Segment file not found" in result.output
    assert "faceanalyze2 segment run --video" in result.output
