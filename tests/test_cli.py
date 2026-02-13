from pathlib import Path

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
