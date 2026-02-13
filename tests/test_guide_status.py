from pathlib import Path

from typer.testing import CliRunner

from faceanalyze2.cli import app

runner = CliRunner()


def test_guide_command_includes_task_choices() -> None:
    result = runner.invoke(app, ["guide"])
    assert result.exit_code == 0
    assert "smile" in result.output
    assert "brow" in result.output
    assert "eyeclose" in result.output


def test_status_command_prints_artifact_keywords(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "status",
            "--video",
            str(tmp_path / "sample.mp4"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 0
    assert "landmarks.npz" in result.output
    assert "segment.json" in result.output
    assert "다음으로 추천하는 커맨드" in result.output


def test_run_dry_run_prints_plan_without_real_files() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dry-run",
            "--video",
            "dummy.mp4",
            "--task",
            "smile",
        ],
    )
    assert result.exit_code == 0
    assert "DRY-RUN" in result.output
    assert "Step 1 - landmarks extract" in result.output
    assert "Step 4 - metrics run" in result.output
