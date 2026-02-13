from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    smile = "smile"
    brow = "brow"
    eyeclose = "eyeclose"


class RunConfig(BaseModel):
    video_path: Path = Field(description="Input video file path")
    task: TaskType
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Video file does not exist: {value}")
        if not value.is_file():
            raise ValueError(f"Video path is not a file: {value}")
        return value

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        if value.exists() and not value.is_dir():
            raise ValueError(f"Output path is not a directory: {value}")
        return value

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def as_summary(self) -> dict[str, str]:
        return {
            "video_path": str(self.video_path),
            "task": self.task.value,
            "output_dir": str(self.output_dir),
        }


def artifact_dir_for_video(
    video_path: str | Path,
    artifact_root: str | Path = Path("artifacts"),
) -> Path:
    return Path(artifact_root) / Path(video_path).stem


def artifact_paths_for_video(
    video_path: str | Path,
    artifact_root: str | Path = Path("artifacts"),
) -> dict[str, Path]:
    artifact_dir = artifact_dir_for_video(video_path=video_path, artifact_root=artifact_root)
    return {
        "artifact_dir": artifact_dir,
        "landmarks_npz": artifact_dir / "landmarks.npz",
        "meta_json": artifact_dir / "meta.json",
        "segment_json": artifact_dir / "segment.json",
        "signals_csv": artifact_dir / "signals.csv",
        "signals_plot_png": artifact_dir / "signals_plot.png",
        "aligned_npz": artifact_dir / "landmarks_aligned.npz",
        "alignment_json": artifact_dir / "alignment.json",
        "alignment_check_png": artifact_dir / "alignment_check.png",
        "trajectory_plot_png": artifact_dir / "trajectory_plot.png",
        "alignment_overlay_mp4": artifact_dir / "alignment_overlay.mp4",
        "alignment_overlay_avi": artifact_dir / "alignment_overlay.avi",
        "plots_dir": artifact_dir / "plots",
        "metrics_csv": artifact_dir / "metrics.csv",
        "metrics_json": artifact_dir / "metrics.json",
        "timeseries_csv": artifact_dir / "timeseries.csv",
    }


def normalize_task_value(task: Any) -> str:
    if isinstance(task, TaskType):
        return task.value
    value = str(task.value) if hasattr(task, "value") else str(task)
    valid_values = {member.value for member in TaskType}
    if value not in valid_values:
        valid = ", ".join(member.value for member in TaskType)
        raise ValueError(f"Unsupported task '{value}'. Expected one of: {valid}")
    return value
