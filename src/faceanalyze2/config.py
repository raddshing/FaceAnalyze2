from __future__ import annotations

from enum import Enum
from pathlib import Path

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
