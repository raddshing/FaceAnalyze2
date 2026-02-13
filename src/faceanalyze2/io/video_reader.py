from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class VideoReader(ABC):
    """Interface for reading frames from a video source."""

    @abstractmethod
    def open(self, video_path: Path) -> None:
        """Open a video source."""

    @abstractmethod
    def frames(self) -> Iterator[Any]:
        """Yield decoded frames in order."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
