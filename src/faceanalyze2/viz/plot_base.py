from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

Metrics = Mapping[str, float]
Landmarks = Mapping[str, tuple[float, float]]


class Plotter(ABC):
    """Interface for analysis visualization outputs."""

    @abstractmethod
    def render_overlay(self, frame: Any, landmarks: Landmarks) -> Any:
        """Render frame overlay for debugging or presentation."""

    @abstractmethod
    def render_summary(self, metrics: Metrics, output_path: Path) -> None:
        """Render chart/report artifact to the output path."""
