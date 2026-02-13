from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

Landmarks = Mapping[str, tuple[float, float]]
Metrics = Mapping[str, float]


class MetricCalculator(ABC):
    """Interface for task-specific metric computation."""

    @abstractmethod
    def find_neutral_baseline(self, landmarks_series: Sequence[Landmarks]) -> int:
        """Return frame index used as neutral baseline."""

    @abstractmethod
    def find_peak_frame(self, landmarks_series: Sequence[Landmarks]) -> int:
        """Return frame index with strongest movement response."""

    @abstractmethod
    def compute_metrics(self, landmarks_series: Sequence[Landmarks]) -> Metrics:
        """Compute summary metrics from the full landmark sequence."""
