from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

Landmarks = Mapping[str, tuple[float, float]]


class LandmarkProvider(ABC):
    """Interface for frame-level facial landmark extraction."""

    @abstractmethod
    def extract(self, frame: Any) -> Landmarks:
        """Return normalized landmarks for a single frame."""
