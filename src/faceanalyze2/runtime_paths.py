"""Portable path resolution for both dev and PyInstaller-frozen environments.

All path resolution in FaceAnalyze2 should go through the helpers in this
module so that the application works correctly when bundled as a standalone
exe via PyInstaller (onedir mode).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def is_frozen() -> bool:
    """Return ``True`` when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) is True


def get_base_dir() -> Path:
    """Return the base directory used for resolving relative paths.

    * **Frozen (PyInstaller)**: the directory containing the exe.
    * **Development**: the project root (two levels up from this file,
      i.e. ``src/faceanalyze2/`` -> project root).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # src/faceanalyze2/runtime_paths.py -> src/faceanalyze2 -> src -> project root
    return Path(__file__).resolve().parent.parent.parent


def get_model_path(relative: str = "models/face_landmarker.task") -> Path:
    """Return the absolute path to a model file relative to *base_dir*."""
    return get_base_dir() / relative


def get_artifact_root() -> Path:
    """Return the default root directory for pipeline artifacts."""
    return get_base_dir() / "artifacts"


def get_temp_dir() -> Path:
    """Return a writable temporary directory.

    * **Frozen**: ``<base_dir>/.temp`` (keeps temp files next to the exe so
      the app works from a USB stick without relying on OS temp).
    * **Development**: the system temp directory.
    """
    if is_frozen():
        tmp = get_base_dir() / ".temp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    return Path(tempfile.gettempdir())
