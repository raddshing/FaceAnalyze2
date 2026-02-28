"""Application entry-point for the FaceAnalyze2 desktop GUI.

Usage (development)::

    python -m faceanalyze2.desktop.app
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from faceanalyze2.desktop.main_window import MainWindow
from faceanalyze2.runtime_paths import get_base_dir


def main() -> None:
    """Launch the FaceAnalyze2 desktop application."""
    # Ensure CWD matches base_dir so that relative artifact paths ("artifacts/")
    # used by pipeline steps 1-4 align with the absolute paths used by
    # dynamicAnalysis() (via get_artifact_root()).  Critical for frozen/exe mode
    # where the OS may set CWD to an arbitrary directory.
    os.chdir(get_base_dir())

    # Use INI file instead of Windows registry for portable settings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    config_dir = get_base_dir() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(config_dir))

    app = QApplication(sys.argv)
    app.setApplicationName("FaceAnalyze2")
    app.setOrganizationName("FaceAnalyze2")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
