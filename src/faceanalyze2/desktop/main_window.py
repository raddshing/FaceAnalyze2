"""Main application window with input/results splitter and drag-drop support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter, QStatusBar

from faceanalyze2.desktop.input_panel import InputPanel
from faceanalyze2.desktop.pipeline_worker import TOTAL_STEPS, PipelineWorker
from faceanalyze2.desktop.results_panel import ResultsPanel

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class MainWindow(QMainWindow):
    """FaceAnalyze2 main window: left input panel, right results panel."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FaceAnalyze2")
        self.resize(1200, 780)
        self.setAcceptDrops(True)

        self._worker: PipelineWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._input_panel = InputPanel()
        self._results_panel = ResultsPanel()

        splitter.addWidget(self._input_panel)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 880])

        self.setCentralWidget(splitter)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Connect signals
        self._input_panel.run_requested.connect(self._on_run_requested)

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------
    def _on_run_requested(self, video: str, motion: str, model: str, stride: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._input_panel.set_running(True)
        self._results_panel.clear()
        self.statusBar().showMessage("Running pipeline...")

        self._worker = PipelineWorker(
            video_path=video,
            motion=motion,
            model_path=model,
            stride=stride,
            artifact_root="artifacts",
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: int, message: str) -> None:
        self._input_panel.update_progress(step, message)
        self.statusBar().showMessage(f"Step {step}/{TOTAL_STEPS}: {message}")

    def _on_finished(self, result: dict[str, Any]) -> None:
        self._input_panel.set_finished()
        self._results_panel.display_results(result)
        self.statusBar().showMessage("Analysis complete", 5000)

    def _on_error(self, message: str) -> None:
        short = message.split("\n")[0][:200]
        self._input_panel.set_error(short)
        self.statusBar().showMessage("Pipeline error")
        QMessageBox.critical(self, "Pipeline Error", message[:2000])

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 – Qt override
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: Any) -> None:  # noqa: N802 – Qt override
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                self._input_panel.set_video_path(path)
                event.acceptProposedAction()
                return
        event.ignore()
