"""Input panel widget for video selection and pipeline configuration."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from faceanalyze2.runtime_paths import get_base_dir, get_model_path

VIDEO_FILTERS = "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v);;All Files (*)"

MOTIONS = [
    ("big-smile", "Big Smile"),
    ("blinking-motion", "Blinking Motion"),
    ("eyebrow-motion", "Eyebrow Motion"),
]


class InputPanel(QWidget):
    """Left-side panel with video/motion selection and an 'Analyze' button."""

    run_requested = Signal(str, str, str, int)  # video, motion, model, stride

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # -- Video file row --
        video_row = QHBoxLayout()
        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("Select a video file...")
        self._video_edit.setReadOnly(True)
        video_row.addWidget(self._video_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        video_row.addWidget(browse_btn)

        root.addWidget(QLabel("Video File"))
        root.addLayout(video_row)

        # -- Form fields --
        form = QFormLayout()

        self._motion_combo = QComboBox()
        for value, label in MOTIONS:
            self._motion_combo.addItem(label, userData=value)
        form.addRow("Motion:", self._motion_combo)

        # Model path
        model_row = QHBoxLayout()
        self._model_edit = QLineEdit(str(get_model_path()))
        model_row.addWidget(self._model_edit)
        model_browse_btn = QPushButton("...")
        model_browse_btn.setFixedWidth(32)
        model_browse_btn.clicked.connect(self._on_browse_model)
        model_row.addWidget(model_browse_btn)
        form.addRow("Model:", model_row)

        # Stride slider
        stride_row = QHBoxLayout()
        self._stride_slider = QSlider()
        self._stride_slider.setOrientation(Qt.Orientation.Horizontal)
        self._stride_slider.setRange(1, 8)
        self._stride_slider.setValue(2)
        self._stride_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._stride_slider.setTickInterval(1)
        stride_row.addWidget(self._stride_slider)
        self._stride_label = QLabel("2")
        self._stride_label.setFixedWidth(20)
        stride_row.addWidget(self._stride_label)
        self._stride_slider.valueChanged.connect(
            lambda v: self._stride_label.setText(str(v))
        )
        form.addRow("Stride:", stride_row)

        root.addLayout(form)

        # -- Run button --
        self._run_btn = QPushButton("Analyze")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        root.addWidget(self._run_btn)

        # -- Progress --
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 6)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m")
        root.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        root.addWidget(self._status_label)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        start_dir = str(get_base_dir())
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", start_dir, VIDEO_FILTERS)
        if path:
            self.set_video_path(path)

    def _on_browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", str(get_base_dir()), "Task Files (*.task);;All Files (*)"
        )
        if path:
            self._model_edit.setText(path)

    def _on_run(self) -> None:
        video = self._video_edit.text().strip()
        if not video:
            return
        motion = self._motion_combo.currentData()
        model = self._model_edit.text().strip()
        stride = self._stride_slider.value()
        self.run_requested.emit(video, motion, model, stride)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_video_path(self, path: str) -> None:
        """Set the video path and enable the run button if the file exists."""
        self._video_edit.setText(path)
        self._run_btn.setEnabled(bool(path) and Path(path).is_file())

    def video_path(self) -> str:
        return self._video_edit.text().strip()

    def set_running(self, running: bool) -> None:
        """Toggle controls on/off while the pipeline is active."""
        self._run_btn.setEnabled(not running)
        if running:
            self._progress_bar.setValue(0)
            self._status_label.setText("")

    def update_progress(self, step: int, message: str) -> None:
        self._progress_bar.setValue(step)
        self._status_label.setText(message)

    def set_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._status_label.setText("Complete")

    def set_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status_label.setText(f"Error: {message[:120]}")
