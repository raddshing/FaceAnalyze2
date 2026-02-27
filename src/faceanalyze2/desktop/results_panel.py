"""Results panel with tabs for images, metrics, and raw JSON."""

from __future__ import annotations

import csv
import json
import shutil
import webbrowser
from io import StringIO
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from faceanalyze2.desktop.image_utils import base64_png_to_pixmap

IMAGE_KEYS = [
    ("key rest", "Neutral (Rest)"),
    ("key exp", "Peak Expression"),
    ("key value graph", "L/R Displacement Graph"),
    ("before regi", "Before Registration"),
    ("after regi", "After Registration"),
]

METRIC_COLUMNS = ["ROI", "L_peak", "R_peak", "AI", "Score"]


class ResultsPanel(QWidget):
    """Right-side panel showing analysis results in three tabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: dict[str, Any] | None = None
        self._image_pixmaps: dict[str, QPixmap] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Tab 1 – Images
        self._images_tab = QWidget()
        self._images_layout = QVBoxLayout(self._images_tab)
        self._image_labels: dict[str, QLabel] = {}
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._images_tab)
        self._tabs.addTab(self._scroll, "Images")

        # Tab 2 – Metrics table
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(len(METRIC_COLUMNS))
        self._metrics_table.setHorizontalHeaderLabels(METRIC_COLUMNS)
        self._metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tabs.addTab(self._metrics_table, "Metrics")

        # Tab 3 – JSON
        self._json_edit = QTextEdit()
        self._json_edit.setReadOnly(True)
        self._json_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._tabs.addTab(self._json_edit, "JSON")

        # Bottom button row
        btn_row = QHBoxLayout()
        self._csv_btn = QPushButton("Export CSV")
        self._csv_btn.setEnabled(False)
        self._csv_btn.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self._csv_btn)

        self._html_btn = QPushButton("Export Viewer HTML")
        self._html_btn.setEnabled(False)
        self._html_btn.clicked.connect(self._on_export_html)
        btn_row.addWidget(self._html_btn)

        self._viewer_btn = QPushButton("Open 3D Viewer")
        self._viewer_btn.setEnabled(False)
        self._viewer_btn.clicked.connect(self._on_open_3d_viewer)
        btn_row.addWidget(self._viewer_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def display_results(self, result: dict[str, Any]) -> None:
        """Populate all tabs with a ``dynamicAnalysis`` result dict."""
        self._result = result

        # -- Images tab --
        # Remove old labels
        for lbl in self._image_labels.values():
            self._images_layout.removeWidget(lbl)
            lbl.deleteLater()
        self._image_labels.clear()
        self._image_pixmaps.clear()

        for key, title in IMAGE_KEYS:
            b64 = result.get(key)
            if not b64:
                continue
            title_lbl = QLabel(f"<b>{title}</b>")
            self._images_layout.addWidget(title_lbl)
            self._image_labels[f"{key}_title"] = title_lbl

            pixmap = base64_png_to_pixmap(b64)
            self._image_pixmaps[key] = pixmap
            img_lbl = QLabel()
            img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._images_layout.addWidget(img_lbl)
            self._image_labels[key] = img_lbl

        self._rescale_images()

        # -- Metrics tab --
        metrics = result.get("metrics", {})
        roi_metrics: dict[str, dict[str, float]] = metrics.get("roi_metrics", {})
        self._metrics_table.setRowCount(len(roi_metrics))
        for row, (roi_name, values) in enumerate(roi_metrics.items()):
            self._metrics_table.setItem(row, 0, QTableWidgetItem(roi_name))
            self._metrics_table.setItem(row, 1, QTableWidgetItem(f"{values.get('L_peak', 0):.4f}"))
            self._metrics_table.setItem(row, 2, QTableWidgetItem(f"{values.get('R_peak', 0):.4f}"))
            self._metrics_table.setItem(row, 3, QTableWidgetItem(f"{values.get('AI', 0):.4f}"))
            self._metrics_table.setItem(row, 4, QTableWidgetItem(f"{values.get('score', 0):.1f}"))
        self._metrics_table.resizeColumnsToContents()

        # -- JSON tab --
        json_payload = {
            "metrics": metrics,
        }
        self._json_edit.setPlainText(json.dumps(json_payload, indent=2, ensure_ascii=False))

        self._csv_btn.setEnabled(True)
        self._html_btn.setEnabled(True)

        viewer_path = result.get("viewer_html_path")
        self._viewer_btn.setEnabled(bool(viewer_path and Path(viewer_path).is_file()))

        self._tabs.setCurrentIndex(0)

    def clear(self) -> None:
        for lbl in self._image_labels.values():
            self._images_layout.removeWidget(lbl)
            lbl.deleteLater()
        self._image_labels.clear()
        self._image_pixmaps.clear()
        self._metrics_table.setRowCount(0)
        self._json_edit.clear()
        self._csv_btn.setEnabled(False)
        self._html_btn.setEnabled(False)
        self._viewer_btn.setEnabled(False)
        self._result = None

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 – Qt override
        super().resizeEvent(event)
        self._rescale_images()

    def _rescale_images(self) -> None:
        """Rescale all displayed images to fit the current scroll area width."""
        if not self._image_pixmaps:
            return
        # Available width minus scroll bar and layout margins
        avail = self._scroll.viewport().width() - 20
        if avail < 100:
            return
        for key, pixmap in self._image_pixmaps.items():
            lbl = self._image_labels.get(key)
            if lbl is None:
                continue
            w = min(avail, pixmap.width())
            lbl.setPixmap(
                pixmap.scaled(
                    w,
                    pixmap.height() * w // max(pixmap.width(), 1),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    # ------------------------------------------------------------------
    # Export slots
    # ------------------------------------------------------------------
    def _on_export_csv(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "metrics.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            roi_metrics = self._result.get("metrics", {}).get("roi_metrics", {})
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(METRIC_COLUMNS)
            for roi_name, values in roi_metrics.items():
                writer.writerow([
                    roi_name,
                    f"{values.get('L_peak', 0):.6f}",
                    f"{values.get('R_peak', 0):.6f}",
                    f"{values.get('AI', 0):.6f}",
                    f"{values.get('score', 0):.2f}",
                ])
            Path(path).write_text(buf.getvalue(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _on_export_html(self) -> None:
        if not self._result:
            return
        viewer_path = self._result.get("viewer_html_path")
        default_name = "motion_viewer.html" if viewer_path else "viewer.html"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Viewer HTML", default_name, "HTML (*.html)"
        )
        if not path:
            return
        try:
            if viewer_path and Path(viewer_path).is_file():
                shutil.copy2(viewer_path, path)
            else:
                self._write_viewer_html(path)
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _on_open_3d_viewer(self) -> None:
        if not self._result:
            return
        viewer_path = self._result.get("viewer_html_path")
        if viewer_path and Path(viewer_path).is_file():
            webbrowser.open(Path(viewer_path).resolve().as_uri())

    def _write_viewer_html(self, path: str) -> None:
        """Generate a self-contained HTML viewer with embedded images."""
        result = self._result
        if not result:
            return
        metrics = result.get("metrics", {})

        parts = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            "<title>FaceAnalyze2 Results</title>",
            "<style>body{font-family:sans-serif;max-width:960px;margin:auto;padding:20px}",
            "img{max-width:100%;margin:8px 0}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}",
            "th{background:#f5f5f5}</style></head><body>",
            "<h1>FaceAnalyze2 Results</h1>",
        ]

        for key, title in IMAGE_KEYS:
            b64 = result.get(key)
            if b64:
                parts.append(f"<h2>{title}</h2>")
                parts.append(f"<img src='data:image/png;base64,{b64}' alt='{title}'>")

        roi_metrics = metrics.get("roi_metrics", {})
        if roi_metrics:
            parts.append("<h2>Metrics</h2><table><tr>")
            for col in METRIC_COLUMNS:
                parts.append(f"<th>{col}</th>")
            parts.append("</tr>")
            for roi_name, values in roi_metrics.items():
                parts.append(
                    f"<tr><td>{roi_name}</td>"
                    f"<td>{values.get('L_peak', 0):.4f}</td>"
                    f"<td>{values.get('R_peak', 0):.4f}</td>"
                    f"<td>{values.get('AI', 0):.4f}</td>"
                    f"<td>{values.get('score', 0):.1f}</td></tr>"
                )
            parts.append("</table>")

        parts.append("<h2>Raw JSON</h2><pre>")
        parts.append(json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False))
        parts.append("</pre></body></html>")

        Path(path).write_text("\n".join(parts), encoding="utf-8")
