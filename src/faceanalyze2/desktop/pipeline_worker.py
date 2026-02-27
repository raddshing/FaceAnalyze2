"""Background QThread worker that runs the full analysis pipeline."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

MOTION_TO_TASK = {
    "big-smile": "smile",
    "blinking-motion": "eyeclose",
    "eyebrow-motion": "brow",
}

TOTAL_STEPS = 6


class PipelineWorker(QThread):
    """Run the FaceAnalyze2 pipeline in a background thread.

    Signals
    -------
    progress(int, str)
        Emitted before each pipeline step with (step_number, description).
    finished(dict)
        Emitted when the pipeline completes successfully with the
        ``dynamicAnalysis`` result dict.
    error(str)
        Emitted when the pipeline fails with a human-readable message.
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        video_path: str,
        motion: str,
        model_path: str = "models/face_landmarker.task",
        stride: int = 1,
        artifact_root: str = "artifacts",
        *,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._motion = motion
        self._model_path = model_path
        self._stride = stride
        self._artifact_root = artifact_root

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 – overridden Qt method
        """Execute pipeline steps sequentially (called by QThread.start)."""
        try:
            task_value = MOTION_TO_TASK.get(self._motion)
            if task_value is None:
                self.error.emit(f"Unsupported motion: {self._motion}")
                return

            video = Path(self._video_path)
            if not video.exists():
                self.error.emit(f"Video file not found: {video}")
                return

            # Step 1 – Landmark extraction
            self.progress.emit(1, "Extracting face landmarks...")
            from faceanalyze2.landmarks.mediapipe_face_landmarker import (
                extract_face_landmarks_from_video,
            )

            extract_face_landmarks_from_video(
                video_path=str(video),
                model_path=self._model_path,
                stride=self._stride,
                start_frame=0,
                end_frame=None,
                num_faces=1,
                output_blendshapes=False,
                output_transforms=False,
                use_gpu_delegate=False,
                artifact_root=self._artifact_root,
            )

            # Step 2 – Segmentation
            self.progress.emit(2, "Computing segmentation signal...")
            from faceanalyze2.analysis.segmentation import run_segmentation

            run_segmentation(
                video_path=video,
                task=task_value,
                artifact_root=self._artifact_root,
            )

            # Step 3 – Alignment
            self.progress.emit(3, "Aligning landmarks...")
            from faceanalyze2.analysis.alignment import run_alignment

            run_alignment(
                video_path=video,
                artifact_root=self._artifact_root,
            )

            # Step 4 – Metrics
            self.progress.emit(4, "Computing ROI metrics...")
            from faceanalyze2.analysis.metrics import run_metrics

            run_metrics(
                video_path=video,
                task=task_value,
                artifact_root=self._artifact_root,
                normalize=True,
                rois="all",
            )

            # Step 5 – Generate visualization payload
            self.progress.emit(5, "Generating visualization...")
            from faceanalyze2.api.dynamic_analysis import dynamicAnalysis

            result = dynamicAnalysis(vd_path=video, motion=self._motion)

            # Step 6 – Generate 3D motion viewer HTML
            self.progress.emit(6, "Generating 3D motion viewer...")
            try:
                from faceanalyze2.viz.motion_viewer import generate_motion_viewer

                viewer_result = generate_motion_viewer(
                    video_path=video,
                    artifact_root=self._artifact_root,
                )
                result["viewer_html_path"] = viewer_result["html_path"]
            except Exception:
                result["viewer_html_path"] = None

            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n\n{tb}")
