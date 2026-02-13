from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faceanalyze2.io.video_reader import VideoInfo
from faceanalyze2.landmarks.mediapipe_face_landmarker import (
    OFFICIAL_FACE_LANDMARKER_MODEL_URL,
    extract_face_landmarks_from_video,
    save_landmark_artifacts,
)


def test_extract_reports_helpful_model_missing_message(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.task"

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_face_landmarks_from_video(
            video_path=str(tmp_path / "unused.mp4"),
            model_path=str(missing_model),
        )

    message = str(exc_info.value)
    assert "Model file not found" in message
    assert OFFICIAL_FACE_LANDMARKER_MODEL_URL in message
    assert "Invoke-WebRequest" in message


def test_save_landmark_artifacts_npz_layout(tmp_path: Path) -> None:
    video_info = VideoInfo(
        path=tmp_path / "sample.mp4",
        fps=30.0,
        frame_count=3,
        width=64,
        height=48,
        duration_ms=100,
    )
    timestamps_ms = np.asarray([0, 33, 67], dtype=np.int64)
    frame_indices = np.asarray([0, 1, 2], dtype=np.int64)
    landmarks_xyz = np.zeros((3, 478, 3), dtype=np.float32)
    presence = np.asarray([True, False, True], dtype=bool)
    blendshapes = np.zeros((3, 52), dtype=np.float32)
    transforms = np.zeros((3, 4, 4), dtype=np.float32)

    npz_path, meta_path = save_landmark_artifacts(
        video_info=video_info,
        model_path="models/face_landmarker.task",
        stride=1,
        start_frame=0,
        end_frame=None,
        num_faces=1,
        output_blendshapes=True,
        output_transforms=True,
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
        landmarks_xyz=landmarks_xyz,
        presence=presence,
        blendshapes=blendshapes,
        transforms=transforms,
        artifact_root=tmp_path / "artifacts",
        extract_time="2026-02-13T00:00:00+00:00",
        elapsed_seconds=0.25,
    )

    assert npz_path.exists()
    assert meta_path.exists()

    saved = np.load(npz_path)
    assert saved["timestamps_ms"].shape == (3,)
    assert saved["frame_indices"].shape == (3,)
    assert saved["landmarks_xyz"].shape == (3, 478, 3)
    assert saved["presence"].shape == (3,)
    assert saved["blendshapes"].shape == (3, 52)
    assert saved["transforms"].shape == (3, 4, 4)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["fps"] == 30.0
    assert meta["width"] == 64
    assert meta["height"] == 48
    assert meta["frame_count"] == 3
    assert meta["stride"] == 1
    assert meta["model_path"] == "models/face_landmarker.task"
    assert meta["extract_time"] == "2026-02-13T00:00:00+00:00"
