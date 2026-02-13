from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from faceanalyze2.analysis.segmentation import (
    SegmentParams,
    detect_segments_from_signal,
    load_landmark_artifacts,
    run_segmentation,
)
from faceanalyze2.roi.indices import INNER_LIP, MOUTH_CORNERS


def test_detect_segments_from_synthetic_signal() -> None:
    baseline = np.zeros(35, dtype=np.float32)
    rise = np.linspace(0.0, 3.0, 14, dtype=np.float32)
    hold = np.full(8, 3.0, dtype=np.float32)
    fall = np.linspace(3.0, 0.0, 20, dtype=np.float32)
    tail = np.zeros(25, dtype=np.float32)
    signal = np.concatenate([baseline, rise, hold, fall, tail], axis=0)

    result = detect_segments_from_signal(
        signal,
        fps=30.0,
        params=SegmentParams(pre_seconds=1.5),
        missing_ratio=0.0,
    )

    assert result.neutral_idx < result.peak_idx
    assert result.onset_idx <= result.peak_idx <= result.offset_idx
    assert result.amplitude > 2.0
    assert result.flags["neutral_after_peak"] is False


def test_load_landmarks_reports_missing_file_with_guidance(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_landmark_artifacts(video_path=video_path, artifact_root=tmp_path / "artifacts")

    message = str(exc_info.value)
    assert "Landmarks file not found" in message
    assert "faceanalyze2 landmarks extract --video" in message


def test_load_landmarks_validates_shape(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    npz_path = artifact_dir / "landmarks.npz"
    np.savez_compressed(
        npz_path,
        timestamps_ms=np.asarray([0, 33], dtype=np.int64),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        landmarks_xyz=np.zeros((2, 477, 3), dtype=np.float32),
        presence=np.asarray([True, True], dtype=bool),
    )
    meta_path = artifact_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"fps": 30.0, "width": 640, "height": 480, "frame_count": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_landmark_artifacts(video_path=tmp_path / "sample.mp4", artifact_root=tmp_path / "artifacts")

    assert "landmarks_xyz must have shape" in str(exc_info.value)


def test_run_segmentation_writes_outputs(tmp_path: Path) -> None:
    video_stem = "sample"
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / video_stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t = 40
    timestamps_ms = (np.arange(t) * 33).astype(np.int64)
    frame_indices = np.arange(t, dtype=np.int64)
    presence = np.ones((t,), dtype=bool)
    landmarks_xyz = np.full((t, 478, 3), np.nan, dtype=np.float32)

    left_eye = np.asarray([0.40, 0.40, 0.0], dtype=np.float32)
    right_eye = np.asarray([0.60, 0.40, 0.0], dtype=np.float32)
    center = np.asarray([0.50, 0.62], dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, len(INNER_LIP), endpoint=False)

    for idx in range(t):
        strength = np.exp(-0.5 * ((idx - 18.0) / 4.0) ** 2)
        mouth_half_width = 0.055 + (0.035 * strength)
        lip_rx = 0.018 + (0.012 * strength)
        lip_ry = 0.010 + (0.022 * strength)

        landmarks_xyz[idx, 33] = left_eye
        landmarks_xyz[idx, 263] = right_eye
        landmarks_xyz[idx, MOUTH_CORNERS[0]] = np.asarray(
            [center[0] - mouth_half_width, center[1], 0.0],
            dtype=np.float32,
        )
        landmarks_xyz[idx, MOUTH_CORNERS[1]] = np.asarray(
            [center[0] + mouth_half_width, center[1], 0.0],
            dtype=np.float32,
        )

        for lip_idx, angle in zip(INNER_LIP, angles):
            landmarks_xyz[idx, lip_idx] = np.asarray(
                [center[0] + lip_rx * np.cos(angle), center[1] + lip_ry * np.sin(angle), 0.0],
                dtype=np.float32,
            )

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
        landmarks_xyz=landmarks_xyz,
        presence=presence,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": 640, "height": 480, "frame_count": t}),
        encoding="utf-8",
    )

    result = run_segmentation(
        video_path=tmp_path / "sample.mp4",
        task="smile",
        artifact_root=artifact_root,
    )

    signals_csv = Path(result["signals_csv_path"])
    segment_json = Path(result["segment_json_path"])
    plot_path = Path(result["plot_path"])

    assert signals_csv.exists()
    assert segment_json.exists()
    assert plot_path.exists()

    with signals_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["frame_idx", "timestamp_ms", "signal_raw", "signal_smooth", "task"]
    assert len(rows) == t + 1

    payload = json.loads(segment_json.read_text(encoding="utf-8"))
    assert payload["task"] == "smile"
    assert payload["neutral_idx"] < payload["peak_idx"]
    assert payload["onset_idx"] <= payload["peak_idx"] <= payload["offset_idx"]
