from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceanalyze2.analysis.alignment_viz import run_alignment_visualization
from faceanalyze2.analysis.segmentation import DEFAULT_ARTIFACT_ROOT


def _write_dummy_video(
    path: Path,
    *,
    codec: str,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        return False
    try:
        for idx in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (idx * 11) % 255
            frame[:, :, 1] = (idx * 17) % 255
            frame[:, :, 2] = (idx * 23) % 255
            writer.write(frame)
    finally:
        writer.release()
    return path.exists() and path.stat().st_size > 0


def _make_dummy_video(tmp_path: Path) -> tuple[Path, float, int, int, int]:
    fps = 12.0
    width = 96
    height = 72
    frame_count = 12
    mp4_path = tmp_path / "sample.mp4"
    if _write_dummy_video(
        mp4_path, codec="mp4v", fps=fps, width=width, height=height, frame_count=frame_count
    ):
        return mp4_path, fps, width, height, frame_count

    avi_path = tmp_path / "sample.avi"
    if _write_dummy_video(
        avi_path, codec="MJPG", fps=fps, width=width, height=height, frame_count=frame_count
    ):
        return avi_path, fps, width, height, frame_count

    pytest.skip("No available OpenCV writer codec for test video generation")


def test_run_alignment_visualization_creates_outputs(tmp_path: Path) -> None:
    video_path, fps, width, height, frame_count = _make_dummy_video(tmp_path)
    artifact_root = tmp_path / DEFAULT_ARTIFACT_ROOT
    artifact_dir = artifact_root / video_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = np.arange(frame_count, dtype=np.int64)
    timestamps_ms = (frame_indices * round(1000.0 / fps)).astype(np.int64)
    presence = np.ones((frame_count,), dtype=bool)
    landmarks_xyz = np.full((frame_count, 478, 3), np.nan, dtype=np.float32)

    base_points = {
        33: (0.35, 0.38),
        133: (0.42, 0.38),
        263: (0.58, 0.38),
        362: (0.65, 0.38),
        61: (0.44, 0.64),
        291: (0.56, 0.64),
    }
    for t in range(frame_count):
        dx = 0.005 * np.sin(t / 4.0)
        dy = 0.004 * np.cos(t / 5.0)
        for idx, (x, y) in base_points.items():
            landmarks_xyz[t, idx] = np.asarray([x + dx, y + dy, 0.0], dtype=np.float32)

    raw_xy_pixel = landmarks_xyz[..., :2].copy()
    raw_xy_pixel[..., 0] *= float(width)
    raw_xy_pixel[..., 1] *= float(height)
    stable_xy_ref = raw_xy_pixel[0, [33, 133, 263, 362], :]

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        landmarks_xyz=landmarks_xyz,
        presence=presence,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": fps, "width": width, "height": height, "frame_count": frame_count}),
        encoding="utf-8",
    )
    (artifact_dir / "segment.json").write_text(json.dumps({"neutral_idx": 0}), encoding="utf-8")
    np.savez_compressed(
        artifact_dir / "landmarks_aligned.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        presence=presence,
        landmarks_xy_aligned=raw_xy_pixel.astype(np.float32),
        landmarks_z_aligned=landmarks_xyz[..., 2].astype(np.float32),
        stable_xy_ref=stable_xy_ref.astype(np.float32),
        stable_xy_t=raw_xy_pixel[:, [33, 133, 263, 362], :].astype(np.float32),
        transform_scale=np.ones((frame_count,), dtype=np.float32),
        transform_R=np.repeat(np.eye(2, dtype=np.float32)[None, :, :], frame_count, axis=0),
        transform_t=np.zeros((frame_count, 2), dtype=np.float32),
    )

    result = run_alignment_visualization(
        video_path=video_path,
        artifact_root=artifact_root,
        max_frames=8,
        stride=1,
        n_samples=6,
    )

    check_path = Path(result["alignment_check_path"])
    trajectory_path = Path(result["trajectory_plot_path"])
    overlay_path = Path(result["overlay_path"])

    assert check_path.exists()
    assert trajectory_path.exists()
    assert overlay_path.exists()
    assert overlay_path.suffix.lower() in {".mp4", ".avi"}
