from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from faceanalyze2.io.video_reader import iter_frames, probe_video


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
            frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            frame_bgr[:, :, 0] = (idx * 25) % 255
            frame_bgr[:, :, 1] = (idx * 50) % 255
            frame_bgr[:, :, 2] = (idx * 75) % 255
            writer.write(frame_bgr)
    finally:
        writer.release()

    return path.exists() and path.stat().st_size > 0


def _make_dummy_video(tmp_path: Path) -> tuple[Path, float, int, int, int]:
    fps = 10.0
    width = 64
    height = 48
    frame_count = 6

    mp4_path = tmp_path / "dummy.mp4"
    if _write_dummy_video(
        mp4_path,
        codec="mp4v",
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    ):
        return mp4_path, fps, width, height, frame_count

    avi_path = tmp_path / "dummy.avi"
    if _write_dummy_video(
        avi_path,
        codec="MJPG",
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    ):
        return avi_path, fps, width, height, frame_count

    pytest.skip("No available OpenCV writer codec for test video generation")


def test_probe_video_reads_metadata(tmp_path: Path) -> None:
    video_path, _, expected_width, expected_height, _ = _make_dummy_video(tmp_path)

    info = probe_video(video_path)

    assert info.path == video_path
    assert info.fps > 0
    assert info.width == expected_width
    assert info.height == expected_height
    assert info.frame_count > 0
    assert info.duration_ms == round(info.frame_count * 1000 / info.fps)


def test_iter_frames_stride_and_timestamps(tmp_path: Path) -> None:
    video_path, _, expected_width, expected_height, _ = _make_dummy_video(tmp_path)
    info = probe_video(video_path)

    frames = list(iter_frames(video_path, stride=2))
    assert frames

    indices = [frame.idx for frame in frames]
    assert indices == list(range(0, indices[-1] + 1, 2))

    for frame in frames:
        assert frame.image_rgb.shape == (expected_height, expected_width, 3)
        assert frame.image_rgb.dtype == np.uint8
        assert frame.timestamp_ms == round(frame.idx * 1000 / info.fps)

    timestamp_values = [frame.timestamp_ms for frame in frames]
    assert timestamp_values == sorted(timestamp_values)
