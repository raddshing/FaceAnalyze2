from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: int


@dataclass(frozen=True)
class Frame:
    idx: int
    timestamp_ms: int
    image_rgb: np.ndarray


def _normalize_path(path: str | Path) -> Path:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video extension '{video_path.suffix}'. Supported: {supported}"
        )
    return video_path


def probe_video(path: str | Path) -> VideoInfo:
    video_path = _normalize_path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video file with OpenCV: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS from video metadata: {video_path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid frame size from video metadata: {video_path}")
    if frame_count < 0:
        raise RuntimeError(f"Invalid frame count from video metadata: {video_path}")

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_ms=round(frame_count * 1000 / fps),
    )


def iter_frames(
    path: str | Path,
    *,
    stride: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    to_rgb: bool = True,
) -> Iterator[Frame]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got: {stride}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got: {start_frame}")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")

    info = probe_video(path)
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video file with OpenCV: {info.path}")

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break

            ok, image = cap.read()
            if not ok:
                break

            if (frame_idx - start_frame) % stride == 0:
                image_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if to_rgb else image
                yield Frame(
                    idx=frame_idx,
                    timestamp_ms=round(frame_idx * 1000 / info.fps),
                    image_rgb=image_out,
                )

            frame_idx += 1
    finally:
        cap.release()


def extract_frame(path: str | Path, frame_idx: int, *, to_rgb: bool = True) -> Frame:
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got: {frame_idx}")
    iterator = iter_frames(
        path, stride=1, start_frame=frame_idx, end_frame=frame_idx + 1, to_rgb=to_rgb
    )
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(f"Frame index out of range: {frame_idx}") from exc


def save_frame_png(path: str | Path, frame_idx: int, out: str | Path) -> Path:
    out_path = Path(out)
    if out_path.suffix.lower() != ".png":
        raise ValueError(f"Output file must be a .png file: {out_path}")

    frame = extract_frame(path, frame_idx, to_rgb=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write PNG frame to: {out_path}")

    return out_path
