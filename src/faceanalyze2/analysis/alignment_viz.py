from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from faceanalyze2.analysis.alignment import STABLE_IDXS
from faceanalyze2.analysis.segmentation import DEFAULT_ARTIFACT_ROOT, load_landmark_artifacts
from faceanalyze2.io.video_reader import iter_frames
from faceanalyze2.roi.indices import (
    INNER_LIP,
    LEFT_BROW,
    LEFT_EYE,
    MOUTH_CORNERS,
    RIGHT_BROW,
    RIGHT_EYE,
)

REQUIRED_ALIGNED_KEYS = {
    "frame_indices",
    "timestamps_ms",
    "presence",
    "landmarks_xy_aligned",
    "transform_scale",
    "transform_R",
    "transform_t",
}
OVERLAY_POINT_INDICES = sorted(
    set(
        STABLE_IDXS
        + [MOUTH_CORNERS[0], MOUTH_CORNERS[1]]
        + LEFT_EYE
        + RIGHT_EYE
        + LEFT_BROW
        + RIGHT_BROW
        + INNER_LIP
    )
)


def _resolve_artifact_dir(
    *,
    video_path: str | Path,
    artifact_root: str | Path,
    landmarks_path: str | Path | None,
    segment_path: str | Path | None,
    aligned_path: str | Path | None,
) -> Path:
    if aligned_path is not None:
        return Path(aligned_path).parent
    if landmarks_path is not None:
        return Path(landmarks_path).parent
    if segment_path is not None:
        return Path(segment_path).parent
    return Path(artifact_root) / Path(video_path).stem


def _missing_landmarks_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Landmarks file not found: {path}\n"
        "Run this first: landmarks extract.\n"
        f'faceanalyze2 landmarks extract --video "{video_path}"'
    )


def _missing_segment_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Segment file not found: {path}\n"
        "Run this first: segment run.\n"
        f'faceanalyze2 segment run --video "{video_path}" --task <smile|brow|eyeclose>'
    )


def _missing_aligned_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Aligned landmarks file not found: {path}\n"
        "Run this first: align run.\n"
        f'faceanalyze2 align run --video "{video_path}"'
    )


def _uniform_sample(indices: np.ndarray, n_samples: int) -> np.ndarray:
    if indices.size == 0:
        return indices
    n_samples = max(1, int(n_samples))
    if indices.size <= n_samples:
        return indices
    positions = np.linspace(0, indices.size - 1, n_samples).round().astype(int)
    return indices[np.unique(positions)]


def _load_segment_neutral_idx(segment_path: Path, frame_count: int) -> int:
    payload = json.loads(segment_path.read_text(encoding="utf-8"))
    if "neutral_idx" not in payload:
        raise ValueError(f"segment.json is missing required field 'neutral_idx': {segment_path}")
    neutral_idx = int(payload["neutral_idx"])
    if neutral_idx < 0 or neutral_idx >= frame_count:
        raise ValueError(
            f"segment.json neutral_idx out of range: {neutral_idx} (frame_count={frame_count})"
        )
    return neutral_idx


def _load_aligned_arrays(aligned_path: Path) -> dict[str, np.ndarray]:
    with np.load(aligned_path) as data:
        arrays = {name: data[name] for name in data.files}
    missing = REQUIRED_ALIGNED_KEYS.difference(arrays)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"landmarks_aligned.npz is missing required keys: {missing_keys}")
    return arrays


def _validate_alignment_compatibility(
    *,
    raw_frame_indices: np.ndarray,
    raw_timestamps_ms: np.ndarray,
    aligned_arrays: dict[str, np.ndarray],
) -> None:
    aligned_frame_indices = np.asarray(aligned_arrays["frame_indices"])
    aligned_timestamps_ms = np.asarray(aligned_arrays["timestamps_ms"])
    landmarks_xy_aligned = np.asarray(aligned_arrays["landmarks_xy_aligned"])

    if aligned_frame_indices.shape != raw_frame_indices.shape:
        raise ValueError("landmarks_aligned frame_indices shape mismatch with landmarks")
    if aligned_timestamps_ms.shape != raw_timestamps_ms.shape:
        raise ValueError("landmarks_aligned timestamps_ms shape mismatch with landmarks")
    if landmarks_xy_aligned.ndim != 3 or landmarks_xy_aligned.shape[1:] != (478, 2):
        raise ValueError(
            f"landmarks_xy_aligned must have shape (T, 478, 2), got {landmarks_xy_aligned.shape}"
        )
    if landmarks_xy_aligned.shape[0] != raw_frame_indices.shape[0]:
        raise ValueError("landmarks_xy_aligned frame count mismatch with landmarks")

    if not np.array_equal(
        aligned_frame_indices.astype(np.int64), raw_frame_indices.astype(np.int64)
    ):
        raise ValueError(
            "landmarks_aligned frame_indices values do not match landmarks frame_indices"
        )
    if not np.array_equal(
        aligned_timestamps_ms.astype(np.int64), raw_timestamps_ms.astype(np.int64)
    ):
        raise ValueError(
            "landmarks_aligned timestamps_ms values do not match landmarks timestamps_ms"
        )


def _to_raw_pixel_xy(landmarks_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    raw_xy = np.asarray(landmarks_xyz[..., :2], dtype=np.float32).copy()
    raw_xy[..., 0] *= float(width)
    raw_xy[..., 1] *= float(height)
    return raw_xy


def _draw_points(image_rgb: np.ndarray, points_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    output = image_rgb.copy()
    stable_set = set(STABLE_IDXS)
    for idx in indices:
        point = points_xy[idx]
        if not np.isfinite(point).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        color = (255, 90, 40) if idx in stable_set else (80, 210, 255)
        radius = 3 if idx in stable_set else 2
        cv2.circle(output, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return output


def _save_alignment_check_plot(
    *,
    output_path: Path,
    raw_xy: np.ndarray,
    aligned_xy: np.ndarray,
    sample_rows: np.ndarray,
    width: int,
    height: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for row in sample_rows:
        raw_points = raw_xy[row]
        aligned_points = aligned_xy[row]
        axes[0].scatter(raw_points[:, 0], raw_points[:, 1], s=4, alpha=0.06, c="#1f77b4")
        axes[1].scatter(aligned_points[:, 0], aligned_points[:, 1], s=4, alpha=0.06, c="#d62728")

        stable_raw = raw_xy[row, STABLE_IDXS]
        stable_aligned = aligned_xy[row, STABLE_IDXS]
        axes[0].scatter(stable_raw[:, 0], stable_raw[:, 1], s=14, alpha=0.5, c="#ff7f0e")
        axes[1].scatter(stable_aligned[:, 0], stable_aligned[:, 1], s=14, alpha=0.5, c="#2ca02c")

    axes[0].set_title("Raw landmarks overlay")
    axes[1].set_title("Aligned landmarks overlay")
    for axis in axes:
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_xlabel("x (pixel)")
    axes[0].set_ylabel("y (pixel)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_trajectory_plot(
    *,
    output_path: Path,
    frame_indices: np.ndarray,
    raw_xy: np.ndarray,
    aligned_xy: np.ndarray,
    neutral_idx: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neutral_raw = raw_xy[neutral_idx, STABLE_IDXS, :]
    neutral_aligned = aligned_xy[neutral_idx, STABLE_IDXS, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for local_idx, stable_idx in enumerate(STABLE_IDXS):
        raw_dist = np.linalg.norm(raw_xy[:, stable_idx, :] - neutral_raw[local_idx], axis=1)
        aligned_dist = np.linalg.norm(
            aligned_xy[:, stable_idx, :] - neutral_aligned[local_idx], axis=1
        )
        raw_dist[~np.isfinite(raw_dist)] = np.nan
        aligned_dist[~np.isfinite(aligned_dist)] = np.nan
        axes[0].plot(frame_indices, raw_dist, linewidth=1.6, label=f"idx {stable_idx}")
        axes[1].plot(frame_indices, aligned_dist, linewidth=1.6, label=f"idx {stable_idx}")

    axes[0].set_title("Stable point trajectory (raw): ||p(t)-p(neutral)||")
    axes[1].set_title("Stable point trajectory (aligned): ||p(t)-p(neutral)||")
    for axis in axes:
        axis.set_ylabel("distance (pixel)")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[1].set_xlabel("frame index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _open_overlay_writer(
    artifact_dir: Path, width: int, height: int, fps: float
) -> tuple[Any, Path]:
    overlay_mp4 = artifact_dir / "alignment_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width * 2, height),
    )
    if writer.isOpened():
        return writer, overlay_mp4
    writer.release()

    overlay_avi = artifact_dir / "alignment_overlay.avi"
    writer = cv2.VideoWriter(
        str(overlay_avi),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width * 2, height),
    )
    if writer.isOpened():
        return writer, overlay_avi
    writer.release()
    raise RuntimeError("Failed to initialize overlay video writer for mp4 and avi")


def _build_warp_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.asarray(rotation, dtype=np.float32)
    t = np.asarray(translation, dtype=np.float32)
    return np.asarray(
        [
            [scale * mat[0, 0], scale * mat[0, 1], t[0]],
            [scale * mat[1, 0], scale * mat[1, 1], t[1]],
        ],
        dtype=np.float32,
    )


def _write_overlay_video(
    *,
    artifact_dir: Path,
    video_path: str | Path,
    width: int,
    height: int,
    fps: float,
    frame_indices: np.ndarray,
    raw_xy: np.ndarray,
    aligned_xy: np.ndarray,
    presence: np.ndarray,
    transform_scale: np.ndarray,
    transform_r: np.ndarray,
    transform_t: np.ndarray,
    max_frames: int,
    stride: int,
) -> Path:
    selected_rows = np.arange(frame_indices.shape[0], dtype=int)
    selected_rows = selected_rows[:: max(1, int(stride))]
    if max_frames > 0:
        selected_rows = selected_rows[: int(max_frames)]
    if selected_rows.size == 0:
        raise ValueError("No frames selected for overlay video generation")

    row_by_frame_idx = {int(frame_indices[row]): int(row) for row in selected_rows}
    min_frame_idx = int(frame_indices[selected_rows[0]])
    max_frame_idx = int(frame_indices[selected_rows[-1]]) + 1

    writer, output_path = _open_overlay_writer(artifact_dir, width, height, fps)
    written = 0
    try:
        for frame in iter_frames(
            video_path,
            stride=1,
            start_frame=min_frame_idx,
            end_frame=max_frame_idx,
            to_rgb=True,
        ):
            row = row_by_frame_idx.get(frame.idx)
            if row is None:
                continue

            raw_canvas = frame.image_rgb.copy()
            raw_canvas = _draw_points(raw_canvas, raw_xy[row], OVERLAY_POINT_INDICES)

            valid_transform = (
                bool(presence[row])
                and np.isfinite(transform_scale[row])
                and (transform_scale[row] > 0)
                and np.isfinite(transform_r[row]).all()
                and np.isfinite(transform_t[row]).all()
            )
            if valid_transform:
                warp_matrix = _build_warp_matrix(
                    scale=float(transform_scale[row]),
                    rotation=transform_r[row],
                    translation=transform_t[row],
                )
                aligned_frame = cv2.warpAffine(
                    frame.image_rgb,
                    warp_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
            else:
                aligned_frame = np.zeros((height, width, 3), dtype=np.uint8)

            aligned_canvas = _draw_points(aligned_frame, aligned_xy[row], OVERLAY_POINT_INDICES)
            split_rgb = np.hstack([raw_canvas, aligned_canvas])
            split_bgr = cv2.cvtColor(split_rgb, cv2.COLOR_RGB2BGR)
            writer.write(split_bgr)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to alignment overlay video")
    return output_path


def run_alignment_visualization(
    *,
    video_path: str | Path,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    landmarks_path: str | Path | None = None,
    segment_path: str | Path | None = None,
    aligned_path: str | Path | None = None,
    max_frames: int = 300,
    stride: int = 2,
    n_samples: int = 15,
) -> dict[str, str]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got: {stride}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got: {max_frames}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got: {n_samples}")

    artifact_dir = _resolve_artifact_dir(
        video_path=video_path,
        artifact_root=artifact_root,
        landmarks_path=landmarks_path,
        segment_path=segment_path,
        aligned_path=aligned_path,
    )
    resolved_landmarks = (
        Path(landmarks_path) if landmarks_path is not None else artifact_dir / "landmarks.npz"
    )
    resolved_segment = (
        Path(segment_path) if segment_path is not None else artifact_dir / "segment.json"
    )
    resolved_aligned = (
        Path(aligned_path) if aligned_path is not None else artifact_dir / "landmarks_aligned.npz"
    )

    if not resolved_landmarks.exists():
        raise FileNotFoundError(_missing_landmarks_message(resolved_landmarks, video_path))
    if not resolved_segment.exists():
        raise FileNotFoundError(_missing_segment_message(resolved_segment, video_path))
    if not resolved_aligned.exists():
        raise FileNotFoundError(_missing_aligned_message(resolved_aligned, video_path))

    loaded = load_landmark_artifacts(
        video_path=video_path,
        landmarks_path=resolved_landmarks,
        artifact_root=artifact_root,
    )
    aligned_arrays = _load_aligned_arrays(resolved_aligned)
    _validate_alignment_compatibility(
        raw_frame_indices=loaded.frame_indices,
        raw_timestamps_ms=loaded.timestamps_ms,
        aligned_arrays=aligned_arrays,
    )

    width = int(loaded.meta["width"])
    height = int(loaded.meta["height"])
    fps = float(loaded.meta["fps"])
    raw_xy = _to_raw_pixel_xy(loaded.landmarks_xyz, width=width, height=height)
    aligned_xy = np.asarray(aligned_arrays["landmarks_xy_aligned"], dtype=np.float32)

    neutral_idx = _load_segment_neutral_idx(
        resolved_segment, frame_count=loaded.frame_indices.shape[0]
    )
    valid_rows = np.where(
        loaded.presence
        & np.isfinite(raw_xy[:, STABLE_IDXS, :]).all(axis=(1, 2))
        & np.isfinite(aligned_xy[:, STABLE_IDXS, :]).all(axis=(1, 2))
    )[0]
    if valid_rows.size == 0:
        raise ValueError(
            "No valid presence=True frames with finite stable landmarks for visualization"
        )

    sampled_rows = _uniform_sample(valid_rows, n_samples=n_samples)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    alignment_check_path = artifact_dir / "alignment_check.png"
    trajectory_plot_path = artifact_dir / "trajectory_plot.png"
    overlay_path = _write_overlay_video(
        artifact_dir=artifact_dir,
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_indices=loaded.frame_indices,
        raw_xy=raw_xy,
        aligned_xy=aligned_xy,
        presence=loaded.presence,
        transform_scale=np.asarray(aligned_arrays["transform_scale"], dtype=np.float32),
        transform_r=np.asarray(aligned_arrays["transform_R"], dtype=np.float32),
        transform_t=np.asarray(aligned_arrays["transform_t"], dtype=np.float32),
        max_frames=max_frames,
        stride=stride,
    )

    _save_alignment_check_plot(
        output_path=alignment_check_path,
        raw_xy=raw_xy,
        aligned_xy=aligned_xy,
        sample_rows=sampled_rows,
        width=width,
        height=height,
    )
    _save_trajectory_plot(
        output_path=trajectory_plot_path,
        frame_indices=loaded.frame_indices,
        raw_xy=raw_xy,
        aligned_xy=aligned_xy,
        neutral_idx=neutral_idx,
    )

    return {
        "alignment_check_path": str(alignment_check_path),
        "trajectory_plot_path": str(trajectory_plot_path),
        "overlay_path": str(overlay_path),
    }
