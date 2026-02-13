from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.analysis.segmentation import DEFAULT_ARTIFACT_ROOT, LoadedLandmarks, load_landmark_artifacts

STABLE_IDXS = [33, 133, 263, 362]
LANDMARK_COUNT = 478
DEFAULT_SCALE_OUTLIER_FACTOR = 3.0


@dataclass(frozen=True)
class SegmentInfo:
    segment_path: Path
    neutral_idx: int
    payload: dict[str, Any]


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("src_pts and dst_pts must both have shape (N, 2)")
    if src.shape != dst.shape:
        raise ValueError("src_pts and dst_pts must have the same shape")
    if src.shape[0] < 2:
        raise ValueError("At least 2 points are required for similarity transform estimation")
    if (not np.isfinite(src).all()) or (not np.isfinite(dst).all()):
        raise ValueError("src_pts and dst_pts must be finite")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    var_src = float(np.mean(np.sum(src_centered**2, axis=1)))
    if var_src <= 1e-12:
        raise ValueError("Degenerate source points: variance is too small")

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.eye(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1.0

    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular_values) @ sign) / var_src)
    translation = dst_mean - (scale * (rotation @ src_mean))
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _resolve_segment_path(
    *,
    video_path: str | Path,
    segment_path: str | Path | None,
    artifact_dir: Path,
) -> Path:
    if segment_path is not None:
        return Path(segment_path)
    return artifact_dir / "segment.json"


def _missing_segment_message(segment_path: Path, video_path: str | Path) -> str:
    return (
        f"Segment file not found: {segment_path}\n"
        f"Run this first:\nfaceanalyze2 segment run --video \"{video_path}\" --task <smile|brow|eyeclose>"
    )


def load_segment_info(
    *,
    video_path: str | Path,
    artifact_dir: Path,
    segment_path: str | Path | None = None,
    frame_count: int,
) -> SegmentInfo:
    resolved = _resolve_segment_path(video_path=video_path, segment_path=segment_path, artifact_dir=artifact_dir)
    if not resolved.exists():
        raise FileNotFoundError(_missing_segment_message(resolved, video_path))

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if "neutral_idx" not in payload:
        raise ValueError(f"segment.json is missing required field 'neutral_idx': {resolved}")
    neutral_idx = int(payload["neutral_idx"])
    if neutral_idx < 0 or neutral_idx >= frame_count:
        raise ValueError(
            f"segment.json neutral_idx is out of range: {neutral_idx} (frame_count={frame_count})"
        )

    return SegmentInfo(segment_path=resolved, neutral_idx=neutral_idx, payload=payload)


def _landmarks_to_pixel_xy(landmarks_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(landmarks_xyz[..., :2], dtype=np.float32).copy()
    xy[..., 0] *= float(width)
    xy[..., 1] *= float(height)
    return xy


def _apply_similarity_transform(
    points_xy: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    out = np.full(points_xy.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(points_xy).all(axis=1)
    if valid.any():
        transformed = (scale * (rotation @ points_xy[valid].T)).T + translation
        out[valid] = transformed.astype(np.float32)
    return out


def _build_alignment_json(
    *,
    segment: SegmentInfo,
    loaded: LoadedLandmarks,
    neutral_idx: int,
    scale_z: bool,
    scales: np.ndarray,
    invalid_presence_mask: np.ndarray,
    invalid_stable_mask: np.ndarray,
    invalid_transform_mask: np.ndarray,
    scale_outlier_mask: np.ndarray,
    scale_median: float | None,
) -> dict[str, Any]:
    transformed_mask = np.isfinite(scales)
    transformed_frames = int(np.sum(transformed_mask))
    total_frames = int(loaded.frame_indices.shape[0])
    present_frames = int(np.sum(loaded.presence))
    scale_outlier_frames = int(np.sum(scale_outlier_mask))

    return {
        "stable_indices": STABLE_IDXS,
        "neutral_idx": neutral_idx,
        "neutral_frame_idx": int(loaded.frame_indices[neutral_idx]),
        "neutral_timestamp_ms": int(loaded.timestamps_ms[neutral_idx]),
        "method": "umeyama_similarity_2d",
        "output_xy_unit": "pixel",
        "z_mode": "scaled_by_similarity_scale" if scale_z else "keep_original_z",
        "landmarks_path": str(loaded.npz_path),
        "segment_path": str(segment.segment_path),
        "quality_stats": {
            "total_frames": total_frames,
            "present_frames": present_frames,
            "transformed_frames": transformed_frames,
            "invalid_presence_frames": int(np.sum(invalid_presence_mask)),
            "invalid_stable_frames": int(np.sum(invalid_stable_mask)),
            "invalid_transform_frames": int(np.sum(invalid_transform_mask)),
            "scale_median": scale_median,
            "scale_outlier_factor": DEFAULT_SCALE_OUTLIER_FACTOR,
            "scale_outlier_frames": scale_outlier_frames,
            "scale_outlier_ratio": (
                float(scale_outlier_frames / transformed_frames) if transformed_frames > 0 else None
            ),
            "scale_outlier_frame_indices": loaded.frame_indices[scale_outlier_mask].astype(int).tolist(),
            "invalid_stable_frame_indices": loaded.frame_indices[invalid_stable_mask].astype(int).tolist(),
        },
    }


def run_alignment(
    *,
    video_path: str | Path,
    landmarks_path: str | Path | None = None,
    segment_path: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    scale_z: bool = False,
    scale_outlier_factor: float = DEFAULT_SCALE_OUTLIER_FACTOR,
) -> dict[str, Any]:
    if scale_outlier_factor <= 1.0:
        raise ValueError(f"scale_outlier_factor must be > 1.0, got: {scale_outlier_factor}")

    loaded = load_landmark_artifacts(
        video_path=video_path,
        landmarks_path=landmarks_path,
        artifact_root=artifact_root,
    )
    segment = load_segment_info(
        video_path=video_path,
        artifact_dir=loaded.artifact_dir,
        segment_path=segment_path,
        frame_count=loaded.frame_indices.shape[0],
    )
    neutral_idx = segment.neutral_idx

    width = int(loaded.meta["width"])
    height = int(loaded.meta["height"])
    pixel_xy = _landmarks_to_pixel_xy(loaded.landmarks_xyz, width=width, height=height)
    z_values = np.asarray(loaded.landmarks_xyz[..., 2], dtype=np.float32)

    stable_xy_ref = pixel_xy[neutral_idx, STABLE_IDXS, :]
    if not loaded.presence[neutral_idx]:
        raise ValueError(f"neutral_idx {neutral_idx} is marked as presence=False in landmarks")
    if not np.isfinite(stable_xy_ref).all():
        raise ValueError("neutral reference stable landmarks contain NaN; cannot run alignment")

    t_count = loaded.frame_indices.shape[0]
    k_count = len(STABLE_IDXS)
    landmarks_xy_aligned = np.full((t_count, LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
    landmarks_z_aligned = np.full((t_count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    stable_xy_t = np.full((t_count, k_count, 2), np.nan, dtype=np.float32)
    transform_scale = np.full((t_count,), np.nan, dtype=np.float32)
    transform_r = np.full((t_count, 2, 2), np.nan, dtype=np.float32)
    transform_t = np.full((t_count, 2), np.nan, dtype=np.float32)

    invalid_presence_mask = ~loaded.presence.astype(bool)
    invalid_stable_mask = np.zeros((t_count,), dtype=bool)
    invalid_transform_mask = np.zeros((t_count,), dtype=bool)

    for idx in range(t_count):
        if invalid_presence_mask[idx]:
            continue

        stable_src = pixel_xy[idx, STABLE_IDXS, :]
        if not np.isfinite(stable_src).all():
            invalid_stable_mask[idx] = True
            continue

        try:
            scale, rotation, translation = estimate_similarity_transform(stable_src, stable_xy_ref)
        except ValueError:
            invalid_transform_mask[idx] = True
            continue

        if (not np.isfinite(scale)) or (scale <= 0):
            invalid_transform_mask[idx] = True
            continue

        aligned_xy = _apply_similarity_transform(
            pixel_xy[idx],
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        landmarks_xy_aligned[idx] = aligned_xy
        stable_xy_t[idx] = aligned_xy[STABLE_IDXS]

        z_out = z_values[idx] * float(scale) if scale_z else z_values[idx]
        finite_z = np.isfinite(z_out)
        landmarks_z_aligned[idx, finite_z] = z_out[finite_z].astype(np.float32)

        transform_scale[idx] = np.float32(scale)
        transform_r[idx] = rotation.astype(np.float32)
        transform_t[idx] = translation.astype(np.float32)

    finite_scale_mask = np.isfinite(transform_scale) & (transform_scale > 0)
    if finite_scale_mask.any():
        scale_median = float(np.median(transform_scale[finite_scale_mask]))
        scale_threshold = scale_median * float(scale_outlier_factor)
        scale_outlier_mask = finite_scale_mask & (transform_scale > scale_threshold)
    else:
        scale_median = None
        scale_outlier_mask = np.zeros((t_count,), dtype=bool)

    aligned_npz_path = loaded.artifact_dir / "landmarks_aligned.npz"
    np.savez_compressed(
        aligned_npz_path,
        frame_indices=loaded.frame_indices.astype(np.int64),
        timestamps_ms=loaded.timestamps_ms.astype(np.int64),
        presence=loaded.presence.astype(bool),
        landmarks_xy_aligned=landmarks_xy_aligned.astype(np.float32),
        landmarks_z_aligned=landmarks_z_aligned.astype(np.float32),
        stable_xy_ref=stable_xy_ref.astype(np.float32),
        stable_xy_t=stable_xy_t.astype(np.float32),
        transform_scale=transform_scale.astype(np.float32),
        transform_R=transform_r.astype(np.float32),
        transform_t=transform_t.astype(np.float32),
        scale_outlier_mask=scale_outlier_mask.astype(bool),
        invalid_stable_mask=invalid_stable_mask.astype(bool),
        invalid_transform_mask=invalid_transform_mask.astype(bool),
    )

    alignment_payload = _build_alignment_json(
        segment=segment,
        loaded=loaded,
        neutral_idx=neutral_idx,
        scale_z=scale_z,
        scales=transform_scale,
        invalid_presence_mask=invalid_presence_mask,
        invalid_stable_mask=invalid_stable_mask,
        invalid_transform_mask=invalid_transform_mask,
        scale_outlier_mask=scale_outlier_mask,
        scale_median=scale_median,
    )
    alignment_payload["quality_stats"]["scale_outlier_factor"] = float(scale_outlier_factor)
    alignment_json_path = loaded.artifact_dir / "alignment.json"
    alignment_json_path.write_text(json.dumps(alignment_payload, indent=2), encoding="utf-8")

    return {
        "aligned_npz_path": str(aligned_npz_path),
        "alignment_json_path": str(alignment_json_path),
        "quality_stats": alignment_payload["quality_stats"],
    }
