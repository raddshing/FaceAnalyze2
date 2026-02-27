from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.analysis.segmentation import DEFAULT_ARTIFACT_ROOT
from faceanalyze2.roi.indices import (
    REGION_JAW,
    REGION_LEFT_BROW,
    REGION_LEFT_EYE,
    REGION_MOUTH,
    REGION_NOSE,
    REGION_RIGHT_BROW,
    REGION_RIGHT_EYE,
)

LANDMARK_COUNT = 478
REQUIRED_LANDMARK_KEYS = {"timestamps_ms", "frame_indices", "landmarks_xyz", "presence"}
STABLE_IDXS = [33, 133, 263, 362]
EPSILON = 1e-6

DEFAULT_FLIP_Y = True
DEFAULT_FLIP_Z = True
DEFAULT_Z_SCALE = 1.6
DEFAULT_CONE_INTEROCULAR_SCALE = 0.03
MIN_CONE_LENGTH_PX = 6.0
MAX_CONE_LENGTH_PX = 12.0
MAX_2D_FRAMES = 120
FRAME_JPEG_QUALITY = 80

REGION_SEEDS: list[tuple[str, list[int]]] = [
    ("left_eye", REGION_LEFT_EYE),
    ("right_eye", REGION_RIGHT_EYE),
    ("left_brow", REGION_LEFT_BROW),
    ("right_brow", REGION_RIGHT_BROW),
    ("mouth", REGION_MOUTH),
    ("nose", REGION_NOSE),
    ("jaw", REGION_JAW),
]


@dataclass(frozen=True)
class MotionViewerPaths:
    artifact_dir: Path
    landmarks_path: Path
    meta_path: Path
    segment_path: Path
    output_html_path: Path


def _resolve_artifact_dir(
    *,
    video_path: str | Path,
    artifact_root: str | Path,
    landmarks_path: str | Path | None,
    segment_path: str | Path | None,
) -> Path:
    if landmarks_path is not None:
        return Path(landmarks_path).parent
    if segment_path is not None:
        return Path(segment_path).parent
    return Path(artifact_root) / Path(video_path).stem


def _resolve_paths(
    *,
    video_path: str | Path,
    artifact_root: str | Path,
    landmarks_path: str | Path | None,
    segment_path: str | Path | None,
) -> MotionViewerPaths:
    artifact_dir = _resolve_artifact_dir(
        video_path=video_path,
        artifact_root=artifact_root,
        landmarks_path=landmarks_path,
        segment_path=segment_path,
    )
    resolved_landmarks = (
        Path(landmarks_path) if landmarks_path is not None else artifact_dir / "landmarks.npz"
    )
    resolved_segment = (
        Path(segment_path) if segment_path is not None else artifact_dir / "segment.json"
    )
    return MotionViewerPaths(
        artifact_dir=artifact_dir,
        landmarks_path=resolved_landmarks,
        meta_path=artifact_dir / "meta.json",
        segment_path=resolved_segment,
        output_html_path=artifact_dir / "motion_viewer.html",
    )


def _missing_landmarks_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Landmarks file not found: {path}\n"
        "Run this first:\n"
        f'faceanalyze2 landmarks extract --video "{video_path}"'
    )


def _missing_segment_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Segment file not found: {path}\n"
        "Run this first:\n"
        f'faceanalyze2 segment run --video "{video_path}" --task <smile|brow|eyeclose>'
    )


def _missing_meta_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Metadata file not found: {path}\n"
        "Run this first:\n"
        f'faceanalyze2 landmarks extract --video "{video_path}"'
    )


def _load_landmark_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}

    missing_keys = REQUIRED_LANDMARK_KEYS.difference(arrays)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"landmarks.npz is missing required keys: {missing}")

    timestamps_ms = np.asarray(arrays["timestamps_ms"])
    frame_indices = np.asarray(arrays["frame_indices"])
    landmarks_xyz = np.asarray(arrays["landmarks_xyz"])
    presence = np.asarray(arrays["presence"])

    if timestamps_ms.ndim != 1:
        raise ValueError("timestamps_ms must be 1D")
    if frame_indices.ndim != 1:
        raise ValueError("frame_indices must be 1D")
    if presence.ndim != 1:
        raise ValueError("presence must be 1D")
    if landmarks_xyz.ndim != 3 or landmarks_xyz.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(
            f"landmarks_xyz must have shape (T, {LANDMARK_COUNT}, 3), got {landmarks_xyz.shape}"
        )

    t_count = timestamps_ms.shape[0]
    if (
        frame_indices.shape[0] != t_count
        or presence.shape[0] != t_count
        or landmarks_xyz.shape[0] != t_count
    ):
        raise ValueError("Array length mismatch in landmarks.npz")

    return {
        "timestamps_ms": timestamps_ms.astype(np.int64),
        "frame_indices": frame_indices.astype(np.int64),
        "landmarks_xyz": landmarks_xyz.astype(np.float32),
        "presence": presence.astype(bool),
    }


def _load_meta(meta_path: Path) -> dict[str, Any]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    for field in ("width", "height", "fps"):
        if field not in payload:
            raise ValueError(f"meta.json is missing required field: {field}")
    width = int(payload["width"])
    height = int(payload["height"])
    fps = float(payload["fps"])
    if width <= 0 or height <= 0:
        raise ValueError("meta.json contains invalid width/height")
    if fps <= 0:
        raise ValueError("meta.json contains invalid fps")
    payload["width"] = width
    payload["height"] = height
    payload["fps"] = fps
    return payload


def _load_segment(segment_path: Path, frame_count: int) -> dict[str, Any]:
    payload = json.loads(segment_path.read_text(encoding="utf-8"))
    for field in ("neutral_idx", "peak_idx"):
        if field not in payload:
            raise ValueError(f"segment.json is missing required field: {field}")

    neutral_idx = int(payload["neutral_idx"])
    peak_idx = int(payload["peak_idx"])
    if neutral_idx < 0 or neutral_idx >= frame_count:
        raise ValueError(
            f"segment.json neutral_idx out of range: {neutral_idx} (frame_count={frame_count})"
        )
    if peak_idx < 0 or peak_idx >= frame_count:
        raise ValueError(
            f"segment.json peak_idx out of range: {peak_idx} (frame_count={frame_count})"
        )
    payload["neutral_idx"] = neutral_idx
    payload["peak_idx"] = peak_idx
    return payload


def _to_pseudo_pixel_3d(landmarks_xyz: np.ndarray, *, width: int, height: int) -> np.ndarray:
    xyz = np.asarray(landmarks_xyz, dtype=np.float32).copy()
    xyz[..., 0] *= float(width)
    xyz[..., 1] *= float(height)
    xyz[..., 2] *= float(width)
    return xyz


def estimate_rigid_transform_3d(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must both have shape (N, 3)")
    if src.shape != dst.shape:
        raise ValueError("src_pts and dst_pts must have identical shape")
    if src.shape[0] < 3:
        raise ValueError("At least 3 points are required for rigid 3D transform")
    if (not np.isfinite(src).all()) or (not np.isfinite(dst).all()):
        raise ValueError("src_pts and dst_pts must be finite")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = dst_mean - (rotation @ src_mean)
    transformed = (rotation @ src.T).T + translation
    residual = float(np.sqrt(np.mean(np.sum((transformed - dst) ** 2, axis=1))))
    return rotation.astype(np.float32), translation.astype(np.float32), residual


def _align_to_neutral(
    xyz: np.ndarray,
    *,
    presence: np.ndarray,
    neutral_idx: int,
) -> dict[str, np.ndarray]:
    t_count, n_landmarks, _ = xyz.shape
    aligned = np.full((t_count, n_landmarks, 3), np.nan, dtype=np.float32)
    rotations = np.full((t_count, 3, 3), np.nan, dtype=np.float32)
    translations = np.full((t_count, 3), np.nan, dtype=np.float32)
    residuals = np.full((t_count,), np.nan, dtype=np.float32)
    valid_mask = np.zeros((t_count,), dtype=bool)
    invalid_presence = np.zeros((t_count,), dtype=bool)
    invalid_stable = np.zeros((t_count,), dtype=bool)
    invalid_transform = np.zeros((t_count,), dtype=bool)

    if not presence[neutral_idx]:
        raise ValueError(f"neutral_idx {neutral_idx} is marked as presence=False in landmarks")
    stable_ref = xyz[neutral_idx, STABLE_IDXS, :]
    if not np.isfinite(stable_ref).all():
        raise ValueError("neutral frame stable landmarks contain NaN; cannot run viewer alignment")

    for idx in range(t_count):
        if not presence[idx]:
            invalid_presence[idx] = True
            continue
        stable_src = xyz[idx, STABLE_IDXS, :]
        if not np.isfinite(stable_src).all():
            invalid_stable[idx] = True
            continue
        try:
            rotation, translation, residual = estimate_rigid_transform_3d(stable_src, stable_ref)
        except ValueError:
            invalid_transform[idx] = True
            continue

        frame = xyz[idx]
        finite_mask = np.isfinite(frame).all(axis=1)
        transformed = np.full_like(frame, np.nan, dtype=np.float32)
        if finite_mask.any():
            transformed_valid = (rotation @ frame[finite_mask].T).T + translation
            transformed[finite_mask] = transformed_valid.astype(np.float32)

        aligned[idx] = transformed
        rotations[idx] = rotation
        translations[idx] = translation
        residuals[idx] = np.float32(residual)
        valid_mask[idx] = True

    return {
        "aligned_xyz": aligned,
        "R": rotations,
        "t": translations,
        "residuals": residuals,
        "valid_mask": valid_mask,
        "invalid_presence": invalid_presence,
        "invalid_stable": invalid_stable,
        "invalid_transform": invalid_transform,
    }


def _select_valid_frame_index(target_idx: int, valid_mask: np.ndarray) -> tuple[int, bool]:
    if valid_mask[target_idx]:
        return target_idx, False
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid aligned frames available for viewer generation")
    nearest_pos = int(np.argmin(np.abs(valid_indices - int(target_idx))))
    return int(valid_indices[nearest_pos]), True


def _compute_center_from_neutral(neutral_xyz: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(neutral_xyz).all(axis=1)
    if not finite_mask.any():
        raise ValueError("neutral aligned frame has no finite landmarks")
    center = np.median(neutral_xyz[finite_mask], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("Failed to compute neutral center for 3D viewer transform")
    return center.astype(np.float32)


def _apply_threejs_point_transform(
    points_xyz: np.ndarray,
    *,
    center: np.ndarray,
    flip_y: bool,
    flip_z: bool,
    z_scale: float,
) -> np.ndarray:
    transformed = np.asarray(points_xyz, dtype=np.float32).copy()
    transformed = transformed - center[None, :]
    transformed[:, 2] *= float(z_scale)
    if flip_y:
        transformed[:, 1] *= -1.0
    if flip_z:
        transformed[:, 2] *= -1.0
    return transformed


def _apply_threejs_vector_transform(
    vectors_xyz: np.ndarray,
    *,
    flip_y: bool,
    flip_z: bool,
    z_scale: float,
) -> np.ndarray:
    transformed = np.asarray(vectors_xyz, dtype=np.float32).copy()
    transformed[:, 2] *= float(z_scale)
    if flip_y:
        transformed[:, 1] *= -1.0
    if flip_z:
        transformed[:, 2] *= -1.0
    return transformed


def _build_region_assignment(neutral_xyz: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    n_points = neutral_xyz.shape[0]
    region_ids = np.full((n_points,), -1, dtype=np.int32)
    region_names = [name for name, _ in REGION_SEEDS]
    centroids = np.full((len(REGION_SEEDS), 3), np.nan, dtype=np.float32)

    for region_idx, (_, seeds) in enumerate(REGION_SEEDS):
        for landmark_idx in seeds:
            if 0 <= int(landmark_idx) < n_points and region_ids[int(landmark_idx)] < 0:
                region_ids[int(landmark_idx)] = np.int32(region_idx)

    for region_idx, (_, seeds) in enumerate(REGION_SEEDS):
        unique_seeds = sorted({int(idx) for idx in seeds if 0 <= int(idx) < n_points})
        if not unique_seeds:
            continue
        points = neutral_xyz[unique_seeds]
        finite_mask = np.isfinite(points).all(axis=1)
        if finite_mask.any():
            centroids[region_idx] = np.mean(points[finite_mask], axis=0).astype(np.float32)

    global_finite = np.isfinite(neutral_xyz).all(axis=1)
    global_center = (
        np.median(neutral_xyz[global_finite], axis=0).astype(np.float32)
        if global_finite.any()
        else np.zeros((3,), dtype=np.float32)
    )
    for region_idx in range(centroids.shape[0]):
        if not np.isfinite(centroids[region_idx]).all():
            centroids[region_idx] = global_center

    unassigned = np.flatnonzero(region_ids < 0)
    for idx in unassigned:
        point = neutral_xyz[idx]
        if not np.isfinite(point).all():
            region_ids[idx] = np.int32(0)
            continue
        distances = np.linalg.norm(centroids - point[None, :], axis=1)
        region_ids[idx] = np.int32(int(np.argmin(distances)))

    return region_ids.astype(np.int32), region_names, centroids.astype(np.float32)


def _safe_quantile_scale(values: np.ndarray, quantile: float = 0.95) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    scale = float(np.quantile(finite, quantile))
    if scale <= EPSILON:
        scale = float(np.max(finite))
    if scale <= EPSILON:
        scale = 1.0
    return scale


def _normalize_magnitude(
    *,
    magnitude: np.ndarray,
    region_ids: np.ndarray,
    n_regions: int,
) -> dict[str, Any]:
    global_scale = _safe_quantile_scale(magnitude)
    mag_global = np.zeros(magnitude.shape, dtype=np.float32)
    finite_global = np.isfinite(magnitude)
    mag_global[finite_global] = np.clip(magnitude[finite_global] / global_scale, 0.0, 1.0).astype(
        np.float32
    )

    mag_region = np.zeros(magnitude.shape, dtype=np.float32)
    region_scales = np.full((n_regions,), global_scale, dtype=np.float32)
    for region_idx in range(n_regions):
        mask = (region_ids == region_idx) & np.isfinite(magnitude)
        if not mask.any():
            continue
        region_scale = _safe_quantile_scale(magnitude[mask])
        region_scales[region_idx] = np.float32(region_scale)
        mag_region[mask] = np.clip(magnitude[mask] / region_scale, 0.0, 1.0).astype(np.float32)

    return {
        "mag_global_norm": mag_global.astype(np.float32),
        "mag_region_norm": mag_region.astype(np.float32),
        "global_scale": float(global_scale),
        "region_scales": region_scales.astype(np.float32),
    }


def _extract_facemesh_edges() -> list[list[int]]:
    raw_edges: Any
    try:
        from mediapipe.python.solutions import face_mesh_connections

        raw_edges = face_mesh_connections.FACEMESH_TESSELATION
    except Exception:
        try:
            import mediapipe as mp

            raw_edges = mp.solutions.face_mesh.FACEMESH_TESSELATION  # type: ignore[attr-defined]
        except Exception:
            try:
                from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_lib

                raw_edges = face_landmarker_lib.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            except Exception as second_error:
                raise ImportError(
                    "Unable to import MediaPipe FACEMESH_TESSELATION-compatible edges."
                ) from second_error

    edges: set[tuple[int, int]] = set()
    for pair in raw_edges:
        if isinstance(pair, tuple) or isinstance(pair, list):
            a, b = int(pair[0]), int(pair[1])
        elif hasattr(pair, "start") and hasattr(pair, "end"):
            a, b = int(pair.start), int(pair.end)
        else:
            continue
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        edges.add((lo, hi))

    if not edges:
        raise RuntimeError("FACEMESH_TESSELATION is empty; cannot build wireframe edges")

    return [[a, b] for a, b in sorted(edges)]


def _extract_facemesh_faces(edges: list[list[int]]) -> list[list[int]]:
    edge_set: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = {}

    for pair in edges:
        if len(pair) != 2:
            continue
        a = int(pair[0])
        b = int(pair[1])
        if a < 0 or b < 0 or a >= LANDMARK_COUNT or b >= LANDMARK_COUNT or a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        edge_set.add((lo, hi))
        adjacency.setdefault(lo, set()).add(hi)
        adjacency.setdefault(hi, set()).add(lo)

    face_set: set[tuple[int, int, int]] = set()
    for j, neighbors in adjacency.items():
        sorted_neighbors = sorted(neighbors)
        neighbor_count = len(sorted_neighbors)
        for i_pos in range(neighbor_count):
            i = sorted_neighbors[i_pos]
            for k_pos in range(i_pos + 1, neighbor_count):
                k = sorted_neighbors[k_pos]
                lo, hi = (i, k) if i < k else (k, i)
                if (lo, hi) not in edge_set:
                    continue
                tri = tuple(sorted((int(i), int(j), int(k))))
                face_set.add(tri)

    faces = [[a, b, c] for a, b, c in sorted(face_set)]
    if not faces:
        raise RuntimeError("Failed to derive FACEMESH faces from tessellation edges")
    return faces


def _build_uv_coords_from_normalized_xy(normalized_xy: np.ndarray) -> list[list[float]]:
    xy = np.asarray(normalized_xy, dtype=np.float32)
    if xy.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"normalized_xy must have shape ({LANDMARK_COUNT}, 2), got {xy.shape}")

    uv = np.zeros((LANDMARK_COUNT, 2), dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1)
    uv[valid, 0] = xy[valid, 0]
    uv[valid, 1] = 1.0 - xy[valid, 1]
    return uv.astype(np.float32).tolist()


def _safe_norm_xy_for_payload(row_norm_xy: np.ndarray, *, row_presence: bool) -> list[list[float]]:
    if not row_presence:
        return [[0.0, 0.0] for _ in range(LANDMARK_COUNT)]
    xy = np.asarray(row_norm_xy, dtype=np.float32)
    if xy.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"row_norm_xy must have shape ({LANDMARK_COUNT}, 2), got {xy.shape}")
    out = np.zeros((LANDMARK_COUNT, 2), dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1)
    out[valid] = xy[valid]
    return out.astype(np.float32).tolist()


def _segment_row_indices(
    *,
    neutral_idx: int,
    peak_idx: int,
    max_frames: int = MAX_2D_FRAMES,
) -> np.ndarray:
    step = 1 if peak_idx >= neutral_idx else -1
    rows = np.arange(neutral_idx, peak_idx + step, step, dtype=np.int64)
    if rows.size <= max_frames:
        return rows
    sample_positions = np.linspace(0, rows.size - 1, num=max_frames)
    sample_positions = np.round(sample_positions).astype(np.int64)
    sample_positions = np.clip(sample_positions, 0, rows.size - 1)
    dedup_positions = np.unique(sample_positions)
    sampled_rows = rows[dedup_positions]
    if sampled_rows[0] != neutral_idx:
        sampled_rows[0] = neutral_idx
    if sampled_rows[-1] != peak_idx:
        sampled_rows[-1] = peak_idx
    return sampled_rows


def _build_all_norm_xy_payload(
    *,
    landmarks_xyz: np.ndarray,
    presence: np.ndarray,
    row_indices: np.ndarray,
) -> list[list[list[float]]]:
    output: list[list[list[float]]] = []
    for row_idx in row_indices.astype(np.int64).tolist():
        row_xy = np.asarray(landmarks_xyz[row_idx, :, :2], dtype=np.float32)
        output.append(
            _safe_norm_xy_for_payload(
                row_xy,
                row_presence=bool(presence[row_idx]),
            )
        )
    return output


def _encode_bgr_to_jpeg_base64(
    frame_bgr: np.ndarray,
    *,
    quality: int = FRAME_JPEG_QUALITY,
) -> str | None:
    try:
        import cv2
    except Exception:
        return None
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _black_frame_base64(width: int, height: int) -> str | None:
    safe_w = max(1, int(width))
    safe_h = max(1, int(height))
    black = np.zeros((safe_h, safe_w, 3), dtype=np.uint8)
    return _encode_bgr_to_jpeg_base64(black)


def _extract_video_frames_base64(
    *,
    video_path: str | Path,
    frame_indices: np.ndarray,
    width: int,
    height: int,
) -> list[str] | None:
    video = Path(video_path)
    if (not video.exists()) or (not video.is_file()):
        return None

    try:
        import cv2
    except Exception:
        return None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return None

    black_fallback = _black_frame_base64(width, height)
    frame_images: list[str] = []
    last_success: str | None = None

    try:
        for frame_idx in frame_indices.astype(np.int64).tolist():
            frame_base64: str | None = None
            if frame_idx >= 0 and capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx)):
                ok, frame_bgr = capture.read()
                if ok and frame_bgr is not None:
                    frame_base64 = _encode_bgr_to_jpeg_base64(frame_bgr, quality=FRAME_JPEG_QUALITY)
            if frame_base64 is None:
                frame_base64 = last_success if last_success is not None else black_fallback
            if frame_base64 is None:
                return None
            frame_images.append(frame_base64)
            last_success = frame_base64
    finally:
        capture.release()

    return frame_images


def _extract_neutral_frame_base64(video_path: str | Path, frame_idx: int) -> str | None:
    video = Path(video_path)
    if not video.exists() or not video.is_file():
        return None
    if frame_idx < 0:
        return None

    try:
        import cv2
    except Exception:
        return None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return None

    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx)):
            return None
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            return None
        encoded_ok, encoded = cv2.imencode(".png", frame_bgr)
        if not encoded_ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    finally:
        capture.release()


def _compute_interocular_distance(neutral_xyz: np.ndarray) -> float:
    a = neutral_xyz[33]
    b = neutral_xyz[263]
    if np.isfinite(a).all() and np.isfinite(b).all():
        dist = float(np.linalg.norm(a - b))
        if dist > EPSILON:
            return dist
    finite = neutral_xyz[np.isfinite(neutral_xyz).all(axis=1)]
    if finite.shape[0] < 2:
        return 1.0
    spread = np.max(finite, axis=0) - np.min(finite, axis=0)
    fallback = float(np.linalg.norm(spread))
    return fallback if fallback > EPSILON else 1.0


def _build_quality_payload(
    *,
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    neutral_idx_requested: int,
    peak_idx_requested: int,
    neutral_idx_used: int,
    peak_idx_used: int,
    invalid_presence: np.ndarray,
    invalid_stable: np.ndarray,
    invalid_transform: np.ndarray,
) -> dict[str, Any]:
    finite_residuals = residuals[np.isfinite(residuals)]
    return {
        "total_frames": int(valid_mask.shape[0]),
        "valid_frames": int(np.sum(valid_mask)),
        "invalid_presence_frames": int(np.sum(invalid_presence)),
        "invalid_stable_frames": int(np.sum(invalid_stable)),
        "invalid_transform_frames": int(np.sum(invalid_transform)),
        "neutral_idx_requested": int(neutral_idx_requested),
        "peak_idx_requested": int(peak_idx_requested),
        "neutral_idx_used": int(neutral_idx_used),
        "peak_idx_used": int(peak_idx_used),
        "neutral_replaced": bool(neutral_idx_requested != neutral_idx_used),
        "peak_replaced": bool(peak_idx_requested != peak_idx_used),
        "residual_mean": float(np.mean(finite_residuals)) if finite_residuals.size else None,
        "residual_median": float(np.median(finite_residuals)) if finite_residuals.size else None,
        "residual_max": float(np.max(finite_residuals)) if finite_residuals.size else None,
        "residual_neutral": (
            float(residuals[neutral_idx_used]) if np.isfinite(residuals[neutral_idx_used]) else None
        ),
        "residual_peak": float(residuals[peak_idx_used])
        if np.isfinite(residuals[peak_idx_used])
        else None,
    }


def _build_viewer_payload(
    *,
    aligned_xyz: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_ms: np.ndarray,
    neutral_idx: int,
    peak_idx: int,
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    invalid_presence: np.ndarray,
    invalid_stable: np.ndarray,
    invalid_transform: np.ndarray,
    flip_y: bool,
    flip_z: bool,
    z_scale: float,
) -> dict[str, Any]:
    neutral_idx_used, neutral_replaced = _select_valid_frame_index(neutral_idx, valid_mask)
    peak_idx_used, peak_replaced = _select_valid_frame_index(peak_idx, valid_mask)

    neutral_xyz = aligned_xyz[neutral_idx_used]
    peak_xyz = aligned_xyz[peak_idx_used]
    if not np.isfinite(neutral_xyz).all(axis=1).any():
        raise ValueError("neutral frame does not contain finite aligned landmarks")
    if not np.isfinite(peak_xyz).all(axis=1).any():
        raise ValueError("peak frame does not contain finite aligned landmarks")

    raw_disp = peak_xyz - neutral_xyz
    raw_mag = np.linalg.norm(raw_disp, axis=1)
    invalid_mag = ~np.isfinite(raw_disp).all(axis=1)
    raw_mag[invalid_mag] = np.nan

    interocular = _compute_interocular_distance(neutral_xyz)
    magnitude = raw_mag / float(interocular) if interocular > EPSILON else raw_mag

    region_ids, region_names, region_centroids = _build_region_assignment(neutral_xyz)
    normalization = _normalize_magnitude(
        magnitude=magnitude,
        region_ids=region_ids,
        n_regions=len(region_names),
    )

    center = _compute_center_from_neutral(neutral_xyz)
    base_xyz = _apply_threejs_point_transform(
        neutral_xyz,
        center=center,
        flip_y=flip_y,
        flip_z=flip_z,
        z_scale=z_scale,
    )
    peak_xyz_three = _apply_threejs_point_transform(
        peak_xyz,
        center=center,
        flip_y=flip_y,
        flip_z=flip_z,
        z_scale=z_scale,
    )
    disp_xyz_three = _apply_threejs_vector_transform(
        raw_disp,
        flip_y=flip_y,
        flip_z=flip_z,
        z_scale=z_scale,
    )

    unit_dir = np.zeros_like(disp_xyz_three, dtype=np.float32)
    disp_norm = np.linalg.norm(disp_xyz_three, axis=1)
    valid_dir = (
        np.isfinite(disp_xyz_three).all(axis=1) & np.isfinite(disp_norm) & (disp_norm > EPSILON)
    )
    unit_dir[valid_dir] = (disp_xyz_three[valid_dir] / disp_norm[valid_dir, None]).astype(
        np.float32
    )

    cone_length = float(
        np.clip(
            DEFAULT_CONE_INTEROCULAR_SCALE * float(interocular),
            MIN_CONE_LENGTH_PX,
            MAX_CONE_LENGTH_PX,
        )
    )

    quality = _build_quality_payload(
        residuals=residuals,
        valid_mask=valid_mask,
        neutral_idx_requested=neutral_idx,
        peak_idx_requested=peak_idx,
        neutral_idx_used=neutral_idx_used,
        peak_idx_used=peak_idx_used,
        invalid_presence=invalid_presence,
        invalid_stable=invalid_stable,
        invalid_transform=invalid_transform,
    )
    quality["neutral_replaced"] = bool(neutral_replaced)
    quality["peak_replaced"] = bool(peak_replaced)

    finite_base = np.isfinite(base_xyz).all(axis=1)
    bbox_points = base_xyz[finite_base]
    if bbox_points.shape[0] >= 2:
        extent = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
        radius_hint = float(max(np.linalg.norm(extent), 1.0))
    else:
        radius_hint = 400.0

    return {
        "frame_indices": frame_indices.astype(np.int64).tolist(),
        "timestamps_ms": timestamps_ms.astype(np.int64).tolist(),
        "neutral_idx": int(neutral_idx_used),
        "peak_idx": int(peak_idx_used),
        "base_xyz": np.nan_to_num(base_xyz, nan=0.0).astype(np.float32).tolist(),
        "peak_xyz": np.nan_to_num(peak_xyz_three, nan=0.0).astype(np.float32).tolist(),
        "disp_xyz": np.nan_to_num(disp_xyz_three, nan=0.0).astype(np.float32).tolist(),
        "unit_dir_xyz": np.nan_to_num(unit_dir, nan=0.0).astype(np.float32).tolist(),
        "magnitude": np.nan_to_num(magnitude, nan=0.0).astype(np.float32).tolist(),
        "mag_global_norm": normalization["mag_global_norm"].astype(np.float32).tolist(),
        "mag_region_norm": normalization["mag_region_norm"].astype(np.float32).tolist(),
        "region_ids": region_ids.astype(np.int32).tolist(),
        "region_names": region_names,
        "region_centroids": np.nan_to_num(region_centroids, nan=0.0).astype(np.float32).tolist(),
        "region_scales": normalization["region_scales"].astype(np.float32).tolist(),
        "global_scale": float(normalization["global_scale"]),
        "edges": _extract_facemesh_edges(),
        "cone_length_px": float(cone_length),
        "radius_hint": radius_hint,
        "quality": quality,
        "params": {
            "flip_y": bool(flip_y),
            "flip_z": bool(flip_z),
            "z_scale": float(z_scale),
            "normalize_default": "region",
        },
    }


def _render_motion_viewer_html(viewer_payload: dict[str, Any]) -> str:
    data_json = json.dumps(viewer_payload, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FaceAnalyze2 Motion Viewer</title>
  <style>
    :root{--bg:#0b1020;--panel:#111a31;--line:#2a3557;--text:#e8eeff;--muted:#93a1c8}
    html,body,#root{margin:0;width:100%;height:100%;background:var(--bg);color:var(--text)}
    body{font-family:"Segoe UI","Noto Sans KR",sans-serif}
    .app{display:grid;grid-template-columns:340px 1fr;height:100%}
    .panel{border-right:1px solid var(--line);padding:12px;overflow:auto;background:linear-gradient(180deg,#151e36 0%,#10182d 100%)}
    .viewer{position:relative;height:100%}
    .mount{width:100%;height:100%}
    .row{margin:8px 0}
    .row label{display:flex;align-items:center;gap:8px;font-size:13px}
    .row input[type=range],.row select,.row button{width:100%}
    .row select,.row button{padding:7px;border-radius:8px;border:1px solid var(--line);background:#1a2442;color:var(--text)}
    .checks{display:grid;grid-template-columns:1fr 1fr;gap:6px;border:1px solid var(--line);border-radius:10px;padding:8px}
    .checks label{font-size:12px;color:var(--muted)}
    .meta{border:1px solid var(--line);border-radius:10px;padding:8px;font-size:12px;line-height:1.45;color:var(--muted)}
    .legend{height:8px;border-radius:6px;background:linear-gradient(90deg,#2ecc71 0%,#f1c40f 50%,#e74c3c 100%);border:1px solid var(--line)}
    .tt{position:absolute;display:none;pointer-events:none;z-index:10;background:rgba(7,10,20,.94);border:1px solid #334567;border-radius:8px;padding:6px 8px;font-size:12px}
    .muted{font-size:12px;color:var(--muted)}
    @media (max-width:980px){.app{grid-template-columns:1fr;grid-template-rows:auto 1fr}.panel{border-right:none;border-bottom:1px solid var(--line);max-height:45vh}}
  </style>
</head>
<body>
  <div id="root"></div>
  <script>window.MOTION_VIEWER_DATA=__MOTION_VIEWER_DATA__;</script>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone@7.24.7/babel.min.js"></script>
  <script type="text/babel">
    const data = window.MOTION_VIEWER_DATA;
    class SimpleOrbitControls{
      constructor(camera,dom){this.camera=camera;this.dom=dom;this.target=new THREE.Vector3(0,0,0);this.state="none";this.enabled=true;this.enableRotate=true;this.enablePan=true;this.enableZoom=true;this.sph=new THREE.Spherical();const off=new THREE.Vector3().copy(camera.position).sub(this.target);this.sph.setFromVector3(off);this.lastX=0;this.lastY=0;this.onDown=this.onDown.bind(this);this.onMove=this.onMove.bind(this);this.onUp=this.onUp.bind(this);this.onWheel=this.onWheel.bind(this);this.onCtx=e=>e.preventDefault();dom.addEventListener("mousedown",this.onDown);window.addEventListener("mousemove",this.onMove);window.addEventListener("mouseup",this.onUp);dom.addEventListener("wheel",this.onWheel,{passive:false});dom.addEventListener("contextmenu",this.onCtx)}
      onDown(e){if(!this.enabled)return;this.lastX=e.clientX;this.lastY=e.clientY;if(e.button===2){this.state=this.enablePan?"pan":"none";return}this.state=this.enableRotate?"rotate":"none"}
      onMove(e){if(!this.enabled||this.state==="none")return;const dx=e.clientX-this.lastX;const dy=e.clientY-this.lastY;this.lastX=e.clientX;this.lastY=e.clientY;if(this.state==="rotate"){this.sph.theta-=dx*0.005;this.sph.phi-=dy*0.005;this.sph.phi=Math.max(0.001,Math.min(Math.PI-0.001,this.sph.phi));return}const rect=this.dom.getBoundingClientRect();let sc;if(this.camera.isPerspectiveCamera){const d=this.camera.position.distanceTo(this.target);const f=this.camera.fov*Math.PI/180;sc=2*Math.tan(f/2)*d;}else{sc=(this.camera.right-this.camera.left);}const panX=(-dx/Math.max(rect.width,1))*sc;const panY=(dy/Math.max(rect.height,1))*sc;const fw=new THREE.Vector3();this.camera.getWorldDirection(fw);const right=new THREE.Vector3().crossVectors(fw,this.camera.up).normalize();const up=new THREE.Vector3().copy(this.camera.up).normalize();const pan=new THREE.Vector3();pan.addScaledVector(right,panX);pan.addScaledVector(up,panY);this.target.add(pan);this.camera.position.add(pan)}
      onUp(){this.state="none"}
      onWheel(e){if(!this.enabled||!this.enableZoom)return;e.preventDefault();const sign=Math.sign(e.deltaY);if(this.camera.isPerspectiveCamera){this.sph.radius=Math.max(5,this.sph.radius*(1+sign*0.1));return}const factor=1+sign*0.1;this.camera.left*=factor;this.camera.right*=factor;this.camera.top*=factor;this.camera.bottom*=factor;this.camera.updateProjectionMatrix();}
      update(){const off=new THREE.Vector3().setFromSpherical(this.sph);this.camera.position.copy(this.target).add(off);this.camera.lookAt(this.target)}
      dispose(){this.dom.removeEventListener("mousedown",this.onDown);window.removeEventListener("mousemove",this.onMove);window.removeEventListener("mouseup",this.onUp);this.dom.removeEventListener("wheel",this.onWheel);this.dom.removeEventListener("contextmenu",this.onCtx)}
    }
    const clamp01=v=>Math.max(0,Math.min(1,v));
    const colorFromValue=v=>{const t=clamp01(v);const r=t,g=1-t,b=.12;return{r,g,b,hex:(Math.round(r*255)<<16)|(Math.round(g*255)<<8)|Math.round(b*255)}};
    const friendlyRegion=name=>name.replaceAll("_"," ");
    function App(){
      const mountRef=React.useRef(null),canvas2dRef=React.useRef(null),tipRef=React.useRef(null),viewerRef=React.useRef(null);
      const images2dRef=React.useRef([]);
      const twoDRef=React.useRef({points:[],mag:[]});
      const hasTexture=Boolean(data.neutral_image_base64)&&Array.isArray(data.uv_coords)&&data.uv_coords.length===data.base_xyz.length&&Array.isArray(data.faces)&&data.faces.length>0;
      const has2d=Array.isArray(data.frame_images)&&Array.isArray(data.all_norm_xy)&&data.frame_images.length>0&&data.all_norm_xy.length===data.frame_images.length;
      const initialRegions=React.useMemo(()=>{const m={};data.region_names.forEach(n=>{m[n]=true});return m},[]);
      const [t,setT]=React.useState(1.0);
      const [showWire,setShowWire]=React.useState(true);
      const [showCones,setShowCones]=React.useState(true);
      const [normalizeMode,setNormalizeMode]=React.useState(data.params.normalize_default||"region");
      const [renderMode,setRenderMode]=React.useState(has2d?"2d":"points");
      const [showTextureOverlay,setShowTextureOverlay]=React.useState(true);
      const [showBackground2d,setShowBackground2d]=React.useState(true);
      const [coneScale,setConeScale]=React.useState(2.5);
      const [pointSize,setPointSize]=React.useState(2.8);
      const [flattenVectors,setFlattenVectors]=React.useState(true);
      const [regionEnabled,setRegionEnabled]=React.useState(initialRegions);
      const [isPlaying,setIsPlaying]=React.useState(false);
      const [playSpeed,setPlaySpeed]=React.useState(1.0);
      const [images2dReady,setImages2dReady]=React.useState(false);
      const settingsRef=React.useRef({t,showWire,showCones,normalizeMode,renderMode,showTextureOverlay,showBackground2d,coneScale,pointSize,flattenVectors,regionEnabled});
      React.useEffect(()=>{if(!has2d&&renderMode==="2d"){setRenderMode("points")}},[has2d,renderMode]);
      React.useEffect(()=>{if(!hasTexture&&renderMode==="texture"){setRenderMode(has2d?"2d":"points")}},[hasTexture,has2d,renderMode]);
      React.useEffect(()=>{settingsRef.current={t,showWire,showCones,normalizeMode,renderMode,showTextureOverlay,showBackground2d,coneScale,pointSize,flattenVectors,regionEnabled};if(viewerRef.current?.updateScene)viewerRef.current.updateScene()},[t,showWire,showCones,normalizeMode,renderMode,showTextureOverlay,showBackground2d,coneScale,pointSize,flattenVectors,regionEnabled]);
      React.useEffect(()=>{if(!isPlaying)return;const interval=setInterval(()=>{setT(prev=>{const next=prev+0.01*playSpeed;return next>1?0:next;});},30);return()=>clearInterval(interval);},[isPlaying,playSpeed]);
      React.useEffect(()=>{
        if(!has2d){images2dRef.current=[];setImages2dReady(false);return;}
        let cancelled=false;
        const frames=data.frame_images||[];
        Promise.all(frames.map((b64)=>new Promise((resolve)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>resolve(null);img.src=`data:image/jpeg;base64,${b64}`;}))).then((images)=>{
          if(cancelled)return;
          images2dRef.current=images;
          setImages2dReady(true);
        });
        return()=>{cancelled=true;};
      },[has2d]);
      React.useEffect(()=>{
        if(!mountRef.current)return;
        const container=mountRef.current;
        const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
        renderer.setSize(container.clientWidth,container.clientHeight);
        container.appendChild(renderer.domElement);

        const scene=new THREE.Scene();
        scene.background=new THREE.Color(0x0b1020);
        const perspectiveCamera=new THREE.PerspectiveCamera(42,container.clientWidth/Math.max(container.clientHeight,1),0.1,10000);
        const orthoCamera=new THREE.OrthographicCamera(-320,320,240,-240,-5000,5000);
        const radius=Math.max(data.radius_hint||400,200);
        perspectiveCamera.position.set(radius*.8,radius*.45,radius*1.1);
        const controlsPerspective=new SimpleOrbitControls(perspectiveCamera,renderer.domElement);
        controlsPerspective.target.set(0,0,0);
        controlsPerspective.update();
        orthoCamera.position.set(0,0,1000);
        const controlsOrtho=new SimpleOrbitControls(orthoCamera,renderer.domElement);
        controlsOrtho.target.set(0,0,0);
        controlsOrtho.enableRotate=false;
        controlsOrtho.update();

        scene.add(new THREE.HemisphereLight(0xffffff,0x233554,.78));
        const dl=new THREE.DirectionalLight(0xffffff,.56);
        dl.position.set(1,1,1);
        scene.add(dl);
        const setOverlayLayer=(root,order)=>{
          if(!root){return;}
          root.renderOrder=order;
          root.traverse((node)=>{
            node.renderOrder=order;
            const mats=Array.isArray(node.material)?node.material:[node.material];
            for(let mi=0;mi<mats.length;mi+=1){
              const mat=mats[mi];
              if(!mat){continue;}
              mat.depthTest=false;
              mat.depthWrite=false;
              mat.transparent=true;
              mat.needsUpdate=true;
            }
          });
        };

        const base=data.base_xyz,disp=data.disp_xyz,dirs=data.unit_dir_xyz,magRegion=data.mag_region_norm,magGlobal=data.mag_global_norm,regionIds=data.region_ids,regionNames=data.region_names,edges=data.edges,pointCount=base.length;
        const baseNorm=data.base_norm_xy||[];
        const peakNorm=data.peak_norm_xy||[];
        let frameW=1024;
        let frameH=1024;
        const quantile95=(arr)=>{const vals=arr.filter(v=>Number.isFinite(v));if(!vals.length)return 1.0;vals.sort((a,b)=>a-b);const p=Math.min(vals.length-1,Math.max(0,Math.floor(0.95*(vals.length-1))));const q=vals[p];if(q>1e-6)return q;const mx=Math.max(...vals);return mx>1e-6?mx:1.0;};
        const compute2dMagnitudes=(w,h)=>{
          const raw=new Array(pointCount).fill(0);
          for(let i=0;i<pointCount;i+=1){
            const b=baseNorm[i]||[0,0],p=peakNorm[i]||[0,0];
            const dx=(p[0]-b[0])*w,dy=(p[1]-b[1])*h;
            const m=Math.sqrt(dx*dx+dy*dy);
            raw[i]=Number.isFinite(m)?m:0;
          }
          const gScale=quantile95(raw);
          const global=raw.map(v=>Math.max(0,Math.min(1,v/gScale)));
          const regionScale=new Array(regionNames.length).fill(gScale);
          const region=new Array(pointCount).fill(0);
          for(let r=0;r<regionNames.length;r+=1){
            const vals=[];for(let i=0;i<pointCount;i+=1){if(regionIds[i]===r&&Number.isFinite(raw[i]))vals.push(raw[i]);}
            const rs=quantile95(vals);regionScale[r]=rs;
            for(let i=0;i<pointCount;i+=1){if(regionIds[i]===r){region[i]=Math.max(0,Math.min(1,raw[i]/rs));}}
          }
          return {raw,global,region,regionScale};
        };
        let mag2d=compute2dMagnitudes(frameW,frameH);
        const pointsGeom=new THREE.BufferGeometry();
        const pointsMat=new THREE.PointsMaterial({size:2.8,vertexColors:true,transparent:true,opacity:.96,depthWrite:false});
        const pointsMesh=new THREE.Points(pointsGeom,pointsMat);
        setOverlayLayer(pointsMesh,20);
        scene.add(pointsMesh);

        const wireGeom=new THREE.BufferGeometry();
        const wireMat=new THREE.LineBasicMaterial({color:0x9ba7c6,transparent:true,opacity:.35});
        const wireMesh=new THREE.LineSegments(wireGeom,wireMat);
        setOverlayLayer(wireMesh,10);
        scene.add(wireMesh);

        const arrows=new THREE.Group();
        arrows.renderOrder=30;
        scene.add(arrows);

        const updateOrthoFrustum=()=>{
          const aspect=Math.max(container.clientWidth,1)/Math.max(container.clientHeight,1);
          const halfW=frameW*0.55;
          const halfH=frameH*0.55;
          if(aspect>=1){
            orthoCamera.left=-halfW*aspect;orthoCamera.right=halfW*aspect;orthoCamera.top=halfH;orthoCamera.bottom=-halfH;
          }else{
            orthoCamera.left=-halfW;orthoCamera.right=halfW;orthoCamera.top=halfH/aspect;orthoCamera.bottom=-halfH/aspect;
          }
          orthoCamera.updateProjectionMatrix();
        };

        let textureMesh=null,overlayMesh=null,textureGeometry=null,texturePositionAttr=null,overlayColorAttr=null,bgPlane=null,bgPlaneMat=null;
        if(hasTexture){
          const uvFlat=[];for(let i=0;i<data.uv_coords.length;i+=1){uvFlat.push(data.uv_coords[i][0],data.uv_coords[i][1]);}
          const idxFlat=[];for(let i=0;i<data.faces.length;i+=1){idxFlat.push(data.faces[i][0],data.faces[i][1],data.faces[i][2]);}
          textureGeometry=new THREE.BufferGeometry();
          texturePositionAttr=new THREE.Float32BufferAttribute(new Float32Array(pointCount*3),3);
          overlayColorAttr=new THREE.Float32BufferAttribute(new Float32Array(pointCount*3),3);
          textureGeometry.setAttribute("position",texturePositionAttr);
          textureGeometry.setAttribute("uv",new THREE.Float32BufferAttribute(uvFlat,2));
          textureGeometry.setAttribute("color",overlayColorAttr);
          textureGeometry.setIndex(idxFlat);
          const tex=new THREE.TextureLoader().load(`data:image/png;base64,${data.neutral_image_base64}`,()=>{if(tex.image){frameW=tex.image.width||frameW;frameH=tex.image.height||frameH;mag2d=compute2dMagnitudes(frameW,frameH);updateOrthoFrustum();if(viewerRef.current?.updateScene)viewerRef.current.updateScene();}});
          tex.minFilter=THREE.LinearFilter;tex.magFilter=THREE.LinearFilter;
          const textureMat=new THREE.MeshBasicMaterial({map:tex,side:THREE.DoubleSide,transparent:true,opacity:1.0});
          const overlayMat=new THREE.MeshBasicMaterial({vertexColors:true,transparent:true,opacity:.4,depthWrite:false,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1});
          textureMesh=new THREE.Mesh(textureGeometry,textureMat);
          overlayMesh=new THREE.Mesh(textureGeometry,overlayMat);
          textureMesh.renderOrder=0;
          if(textureMesh.material){
            textureMesh.material.polygonOffset=true;
            textureMesh.material.polygonOffsetFactor=1;
            textureMesh.material.polygonOffsetUnits=1;
          }
          setOverlayLayer(overlayMesh,5);
          textureMesh.visible=false;overlayMesh.visible=false;
          bgPlaneMat=new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:1.0,depthWrite:false,side:THREE.DoubleSide});
          bgPlane=new THREE.Mesh(new THREE.PlaneGeometry(1,1),bgPlaneMat);
          bgPlane.position.set(0,0,-2);
          bgPlane.visible=false;
          scene.add(bgPlane);
          scene.add(textureMesh);scene.add(overlayMesh);
        }
        updateOrthoFrustum();

        const activeCamera=()=>perspectiveCamera;
        const ray=new THREE.Raycaster();
        ray.params.Points.threshold=6;
        const mouse=new THREE.Vector2();
        const visiblePointLut=[];
        let currentCoords=new Array(pointCount).fill([0,0,0]);
        let currentMag=magRegion;

        const currentPoint=(i,k)=>[base[i][0]+k*disp[i][0],base[i][1]+k*disp[i][1],base[i][2]+k*disp[i][2]];
        const clearArrows=()=>{while(arrows.children.length)arrows.remove(arrows.children[0])};
        const updateScene=()=>{
          const s=settingsRef.current;
          const coords=new Array(pointCount);
          const visible=new Array(pointCount).fill(false);
          const use2d=false;
          const mag=use2d?(s.normalizeMode==="region"?mag2d.region:mag2d.global):(s.normalizeMode==="region"?magRegion:magGlobal);
          currentCoords=coords;currentMag=mag;

          for(let i=0;i<pointCount;i+=1){
            let p;
            if(use2d){
              const b=baseNorm[i]||[0,0],pk=peakNorm[i]||[0,0];
              const nx=((1.0-s.t)*b[0])+(s.t*pk[0]);
              const ny=((1.0-s.t)*b[1])+(s.t*pk[1]);
              p=[(nx-0.5)*frameW,(0.5-ny)*frameH,0.0];
            }else{
              p=currentPoint(i,s.t);
            }
            coords[i]=p;
            const finite=Number.isFinite(p[0])&&Number.isFinite(p[1])&&Number.isFinite(p[2]);
            if(!finite)continue;
            visible[i]=!!s.regionEnabled[regionNames[regionIds[i]]];
          }

          const pos=[];const col=[];visiblePointLut.length=0;
          for(let i=0;i<pointCount;i+=1){
            if(!visible[i])continue;
            const p=coords[i];pos.push(p[0],p[1],p[2]);
            const c=colorFromValue(s.t*mag[i]);col.push(c.r,c.g,c.b);visiblePointLut.push(i);
          }
          pointsGeom.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));
          pointsGeom.setAttribute("color",new THREE.Float32BufferAttribute(col,3));
          pointsGeom.computeBoundingSphere();
          pointsMat.size=s.pointSize;
          pointsMesh.visible=(s.renderMode==="points"||s.renderMode==="texture");

          const wPos=[];
          if(s.showWire){for(let i=0;i<edges.length;i+=1){const a=edges[i][0],b=edges[i][1];if(!visible[a]||!visible[b])continue;const pa=coords[a],pb=coords[b];wPos.push(pa[0],pa[1],pa[2],pb[0],pb[1],pb[2]);}}
          wireGeom.setAttribute("position",new THREE.Float32BufferAttribute(wPos,3));
          wireGeom.computeBoundingSphere();
          wireMesh.visible=s.showWire&&(s.renderMode==="points"||s.renderMode==="texture");

          clearArrows();
          if(s.showCones&&s.t>0){
            const len=data.cone_length_px*s.coneScale*s.t;
            for(let i=0;i<pointCount;i+=1){
              if(!visible[i])continue;
              let u=dirs[i];
              if(use2d){
                const b=baseNorm[i]||[0,0],pk=peakNorm[i]||[0,0];
                const dx=(pk[0]-b[0])*frameW;
                const dy=(b[1]-pk[1])*frameH;
                u=[dx,dy,0.0];
              }
              if(s.flattenVectors){u=[u[0],u[1],0.0];}
              if(!Number.isFinite(u[0])||!Number.isFinite(u[1])||!Number.isFinite(u[2]))continue;
              const magU=Math.sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
              if(magU<1e-6)continue;
              const d=new THREE.Vector3(u[0],u[1],u[2]).normalize();
              const o=coords[i];
              const color=colorFromValue(s.t*mag[i]).hex;
              const arrow=new THREE.ArrowHelper(d,new THREE.Vector3(o[0],o[1],o[2]),len,color,Math.max(len*.35,1.1),Math.max(len*.2,.75));
              setOverlayLayer(arrow,30);
              arrows.add(arrow);
            }
          }
          arrows.visible=s.showCones&&(s.renderMode==="points"||s.renderMode==="texture");

          if(hasTexture&&texturePositionAttr&&overlayColorAttr&&textureMesh&&overlayMesh){
            for(let i=0;i<pointCount;i+=1){
              const p=coords[i];
              const finite=Number.isFinite(p[0])&&Number.isFinite(p[1])&&Number.isFinite(p[2]);
              texturePositionAttr.setXYZ(i,finite?p[0]:0,finite?p[1]:0,finite?p[2]:0);
              const c=colorFromValue(s.t*mag[i]);
              overlayColorAttr.setXYZ(i,visible[i]?c.r:0,visible[i]?c.g:0,visible[i]?c.b:0);
            }
            texturePositionAttr.needsUpdate=true;
            overlayColorAttr.needsUpdate=true;
            textureGeometry.computeBoundingSphere();
            textureMesh.visible=s.renderMode==="texture";
            overlayMesh.visible=s.renderMode==="texture"&&s.showTextureOverlay;
            if(textureMesh.material){textureMesh.material.opacity=1.0;}
            if(bgPlane&&bgPlaneMat){
              bgPlane.visible=false;
              bgPlane.scale.set(frameW,frameH,1);
            }
          }
          if(s.renderMode!=="2d"){const cw=container.clientWidth,ch=container.clientHeight;if(cw>0&&ch>0){const sz=new THREE.Vector2();renderer.getSize(sz);if(Math.abs(sz.x-cw)>1||Math.abs(sz.y-ch)>1){renderer.setSize(cw,ch);perspectiveCamera.aspect=cw/Math.max(ch,1);}}perspectiveCamera.updateProjectionMatrix();controlsPerspective.update();renderer.render(scene,perspectiveCamera);}
        };

        const hideTip=()=>{if(tipRef.current)tipRef.current.style.display="none"};
        const showTip=(event,idx)=>{if(!tipRef.current)return;const rect=renderer.domElement.getBoundingClientRect();tipRef.current.style.display="block";tipRef.current.style.left=`${event.clientX-rect.left+14}px`;tipRef.current.style.top=`${event.clientY-rect.top+14}px`;tipRef.current.innerHTML=`<b>Landmark #${idx}</b><br/>Region: ${friendlyRegion(regionNames[regionIds[idx]])}<br/>Norm mag: ${(settingsRef.current.t*currentMag[idx]).toFixed(3)}`};
        const onMove=e=>{
          const rect=renderer.domElement.getBoundingClientRect();
          mouse.x=((e.clientX-rect.left)/Math.max(rect.width,1))*2-1;
          mouse.y=-((e.clientY-rect.top)/Math.max(rect.height,1))*2+1;
          ray.setFromCamera(mouse,activeCamera());
          const s=settingsRef.current;
          if(s.renderMode==="texture"&&hasTexture&&textureMesh){
            const hits=ray.intersectObject(textureMesh);
            if(!hits.length||!hits[0].face){hideTip();return}
            const hit=hits[0],cand=[hit.face.a,hit.face.b,hit.face.c];
            let best=cand[0],bestD=Number.POSITIVE_INFINITY;
            for(let i=0;i<cand.length;i+=1){const idx=cand[i];const p=currentCoords[idx];if(!p)continue;const d=Math.sqrt((p[0]-hit.point.x)**2+(p[1]-hit.point.y)**2+(p[2]-hit.point.z)**2);if(d<bestD){bestD=d;best=idx}}
            showTip(e,best);return;
          }
          const hits=ray.intersectObject(pointsMesh);
          if(!hits.length||hits[0].index==null){hideTip();return}
          const vIdx=hits[0].index;const lIdx=visiblePointLut[vIdx];
          if(lIdx==null){hideTip();return}
          showTip(e,lIdx);
        };

        const onResize=()=>{const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);perspectiveCamera.aspect=w/Math.max(h,1);perspectiveCamera.updateProjectionMatrix();updateOrthoFrustum()};
        renderer.domElement.addEventListener("mousemove",onMove);
        renderer.domElement.addEventListener("mouseleave",hideTip);
        window.addEventListener("resize",onResize);

        let raf=0;
        const loop=()=>{
          const is2d=settingsRef.current.renderMode==="2d";
          controlsPerspective.enabled=!is2d;
          controlsOrtho.enabled=false;
          controlsPerspective.update();
          if(!is2d){renderer.render(scene,activeCamera());}
          raf=requestAnimationFrame(loop);
        };
        viewerRef.current={updateScene,exportPng:()=>{
          const is2d=settingsRef.current.renderMode==="2d";
          if(is2d&&canvas2dRef.current){
            const url=canvas2dRef.current.toDataURL("image/png");
            const a=document.createElement("a");
            a.href=url;
            a.download="motion_viewer.png";
            a.click();
            return;
          }
          renderer.render(scene,activeCamera());
          const url=renderer.domElement.toDataURL("image/png");
          const a=document.createElement("a");
          a.href=url;
          a.download="motion_viewer.png";
          a.click();
        }};
        updateScene();
        loop();
        return()=>{cancelAnimationFrame(raf);renderer.domElement.removeEventListener("mousemove",onMove);renderer.domElement.removeEventListener("mouseleave",hideTip);window.removeEventListener("resize",onResize);controlsPerspective.dispose();controlsOrtho.dispose();renderer.dispose();if(container.contains(renderer.domElement))container.removeChild(renderer.domElement)};
      },[]);
      React.useEffect(()=>{
        if(renderMode!=="2d"||!has2d||!images2dReady||!canvas2dRef.current){return;}
        const canvas=canvas2dRef.current;
        const ctx=canvas.getContext("2d");
        if(!ctx){return;}

        const frames=data.frame_images||[];
        const xySeries=data.all_norm_xy||[];
        const images=images2dRef.current||[];
        const pointCount=(Array.isArray(data.base_xyz)?data.base_xyz.length:0);
        const n=Math.min(frames.length,xySeries.length,images.length);
        if(n<=0||pointCount<=0){return;}

        const firstImage=images.find((img)=>img&&img.width>0&&img.height>0);
        const width=firstImage?firstImage.width:640;
        const height=firstImage?firstImage.height:480;
        canvas.width=width;
        canvas.height=height;

        const regionIds=data.region_ids||[];
        const regionNames=data.region_names||[];
        const faces=data.faces||[];
        const edges=data.edges||[];
        const baseNorm=data.base_norm_xy||[];
        const peakNorm=data.peak_norm_xy||[];
        const isRegionEnabled=(landmarkIdx)=>{
          if(landmarkIdx<0||landmarkIdx>=pointCount){return false;}
          const regionIdx=regionIds[landmarkIdx];
          if(regionIdx==null||regionIdx<0||regionIdx>=regionNames.length){return false;}
          return !!regionEnabled[regionNames[regionIdx]];
        };
        const regionLabel=(landmarkIdx)=>{
          const regionIdx=regionIds[landmarkIdx];
          if(regionIdx==null||regionIdx<0||regionIdx>=regionNames.length){return "unknown";}
          return friendlyRegion(regionNames[regionIdx]);
        };

        const frameIdx=Math.max(0,Math.min(n-1,Math.round(t*(n-1))));
        const currentImage=images[frameIdx];
        const normXY=xySeries[frameIdx];
        const pointsPx=new Array(pointCount);
        for(let i=0;i<pointCount;i+=1){
          const p=normXY&&normXY[i]?normXY[i]:[0,0];
          pointsPx[i]=[p[0]*width,p[1]*height];
        }

        const rawDisp2d=new Array(pointCount).fill(0);
        for(let i=0;i<pointCount;i+=1){
          const b=baseNorm[i]||[0,0];
          const curr=normXY&&normXY[i]?normXY[i]:[0,0];
          const dx=(curr[0]-b[0])*width;
          const dy=(curr[1]-b[1])*height;
          const m=Math.sqrt(dx*dx+dy*dy);
          rawDisp2d[i]=Number.isFinite(m)?m:0;
        }
        const peakDisp2d=new Array(pointCount).fill(0);
        for(let i=0;i<pointCount;i+=1){
          const b=baseNorm[i]||[0,0];
          const p=peakNorm[i]||[0,0];
          const dx=(p[0]-b[0])*width;
          const dy=(p[1]-b[1])*height;
          const m=Math.sqrt(dx*dx+dy*dy);
          peakDisp2d[i]=Number.isFinite(m)?m:0;
        }
        const quantile95=(arr)=>{const vals=arr.filter(v=>Number.isFinite(v));if(!vals.length)return 1.0;vals.sort((a,b)=>a-b);const idx=Math.max(0,Math.min(vals.length-1,Math.floor(0.95*(vals.length-1))));const q=vals[idx];if(q>1e-6)return q;const mx=Math.max(...vals);return mx>1e-6?mx:1.0;};
        const globalScale=quantile95(peakDisp2d);
        const regionScale=new Array(regionNames.length).fill(globalScale);
        for(let r=0;r<regionNames.length;r+=1){
          const vals=[];for(let i=0;i<pointCount;i+=1){if(regionIds[i]===r){vals.push(peakDisp2d[i]);}}
          regionScale[r]=quantile95(vals);
        }
        const magGlobal=rawDisp2d.map(v=>clamp01(v/globalScale));
        const magRegion=new Array(pointCount).fill(0);
        for(let i=0;i<pointCount;i+=1){
          const regionIdx=regionIds[i];
          const scale=(regionIdx!=null&&regionIdx>=0&&regionIdx<regionScale.length)?regionScale[regionIdx]:globalScale;
          magRegion[i]=clamp01(rawDisp2d[i]/scale);
        }
        const magArray=(normalizeMode==="region")?magRegion:magGlobal;
        const displayMag=magArray.map(v=>clamp01(t*v));
        twoDRef.current={points:pointsPx,mag:displayMag};

        if(showBackground2d&&currentImage&&currentImage.width>0){
          ctx.drawImage(currentImage,0,0,width,height);
        }else{
          ctx.fillStyle="rgba(0,0,0,1)";
          ctx.fillRect(0,0,width,height);
        }

        if(showTextureOverlay){
          for(let fi=0;fi<faces.length;fi+=1){
            const tri=faces[fi];const a=tri[0],b=tri[1],c=tri[2];
            if(a<0||b<0||c<0||a>=pointCount||b>=pointCount||c>=pointCount){continue;}
            if(!isRegionEnabled(a)||!isRegionEnabled(b)||!isRegionEnabled(c)){continue;}
            const pa=pointsPx[a],pb=pointsPx[b],pc=pointsPx[c];
            const m=(displayMag[a]+displayMag[b]+displayMag[c])/3.0;
            if(!(m>0.001)){continue;}
            const color=colorFromValue(m);
            ctx.fillStyle=`rgba(${Math.round(color.r*255)},${Math.round(color.g*255)},${Math.round(color.b*255)},0.35)`;
            ctx.beginPath();
            ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.lineTo(pc[0],pc[1]);ctx.closePath();ctx.fill();
          }
        }

        if(showWire){
          ctx.strokeStyle="rgba(255,255,255,0.62)";
          ctx.lineWidth=1.0;
          for(let ei=0;ei<edges.length;ei+=1){
            const e=edges[ei];const a=e[0],b=e[1];
            if(a<0||b<0||a>=pointCount||b>=pointCount){continue;}
            if(!isRegionEnabled(a)||!isRegionEnabled(b)){continue;}
            const pa=pointsPx[a],pb=pointsPx[b];
            ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();
          }
        }

        const pointRadius=Math.max(0.5,pointSize);
        for(let i=0;i<pointCount;i+=1){
          if(!isRegionEnabled(i)){continue;}
          const pxy=pointsPx[i];
          const col=colorFromValue(displayMag[i]);
          ctx.fillStyle=`rgba(${Math.round(col.r*255)},${Math.round(col.g*255)},${Math.round(col.b*255)},0.95)`;
          ctx.beginPath();
          ctx.arc(pxy[0],pxy[1],pointRadius,0,Math.PI*2);
          ctx.fill();
        }

        if(showCones){
          const fullLen=Math.max(0,data.cone_length_px*coneScale);
          const len=t*fullLen;
          if(t>0.001&&len>=1){
            const headLen=Math.max(3,Math.min(12,len*0.25));
            const headWidth=headLen*0.6;
            for(let i=0;i<pointCount;i+=1){
              if(!isRegionEnabled(i)){continue;}
              const b=baseNorm[i]||[0,0];const p=peakNorm[i]||[0,0];
              const dx=(p[0]-b[0])*width;const dy=(p[1]-b[1])*height;
              const norm=Math.sqrt(dx*dx+dy*dy);
              if(!(norm>1e-6)){continue;}
              const ux=dx/norm;const uy=dy/norm;
              const sxy=pointsPx[i];
              const ex=sxy[0]+ux*len;const ey=sxy[1]+uy*len;
              const col=colorFromValue(displayMag[i]);
              ctx.strokeStyle=`rgba(${Math.round(col.r*255)},${Math.round(col.g*255)},${Math.round(col.b*255)},0.92)`;
              ctx.fillStyle=ctx.strokeStyle;
              ctx.lineWidth=1.25;
              ctx.beginPath();ctx.moveTo(sxy[0],sxy[1]);ctx.lineTo(ex,ey);ctx.stroke();

              const px=-uy;const py=ux;
              const bx=ex-(ux*headLen);const by=ey-(uy*headLen);
              const lx=bx+(px*headWidth*0.5);const ly=by+(py*headWidth*0.5);
              const rx=bx-(px*headWidth*0.5);const ry=by-(py*headWidth*0.5);
              ctx.beginPath();
              ctx.moveTo(ex,ey);
              ctx.lineTo(lx,ly);
              ctx.lineTo(rx,ry);
              ctx.closePath();
              ctx.fill();
            }
          }
        }

        const hideTip=()=>{if(tipRef.current){tipRef.current.style.display="none";}};
        const onMove=(event)=>{
          const rect=canvas.getBoundingClientRect();
          const x=(event.clientX-rect.left)*(canvas.width/Math.max(rect.width,1));
          const y=(event.clientY-rect.top)*(canvas.height/Math.max(rect.height,1));
          let best=-1;let bestDist=Number.POSITIVE_INFINITY;
          for(let i=0;i<pointCount;i+=1){
            if(!isRegionEnabled(i)){continue;}
            const p=twoDRef.current.points[i];if(!p){continue;}
            const dx=p[0]-x,dy=p[1]-y;const d=Math.sqrt(dx*dx+dy*dy);
            if(d<bestDist){bestDist=d;best=i;}
          }
          if(best<0||bestDist>20){hideTip();return;}
          if(!tipRef.current){return;}
          tipRef.current.style.display="block";
          tipRef.current.style.left=`${event.clientX-rect.left+14}px`;
          tipRef.current.style.top=`${event.clientY-rect.top+14}px`;
          const magVal=twoDRef.current.mag[best]||0;
          tipRef.current.innerHTML=`<b>Landmark #${best}</b><br/>Region: ${regionLabel(best)}<br/>Norm mag: ${magVal.toFixed(3)}`;
        };
        canvas.addEventListener("mousemove",onMove);
        canvas.addEventListener("mouseleave",hideTip);
        return()=>{canvas.removeEventListener("mousemove",onMove);canvas.removeEventListener("mouseleave",hideTip);hideTip();};
      },[renderMode,has2d,images2dReady,t,showWire,showCones,showTextureOverlay,normalizeMode,regionEnabled,showBackground2d,coneScale,pointSize]);
      const toggleRegion=name=>setRegionEnabled(prev=>({...prev,[name]:!prev[name]}));
      return(
        <div className="app">
          <aside className="panel">
            <h3 style={{margin:"0 0 8px 0"}}>FaceAnalyze2 Motion Viewer</h3>
            <div className="row" style={{fontSize:12,color:"#93a1c8"}}>neutral #{data.frame_indices[data.neutral_idx]} - peak #{data.frame_indices[data.peak_idx]}</div>
            <div className="row">
              <label>mode</label>
              <select value={renderMode} onChange={e=>setRenderMode(e.target.value)}>
                <option value="2d" disabled={!has2d}>2d</option>
                <option value="texture" disabled={!hasTexture}>texture</option>
                <option value="points">points</option>
              </select>
            </div>
            {!has2d && <div className="row muted">2d unavailable (video/frame read failed)</div>}
            {!hasTexture && <div className="row muted">texture unavailable (neutral frame unavailable)</div>}
            <div className="row"><label>t = {t.toFixed(2)}</label><input type="range" min="0" max="1" step="0.01" value={t} onChange={e=>{setIsPlaying(false);setT(parseFloat(e.target.value))}}/></div>
            <div className="row" style={{display:"flex",gap:"6px"}}><button style={{flex:1}} onClick={()=>setIsPlaying(p=>!p)}>{isPlaying?"\u23f8 Pause":"\u25b6 Play"}</button><button style={{flex:0,padding:"7px 10px",opacity:playSpeed===0.5?1:0.5}} onClick={()=>setPlaySpeed(0.5)}>0.5x</button><button style={{flex:0,padding:"7px 10px",opacity:playSpeed===1.0?1:0.5}} onClick={()=>setPlaySpeed(1.0)}>1x</button><button style={{flex:0,padding:"7px 10px",opacity:playSpeed===2.0?1:0.5}} onClick={()=>setPlaySpeed(2.0)}>2x</button></div>
            <div className="row"><label><input type="checkbox" checked={showWire} onChange={e=>setShowWire(e.target.checked)}/>wireframe</label></div>
            <div className="row"><label><input type="checkbox" checked={showCones} onChange={e=>setShowCones(e.target.checked)}/>cones</label></div>
            <div className="row"><label><input type="checkbox" checked={flattenVectors} onChange={e=>setFlattenVectors(e.target.checked)}/>2D vectors (flatten z)</label></div>
            {has2d&&renderMode==="2d"&&(<div className="row"><label><input type="checkbox" checked={showBackground2d} onChange={e=>setShowBackground2d(e.target.checked)}/>video overlay</label></div>)}
            {renderMode!=="points"&&(<div className="row"><label><input type="checkbox" checked={showTextureOverlay} onChange={e=>setShowTextureOverlay(e.target.checked)}/>displacement heatmap</label></div>)}
            <div className="row"><label>normalize mode</label><select value={normalizeMode} onChange={e=>setNormalizeMode(e.target.value)}><option value="region">region normalize</option><option value="global">global normalize</option></select></div>
            <div className="row"><label>cone scale: {coneScale.toFixed(2)}</label><input type="range" min="0.2" max="3.0" step="0.05" value={coneScale} onChange={e=>setConeScale(parseFloat(e.target.value))}/></div>
            <div className="row"><label>point size: {pointSize.toFixed(1)}</label><input type="range" min="1" max="8" step="0.2" value={pointSize} onChange={e=>setPointSize(parseFloat(e.target.value))}/></div>
            <div className="row"><button onClick={()=>viewerRef.current?.exportPng?.()}>export png</button></div>
            <div className="checks">{data.region_names.map((name,idx)=><label key={name}><input type="checkbox" checked={!!regionEnabled[name]} onChange={()=>toggleRegion(name)}/>{friendlyRegion(name)} ({data.region_scales[idx].toFixed(3)})</label>)}</div>
            <div className="row" style={{marginTop:10}}><div className="legend"/></div>
            <div className="meta">
              <div><b>residual max:</b> {data.quality.residual_max==null?"n/a":data.quality.residual_max.toFixed(4)}</div>
              <div><b>residual mean:</b> {data.quality.residual_mean==null?"n/a":data.quality.residual_mean.toFixed(4)}</div>
              <div><b>valid frames:</b> {data.quality.valid_frames}/{data.quality.total_frames}</div>
              <div><b>neutral replaced:</b> {String(data.quality.neutral_replaced)}</div>
              <div><b>peak replaced:</b> {String(data.quality.peak_replaced)}</div>
              <div><b>cone length px:</b> {data.cone_length_px.toFixed(2)}</div>
              <div><b>flipY:</b> {String(data.params.flip_y)} | <b>flipZ:</b> {String(data.params.flip_z)} | <b>z_scale:</b> {data.params.z_scale}</div>
            </div>
          </aside>
          <section className="viewer">
            <div ref={mountRef} className="mount" style={{display:renderMode==="2d"?"none":"block"}}/>
            <canvas ref={canvas2dRef} className="mount" style={{display:renderMode==="2d"?"block":"none"}}/>
            <div ref={tipRef} className="tt"/>
          </section>
        </div>
      );
    }
    ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
  </script>
</body>
</html>
"""
    return template.replace("__MOTION_VIEWER_DATA__", data_json)


def generate_motion_viewer(
    *,
    video_path: str | Path,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    landmarks_path: str | Path | None = None,
    segment_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = _resolve_paths(
        video_path=video_path,
        artifact_root=artifact_root,
        landmarks_path=landmarks_path,
        segment_path=segment_path,
    )

    if not paths.landmarks_path.exists():
        raise FileNotFoundError(_missing_landmarks_message(paths.landmarks_path, video_path))
    if not paths.segment_path.exists():
        raise FileNotFoundError(_missing_segment_message(paths.segment_path, video_path))
    if not paths.meta_path.exists():
        raise FileNotFoundError(_missing_meta_message(paths.meta_path, video_path))

    arrays = _load_landmark_arrays(paths.landmarks_path)
    meta = _load_meta(paths.meta_path)
    segment = _load_segment(paths.segment_path, frame_count=arrays["frame_indices"].shape[0])

    xyz3 = _to_pseudo_pixel_3d(
        arrays["landmarks_xyz"],
        width=int(meta["width"]),
        height=int(meta["height"]),
    )
    aligned = _align_to_neutral(
        xyz3,
        presence=arrays["presence"],
        neutral_idx=int(segment["neutral_idx"]),
    )

    viewer_payload = _build_viewer_payload(
        aligned_xyz=aligned["aligned_xyz"],
        frame_indices=arrays["frame_indices"],
        timestamps_ms=arrays["timestamps_ms"],
        neutral_idx=int(segment["neutral_idx"]),
        peak_idx=int(segment["peak_idx"]),
        residuals=aligned["residuals"],
        valid_mask=aligned["valid_mask"],
        invalid_presence=aligned["invalid_presence"],
        invalid_stable=aligned["invalid_stable"],
        invalid_transform=aligned["invalid_transform"],
        flip_y=DEFAULT_FLIP_Y,
        flip_z=DEFAULT_FLIP_Z,
        z_scale=DEFAULT_Z_SCALE,
    )

    neutral_row_idx = int(viewer_payload["neutral_idx"])
    peak_row_idx = int(viewer_payload["peak_idx"])
    neutral_video_frame_idx = int(arrays["frame_indices"][neutral_row_idx])
    neutral_xy_norm = np.asarray(arrays["landmarks_xyz"][neutral_row_idx, :, :2], dtype=np.float32)
    edges = _extract_facemesh_edges()
    viewer_payload["neutral_image_base64"] = _extract_neutral_frame_base64(
        video_path=video_path,
        frame_idx=neutral_video_frame_idx,
    )
    viewer_payload["uv_coords"] = _build_uv_coords_from_normalized_xy(neutral_xy_norm)
    viewer_payload["edges"] = edges
    viewer_payload["faces"] = _extract_facemesh_faces(edges)

    row_indices_2d = _segment_row_indices(
        neutral_idx=neutral_row_idx,
        peak_idx=peak_row_idx,
        max_frames=MAX_2D_FRAMES,
    )
    frame_indices_2d = arrays["frame_indices"][row_indices_2d].astype(np.int64)
    frame_images = _extract_video_frames_base64(
        video_path=video_path,
        frame_indices=frame_indices_2d,
        width=int(meta["width"]),
        height=int(meta["height"]),
    )
    if frame_images is None:
        viewer_payload["frame_images"] = None
        viewer_payload["all_norm_xy"] = None
        viewer_payload["base_norm_xy"] = None
        viewer_payload["peak_norm_xy"] = None
    else:
        all_norm_xy = _build_all_norm_xy_payload(
            landmarks_xyz=arrays["landmarks_xyz"],
            presence=arrays["presence"],
            row_indices=row_indices_2d,
        )
        viewer_payload["frame_images"] = frame_images
        viewer_payload["all_norm_xy"] = all_norm_xy
        viewer_payload["base_norm_xy"] = all_norm_xy[0] if all_norm_xy else None
        viewer_payload["peak_norm_xy"] = all_norm_xy[-1] if all_norm_xy else None

    html = _render_motion_viewer_html(viewer_payload)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    paths.output_html_path.write_text(html, encoding="utf-8")
    return {
        "html_path": str(paths.output_html_path),
        "artifact_dir": str(paths.artifact_dir),
        "landmarks_path": str(paths.landmarks_path),
        "segment_path": str(paths.segment_path),
        "meta_path": str(paths.meta_path),
        "quality": viewer_payload["quality"],
    }
