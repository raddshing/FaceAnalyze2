from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.io.video_reader import VideoInfo, iter_frames, probe_video

OFFICIAL_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
DEFAULT_ARTIFACT_ROOT = Path("artifacts")
LANDMARK_COUNT = 478
BLENDSHAPE_COUNT = 52


def _build_missing_model_message(model_path: Path) -> str:
    return (
        f"Model file not found: {model_path}\n"
        f"Official model URL: {OFFICIAL_FACE_LANDMARKER_MODEL_URL}\n"
        "PowerShell download example:\n"
        f'New-Item -ItemType Directory -Force -Path "{model_path.parent}" | Out-Null\n'
        f'Invoke-WebRequest -Uri "{OFFICIAL_FACE_LANDMARKER_MODEL_URL}" -OutFile "{model_path}"'
    )


def _require_model_file(model_path: str | Path) -> Path:
    resolved = Path(model_path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(_build_missing_model_message(resolved))
    return resolved


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for landmark extraction. Install with: pip install mediapipe"
        ) from exc
    return mp


def _to_blendshape_vector(result: Any) -> np.ndarray:
    vector = np.full((BLENDSHAPE_COUNT,), np.nan, dtype=np.float32)
    if not getattr(result, "face_blendshapes", None):
        return vector

    first = result.face_blendshapes[0]
    categories = getattr(first, "categories", first)
    if categories is None:
        return vector

    for idx, category in enumerate(categories):
        if idx >= BLENDSHAPE_COUNT:
            break
        score = getattr(category, "score", None)
        if score is not None:
            vector[idx] = float(score)
    return vector


def _to_transform_matrix(result: Any) -> np.ndarray:
    matrix = np.full((4, 4), np.nan, dtype=np.float32)
    transforms = getattr(result, "facial_transformation_matrixes", None)
    if not transforms:
        return matrix

    first = transforms[0]
    if hasattr(first, "numpy_view"):
        source = first.numpy_view()
    elif hasattr(first, "data"):
        source = first.data
    elif hasattr(first, "packed_data"):
        source = first.packed_data
    else:
        source = first

    values = np.asarray(source, dtype=np.float32)
    if values.shape == (4, 4):
        return values
    if values.size == 16:
        return values.reshape(4, 4)
    return matrix


def _empty_outputs(
    *,
    output_blendshapes: bool,
    output_transforms: bool,
) -> dict[str, np.ndarray | None]:
    outputs: dict[str, np.ndarray | None] = {
        "timestamps_ms": np.empty((0,), dtype=np.int64),
        "frame_indices": np.empty((0,), dtype=np.int64),
        "landmarks_xyz": np.empty((0, LANDMARK_COUNT, 3), dtype=np.float32),
        "presence": np.empty((0,), dtype=bool),
        "blendshapes": None,
        "transforms": None,
    }
    if output_blendshapes:
        outputs["blendshapes"] = np.empty((0, BLENDSHAPE_COUNT), dtype=np.float32)
    if output_transforms:
        outputs["transforms"] = np.empty((0, 4, 4), dtype=np.float32)
    return outputs


def save_landmark_artifacts(
    *,
    video_info: VideoInfo,
    model_path: str | Path,
    stride: int,
    start_frame: int,
    end_frame: int | None,
    num_faces: int,
    output_blendshapes: bool,
    output_transforms: bool,
    timestamps_ms: np.ndarray,
    frame_indices: np.ndarray,
    landmarks_xyz: np.ndarray,
    presence: np.ndarray,
    blendshapes: np.ndarray | None,
    transforms: np.ndarray | None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    extract_time: str | None = None,
    elapsed_seconds: float | None = None,
) -> tuple[Path, Path]:
    video_stem = Path(video_info.path).stem
    artifact_dir = Path(artifact_root) / video_stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    npz_path = artifact_dir / "landmarks.npz"
    npz_payload: dict[str, np.ndarray] = {
        "timestamps_ms": timestamps_ms,
        "frame_indices": frame_indices,
        "landmarks_xyz": landmarks_xyz,
        "presence": presence,
    }
    if output_blendshapes and blendshapes is not None:
        npz_payload["blendshapes"] = blendshapes
    if output_transforms and transforms is not None:
        npz_payload["transforms"] = transforms
    np.savez_compressed(npz_path, **npz_payload)

    meta_path = artifact_dir / "meta.json"
    meta = {
        "video_path": str(video_info.path),
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "frame_count": video_info.frame_count,
        "duration_ms": video_info.duration_ms,
        "stride": stride,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_faces": num_faces,
        "model_path": str(model_path),
        "output_blendshapes": output_blendshapes,
        "output_transforms": output_transforms,
        "extract_time": extract_time or datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "processed_frames": int(frame_indices.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path, meta_path


def extract_face_landmarks_from_video(
    video_path: str,
    model_path: str = "models/face_landmarker.task",
    stride: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    num_faces: int = 1,
    output_blendshapes: bool = False,
    output_transforms: bool = False,
    use_gpu_delegate: bool = False,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got: {stride}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got: {start_frame}")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if num_faces < 1:
        raise ValueError(f"num_faces must be >= 1, got: {num_faces}")

    model_file = _require_model_file(model_path)
    video_info = probe_video(video_path)

    mp = _import_mediapipe()
    base_options_kwargs: dict[str, Any] = {"model_asset_path": str(model_file)}
    if use_gpu_delegate:
        base_options_kwargs["delegate"] = mp.tasks.BaseOptions.Delegate.GPU
    base_options = mp.tasks.BaseOptions(**base_options_kwargs)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=num_faces,
        output_face_blendshapes=output_blendshapes,
        output_facial_transformation_matrixes=output_transforms,
    )

    timestamps_ms_list: list[int] = []
    frame_indices_list: list[int] = []
    landmarks_xyz_list: list[np.ndarray] = []
    presence_list: list[bool] = []
    blendshapes_list: list[np.ndarray] = []
    transforms_list: list[np.ndarray] = []

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for frame in iter_frames(
            video_path,
            stride=stride,
            start_frame=start_frame,
            end_frame=end_frame,
            to_rgb=True,
        ):
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame.image_rgb),
            )
            result = landmarker.detect_for_video(mp_image, frame.timestamp_ms)

            timestamps_ms_list.append(frame.timestamp_ms)
            frame_indices_list.append(frame.idx)

            frame_landmarks = np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
            detected = bool(result.face_landmarks)
            if detected:
                first_face = result.face_landmarks[0]
                for idx, landmark in enumerate(first_face):
                    if idx >= LANDMARK_COUNT:
                        break
                    frame_landmarks[idx] = np.asarray(
                        [landmark.x, landmark.y, landmark.z],
                        dtype=np.float32,
                    )

            landmarks_xyz_list.append(frame_landmarks)
            presence_list.append(detected)

            if output_blendshapes:
                blendshapes_list.append(_to_blendshape_vector(result))
            if output_transforms:
                transforms_list.append(_to_transform_matrix(result))

    elapsed_seconds = time.perf_counter() - start_time

    if landmarks_xyz_list:
        timestamps_ms = np.asarray(timestamps_ms_list, dtype=np.int64)
        frame_indices = np.asarray(frame_indices_list, dtype=np.int64)
        landmarks_xyz = np.stack(landmarks_xyz_list, axis=0).astype(np.float32, copy=False)
        presence = np.asarray(presence_list, dtype=bool)
        blendshapes = (
            np.stack(blendshapes_list, axis=0).astype(np.float32, copy=False)
            if output_blendshapes
            else None
        )
        transforms = (
            np.stack(transforms_list, axis=0).astype(np.float32, copy=False)
            if output_transforms
            else None
        )
    else:
        empty = _empty_outputs(
            output_blendshapes=output_blendshapes,
            output_transforms=output_transforms,
        )
        timestamps_ms = empty["timestamps_ms"]
        frame_indices = empty["frame_indices"]
        landmarks_xyz = empty["landmarks_xyz"]
        presence = empty["presence"]
        blendshapes = empty["blendshapes"]
        transforms = empty["transforms"]

    npz_path, meta_path = save_landmark_artifacts(
        video_info=video_info,
        model_path=model_file,
        stride=stride,
        start_frame=start_frame,
        end_frame=end_frame,
        num_faces=num_faces,
        output_blendshapes=output_blendshapes,
        output_transforms=output_transforms,
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
        landmarks_xyz=landmarks_xyz,
        presence=presence,
        blendshapes=blendshapes,
        transforms=transforms,
        artifact_root=artifact_root,
        extract_time=started_at,
        elapsed_seconds=elapsed_seconds,
    )

    return {
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "timestamps_ms": timestamps_ms,
        "frame_indices": frame_indices,
        "landmarks_xyz": landmarks_xyz,
        "presence": presence,
        "blendshapes": blendshapes,
        "transforms": transforms,
    }
