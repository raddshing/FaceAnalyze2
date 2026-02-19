from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.analysis.metrics import compute_roi_displacements
from faceanalyze2.config import artifact_dir_for_video
from faceanalyze2.io.video_reader import extract_frame
from faceanalyze2.roi.indices import (
    AREA0_GREEN,
    AREA1_BLUE,
    AREA2_YELLOW,
    AREA3_RED,
    ROI_EYE,
    ROI_EYEBROW,
    ROI_MOUTH,
    resolve_pair_indices,
)

SUPPORTED_MOTIONS = {"big-smile", "blinking-motion", "eyebrow-motion"}
PRIMARY_ROI_BY_MOTION = {
    "big-smile": "mouth",
    "blinking-motion": "eye",
    "eyebrow-motion": "eyebrow",
}
ROI_ORDER_BY_MOTION = {
    "big-smile": ["mouth", "area0_green", "area1_blue", "area2_yellow", "area3_red"],
    "blinking-motion": ["eye"],
    "eyebrow-motion": ["eyebrow"],
}
EPSILON = 1e-6
LANDMARK_COUNT = 478


def _build_pair_roi_groups(landmark_count: int) -> dict[str, tuple[list[int], list[int]]]:
    pair_rois = {
        "mouth": ROI_MOUTH,
        "eye": ROI_EYE,
        "eyebrow": ROI_EYEBROW,
        "area0_green": AREA0_GREEN,
        "area1_blue": AREA1_BLUE,
        "area2_yellow": AREA2_YELLOW,
        "area3_red": AREA3_RED,
    }
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for name, pair_indices in pair_rois.items():
        left, right = resolve_pair_indices(pair_indices)
        left_filtered = [idx for idx in left if 0 <= idx < landmark_count]
        right_filtered = [idx for idx in right if 0 <= idx < landmark_count]
        if left_filtered and right_filtered:
            groups[name] = (left_filtered, right_filtered)
    return groups


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_meta(artifact_dir: Path) -> dict[str, Any]:
    meta_path = artifact_dir / "meta.json"
    if not meta_path.exists():
        return {"width": 640, "height": 480}
    meta = _read_json(meta_path)
    width = int(meta.get("width", 640))
    height = int(meta.get("height", 480))
    return {"width": max(width, 1), "height": max(height, 1)}


def _load_segment(artifact_dir: Path, frame_count: int) -> dict[str, int]:
    segment_path = artifact_dir / "segment.json"
    if not segment_path.exists():
        raise FileNotFoundError(
            f"segment.json not found: {segment_path}\n"
            "Run this first:\n"
            'faceanalyze2 segment run --video "<path>" --task <smile|brow|eyeclose>'
        )
    payload = _read_json(segment_path)
    for key in ("neutral_idx", "peak_idx"):
        if key not in payload:
            raise ValueError(f"segment.json is missing required field: {key}")
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
    return {"neutral_idx": neutral_idx, "peak_idx": peak_idx}


def _load_aligned_xy(artifact_dir: Path, width: int, height: int) -> dict[str, np.ndarray]:
    aligned_path = artifact_dir / "landmarks_aligned.npz"
    if aligned_path.exists():
        with np.load(aligned_path) as npz:
            required = {"frame_indices", "timestamps_ms", "presence", "landmarks_xy_aligned"}
            missing = required.difference(npz.files)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"landmarks_aligned.npz missing required keys: {missing_text}")
            frame_indices = np.asarray(npz["frame_indices"], dtype=np.int64)
            timestamps_ms = np.asarray(npz["timestamps_ms"], dtype=np.int64)
            presence = np.asarray(npz["presence"], dtype=bool)
            aligned_xy = np.asarray(npz["landmarks_xy_aligned"], dtype=np.float32)
        return {
            "frame_indices": frame_indices,
            "timestamps_ms": timestamps_ms,
            "presence": presence,
            "aligned_xy": aligned_xy,
            "source": "landmarks_aligned.npz",
        }

    landmarks_path = artifact_dir / "landmarks.npz"
    if not landmarks_path.exists():
        raise FileNotFoundError(
            f"Aligned and raw landmarks are both missing in: {artifact_dir}\n"
            "Run this first:\n"
            'faceanalyze2 landmarks extract --video "<path>"'
        )
    with np.load(landmarks_path) as npz:
        required = {"frame_indices", "timestamps_ms", "presence", "landmarks_xyz"}
        missing = required.difference(npz.files)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"landmarks.npz missing required keys: {missing_text}")
        frame_indices = np.asarray(npz["frame_indices"], dtype=np.int64)
        timestamps_ms = np.asarray(npz["timestamps_ms"], dtype=np.int64)
        presence = np.asarray(npz["presence"], dtype=bool)
        landmarks_xyz = np.asarray(npz["landmarks_xyz"], dtype=np.float32)

    aligned_xy = landmarks_xyz[:, :, :2].copy()
    aligned_xy[:, :, 0] *= float(width)
    aligned_xy[:, :, 1] *= float(height)
    return {
        "frame_indices": frame_indices,
        "timestamps_ms": timestamps_ms,
        "presence": presence,
        "aligned_xy": aligned_xy.astype(np.float32),
        "source": "landmarks.npz",
    }


def _load_raw_xy_for_overlay(artifact_dir: Path, width: int, height: int) -> np.ndarray | None:
    landmarks_path = artifact_dir / "landmarks.npz"
    if not landmarks_path.exists():
        return None
    with np.load(landmarks_path) as npz:
        if "landmarks_xyz" not in npz.files:
            return None
        raw = np.asarray(npz["landmarks_xyz"], dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (LANDMARK_COUNT, 3):
        return None
    xy = raw[:, :, :2].copy()
    xy[:, :, 0] *= float(width)
    xy[:, :, 1] *= float(height)
    return xy.astype(np.float32)


def _encode_png_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _matplotlib_png_base64(draw_fn: Any, *, width: int = 640, height: int = 480) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(max(width, 1) / 100.0, max(height, 1) / 100.0), dpi=100)
    try:
        draw_fn(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
        return _encode_png_bytes(buffer.getvalue())
    finally:
        plt.close(fig)


def _placeholder_png_base64(text: str, *, width: int, height: int) -> str:
    def _draw(fig: Any) -> None:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#101820")
        ax.text(
            0.5,
            0.5,
            text,
            color="white",
            ha="center",
            va="center",
            fontsize=13,
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return _matplotlib_png_base64(_draw, width=width, height=height)


def _encode_rgb_frame_png_base64(image_rgb: np.ndarray) -> str:
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", image_bgr)
        if ok:
            return _encode_png_bytes(encoded.tobytes())

    def _draw(fig: Any) -> None:
        ax = fig.add_subplot(111)
        ax.imshow(image_rgb)
        ax.axis("off")

    height, width = image_rgb.shape[:2]
    return _matplotlib_png_base64(_draw, width=width, height=height)


def _try_extract_frame_png(video_path: Path, frame_idx: int, *, width: int, height: int) -> str:
    try:
        frame = extract_frame(video_path, frame_idx, to_rgb=True)
        return _encode_rgb_frame_png_base64(frame.image_rgb)
    except Exception:
        return _placeholder_png_base64(
            f"Frame unavailable\nidx={frame_idx}",
            width=width,
            height=height,
        )


def _overlay_landmarks_png(
    neutral_xy: np.ndarray,
    peak_xy: np.ndarray,
    *,
    width: int,
    height: int,
    title: str,
) -> str:
    def _draw(fig: Any) -> None:
        ax = fig.add_subplot(111)
        n_valid = np.isfinite(neutral_xy).all(axis=1)
        p_valid = np.isfinite(peak_xy).all(axis=1)
        if np.any(n_valid):
            ax.scatter(
                neutral_xy[n_valid, 0],
                neutral_xy[n_valid, 1],
                s=8,
                c="#00bcd4",
                alpha=0.55,
                label="neutral",
            )
        if np.any(p_valid):
            ax.scatter(
                peak_xy[p_valid, 0],
                peak_xy[p_valid, 1],
                s=8,
                c="#ff5252",
                alpha=0.55,
                label="peak",
            )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    return _matplotlib_png_base64(_draw, width=width, height=height)


def _series_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _compute_roi_metrics_from_series(
    result: dict[str, Any], roi_names: list[str]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for roi_name in roi_names:
        series = result["series"][roi_name]
        left_peak = _series_max(np.asarray(series["left_disp"], dtype=np.float32))
        right_peak = _series_max(np.asarray(series["right_disp"], dtype=np.float32))
        if np.isfinite(left_peak) and np.isfinite(right_peak):
            ai = abs(left_peak - right_peak) / max(EPSILON, (left_peak + right_peak) * 0.5)
            score = (1.0 - min(1.0, max(0.0, ai))) * 100.0
        else:
            ai = float("nan")
            score = float("nan")
        metrics[roi_name] = {
            "L_peak": left_peak,
            "R_peak": right_peak,
            "AI": float(ai),
            "score": float(score),
        }
    return metrics


def _key_value_graph_png(
    *,
    result: dict[str, Any],
    roi_name: str,
    motion: str,
    width: int,
    height: int,
) -> str:
    series = result["series"][roi_name]
    time_s = np.asarray(result["time_s"], dtype=np.float32)
    left = np.asarray(series["left_disp"], dtype=np.float32)
    right = np.asarray(series["right_disp"], dtype=np.float32)

    def _draw(fig: Any) -> None:
        ax = fig.add_subplot(111)
        ax.plot(time_s, left, color="#d62728", linewidth=2.2, label="Left")
        ax.plot(time_s, right, color="#2ca02c", linewidth=2.2, label="Right")
        ax.set_title(f"{motion} - {roi_name} displacement")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("normalized displacement")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    return _matplotlib_png_base64(_draw, width=width, height=height)


def _validate_motion(motion: str) -> str:
    motion_name = str(motion).strip()
    if motion_name not in SUPPORTED_MOTIONS:
        valid = ", ".join(sorted(SUPPORTED_MOTIONS))
        raise ValueError(f"Unsupported motion '{motion_name}'. Expected one of: {valid}")
    return motion_name


def dynamicAnalysis(vd_path: str | Path, motion: str) -> dict[str, Any]:
    """Frontend adapter for dynamic analysis payload."""
    video_path = Path(vd_path)
    motion_name = _validate_motion(motion)
    artifact_dir = artifact_dir_for_video(video_path=video_path)
    meta = _load_meta(artifact_dir)
    width = int(meta["width"])
    height = int(meta["height"])

    aligned_payload = _load_aligned_xy(artifact_dir=artifact_dir, width=width, height=height)
    aligned_xy = np.asarray(aligned_payload["aligned_xy"], dtype=np.float32)
    if aligned_xy.ndim != 3 or aligned_xy.shape[2] != 2:
        raise ValueError(f"Invalid aligned landmark shape: {aligned_xy.shape}")
    frame_count, landmark_count, _ = aligned_xy.shape
    if frame_count == 0:
        raise ValueError("No frames available for dynamicAnalysis")

    segment = _load_segment(artifact_dir=artifact_dir, frame_count=frame_count)
    neutral_idx = int(segment["neutral_idx"])
    peak_idx = int(segment["peak_idx"])
    roi_names = list(ROI_ORDER_BY_MOTION[motion_name])
    roi_groups_all = _build_pair_roi_groups(landmark_count=landmark_count)
    missing_rois = [roi_name for roi_name in roi_names if roi_name not in roi_groups_all]
    if missing_rois:
        missing = ", ".join(missing_rois)
        raise ValueError(f"Cannot resolve ROI pair indices for: {missing}")

    result = compute_roi_displacements(
        aligned_xy=aligned_xy,
        frame_indices=np.asarray(aligned_payload["frame_indices"], dtype=np.int64),
        timestamps_ms=np.asarray(aligned_payload["timestamps_ms"], dtype=np.int64),
        neutral_idx=neutral_idx,
        peak_idx=peak_idx,
        roi_groups={roi_name: roi_groups_all[roi_name] for roi_name in roi_names},
        presence=np.asarray(aligned_payload["presence"], dtype=bool),
        normalize=True,
        interocular_pair=(33, 263),
        fps=None,
    )

    computed_metrics = _compute_roi_metrics_from_series(result, roi_names)
    primary_roi = PRIMARY_ROI_BY_MOTION[motion_name]
    if primary_roi not in result["series"]:
        primary_roi = roi_names[0]

    frame_indices = np.asarray(aligned_payload["frame_indices"], dtype=np.int64)
    neutral_frame_idx = int(frame_indices[neutral_idx])
    peak_frame_idx = int(frame_indices[peak_idx])

    if video_path.exists() and video_path.is_file():
        key_rest = _try_extract_frame_png(video_path, neutral_frame_idx, width=width, height=height)
        key_exp = _try_extract_frame_png(video_path, peak_frame_idx, width=width, height=height)
    else:
        key_rest = _placeholder_png_base64(
            f"Video not found\n{video_path.name}",
            width=width,
            height=height,
        )
        key_exp = _placeholder_png_base64(
            f"Video not found\n{video_path.name}",
            width=width,
            height=height,
        )

    raw_xy = _load_raw_xy_for_overlay(artifact_dir=artifact_dir, width=width, height=height)
    if raw_xy is None:
        before_regi = _placeholder_png_base64(
            "raw landmarks unavailable", width=width, height=height
        )
    else:
        before_regi = _overlay_landmarks_png(
            raw_xy[neutral_idx],
            raw_xy[peak_idx],
            width=width,
            height=height,
            title="Before registration",
        )

    after_regi = _overlay_landmarks_png(
        aligned_xy[neutral_idx],
        aligned_xy[peak_idx],
        width=width,
        height=height,
        title="After registration",
    )
    key_value_graph = _key_value_graph_png(
        result=result,
        roi_name=primary_roi,
        motion=motion_name,
        width=960,
        height=420,
    )

    metrics_payload: dict[str, Any] = {
        "motion": motion_name,
        "source": str(aligned_payload["source"]),
        "neutral_idx": neutral_idx,
        "peak_idx": peak_idx,
        "neutral_frame_idx": neutral_frame_idx,
        "peak_frame_idx": peak_frame_idx,
        "normalize": True,
        "interocular_px": result["interocular_px"],
        "roi_metrics": computed_metrics,
    }

    return {
        "key rest": key_rest,
        "key exp": key_exp,
        "key value graph": key_value_graph,
        "before regi": before_regi,
        "after regi": after_regi,
        "metrics": metrics_payload,
    }
