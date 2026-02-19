from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.analysis.segmentation import DEFAULT_ARTIFACT_ROOT
from faceanalyze2.roi.indices import (
    LEFT_BROW,
    LEFT_EYE,
    LEFT_MOUTH_CORNER_GROUP,
    RIGHT_BROW,
    RIGHT_EYE,
    RIGHT_MOUTH_CORNER_GROUP,
)

VALID_TASKS = {"smile", "brow", "eyeclose"}
VALID_ROIS = {"mouth", "eye", "brow"}
DEFAULT_ROI_ORDER = {
    "smile": ["mouth", "eye", "brow"],
    "eyeclose": ["eye", "mouth", "brow"],
    "brow": ["brow", "eye", "mouth"],
}
ROI_GROUPS = {
    "mouth": (LEFT_MOUTH_CORNER_GROUP, RIGHT_MOUTH_CORNER_GROUP),
    "eye": (LEFT_EYE, RIGHT_EYE),
    "brow": (LEFT_BROW, RIGHT_BROW),
}
EPSILON = 1e-6


def _normalize_task(task: Any) -> str:
    task_name = str(task.value) if hasattr(task, "value") else str(task)
    if task_name not in VALID_TASKS:
        valid = ", ".join(sorted(VALID_TASKS))
        raise ValueError(f"Unsupported task '{task_name}'. Expected one of: {valid}")
    return task_name


def parse_rois_option(rois: str, task: str) -> list[str]:
    task_name = _normalize_task(task)
    text = str(rois).strip().lower()
    if text == "all":
        return list(DEFAULT_ROI_ORDER[task_name])

    parts = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("rois must be 'all' or comma-separated values from mouth,eye,brow")

    selected: list[str] = []
    for part in parts:
        if part not in VALID_ROIS:
            valid = ", ".join(sorted(VALID_ROIS))
            raise ValueError(f"Unsupported roi '{part}'. Expected one of: {valid}")
        if part not in selected:
            selected.append(part)
    return selected


def _resolve_artifact_dir(
    *,
    video_path: str | Path,
    artifact_root: str | Path,
    aligned_path: str | Path | None,
    segment_path: str | Path | None,
) -> Path:
    if aligned_path is not None:
        return Path(aligned_path).parent
    if segment_path is not None:
        return Path(segment_path).parent
    return Path(artifact_root) / Path(video_path).stem


def _missing_aligned_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Aligned landmarks file not found: {path}\n"
        "Run this pipeline first:\n"
        f'faceanalyze2 landmarks extract --video "{video_path}"\n'
        f'faceanalyze2 segment run --video "{video_path}" --task <smile|brow|eyeclose>\n'
        f'faceanalyze2 align run --video "{video_path}"'
    )


def _missing_segment_message(path: Path, video_path: str | Path) -> str:
    return (
        f"Segment file not found: {path}\n"
        "Run this first: segment run.\n"
        f'faceanalyze2 segment run --video "{video_path}" --task <smile|brow|eyeclose>'
    )


def _load_aligned_arrays(aligned_path: Path) -> dict[str, np.ndarray]:
    if not aligned_path.exists():
        raise FileNotFoundError(aligned_path)
    with np.load(aligned_path) as data:
        arrays = {name: data[name] for name in data.files}

    required = {"frame_indices", "timestamps_ms", "presence", "landmarks_xy_aligned"}
    missing = required.difference(arrays)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"landmarks_aligned.npz is missing required keys: {missing_keys}")

    frame_indices = np.asarray(arrays["frame_indices"])
    timestamps_ms = np.asarray(arrays["timestamps_ms"])
    presence = np.asarray(arrays["presence"])
    aligned_xy = np.asarray(arrays["landmarks_xy_aligned"])

    if frame_indices.ndim != 1 or timestamps_ms.ndim != 1 or presence.ndim != 1:
        raise ValueError("frame_indices, timestamps_ms, and presence must be 1D arrays")
    if aligned_xy.ndim != 3 or aligned_xy.shape[2] != 2:
        raise ValueError(f"landmarks_xy_aligned must have shape (T, N, 2), got {aligned_xy.shape}")

    t_count = frame_indices.shape[0]
    if (
        timestamps_ms.shape[0] != t_count
        or presence.shape[0] != t_count
        or aligned_xy.shape[0] != t_count
    ):
        raise ValueError("Aligned array lengths do not match frame count")

    return {
        "frame_indices": frame_indices.astype(np.int64),
        "timestamps_ms": timestamps_ms.astype(np.int64),
        "presence": presence.astype(bool),
        "landmarks_xy_aligned": aligned_xy.astype(np.float32),
    }


def _load_segment(segment_path: Path, frame_count: int) -> dict[str, Any]:
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)
    payload = json.loads(segment_path.read_text(encoding="utf-8"))
    for key in ("neutral_idx", "peak_idx"):
        if key not in payload:
            raise ValueError(f"segment.json is missing required field '{key}': {segment_path}")
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
    return payload


def _estimate_fps_from_timestamps(timestamps_ms: np.ndarray) -> float | None:
    if timestamps_ms.shape[0] < 2:
        return None
    deltas = np.diff(timestamps_ms.astype(np.float64))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return None
    median_delta = float(np.median(positive))
    if median_delta <= 0:
        return None
    return 1000.0 / median_delta


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=1)
    sums = np.nansum(values, axis=1)
    out = np.full((values.shape[0],), np.nan, dtype=np.float32)
    mask = counts > 0
    out[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return out


def _compute_side_series(
    sequence_xy: np.ndarray,
    neutral_xy: np.ndarray,
    indices: list[int],
    sequence_presence: np.ndarray | None,
    normalize: bool,
    denom: float,
) -> dict[str, np.ndarray]:
    side_seq = sequence_xy[:, indices, :]
    side_ref = neutral_xy[indices, :][None, :, :]
    delta = side_seq - side_ref

    if sequence_presence is not None:
        delta[~sequence_presence, :, :] = np.nan

    mean_dx = _nanmean_rows(delta[:, :, 0])
    mean_dy = _nanmean_rows(delta[:, :, 1])
    displacement = np.linalg.norm(delta, axis=2)
    mean_disp = _nanmean_rows(displacement)

    if normalize:
        mean_dx = (mean_dx / denom).astype(np.float32)
        mean_dy = (mean_dy / denom).astype(np.float32)
        mean_disp = (mean_disp / denom).astype(np.float32)

    return {
        "mean_dx": mean_dx.astype(np.float32),
        "mean_dy": mean_dy.astype(np.float32),
        "mean_disp": mean_disp.astype(np.float32),
    }


def compute_roi_displacements(
    *,
    aligned_xy: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_ms: np.ndarray,
    neutral_idx: int,
    peak_idx: int,
    roi_groups: dict[str, tuple[list[int], list[int]]] | None = None,
    presence: np.ndarray | None = None,
    normalize: bool = True,
    interocular_pair: tuple[int, int] = (33, 263),
    fps: float | None = None,
) -> dict[str, Any]:
    aligned_xy = np.asarray(aligned_xy, dtype=np.float32)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64)
    presence_arr = np.asarray(presence, dtype=bool) if presence is not None else None

    if aligned_xy.ndim != 3 or aligned_xy.shape[2] != 2:
        raise ValueError("aligned_xy must have shape (T, N, 2)")
    t_count, landmark_count, _ = aligned_xy.shape
    if frame_indices.shape[0] != t_count or timestamps_ms.shape[0] != t_count:
        raise ValueError("frame_indices and timestamps_ms must match aligned_xy length")
    if presence_arr is not None and presence_arr.shape[0] != t_count:
        raise ValueError("presence must match aligned_xy length")
    if neutral_idx < 0 or neutral_idx >= t_count:
        raise ValueError(f"neutral_idx out of range: {neutral_idx}")
    if peak_idx < 0 or peak_idx >= t_count:
        raise ValueError(f"peak_idx out of range: {peak_idx}")

    groups = roi_groups or ROI_GROUPS
    for roi_name, (left_indices, right_indices) in groups.items():
        for idx in left_indices + right_indices:
            if idx < 0 or idx >= landmark_count:
                raise ValueError(f"ROI '{roi_name}' contains out-of-range landmark index: {idx}")

    if neutral_idx <= peak_idx:
        row_indices = np.arange(neutral_idx, peak_idx + 1, dtype=np.int64)
        reversed_flag = False
    else:
        row_indices = np.arange(neutral_idx, peak_idx - 1, -1, dtype=np.int64)
        reversed_flag = True

    seq_xy = aligned_xy[row_indices]
    seq_presence = presence_arr[row_indices] if presence_arr is not None else None
    neutral_xy = aligned_xy[neutral_idx]

    interocular_px: float | None = None
    denom = 1.0
    if normalize:
        a_idx, b_idx = interocular_pair
        if a_idx >= landmark_count or b_idx >= landmark_count:
            raise ValueError("interocular_pair index is out of range for aligned landmarks")
        point_a = neutral_xy[a_idx]
        point_b = neutral_xy[b_idx]
        if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
            raise ValueError("Cannot normalize: neutral interocular points contain NaN")
        interocular_px = float(np.linalg.norm(point_a - point_b))
        if interocular_px <= EPSILON:
            raise ValueError(
                f"Cannot normalize: interocular distance is too small ({interocular_px})"
            )
        denom = interocular_px

    fps_value = float(fps) if (fps is not None and fps > 0) else None
    if fps_value is not None:
        time_s = (np.arange(row_indices.shape[0], dtype=np.float32) / fps_value).astype(np.float32)
    else:
        time_s = np.arange(row_indices.shape[0], dtype=np.float32)

    series: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for roi_name, (left_indices, right_indices) in groups.items():
        left = _compute_side_series(
            sequence_xy=seq_xy,
            neutral_xy=neutral_xy,
            indices=left_indices,
            sequence_presence=seq_presence,
            normalize=normalize,
            denom=denom,
        )
        right = _compute_side_series(
            sequence_xy=seq_xy,
            neutral_xy=neutral_xy,
            indices=right_indices,
            sequence_presence=seq_presence,
            normalize=normalize,
            denom=denom,
        )
        series[roi_name] = {
            "left_disp": left["mean_disp"],
            "right_disp": right["mean_disp"],
            "left_dx": left["mean_dx"],
            "left_dy": left["mean_dy"],
            "right_dx": right["mean_dx"],
            "right_dy": right["mean_dy"],
        }

        left_peak = float(left["mean_disp"][-1]) if left["mean_disp"].size else float("nan")
        right_peak = float(right["mean_disp"][-1]) if right["mean_disp"].size else float("nan")
        if np.isfinite(left_peak) and np.isfinite(right_peak):
            mean_lr = 0.5 * (left_peak + right_peak)
            ai = abs(left_peak - right_peak) / (mean_lr + EPSILON)
        else:
            ai = float("nan")
        metrics[roi_name] = {"L_peak": left_peak, "R_peak": right_peak, "AI": float(ai)}

    return {
        "row_indices": row_indices,
        "frame_indices": frame_indices[row_indices],
        "timestamps_ms": timestamps_ms[row_indices],
        "time_s": time_s,
        "neutral_idx": int(neutral_idx),
        "peak_idx": int(peak_idx),
        "reversed": bool(reversed_flag),
        "normalize": bool(normalize),
        "interocular_px": interocular_px,
        "series": series,
        "metrics": metrics,
    }


def _save_roi_plot(
    *,
    output_path: Path,
    roi_name: str,
    time_s: np.ndarray,
    left_disp: np.ndarray,
    right_disp: np.ndarray,
    normalize: bool,
    reversed_flag: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_s, left_disp, color="#d62728", linewidth=2.0, label="Left")
    plt.plot(time_s, right_disp, color="#2ca02c", linewidth=2.0, label="Right")
    title_suffix = " (reversed neutral->peak)" if reversed_flag else ""
    plt.title(f"{roi_name.capitalize()} displacement{title_suffix}")
    plt.xlabel("time_s from neutral")
    ylabel = "mean displacement (normalized)" if normalize else "mean displacement (pixel)"
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_timeseries_csv(
    *,
    output_path: Path,
    roi_names: list[str],
    result: dict[str, Any],
) -> None:
    fieldnames = ["time_s", "frame_idx", "timestamp_ms"]
    for roi_name in roi_names:
        fieldnames.extend(
            [
                f"{roi_name}_L",
                f"{roi_name}_R",
                f"{roi_name}_L_dx",
                f"{roi_name}_L_dy",
                f"{roi_name}_R_dx",
                f"{roi_name}_R_dy",
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        frame_indices = result["frame_indices"]
        timestamps_ms = result["timestamps_ms"]
        time_s = result["time_s"]
        for idx in range(time_s.shape[0]):
            row: dict[str, Any] = {
                "time_s": float(time_s[idx]),
                "frame_idx": int(frame_indices[idx]),
                "timestamp_ms": int(timestamps_ms[idx]),
            }
            for roi_name in roi_names:
                series = result["series"][roi_name]
                row[f"{roi_name}_L"] = float(series["left_disp"][idx])
                row[f"{roi_name}_R"] = float(series["right_disp"][idx])
                row[f"{roi_name}_L_dx"] = float(series["left_dx"][idx])
                row[f"{roi_name}_L_dy"] = float(series["left_dy"][idx])
                row[f"{roi_name}_R_dx"] = float(series["right_dx"][idx])
                row[f"{roi_name}_R_dy"] = float(series["right_dy"][idx])
            writer.writerow(row)


def _save_metrics_csv(
    *,
    output_path: Path,
    roi_names: list[str],
    result: dict[str, Any],
    task: str,
) -> None:
    fieldnames = [
        "roi",
        "L_peak",
        "R_peak",
        "AI",
        "neutral_frame_idx",
        "peak_frame_idx",
        "neutral_idx",
        "peak_idx",
        "reversed",
        "normalize",
        "interocular_px",
        "task",
    ]
    neutral_frame_idx = int(result["frame_indices"][0])
    peak_frame_idx = int(result["frame_indices"][-1])

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for roi_name in roi_names:
            metric = result["metrics"][roi_name]
            writer.writerow(
                {
                    "roi": roi_name,
                    "L_peak": metric["L_peak"],
                    "R_peak": metric["R_peak"],
                    "AI": metric["AI"],
                    "neutral_frame_idx": neutral_frame_idx,
                    "peak_frame_idx": peak_frame_idx,
                    "neutral_idx": int(result["neutral_idx"]),
                    "peak_idx": int(result["peak_idx"]),
                    "reversed": bool(result["reversed"]),
                    "normalize": bool(result["normalize"]),
                    "interocular_px": result["interocular_px"],
                    "task": task,
                }
            )


def _build_metrics_json(
    *,
    task: str,
    roi_names: list[str],
    result: dict[str, Any],
    segment_payload: dict[str, Any],
    aligned_path: Path,
) -> dict[str, Any]:
    metrics_payload: dict[str, Any] = {}
    for roi_name in roi_names:
        metrics_payload[roi_name] = result["metrics"][roi_name]

    return {
        "task": task,
        "rois": roi_names,
        "neutral_idx": int(result["neutral_idx"]),
        "peak_idx": int(result["peak_idx"]),
        "neutral_frame_idx": int(result["frame_indices"][0]),
        "peak_frame_idx": int(result["frame_indices"][-1]),
        "reversed": bool(result["reversed"]),
        "normalize": bool(result["normalize"]),
        "interocular_px": result["interocular_px"],
        "segment_flags": segment_payload.get("flags", {}),
        "metrics": metrics_payload,
        "aligned_path": str(aligned_path),
    }


def run_metrics(
    *,
    video_path: str | Path,
    task: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    aligned_path: str | Path | None = None,
    segment_path: str | Path | None = None,
    rois: str = "all",
    normalize: bool = True,
) -> dict[str, Any]:
    task_name = _normalize_task(task)
    roi_names = parse_rois_option(rois, task_name)
    roi_groups = {name: ROI_GROUPS[name] for name in roi_names}

    artifact_dir = _resolve_artifact_dir(
        video_path=video_path,
        artifact_root=artifact_root,
        aligned_path=aligned_path,
        segment_path=segment_path,
    )
    resolved_aligned = (
        Path(aligned_path) if aligned_path is not None else artifact_dir / "landmarks_aligned.npz"
    )
    resolved_segment = (
        Path(segment_path) if segment_path is not None else artifact_dir / "segment.json"
    )

    if not resolved_aligned.exists():
        raise FileNotFoundError(_missing_aligned_message(resolved_aligned, video_path))
    if not resolved_segment.exists():
        raise FileNotFoundError(_missing_segment_message(resolved_segment, video_path))

    aligned = _load_aligned_arrays(resolved_aligned)
    segment_payload = _load_segment(resolved_segment, frame_count=aligned["frame_indices"].shape[0])
    fps = _estimate_fps_from_timestamps(aligned["timestamps_ms"])

    result = compute_roi_displacements(
        aligned_xy=aligned["landmarks_xy_aligned"],
        frame_indices=aligned["frame_indices"],
        timestamps_ms=aligned["timestamps_ms"],
        neutral_idx=int(segment_payload["neutral_idx"]),
        peak_idx=int(segment_payload["peak_idx"]),
        roi_groups=roi_groups,
        presence=aligned["presence"],
        normalize=normalize,
        interocular_pair=(33, 263),
        fps=fps,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: dict[str, str] = {}
    for roi_name in roi_names:
        plot_path = plots_dir / f"{roi_name}.png"
        _save_roi_plot(
            output_path=plot_path,
            roi_name=roi_name,
            time_s=result["time_s"],
            left_disp=result["series"][roi_name]["left_disp"],
            right_disp=result["series"][roi_name]["right_disp"],
            normalize=normalize,
            reversed_flag=bool(result["reversed"]),
        )
        plot_paths[roi_name] = str(plot_path)

    timeseries_path = artifact_dir / "timeseries.csv"
    _save_timeseries_csv(output_path=timeseries_path, roi_names=roi_names, result=result)

    metrics_csv_path = artifact_dir / "metrics.csv"
    _save_metrics_csv(
        output_path=metrics_csv_path, roi_names=roi_names, result=result, task=task_name
    )

    metrics_json_payload = _build_metrics_json(
        task=task_name,
        roi_names=roi_names,
        result=result,
        segment_payload=segment_payload,
        aligned_path=resolved_aligned,
    )
    metrics_json_path = artifact_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_json_payload, indent=2), encoding="utf-8")

    return {
        "task": task_name,
        "rois": roi_names,
        "plots_dir": str(plots_dir),
        "plot_paths": plot_paths,
        "timeseries_csv_path": str(timeseries_path),
        "metrics_csv_path": str(metrics_csv_path),
        "metrics_json_path": str(metrics_json_path),
        "reversed": bool(result["reversed"]),
        "normalize": bool(result["normalize"]),
    }
