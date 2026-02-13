from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faceanalyze2.roi.indices import (
    INNER_LIP,
    LEFT_BROW,
    LEFT_EYE,
    MOUTH_CORNERS,
    RIGHT_BROW,
    RIGHT_EYE,
)

DEFAULT_ARTIFACT_ROOT = Path("artifacts")
REQUIRED_LANDMARK_KEYS = {"timestamps_ms", "frame_indices", "landmarks_xyz", "presence"}
LANDMARK_COUNT = 478
VALID_TASKS = {"smile", "brow", "eyeclose"}


@dataclass(frozen=True)
class LoadedLandmarks:
    npz_path: Path
    meta_path: Path
    artifact_dir: Path
    timestamps_ms: np.ndarray
    frame_indices: np.ndarray
    landmarks_xyz: np.ndarray
    presence: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class SegmentParams:
    median_window: int = 5
    moving_average_window: int = 9
    pre_seconds: float = 2.0
    baseline_quantile: float = 0.25
    derivative_quantile: float = 0.35
    min_baseline_run_frames: int = 4
    onset_alpha: float = 0.2
    offset_alpha: float = 0.2
    hysteresis_frames: int = 3
    low_amp_threshold: float = 0.75


@dataclass(frozen=True)
class SegmentResult:
    neutral_idx: int
    peak_idx: int
    onset_idx: int
    offset_idx: int
    baseline_value: float
    peak_value: float
    amplitude: float
    flags: dict[str, Any]


def _normalize_task(task: Any) -> str:
    if hasattr(task, "value"):
        task_name = str(task.value)
    else:
        task_name = str(task)
    if task_name not in VALID_TASKS:
        valid = ", ".join(sorted(VALID_TASKS))
        raise ValueError(f"Unsupported task '{task_name}'. Expected one of: {valid}")
    return task_name


def _resolve_landmark_paths(
    *,
    video_path: str | Path,
    landmarks_path: str | Path | None,
    artifact_root: str | Path,
) -> tuple[Path, Path, Path]:
    video = Path(video_path)
    if landmarks_path is None:
        artifact_dir = Path(artifact_root) / video.stem
        npz_path = artifact_dir / "landmarks.npz"
    else:
        npz_path = Path(landmarks_path)
        artifact_dir = npz_path.parent

    meta_path = artifact_dir / "meta.json"
    return npz_path, meta_path, artifact_dir


def _missing_landmarks_message(npz_path: Path, video_path: str | Path) -> str:
    return (
        f"Landmarks file not found: {npz_path}\n"
        f"Run this first:\nfaceanalyze2 landmarks extract --video \"{video_path}\""
    )


def _validate_arrays(npz_path: Path, arrays: dict[str, np.ndarray]) -> None:
    missing_keys = REQUIRED_LANDMARK_KEYS.difference(arrays)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"landmarks.npz is missing required keys: {missing}")

    timestamps_ms = np.asarray(arrays["timestamps_ms"])
    frame_indices = np.asarray(arrays["frame_indices"])
    landmarks_xyz = np.asarray(arrays["landmarks_xyz"])
    presence = np.asarray(arrays["presence"])

    if timestamps_ms.ndim != 1:
        raise ValueError(f"timestamps_ms must be 1D in {npz_path}")
    if frame_indices.ndim != 1:
        raise ValueError(f"frame_indices must be 1D in {npz_path}")
    if presence.ndim != 1:
        raise ValueError(f"presence must be 1D in {npz_path}")
    if landmarks_xyz.ndim != 3 or landmarks_xyz.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(
            f"landmarks_xyz must have shape (T, {LANDMARK_COUNT}, 3) in {npz_path}, "
            f"got {landmarks_xyz.shape}"
        )

    expected_t = timestamps_ms.shape[0]
    if frame_indices.shape[0] != expected_t:
        raise ValueError("frame_indices length must match timestamps_ms length")
    if presence.shape[0] != expected_t:
        raise ValueError("presence length must match timestamps_ms length")
    if landmarks_xyz.shape[0] != expected_t:
        raise ValueError("landmarks_xyz length must match timestamps_ms length")


def load_landmark_artifacts(
    *,
    video_path: str | Path,
    landmarks_path: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> LoadedLandmarks:
    npz_path, meta_path, artifact_dir = _resolve_landmark_paths(
        video_path=video_path,
        landmarks_path=landmarks_path,
        artifact_root=artifact_root,
    )
    if not npz_path.exists():
        raise FileNotFoundError(_missing_landmarks_message(npz_path, video_path))
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}
    _validate_arrays(npz_path, arrays)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for field in ("fps", "width", "height"):
        if field not in meta:
            raise ValueError(f"meta.json is missing required field: {field}")
    if float(meta["fps"]) <= 0:
        raise ValueError("meta.json contains invalid fps")
    if int(meta["width"]) <= 0 or int(meta["height"]) <= 0:
        raise ValueError("meta.json contains invalid width/height")

    return LoadedLandmarks(
        npz_path=npz_path,
        meta_path=meta_path,
        artifact_dir=artifact_dir,
        timestamps_ms=np.asarray(arrays["timestamps_ms"], dtype=np.int64),
        frame_indices=np.asarray(arrays["frame_indices"], dtype=np.int64),
        landmarks_xyz=np.asarray(arrays["landmarks_xyz"], dtype=np.float32),
        presence=np.asarray(arrays["presence"], dtype=bool),
        meta=meta,
    )


def _landmarks_to_pixel_xy(landmarks_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(landmarks_xyz[..., :2], dtype=np.float32).copy()
    xy[..., 0] *= float(width)
    xy[..., 1] *= float(height)
    return xy


def _pairwise_distance(pixel_xy: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    a = pixel_xy[:, idx_a, :]
    b = pixel_xy[:, idx_b, :]
    dist = np.linalg.norm(a - b, axis=1)
    invalid = (~np.isfinite(a).all(axis=1)) | (~np.isfinite(b).all(axis=1))
    dist[invalid] = np.nan
    return dist


def _polygon_area(pixel_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    poly = pixel_xy[:, indices, :]
    valid = np.isfinite(poly).all(axis=(1, 2))
    x = poly[:, :, 0]
    y = poly[:, :, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    area = 0.5 * np.abs(np.sum((x * y_next) - (y * x_next), axis=1))
    area[~valid] = np.nan
    return area


def _mean_y(pixel_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    y = pixel_xy[:, indices, 1]
    with np.errstate(invalid="ignore"):
        return np.nanmean(y, axis=1)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    out[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return out

    clean = values[finite]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = 1.4826 * mad
    if scale <= 1e-6:
        std = float(np.std(clean))
        scale = std if std > 1e-6 else 1.0

    z = (clean - median) / scale
    out[finite] = np.clip(z, -8.0, 8.0).astype(np.float32)
    return out


def compute_task_signal(task: str, pixel_xy: np.ndarray) -> np.ndarray:
    task_name = _normalize_task(task)
    interocular = _pairwise_distance(pixel_xy, 33, 263)

    if task_name == "smile":
        width_norm = _safe_divide(_pairwise_distance(pixel_xy, MOUTH_CORNERS[0], MOUTH_CORNERS[1]), interocular)
        inner_area = _polygon_area(pixel_xy, INNER_LIP)
        area_norm = _safe_divide(inner_area, np.square(interocular))
        area_norm = np.where(area_norm >= 0, area_norm, np.nan)
        return (robust_z(width_norm) + robust_z(np.sqrt(area_norm))).astype(np.float32)

    if task_name == "eyeclose":
        left_eye_area = _polygon_area(pixel_xy, LEFT_EYE)
        right_eye_area = _polygon_area(pixel_xy, RIGHT_EYE)
        eye_area_mean = np.nanmean(np.stack([left_eye_area, right_eye_area], axis=0), axis=0)
        eye_area_norm = _safe_divide(eye_area_mean, np.square(interocular))
        max_area = np.nanmax(eye_area_norm) if np.isfinite(eye_area_norm).any() else np.nan
        inverted = max_area - eye_area_norm
        return robust_z(inverted).astype(np.float32)

    brow_y = np.nanmean(np.stack([_mean_y(pixel_xy, LEFT_BROW), _mean_y(pixel_xy, RIGHT_BROW)], axis=0), axis=0)
    eye_y = _mean_y(pixel_xy, [33, 133, 263, 362])
    gap_norm = _safe_divide(eye_y - brow_y, interocular)
    return robust_z(gap_norm).astype(np.float32)


def _ensure_odd(window: int) -> int:
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = values.shape[0]
    if n == 0:
        return values.copy()

    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    if finite.all():
        return values.copy()

    idx = np.arange(n, dtype=np.float32)
    return np.interp(idx, idx[finite], values[finite]).astype(np.float32)


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = _ensure_odd(window)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[0]):
        out[i] = float(np.median(padded[i : i + window]))
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = _ensure_odd(window)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _robust_clip(values: np.ndarray, z_limit: float = 6.0) -> np.ndarray:
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-6 else max(float(np.nanstd(values)), 1.0)
    low = median - z_limit * scale
    high = median + z_limit * scale
    return np.clip(values, low, high).astype(np.float32)


def smooth_signal(raw_signal: np.ndarray, params: SegmentParams) -> np.ndarray:
    filled = _interpolate_nans(raw_signal)
    medianed = _median_filter_1d(filled, params.median_window)
    clipped = _robust_clip(medianed)
    return _moving_average(clipped, params.moving_average_window)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for idx, value in enumerate(mask):
        if value and run_start is None:
            run_start = idx
        if (not value) and run_start is not None:
            candidate = (run_start, idx)
            if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                best = candidate
            run_start = None
    if run_start is not None:
        candidate = (run_start, mask.shape[0])
        if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
            best = candidate
    return best


def _find_baseline_candidate(
    signal: np.ndarray,
    derivative: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: SegmentParams,
) -> tuple[int, float] | None:
    if end_idx - start_idx < params.min_baseline_run_frames:
        return None

    window_signal = signal[start_idx:end_idx]
    window_derivative = derivative[start_idx:end_idx]
    low_value = float(np.quantile(window_signal, params.baseline_quantile))
    low_derivative = float(np.quantile(window_derivative, params.derivative_quantile))
    mask = (window_signal <= low_value) & (window_derivative <= low_derivative)
    run = _longest_true_run(mask)
    if run is None or (run[1] - run[0]) < params.min_baseline_run_frames:
        return None

    run_indices = np.arange(start_idx + run[0], start_idx + run[1])
    baseline_value = float(np.median(signal[run_indices]))
    nearest = np.argmin(np.abs(signal[run_indices] - baseline_value))
    baseline_idx = int(run_indices[nearest])
    return baseline_idx, baseline_value


def _find_first_sustained_above(
    signal: np.ndarray,
    threshold: float,
    start_idx: int,
    end_idx: int,
    min_len: int,
) -> int | None:
    min_len = max(1, min_len)
    max_start = end_idx - min_len
    for idx in range(start_idx, max_start + 1):
        if np.all(signal[idx : idx + min_len] >= threshold):
            return idx
    return None


def _find_first_sustained_below(
    signal: np.ndarray,
    threshold: float,
    start_idx: int,
    end_idx: int,
    min_len: int,
) -> int | None:
    min_len = max(1, min_len)
    max_start = end_idx - min_len
    for idx in range(start_idx, max_start + 1):
        if np.all(signal[idx : idx + min_len] <= threshold):
            return idx
    return None


def detect_segments_from_signal(
    signal_smooth: np.ndarray,
    *,
    fps: float,
    params: SegmentParams,
    missing_ratio: float,
) -> SegmentResult:
    if signal_smooth.size == 0:
        return SegmentResult(
            neutral_idx=0,
            peak_idx=0,
            onset_idx=0,
            offset_idx=0,
            baseline_value=0.0,
            peak_value=0.0,
            amplitude=0.0,
            flags={
                "low_amp": True,
                "neutral_after_peak": False,
                "missing_ratio": float(missing_ratio),
                "baseline_fallback_full": True,
            },
        )

    signal = _interpolate_nans(signal_smooth)
    derivative = np.abs(np.diff(signal, prepend=signal[0]))
    initial_peak_idx = int(np.argmax(signal))
    pre_frames = max(params.min_baseline_run_frames, int(round(params.pre_seconds * float(fps))))
    pre_start = max(0, initial_peak_idx - pre_frames)
    pre_end = max(pre_start + 1, initial_peak_idx)

    baseline = _find_baseline_candidate(signal, derivative, pre_start, pre_end, params)
    used_full_fallback = False
    if baseline is None:
        baseline = _find_baseline_candidate(signal, derivative, 0, signal.shape[0], params)
        used_full_fallback = True

    if baseline is None:
        baseline_idx = int(np.argmin(signal))
        baseline_value = float(signal[baseline_idx])
        used_full_fallback = True
    else:
        baseline_idx, baseline_value = baseline

    peak_idx = int(np.argmax(signal - baseline_value))
    peak_value = float(signal[peak_idx])

    neutral_after_peak = baseline_idx > peak_idx
    if neutral_after_peak and peak_idx > 0:
        pre_peak = signal[: peak_idx + 1]
        baseline_idx = int(np.argmin(pre_peak))
        baseline_value = float(signal[baseline_idx])

    amplitude = max(0.0, peak_value - baseline_value)
    if amplitude <= 0:
        onset_idx = baseline_idx
        offset_idx = peak_idx
    else:
        onset_threshold = baseline_value + (params.onset_alpha * amplitude)
        offset_threshold = baseline_value + (params.offset_alpha * amplitude)
        onset_idx = _find_first_sustained_above(
            signal,
            threshold=onset_threshold,
            start_idx=0,
            end_idx=peak_idx + 1,
            min_len=params.hysteresis_frames,
        )
        if onset_idx is None:
            onset_idx = max(0, min(baseline_idx, peak_idx))

        offset_idx = _find_first_sustained_below(
            signal,
            threshold=offset_threshold,
            start_idx=peak_idx,
            end_idx=signal.shape[0],
            min_len=params.hysteresis_frames,
        )
        if offset_idx is None:
            offset_idx = signal.shape[0] - 1

    onset_idx = int(min(onset_idx, peak_idx))
    offset_idx = int(max(offset_idx, peak_idx))

    return SegmentResult(
        neutral_idx=int(baseline_idx),
        peak_idx=int(peak_idx),
        onset_idx=int(onset_idx),
        offset_idx=int(offset_idx),
        baseline_value=float(baseline_value),
        peak_value=float(peak_value),
        amplitude=float(amplitude),
        flags={
            "low_amp": bool(amplitude < params.low_amp_threshold),
            "neutral_after_peak": bool(neutral_after_peak),
            "missing_ratio": float(missing_ratio),
            "baseline_fallback_full": bool(used_full_fallback),
        },
    )


def _save_signals_csv(
    *,
    csv_path: Path,
    frame_indices: np.ndarray,
    timestamps_ms: np.ndarray,
    signal_raw: np.ndarray,
    signal_smooth: np.ndarray,
    task: str,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_idx", "timestamp_ms", "signal_raw", "signal_smooth", "task"])
        for idx in range(frame_indices.shape[0]):
            writer.writerow(
                [
                    int(frame_indices[idx]),
                    int(timestamps_ms[idx]),
                    float(signal_raw[idx]) if np.isfinite(signal_raw[idx]) else np.nan,
                    float(signal_smooth[idx]) if np.isfinite(signal_smooth[idx]) else np.nan,
                    task,
                ]
            )


def _save_signals_plot(
    *,
    plot_path: Path,
    frame_indices: np.ndarray,
    signal_raw: np.ndarray,
    signal_smooth: np.ndarray,
    segment: SegmentResult,
    task: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame_indices.astype(np.int64)
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, signal_raw, color="#8ca0b3", linewidth=1.2, alpha=0.55, label="signal_raw")
    plt.plot(x, signal_smooth, color="#1f4e79", linewidth=2.0, label="signal_smooth")

    markers = [
        ("neutral", segment.neutral_idx, "#2f9e44"),
        ("onset", segment.onset_idx, "#f08c00"),
        ("peak", segment.peak_idx, "#e03131"),
        ("offset", segment.offset_idx, "#7048e8"),
    ]
    for label, idx, color in markers:
        if 0 <= idx < x.shape[0]:
            plt.axvline(x=x[idx], color=color, linestyle="--", linewidth=1.2, label=label)

    plt.title(f"Signal Segmentation ({task})")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized signal")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _build_segment_payload(
    *,
    task: str,
    params: SegmentParams,
    segment: SegmentResult,
    frame_indices: np.ndarray,
    timestamps_ms: np.ndarray,
    npz_path: Path,
) -> dict[str, Any]:
    def _map(values: np.ndarray, idx: int) -> int | None:
        return int(values[idx]) if 0 <= idx < values.shape[0] else None

    payload: dict[str, Any] = {
        "task": task,
        "neutral_idx": segment.neutral_idx,
        "peak_idx": segment.peak_idx,
        "onset_idx": segment.onset_idx,
        "offset_idx": segment.offset_idx,
        "neutral_frame_idx": _map(frame_indices, segment.neutral_idx),
        "peak_frame_idx": _map(frame_indices, segment.peak_idx),
        "onset_frame_idx": _map(frame_indices, segment.onset_idx),
        "offset_frame_idx": _map(frame_indices, segment.offset_idx),
        "neutral_timestamp_ms": _map(timestamps_ms, segment.neutral_idx),
        "peak_timestamp_ms": _map(timestamps_ms, segment.peak_idx),
        "onset_timestamp_ms": _map(timestamps_ms, segment.onset_idx),
        "offset_timestamp_ms": _map(timestamps_ms, segment.offset_idx),
        "baseline_value": segment.baseline_value,
        "peak_value": segment.peak_value,
        "amplitude": segment.amplitude,
        "flags": segment.flags,
        "params": asdict(params),
        "landmarks_path": str(npz_path),
    }
    return payload


def run_segmentation(
    *,
    video_path: str | Path,
    task: str,
    landmarks_path: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    params: SegmentParams | None = None,
) -> dict[str, Any]:
    task_name = _normalize_task(task)
    params = params or SegmentParams()

    loaded = load_landmark_artifacts(
        video_path=video_path,
        landmarks_path=landmarks_path,
        artifact_root=artifact_root,
    )
    width = int(loaded.meta["width"])
    height = int(loaded.meta["height"])
    fps = float(loaded.meta["fps"])

    pixel_xy = _landmarks_to_pixel_xy(loaded.landmarks_xyz, width=width, height=height)
    signal_raw = compute_task_signal(task_name, pixel_xy)
    signal_smooth = smooth_signal(signal_raw, params=params)

    missing_mask = (~loaded.presence.astype(bool)) | (~np.isfinite(signal_raw))
    missing_ratio = float(np.mean(missing_mask)) if missing_mask.size else 1.0
    segment = detect_segments_from_signal(
        signal_smooth,
        fps=fps,
        params=params,
        missing_ratio=missing_ratio,
    )

    loaded.artifact_dir.mkdir(parents=True, exist_ok=True)
    signals_csv_path = loaded.artifact_dir / "signals.csv"
    segment_json_path = loaded.artifact_dir / "segment.json"
    plot_path = loaded.artifact_dir / "signals_plot.png"

    _save_signals_csv(
        csv_path=signals_csv_path,
        frame_indices=loaded.frame_indices,
        timestamps_ms=loaded.timestamps_ms,
        signal_raw=signal_raw,
        signal_smooth=signal_smooth,
        task=task_name,
    )
    segment_payload = _build_segment_payload(
        task=task_name,
        params=params,
        segment=segment,
        frame_indices=loaded.frame_indices,
        timestamps_ms=loaded.timestamps_ms,
        npz_path=loaded.npz_path,
    )
    segment_json_path.write_text(json.dumps(segment_payload, indent=2), encoding="utf-8")
    _save_signals_plot(
        plot_path=plot_path,
        frame_indices=loaded.frame_indices,
        signal_raw=signal_raw,
        signal_smooth=signal_smooth,
        segment=segment,
        task=task_name,
    )

    return {
        "signals_csv_path": str(signals_csv_path),
        "segment_json_path": str(segment_json_path),
        "plot_path": str(plot_path),
        "segment": segment_payload,
    }
