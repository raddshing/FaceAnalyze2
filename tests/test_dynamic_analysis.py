from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faceanalyze2.api import dynamicAnalysis
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


def _make_synthetic_artifacts(tmp_path: Path, *, stem: str = "sample", with_aligned: bool = True) -> Path:
    artifact_dir = tmp_path / "artifacts" / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_count = 5
    n_landmarks = 478
    width, height = 640, 480
    frame_indices = np.arange(t_count, dtype=np.int64)
    timestamps_ms = frame_indices * 33
    presence = np.ones((t_count,), dtype=bool)

    idx = np.arange(n_landmarks, dtype=np.float32)
    base = np.zeros((n_landmarks, 2), dtype=np.float32)
    base[:, 0] = 80.0 + 480.0 * ((idx % 40) / 39.0)
    base[:, 1] = 60.0 + 320.0 * ((idx // 40) / 11.0)

    aligned = np.repeat(base[None, :, :], t_count, axis=0).astype(np.float32)
    motion_scale = np.linspace(0.0, 1.0, num=t_count, dtype=np.float32)

    mouth_l, mouth_r = resolve_pair_indices(ROI_MOUTH)
    eye_l, eye_r = resolve_pair_indices(ROI_EYE)
    brow_l, brow_r = resolve_pair_indices(ROI_EYEBROW)
    area0_l, area0_r = resolve_pair_indices(AREA0_GREEN)
    area1_l, area1_r = resolve_pair_indices(AREA1_BLUE)
    area2_l, area2_r = resolve_pair_indices(AREA2_YELLOW)
    area3_l, area3_r = resolve_pair_indices(AREA3_RED)

    for t_idx, scale in enumerate(motion_scale):
        aligned[t_idx, mouth_l, 0] += 6.0 * scale
        aligned[t_idx, mouth_r, 0] += 2.5 * scale
        aligned[t_idx, eye_l, 1] -= 3.0 * scale
        aligned[t_idx, eye_r, 1] -= 1.5 * scale
        aligned[t_idx, brow_l, 1] -= 4.0 * scale
        aligned[t_idx, brow_r, 1] -= 2.0 * scale
        aligned[t_idx, area0_l, :] += np.asarray([2.0 * scale, -1.0 * scale], dtype=np.float32)
        aligned[t_idx, area0_r, :] += np.asarray([1.0 * scale, -0.5 * scale], dtype=np.float32)
        aligned[t_idx, area1_l, :] += np.asarray([3.0 * scale, -1.2 * scale], dtype=np.float32)
        aligned[t_idx, area1_r, :] += np.asarray([1.5 * scale, -0.8 * scale], dtype=np.float32)
        aligned[t_idx, area2_l, :] += np.asarray([1.0 * scale, 1.5 * scale], dtype=np.float32)
        aligned[t_idx, area2_r, :] += np.asarray([0.5 * scale, 0.8 * scale], dtype=np.float32)
        aligned[t_idx, area3_l, :] += np.asarray([2.2 * scale, -1.8 * scale], dtype=np.float32)
        aligned[t_idx, area3_r, :] += np.asarray([1.1 * scale, -1.0 * scale], dtype=np.float32)

    normalized_xyz = np.zeros((t_count, n_landmarks, 3), dtype=np.float32)
    normalized_xyz[:, :, 0] = aligned[:, :, 0] / float(width)
    normalized_xyz[:, :, 1] = aligned[:, :, 1] / float(height)

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        landmarks_xyz=normalized_xyz,
        presence=presence,
    )
    if with_aligned:
        np.savez_compressed(
            artifact_dir / "landmarks_aligned.npz",
            frame_indices=frame_indices,
            timestamps_ms=timestamps_ms,
            presence=presence,
            landmarks_xy_aligned=aligned,
        )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": width, "height": height, "frame_count": t_count}),
        encoding="utf-8",
    )
    (artifact_dir / "segment.json").write_text(
        json.dumps({"neutral_idx": 0, "peak_idx": 4}),
        encoding="utf-8",
    )
    return artifact_dir


def test_dynamic_analysis_big_smile_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_synthetic_artifacts(tmp_path, stem="sample", with_aligned=True)
    monkeypatch.chdir(tmp_path)

    payload = dynamicAnalysis(tmp_path / "sample.mp4", "big-smile")

    for key in ("key rest", "key exp", "key value graph", "before regi", "after regi", "metrics"):
        assert key in payload

    for key in ("key rest", "key exp", "key value graph", "before regi", "after regi"):
        assert isinstance(payload[key], str)
        assert len(payload[key]) > 20
        assert not payload[key].startswith("data:image")

    metrics = payload["metrics"]
    assert metrics["motion"] == "big-smile"
    roi_metrics = metrics["roi_metrics"]
    for roi_name in ("mouth", "area0_green", "area1_blue", "area2_yellow", "area3_red"):
        assert roi_name in roi_metrics
        assert "L_peak" in roi_metrics[roi_name]
        assert "R_peak" in roi_metrics[roi_name]
        assert "AI" in roi_metrics[roi_name]
        assert "score" in roi_metrics[roi_name]


def test_dynamic_analysis_falls_back_to_landmarks_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_synthetic_artifacts(tmp_path, stem="sample", with_aligned=False)
    monkeypatch.chdir(tmp_path)

    payload = dynamicAnalysis(tmp_path / "sample.mp4", "blinking-motion")
    assert payload["metrics"]["motion"] == "blinking-motion"
    assert payload["metrics"]["source"] == "landmarks.npz"
    assert "eye" in payload["metrics"]["roi_metrics"]


def test_dynamic_analysis_rejects_unsupported_motion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_synthetic_artifacts(tmp_path, stem="sample", with_aligned=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        dynamicAnalysis(tmp_path / "sample.mp4", "unknown-motion")

    assert "Unsupported motion" in str(exc_info.value)
