from __future__ import annotations

import numpy as np

from faceanalyze2.analysis.metrics import compute_roi_displacements


def _build_synthetic_xy() -> np.ndarray:
    aligned_xy = np.zeros((5, 8, 2), dtype=np.float32)
    aligned_xy[:, 0, :] = np.asarray([0.0, 0.0], dtype=np.float32)
    aligned_xy[:, 1, :] = np.asarray([2.0, 0.0], dtype=np.float32)

    offsets = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    aligned_xy[:, 2, 0] = 10.0 + offsets
    aligned_xy[:, 2, 1] = 5.0
    aligned_xy[:, 3, 0] = 20.0 + (0.5 * offsets)
    aligned_xy[:, 3, 1] = 7.0
    return aligned_xy


def test_compute_roi_displacements_expected_values() -> None:
    aligned_xy = _build_synthetic_xy()
    frame_indices = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    timestamps_ms = np.asarray([0, 100, 200, 300, 400], dtype=np.int64)

    result = compute_roi_displacements(
        aligned_xy=aligned_xy,
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        neutral_idx=1,
        peak_idx=4,
        roi_groups={"mouth": ([2], [3])},
        presence=np.ones((5,), dtype=bool),
        normalize=True,
        interocular_pair=(0, 1),
        fps=10.0,
    )

    mouth = result["series"]["mouth"]
    assert np.allclose(mouth["left_disp"], np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32))
    assert np.allclose(mouth["right_disp"], np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float32))
    assert np.isclose(mouth["left_dx"][-1], 1.5, atol=1e-6)
    assert np.isclose(mouth["right_dx"][-1], 0.75, atol=1e-6)
    assert result["reversed"] is False
    assert np.array_equal(result["frame_indices"], np.asarray([11, 12, 13, 14], dtype=np.int64))
    assert np.isclose(result["metrics"]["mouth"]["AI"], 2.0 / 3.0, atol=1e-6)
    assert np.isclose(result["metrics"]["mouth"]["score"], 100.0 / 3.0, atol=1e-6)


def test_compute_roi_displacements_reversed_case_and_presence_mask() -> None:
    aligned_xy = _build_synthetic_xy()
    frame_indices = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    timestamps_ms = np.asarray([0, 100, 200, 300, 400], dtype=np.int64)
    presence = np.asarray([True, True, False, True, True], dtype=bool)

    result = compute_roi_displacements(
        aligned_xy=aligned_xy,
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        neutral_idx=4,
        peak_idx=1,
        roi_groups={"mouth": ([2], [3])},
        presence=presence,
        normalize=False,
        interocular_pair=(0, 1),
        fps=10.0,
    )

    mouth = result["series"]["mouth"]
    assert result["reversed"] is True
    assert np.array_equal(result["frame_indices"], np.asarray([14, 13, 12, 11], dtype=np.int64))
    assert np.isclose(mouth["left_disp"][0], 0.0, atol=1e-6)
    assert np.isclose(mouth["left_disp"][1], 1.0, atol=1e-6)
    assert np.isnan(mouth["left_disp"][2])
    assert np.isclose(mouth["left_disp"][3], 3.0, atol=1e-6)
    assert np.isclose(result["metrics"]["mouth"]["L_peak"], 3.0, atol=1e-6)
    assert np.isclose(result["metrics"]["mouth"]["R_peak"], 1.5, atol=1e-6)
    assert np.isclose(result["metrics"]["mouth"]["score"], 100.0 / 3.0, atol=1e-6)
