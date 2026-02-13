from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from faceanalyze2.analysis.alignment import estimate_similarity_transform, run_alignment


def test_estimate_similarity_transform_recovers_known_transform() -> None:
    src = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.5],
            [1.0, 2.0],
            [3.0, 1.5],
        ],
        dtype=np.float32,
    )
    theta = np.deg2rad(21.0)
    true_rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    true_scale = 1.75
    true_translation = np.asarray([14.0, -6.0], dtype=np.float32)
    dst = (true_scale * (true_rotation @ src.T)).T + true_translation

    estimated_scale, estimated_rotation, estimated_translation = estimate_similarity_transform(src, dst)
    recovered = (estimated_scale * (estimated_rotation @ src.T)).T + estimated_translation

    assert np.isclose(estimated_scale, true_scale, rtol=1e-5, atol=1e-5)
    assert np.allclose(estimated_rotation, true_rotation, atol=1e-5)
    assert np.allclose(estimated_translation, true_translation, atol=1e-4)
    assert np.allclose(recovered, dst, atol=1e-4)


def test_run_alignment_handles_presence_false_and_nan(tmp_path: Path) -> None:
    width = 640
    height = 480
    t_count = 4
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    landmarks_xyz = np.full((t_count, 478, 3), np.nan, dtype=np.float32)
    presence = np.asarray([True, True, False, True], dtype=bool)
    frame_indices = np.arange(t_count, dtype=np.int64)
    timestamps_ms = np.arange(t_count, dtype=np.int64) * 33

    neutral_points = {
        33: (0.38, 0.40, 0.01),
        133: (0.44, 0.40, 0.02),
        263: (0.56, 0.40, 0.01),
        362: (0.62, 0.40, 0.02),
        10: (0.50, 0.52, 0.03),
        20: (0.48, 0.57, 0.04),
    }
    for idx, (x, y, z) in neutral_points.items():
        landmarks_xyz[0, idx] = np.asarray([x, y, z], dtype=np.float32)

    theta = np.deg2rad(8.0)
    rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    scale = 1.15
    translation = np.asarray([11.0, -7.0], dtype=np.float32)

    for idx, (x, y, z) in neutral_points.items():
        source_pixel = np.asarray([x * width, y * height], dtype=np.float32)
        transformed_pixel = (scale * (rotation @ source_pixel)) + translation
        landmarks_xyz[1, idx] = np.asarray(
            [transformed_pixel[0] / width, transformed_pixel[1] / height, z],
            dtype=np.float32,
        )

    for idx, (x, y, z) in neutral_points.items():
        landmarks_xyz[3, idx] = np.asarray([x, y, z], dtype=np.float32)
    landmarks_xyz[3, 33] = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        landmarks_xyz=landmarks_xyz,
        presence=presence,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": width, "height": height, "frame_count": t_count}),
        encoding="utf-8",
    )
    (artifact_dir / "segment.json").write_text(
        json.dumps({"neutral_idx": 0}),
        encoding="utf-8",
    )

    result = run_alignment(video_path=tmp_path / "sample.mp4", artifact_root=artifact_root)
    aligned_npz_path = Path(result["aligned_npz_path"])
    alignment_json_path = Path(result["alignment_json_path"])

    assert aligned_npz_path.exists()
    assert alignment_json_path.exists()

    with np.load(aligned_npz_path) as aligned:
        assert aligned["landmarks_xy_aligned"].shape == (t_count, 478, 2)
        assert aligned["transform_scale"].shape == (t_count,)
        assert aligned["transform_R"].shape == (t_count, 2, 2)
        assert aligned["transform_t"].shape == (t_count, 2)

        stable_ref = aligned["stable_xy_ref"]
        stable_t_frame1 = aligned["stable_xy_t"][1]
        assert np.allclose(stable_t_frame1, stable_ref, atol=1e-2)

        assert np.isnan(aligned["landmarks_xy_aligned"][2]).all()
        assert np.isnan(aligned["transform_scale"][2])
        assert aligned["invalid_stable_mask"][3]

    payload = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    quality = payload["quality_stats"]
    assert quality["invalid_presence_frames"] == 1
    assert quality["invalid_stable_frames"] == 1
    assert quality["transformed_frames"] == 2
