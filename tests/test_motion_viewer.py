from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from faceanalyze2.cli import app
from faceanalyze2.viz.motion_viewer import generate_motion_viewer

pytest.importorskip("mediapipe")

runner = CliRunner()


def _write_synthetic_artifacts(tmp_path: Path, stem: str = "sample") -> Path:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_count = 4
    n_landmarks = 478
    frame_indices = np.arange(t_count, dtype=np.int64)
    timestamps_ms = frame_indices * 33
    presence = np.ones((t_count,), dtype=bool)

    idx = np.arange(n_landmarks, dtype=np.float32)
    base = np.zeros((n_landmarks, 3), dtype=np.float32)
    base[:, 0] = 0.25 + 0.5 * ((idx % 26) / 25.0)
    base[:, 1] = 0.2 + 0.6 * ((idx // 26) / 19.0)
    base[:, 2] = 0.02 * np.sin(idx / 21.0)

    landmarks = np.zeros((t_count, n_landmarks, 3), dtype=np.float32)
    for t_idx in range(t_count):
        angle = np.deg2rad(2.0 * float(t_idx))
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        centered_xy = base[:, :2] - np.asarray([0.5, 0.5], dtype=np.float32)
        rotated_xy = (centered_xy @ rotation.T) + np.asarray(
            [0.5 + (0.01 * t_idx), 0.5 - (0.005 * t_idx)],
            dtype=np.float32,
        )

        frame = base.copy()
        frame[:, :2] = rotated_xy
        frame[:, 2] = base[:, 2] + (0.002 * t_idx)

        if t_idx == t_count - 1:
            frame[[61, 291], 0] += np.asarray([-0.02, 0.02], dtype=np.float32)
            frame[[61, 291], 1] -= 0.01
            frame[[13, 14], 1] -= 0.012

        landmarks[t_idx] = frame

    np.savez_compressed(
        artifact_dir / "landmarks.npz",
        frame_indices=frame_indices,
        timestamps_ms=timestamps_ms,
        landmarks_xyz=landmarks,
        presence=presence,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps({"fps": 30.0, "width": 640, "height": 480, "frame_count": t_count}),
        encoding="utf-8",
    )
    (artifact_dir / "segment.json").write_text(
        json.dumps({"neutral_idx": 0, "peak_idx": 3, "onset_idx": 1, "offset_idx": 3}),
        encoding="utf-8",
    )
    return artifact_root


def _write_dummy_video(video_path: Path, *, width: int = 640, height: int = 480, frame_count: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment")
    try:
        for frame_idx in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = np.uint8((40 * frame_idx) % 255)
            frame[..., 1] = np.uint8((90 + 20 * frame_idx) % 255)
            frame[..., 2] = np.uint8((160 + 10 * frame_idx) % 255)
            writer.write(frame)
    finally:
        writer.release()


def _load_payload_from_html(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    match = re.search(r"window\.MOTION_VIEWER_DATA=(.*?);\s*</script>", html, flags=re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_generate_motion_viewer_creates_html(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)
    result = generate_motion_viewer(video_path=tmp_path / "sample.mp4", artifact_root=artifact_root)
    html_path = Path(result["html_path"])

    assert html_path.exists()
    assert html_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    payload = _load_payload_from_html(html_path)
    assert "MOTION_VIEWER_DATA" in html
    assert "three@0.128.0" in html
    assert "region normalize" in html
    assert "\"left_eye\"" in html
    assert "value=\"2d\"" in html
    assert payload["neutral_image_base64"] is None
    assert len(payload["uv_coords"]) == 478
    assert len(payload["faces"]) > 0
    assert payload["frame_images"] is None
    assert payload["all_norm_xy"] is None
    assert payload["base_norm_xy"] is None
    assert payload["peak_norm_xy"] is None

    faces = np.asarray(payload["faces"], dtype=np.int64)
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert np.min(faces) >= 0
    assert np.max(faces) < 478

    with np.load(artifact_root / "sample" / "landmarks.npz") as npz:
        landmarks = np.asarray(npz["landmarks_xyz"], dtype=np.float32)
    test_idx = 61
    expected_u = float(landmarks[0, test_idx, 0])
    expected_v = float(1.0 - landmarks[0, test_idx, 1])
    uv = payload["uv_coords"][test_idx]
    assert np.isclose(float(uv[0]), expected_u, atol=1e-6)
    assert np.isclose(float(uv[1]), expected_v, atol=1e-6)


def test_generate_motion_viewer_embeds_2d_payload_when_video_exists(tmp_path: Path) -> None:
    stem = "sample_video"
    artifact_root = _write_synthetic_artifacts(tmp_path, stem=stem)
    video_path = tmp_path / f"{stem}.avi"
    _write_dummy_video(video_path, width=640, height=480, frame_count=4)

    result = generate_motion_viewer(video_path=video_path, artifact_root=artifact_root)
    payload = _load_payload_from_html(Path(result["html_path"]))

    frame_images = payload["frame_images"]
    all_norm_xy = payload["all_norm_xy"]
    assert isinstance(frame_images, list)
    assert isinstance(all_norm_xy, list)
    assert len(frame_images) > 0
    assert len(all_norm_xy) == len(frame_images)
    assert all(isinstance(item, str) and len(item) > 10 for item in frame_images)
    assert isinstance(payload["base_norm_xy"], list) and len(payload["base_norm_xy"]) == 478
    assert isinstance(payload["peak_norm_xy"], list) and len(payload["peak_norm_xy"]) == 478


def test_generate_motion_viewer_guides_when_segment_missing(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)
    segment_path = artifact_root / "sample" / "segment.json"
    segment_path.unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        generate_motion_viewer(video_path=tmp_path / "sample.mp4", artifact_root=artifact_root)

    message = str(exc_info.value)
    assert "Segment file not found" in message
    assert "faceanalyze2 segment run --video" in message


def test_viewer_generate_cli_runs(tmp_path: Path) -> None:
    artifact_root = _write_synthetic_artifacts(tmp_path)

    result = runner.invoke(
        app,
        [
            "viewer",
            "generate",
            "--video",
            str(tmp_path / "sample.mp4"),
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert result.exit_code == 0
    assert "Saved 3D motion viewer to" in result.output
    assert "Open in browser..." in result.output

