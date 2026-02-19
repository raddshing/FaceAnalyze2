from __future__ import annotations

from faceanalyze2.demo.gradio_app import _decode_base64_payload, _metrics_to_rows, _motion_to_task


def test_decode_base64_payload_png_signature() -> None:
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAusB9Y9N4wsAAAAASUVORK5CYII="
    )
    raw = _decode_base64_payload(tiny_png_base64)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_motion_to_task_mapping_and_metrics_rows() -> None:
    assert _motion_to_task("big-smile") == "smile"
    assert _motion_to_task("blinking-motion") == "eyeclose"
    assert _motion_to_task("eyebrow-motion") == "brow"

    rows = _metrics_to_rows(
        {
            "roi_metrics": {
                "mouth": {"L_peak": 1.0, "R_peak": 0.8, "AI": 0.22, "score": 78.0},
                "eye": {"L_peak": 0.4, "R_peak": 0.3, "AI": 0.28},
            }
        }
    )
    assert len(rows) == 2
    assert rows[0]["roi"] == "eye"
    assert rows[1]["roi"] == "mouth"
    assert "score" in rows[1]
