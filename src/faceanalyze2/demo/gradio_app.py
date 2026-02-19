from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from faceanalyze2 import dynamicAnalysis
from faceanalyze2.config import artifact_dir_for_video

DEFAULT_MODEL_PATH = "models/face_landmarker.task"
DEMO_INPUT_DIR = Path("demo_inputs")
DEMO_OUTPUT_DIR = Path("demo_outputs")
PNG_KEYS = ["key rest", "key exp", "key value graph", "before regi", "after regi"]

PHI_NOTICE = (
    "## FaceAnalyze2 Local Demo\n"
    "환자 영상/프레임(PHI 포함 가능) 처리 화면입니다. 외부 공유 금지. "
    "share 옵션 사용 금지(로컬 실행 전용)."
)

MOTION_TO_TASK = {
    "big-smile": "smile",
    "blinking-motion": "eyeclose",
    "eyebrow-motion": "brow",
}


def _motion_to_task(motion: str) -> str:
    task = MOTION_TO_TASK.get(str(motion).strip())
    if task is None:
        valid = ", ".join(sorted(MOTION_TO_TASK))
        raise ValueError(f"Unsupported motion '{motion}'. Expected one of: {valid}")
    return task


def _decode_base64_payload(payload: str) -> bytes:
    text = str(payload).strip()
    if not text:
        raise ValueError("Empty base64 payload")
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 payload") from exc


def _decode_base64_png_to_pil(payload: str) -> Any:
    from PIL import Image

    raw = _decode_base64_payload(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _safe_filename(name: str) -> str:
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return safe or "upload.mp4"


def _copy_uploaded_video(upload_path: str | Path) -> Path:
    source = Path(upload_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Uploaded file is missing: {source}")
    if source.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 upload is supported for demo: {source.name}")

    DEMO_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(source.name)
    target = DEMO_INPUT_DIR / f"{uuid4().hex[:8]}_{safe_name}"
    shutil.copy2(source, target)
    return target


def _run_subprocess(step_name: str, args: list[str]) -> tuple[bool, str, str]:
    process = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_text = process.stdout.strip()
    stderr_text = process.stderr.strip()
    logs = []
    if stdout_text:
        logs.append("[stdout]\n" + stdout_text)
    if stderr_text:
        logs.append("[stderr]\n" + stderr_text)
    combined = "\n\n".join(logs).strip()
    command = " ".join(args)
    summary = f"{step_name}: {'OK' if process.returncode == 0 else f'FAILED({process.returncode})'}"
    details = (
        f"<details><summary>{step_name} log</summary>\n\n"
        f"`{command}`\n\n```text\n{combined or '(no output)'}\n```\n</details>"
    )
    return process.returncode == 0, summary, details


def _metrics_to_rows(metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    roi_metrics = metrics_payload.get("roi_metrics", {})
    rows: list[dict[str, Any]] = []
    for roi_name in sorted(roi_metrics):
        metric = roi_metrics[roi_name]
        row: dict[str, Any] = {
            "roi": roi_name,
            "L_peak": metric.get("L_peak"),
            "R_peak": metric.get("R_peak"),
            "AI": metric.get("AI"),
        }
        if "score" in metric:
            row["score"] = metric.get("score")
        rows.append(row)
    return rows


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return []
    try:
        import pandas as pd
    except Exception:
        return rows
    return pd.DataFrame(rows)


def _collect_roi_gallery(result: dict[str, Any], artifact_dir: Path) -> list[tuple[Any, str]]:
    gallery: list[tuple[Any, str]] = []
    roi_plots = result.get("roi_plots")
    if isinstance(roi_plots, dict):
        for name, payload in roi_plots.items():
            if not isinstance(payload, str) or not payload.strip():
                continue
            try:
                image = _decode_base64_png_to_pil(payload)
            except Exception:
                continue
            gallery.append((image, str(name)))
        if gallery:
            return gallery

    plots_dir = artifact_dir / "plots"
    if not plots_dir.exists():
        return gallery
    try:
        from PIL import Image
    except Exception:
        return gallery
    for path in sorted(plots_dir.glob("*.png")):
        try:
            gallery.append((Image.open(path).convert("RGB"), path.stem))
        except Exception:
            continue
    return gallery


def _resolve_viewer_file(result: dict[str, Any], artifact_dir: Path) -> str | None:
    candidate = result.get("viewer_html_path")
    if isinstance(candidate, str) and candidate.strip():
        path = Path(candidate)
        if path.exists() and path.is_file():
            return str(path)
    fallback = artifact_dir / "motion_viewer.html"
    if fallback.exists() and fallback.is_file():
        return str(fallback)
    return None


def _resolve_timeseries_file(result: dict[str, Any], artifact_dir: Path, run_id: str) -> str | None:
    payload = result.get("timeseries_csv")
    if isinstance(payload, str) and payload.strip():
        raw = _decode_base64_payload(payload)
        DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_OUTPUT_DIR / f"{run_id}_timeseries.csv"
        out_path.write_bytes(raw)
        return str(out_path)

    for key in ("timeseries_csv_path",):
        candidate = result.get(key)
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate)
            if path.exists() and path.is_file():
                return str(path)

    fallback = artifact_dir / "timeseries.csv"
    if fallback.exists() and fallback.is_file():
        return str(fallback)
    return None


def _empty_demo_outputs(status_md: str, raw_json: str, metrics_rows: list[dict[str, Any]] | None = None) -> tuple:
    return (
        status_md,
        _rows_to_dataframe(metrics_rows or []),
        None,
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        raw_json,
    )


def run_demo(
    video_file: str | None,
    motion: str,
    model_path: str,
    stride: int,
    ensure_pipeline_run: bool,
    generate_viewer_html: bool,
) -> tuple:
    if not video_file:
        return _empty_demo_outputs(
            "### Failed\n- 업로드된 비디오가 없습니다.",
            "{}",
        )

    status_lines: list[str] = []
    log_blocks: list[str] = []
    run_id = uuid4().hex[:8]

    try:
        copied_video = _copy_uploaded_video(video_file)
        status_lines.append(f"입력 비디오 복사: `{copied_video}`")
    except Exception as exc:
        return _empty_demo_outputs(
            f"### Failed\n- 비디오 준비 중 오류: {exc}",
            "{}",
        )

    try:
        task = _motion_to_task(motion)
    except Exception as exc:
        return _empty_demo_outputs(
            f"### Failed\n- motion 설정 오류: {exc}",
            "{}",
        )

    model_value = str(model_path).strip() or DEFAULT_MODEL_PATH
    stride_value = int(stride)
    artifact_dir = artifact_dir_for_video(copied_video)

    if ensure_pipeline_run:
        ok, summary, details = _run_subprocess(
            "Pipeline run",
            [
                sys.executable,
                "-m",
                "faceanalyze2",
                "run",
                "--video",
                str(copied_video),
                "--task",
                task,
                "--model",
                model_value,
                "--stride",
                str(stride_value),
            ],
        )
        status_lines.append(summary)
        log_blocks.append(details)
        if not ok:
            status_md = "### Failed\n" + "\n".join(f"- {line}" for line in status_lines)
            status_md += "\n\n" + "\n\n".join(log_blocks)
            return _empty_demo_outputs(status_md, "{}")
    else:
        status_lines.append("Pipeline run 스킵(사용자 선택).")

    if generate_viewer_html:
        ok, summary, details = _run_subprocess(
            "Viewer generate",
            [
                sys.executable,
                "-m",
                "faceanalyze2",
                "viewer",
                "generate",
                "--video",
                str(copied_video),
            ],
        )
        status_lines.append(summary)
        log_blocks.append(details)
    else:
        status_lines.append("Viewer generate 스킵(사용자 선택).")

    try:
        result = dynamicAnalysis(copied_video, motion)
    except Exception as exc:
        status_md = "### Failed\n- dynamicAnalysis 실행 중 오류가 발생했습니다.\n"
        status_md += f"- 메시지: {exc}\n\n"
        if log_blocks:
            status_md += "\n\n".join(log_blocks)
        return _empty_demo_outputs(status_md, "{}")

    images: list[Any] = []
    decode_warnings: list[str] = []
    for key in PNG_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            try:
                images.append(_decode_base64_png_to_pil(value))
            except Exception:
                images.append(None)
                decode_warnings.append(f"{key} 디코드 실패")
        else:
            images.append(None)

    if decode_warnings:
        status_lines.extend(decode_warnings)

    metrics_payload = result.get("metrics", {})
    metric_rows = _metrics_to_rows(metrics_payload if isinstance(metrics_payload, dict) else {})
    metrics_df = _rows_to_dataframe(metric_rows)

    gallery_items = _collect_roi_gallery(result, artifact_dir=artifact_dir)
    viewer_file = _resolve_viewer_file(result, artifact_dir=artifact_dir)
    timeseries_file = _resolve_timeseries_file(result, artifact_dir=artifact_dir, run_id=run_id)
    raw_json = json.dumps(result, indent=2, ensure_ascii=False)

    status_md = "### Done\n" + "\n".join(f"- {line}" for line in status_lines)
    status_md += f"\n- Artifact directory: `{artifact_dir}`"
    if log_blocks:
        status_md += "\n\n" + "\n\n".join(log_blocks)

    return (
        status_md,
        metrics_df,
        images[0],
        images[1],
        images[2],
        images[3],
        images[4],
        gallery_items,
        viewer_file,
        timeseries_file,
        raw_json,
    )


def create_demo_app() -> Any:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is not installed. Install demo dependencies:\n"
            'python -m pip install -e ".[dev,demo]"'
        ) from exc

    with gr.Blocks(title="FaceAnalyze2 Gradio Demo") as demo:
        gr.Markdown(PHI_NOTICE)

        with gr.Row():
            video_input = gr.File(label="Video (.mp4)", file_types=[".mp4"], type="filepath")
            motion_input = gr.Dropdown(
                label="Motion",
                choices=["big-smile", "blinking-motion", "eyebrow-motion"],
                value="big-smile",
            )

        with gr.Accordion("Advanced", open=False):
            model_input = gr.Textbox(label="Model path", value=DEFAULT_MODEL_PATH)
            stride_input = gr.Slider(label="Stride", minimum=1, maximum=8, step=1, value=2)
            ensure_pipeline_input = gr.Checkbox(label="Ensure pipeline run", value=True)
            viewer_input = gr.Checkbox(label="Generate viewer html", value=True)

        run_button = gr.Button("Run")

        status_output = gr.Markdown(label="Status")
        metrics_output = gr.Dataframe(label="Metrics", interactive=False)

        with gr.Row():
            key_rest_output = gr.Image(label="key rest", type="pil")
            key_exp_output = gr.Image(label="key exp", type="pil")
        with gr.Row():
            key_graph_output = gr.Image(label="key value graph", type="pil")
            before_regi_output = gr.Image(label="before regi", type="pil")
            after_regi_output = gr.Image(label="after regi", type="pil")

        roi_gallery_output = gr.Gallery(label="ROI plots (optional)", columns=3, height=260)
        viewer_file_output = gr.File(label="viewer html (optional)")
        timeseries_file_output = gr.File(label="timeseries csv (optional)")
        with gr.Accordion("Raw JSON", open=False):
            raw_json_output = gr.Code(label="dynamicAnalysis result", language="json")

        run_button.click(
            fn=run_demo,
            inputs=[
                video_input,
                motion_input,
                model_input,
                stride_input,
                ensure_pipeline_input,
                viewer_input,
            ],
            outputs=[
                status_output,
                metrics_output,
                key_rest_output,
                key_exp_output,
                key_graph_output,
                before_regi_output,
                after_regi_output,
                roi_gallery_output,
                viewer_file_output,
                timeseries_file_output,
                raw_json_output,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceAnalyze2 Gradio local demo app")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for local demo server")
    parser.add_argument("--port", type=int, default=7860, help="Port for local demo server")
    args = parser.parse_args()

    demo = create_demo_app()
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()

