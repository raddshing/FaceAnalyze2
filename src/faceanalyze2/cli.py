from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from faceanalyze2.analysis.alignment import run_alignment
from faceanalyze2.analysis.alignment_viz import run_alignment_visualization
from faceanalyze2.analysis.metrics import run_metrics
from faceanalyze2.analysis.segmentation import SegmentParams, run_segmentation
from faceanalyze2.config import RunConfig, TaskType, artifact_paths_for_video, normalize_task_value
from faceanalyze2.landmarks.mediapipe_face_landmarker import extract_face_landmarks_from_video
from faceanalyze2.io.video_reader import probe_video, save_frame_png
from faceanalyze2.viz.motion_viewer import generate_motion_viewer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FaceAnalyze2 CLI scaffold for facial palsy analysis workflows.",
)
video_app = typer.Typer(help="Video I/O utilities.")
landmarks_app = typer.Typer(help="Landmark extraction utilities.")
segment_app = typer.Typer(help="Signal segmentation utilities.")
align_app = typer.Typer(help="Landmark alignment utilities.")
metrics_app = typer.Typer(help="ROI metrics and plots utilities.")
viewer_app = typer.Typer(help="3D motion viewer utilities.")
app.add_typer(video_app, name="video")
app.add_typer(landmarks_app, name="landmarks")
app.add_typer(segment_app, name="segment")
app.add_typer(align_app, name="align")
app.add_typer(metrics_app, name="metrics")
app.add_typer(viewer_app, name="viewer")
console = Console()


def _bool_mark(value: bool) -> str:
    return "[x]" if value else "[ ]"


def _resolve_task_or_prompt(task: TaskType | None) -> TaskType:
    if task is not None:
        return task
    answer = typer.prompt(
        "분석할 동작(task)을 선택하세요",
        type=click.Choice([TaskType.smile.value, TaskType.brow.value, TaskType.eyeclose.value]),
        show_choices=True,
    )
    return TaskType(answer)


def _step_header(step_no: int, step_title: str, description: str) -> None:
    console.print(f"\n[bold cyan]Step {step_no} - {step_title}[/bold cyan]")
    console.print(description)


def _step_footer(outputs: list[Path], next_manual_command: str) -> None:
    for output in outputs:
        console.print(f"- 생성 파일: {output}")
    console.print(f"- 다음 단계(수동 실행): {next_manual_command}")


def _status_recommendation(
    *,
    model_exists: bool,
    landmarks_exists: bool,
    segment_exists: bool,
    aligned_exists: bool,
    metrics_exists: bool,
    plots_exists: bool,
    video: Path,
    task_hint: str,
) -> str:
    if not model_exists:
        return (
            "Invoke-WebRequest -Uri "
            "\"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task\" "
            "-OutFile \"models/face_landmarker.task\""
        )
    if not landmarks_exists:
        return f"faceanalyze2 landmarks extract --video \"{video}\" --model \"models/face_landmarker.task\""
    if not segment_exists:
        return f"faceanalyze2 segment run --video \"{video}\" --task {task_hint}"
    if not aligned_exists:
        return f"faceanalyze2 align run --video \"{video}\""
    if not metrics_exists:
        return f"faceanalyze2 metrics run --video \"{video}\" --task {task_hint}"
    if not plots_exists:
        return f"faceanalyze2 align viz --video \"{video}\""
    return "모든 주요 산출물이 준비되었습니다. 필요하면 --force 옵션으로 재실행하세요."


def _should_skip(force: bool, outputs: list[Path], overlay_optional: bool = False) -> bool:
    if force:
        return False
    if overlay_optional:
        return outputs[0].exists() and outputs[1].exists() and (outputs[2].exists() or outputs[3].exists())
    return all(path.exists() for path in outputs)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    video: Optional[Path] = typer.Option(
        None,
        "--video",
        "-v",
        help="Input video file path (.mp4 expected).",
    ),
    task: Optional[TaskType] = typer.Option(
        None,
        "--task",
        "-t",
        help="Target movement task: smile, brow, eyeclose.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        "-o",
        help="Directory for generated outputs.",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    missing: list[str] = []
    if video is None:
        missing.append("--video")
    if task is None:
        missing.append("--task")
    if missing:
        raise typer.UsageError(f"Missing required options: {', '.join(missing)}")

    try:
        config = RunConfig(video_path=video, task=task, output_dir=output_dir)
    except ValidationError as exc:
        message = exc.errors()[0].get("msg", "Invalid input")
        raise typer.BadParameter(message, param_hint="--video") from exc

    config.ensure_output_dir()

    console.print("[bold green]FaceAnalyze2 M0 stub run[/bold green]")
    console.print_json(data=config.as_summary())


@app.command("guide")
def guide() -> None:
    console.print("[bold]FaceAnalyze2 새 동영상 처리 가이드[/bold]")
    console.print("1) 먼저 동작(task)을 고르세요: smile, brow, eyeclose")
    console.print("2) 추천 원커맨드:")
    console.print('   faceanalyze2 run --video "D:\\local\\sample.mp4" --task smile')
    console.print("3) 수동 실행 순서:")
    console.print('   faceanalyze2 landmarks extract --video "D:\\local\\sample.mp4"')
    console.print('   faceanalyze2 segment run --video "D:\\local\\sample.mp4" --task smile')
    console.print('   faceanalyze2 align run --video "D:\\local\\sample.mp4"')
    console.print('   faceanalyze2 metrics run --video "D:\\local\\sample.mp4" --task smile')
    console.print('   faceanalyze2 align viz --video "D:\\local\\sample.mp4"')
    console.print("4) 현재 상태 확인: faceanalyze2 status --video \"D:\\local\\sample.mp4\"")


@app.command("status")
def status(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    model: Path = typer.Option(
        Path("models/face_landmarker.task"),
        "--model",
        help="MediaPipe model path to check.",
    ),
) -> None:
    paths = artifact_paths_for_video(video, artifact_root=artifact_root)
    landmarks_exists = paths["landmarks_npz"].exists()
    segment_exists = paths["segment_json"].exists()
    aligned_exists = paths["aligned_npz"].exists()
    metrics_exists = paths["metrics_csv"].exists() and paths["metrics_json"].exists()
    plots_exists = paths["plots_dir"].exists() and any(paths["plots_dir"].glob("*.png"))
    model_exists = model.exists()

    task_hint = "smile"
    if segment_exists:
        try:
            import json

            payload = json.loads(paths["segment_json"].read_text(encoding="utf-8"))
            if "task" in payload and str(payload["task"]) in {"smile", "brow", "eyeclose"}:
                task_hint = str(payload["task"])
        except Exception:
            task_hint = "smile"

    table = Table(title=f"처리 상태: {video.stem}")
    table.add_column("항목")
    table.add_column("상태")
    table.add_column("경로")
    table.add_row("모델 파일", _bool_mark(model_exists), str(model))
    table.add_row("landmarks.npz", _bool_mark(landmarks_exists), str(paths["landmarks_npz"]))
    table.add_row("segment.json", _bool_mark(segment_exists), str(paths["segment_json"]))
    table.add_row("landmarks_aligned.npz", _bool_mark(aligned_exists), str(paths["aligned_npz"]))
    table.add_row("metrics.csv/json", _bool_mark(metrics_exists), str(paths["metrics_json"]))
    table.add_row("plots/*.png", _bool_mark(bool(plots_exists)), str(paths["plots_dir"]))
    console.print(table)

    recommendation = _status_recommendation(
        model_exists=model_exists,
        landmarks_exists=landmarks_exists,
        segment_exists=segment_exists,
        aligned_exists=aligned_exists,
        metrics_exists=metrics_exists,
        plots_exists=bool(plots_exists),
        video=video,
        task_hint=task_hint,
    )
    console.print(f"\n[bold]다음으로 추천하는 커맨드[/bold]\n{recommendation}")


@app.command("run")
def run_pipeline(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path.",
    ),
    task: TaskType | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Target task. If omitted, interactive prompt is shown.",
    ),
    model: Path = typer.Option(
        Path("models/face_landmarker.task"),
        "--model",
        help="MediaPipe Face Landmarker .task model path.",
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        min=1,
        help="Frame stride for landmarks extraction.",
    ),
    with_viz: bool = typer.Option(
        False,
        "--with-viz/--no-viz",
        help="Run align viz step after metrics.",
    ),
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="Normalize ROI displacement by interocular distance.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run each step even if outputs already exist.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print step plan only (no execution).",
    ),
    check_video: bool = typer.Option(
        False,
        "--check-video/--no-check-video",
        help="When enabled, verify that --video path exists before running.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
) -> None:
    selected_task = _resolve_task_or_prompt(task)
    task_value = normalize_task_value(selected_task)
    paths = artifact_paths_for_video(video, artifact_root=artifact_root)

    if check_video and not video.exists():
        raise typer.BadParameter(f"Video file does not exist: {video}", param_hint="--video")

    steps = [
        {
            "no": 1,
            "name": "landmarks extract",
            "desc": "비디오 프레임에서 얼굴 랜드마크를 추출해 landmarks.npz를 생성합니다.",
            "outputs": [paths["landmarks_npz"], paths["meta_json"]],
            "manual": (
                f'faceanalyze2 landmarks extract --video "{video}" --model "{model}" '
                f'--stride {stride} --artifact-root "{artifact_root}"'
            ),
        },
        {
            "no": 2,
            "name": "segment run",
            "desc": "신호를 계산해 neutral/peak/onset/offset을 찾고 segment.json을 만듭니다.",
            "outputs": [paths["segment_json"], paths["signals_csv"], paths["signals_plot_png"]],
            "manual": f'faceanalyze2 segment run --video "{video}" --task {task_value}',
        },
        {
            "no": 3,
            "name": "align run",
            "desc": "neutral 기준 similarity transform으로 head motion을 줄입니다.",
            "outputs": [paths["aligned_npz"], paths["alignment_json"]],
            "manual": f'faceanalyze2 align run --video "{video}"',
        },
        {
            "no": 4,
            "name": "metrics run",
            "desc": "ROI 좌/우 이동량과 비대칭 지수(AI)를 계산해 그래프와 metrics를 저장합니다.",
            "outputs": [paths["metrics_csv"], paths["metrics_json"], paths["timeseries_csv"]],
            "manual": f'faceanalyze2 metrics run --video "{video}" --task {task_value} --normalize',
        },
    ]
    if with_viz:
        steps.append(
            {
                "no": 5,
                "name": "align viz",
                "desc": "정합 전/후 품질을 눈으로 확인할 수 있는 QA 시각화 파일을 생성합니다.",
                "outputs": [
                    paths["alignment_check_png"],
                    paths["trajectory_plot_png"],
                    paths["alignment_overlay_mp4"],
                    paths["alignment_overlay_avi"],
                ],
                "manual": f'faceanalyze2 align viz --video "{video}"',
                "overlay_optional": True,
            }
        )

    if dry_run:
        console.print("[bold yellow]DRY-RUN: 실행 계획만 출력합니다.[/bold yellow]")
        for step in steps:
            skip = _should_skip(
                force=force,
                outputs=step["outputs"],
                overlay_optional=bool(step.get("overlay_optional", False)),
            )
            decision = "SKIP(기존 산출물)" if skip else "RUN"
            _step_header(step["no"], step["name"], step["desc"])
            console.print(f"- dry-run 결정: {decision}")
            _step_footer(
                outputs=[Path(path) for path in step["outputs"][:2]],
                next_manual_command=str(step["manual"]),
            )
        return

    for idx, step in enumerate(steps):
        next_manual = steps[idx + 1]["manual"] if idx + 1 < len(steps) else "완료"
        _step_header(step["no"], step["name"], step["desc"])
        skip = _should_skip(
            force=force,
            outputs=step["outputs"],
            overlay_optional=bool(step.get("overlay_optional", False)),
        )
        if skip:
            console.print("- 기존 산출물이 있어 이 단계는 스킵합니다. (--force로 재생성 가능)")
            _step_footer(outputs=[Path(path) for path in step["outputs"][:2]], next_manual_command=next_manual)
            continue

        try:
            if step["name"] == "landmarks extract":
                result = extract_face_landmarks_from_video(
                    video_path=str(video),
                    model_path=str(model),
                    stride=stride,
                    start_frame=0,
                    end_frame=None,
                    num_faces=1,
                    output_blendshapes=False,
                    output_transforms=False,
                    use_gpu_delegate=False,
                    artifact_root=artifact_root,
                )
                outputs = [Path(result["npz_path"]), Path(result["meta_path"])]
            elif step["name"] == "segment run":
                result = run_segmentation(
                    video_path=video,
                    task=task_value,
                    artifact_root=artifact_root,
                )
                outputs = [
                    Path(result["segment_json_path"]),
                    Path(result["signals_csv_path"]),
                    Path(result["plot_path"]),
                ]
            elif step["name"] == "align run":
                result = run_alignment(
                    video_path=video,
                    artifact_root=artifact_root,
                )
                outputs = [Path(result["aligned_npz_path"]), Path(result["alignment_json_path"])]
            elif step["name"] == "metrics run":
                result = run_metrics(
                    video_path=video,
                    task=task_value,
                    artifact_root=artifact_root,
                    normalize=normalize,
                    rois="all",
                )
                outputs = [
                    Path(result["metrics_csv_path"]),
                    Path(result["metrics_json_path"]),
                    Path(result["timeseries_csv_path"]),
                ]
            else:
                result = run_alignment_visualization(
                    video_path=video,
                    artifact_root=artifact_root,
                )
                outputs = [
                    Path(result["alignment_check_path"]),
                    Path(result["trajectory_plot_path"]),
                    Path(result["overlay_path"]),
                ]
        except Exception as exc:
            console.print(f"[bold red]실패[/bold red]: {exc}")
            raise typer.Exit(code=2) from exc

        _step_footer(outputs=outputs, next_manual_command=next_manual)

    console.print("\n[bold green]모든 단계 실행이 완료되었습니다.[/bold green]")


@video_app.command("info")
def video_info(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path.",
    ),
) -> None:
    try:
        info = probe_video(video)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print_json(
        data={
            "path": str(info.path),
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration_ms": info.duration_ms,
        }
    )


@video_app.command("frame")
def video_frame(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path.",
    ),
    frame_idx: int = typer.Option(
        ...,
        "--frame-idx",
        min=0,
        help="Frame index to export.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output PNG file path.",
    ),
) -> None:
    try:
        output_path = save_frame_png(video, frame_idx, out)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc
    except ValueError as exc:
        message = str(exc)
        if "Output file must be a .png" in message:
            raise typer.BadParameter(message, param_hint="--out") from exc
        if "frame_idx" in message or "Frame index out of range" in message:
            raise typer.BadParameter(message, param_hint="--frame-idx") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc

    console.print(f"Saved frame {frame_idx} to {output_path}")


@landmarks_app.command("extract")
def landmarks_extract(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path.",
    ),
    model: Path = typer.Option(
        Path("models/face_landmarker.task"),
        "--model",
        help="MediaPipe Face Landmarker .task model path.",
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        min=1,
        help="Read one frame every N frames.",
    ),
    start_frame: int = typer.Option(
        0,
        "--start-frame",
        min=0,
        help="Start frame index.",
    ),
    end_frame: int | None = typer.Option(
        None,
        "--end-frame",
        min=0,
        help="End frame index (exclusive).",
    ),
    num_faces: int = typer.Option(
        1,
        "--num-faces",
        min=1,
        help="Maximum number of faces to track.",
    ),
    output_blendshapes: bool = typer.Option(
        False,
        "--output-blendshapes/--no-output-blendshapes",
        help="Store blendshape scores when available.",
    ),
    output_transforms: bool = typer.Option(
        False,
        "--output-transforms/--no-output-transforms",
        help="Store facial transformation matrices when available.",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu/--no-use-gpu",
        help="Use GPU delegate when supported. Defaults to CPU.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
) -> None:
    try:
        result = extract_face_landmarks_from_video(
            video_path=str(video),
            model_path=str(model),
            stride=stride,
            start_frame=start_frame,
            end_frame=end_frame,
            num_faces=num_faces,
            output_blendshapes=output_blendshapes,
            output_transforms=output_transforms,
            use_gpu_delegate=use_gpu,
            artifact_root=artifact_root,
        )
    except FileNotFoundError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    except ImportError as exc:
        raise typer.BadParameter(str(exc), param_hint="mediapipe") from exc
    except ValueError as exc:
        message = str(exc)
        if "stride" in message:
            raise typer.BadParameter(message, param_hint="--stride") from exc
        if "start_frame" in message:
            raise typer.BadParameter(message, param_hint="--start-frame") from exc
        if "end_frame" in message:
            raise typer.BadParameter(message, param_hint="--end-frame") from exc
        if "num_faces" in message:
            raise typer.BadParameter(message, param_hint="--num-faces") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved landmarks to {result['npz_path']}")
    console.print(f"Saved metadata to {result['meta_path']}")


@segment_app.command("run")
def segment_run(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    task: TaskType = typer.Option(
        ...,
        "--task",
        "-t",
        help="Target movement task: smile, brow, eyeclose.",
    ),
    landmarks: Path | None = typer.Option(
        None,
        "--landmarks",
        help="Optional landmarks.npz path. Defaults to artifacts/<video_stem>/landmarks.npz.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    median_window: int = typer.Option(
        5,
        "--median-window",
        min=1,
        help="Median filter window for signal smoothing.",
    ),
    moving_average_window: int = typer.Option(
        9,
        "--moving-average-window",
        min=1,
        help="Moving average window for signal smoothing.",
    ),
    pre_seconds: float = typer.Option(
        2.0,
        "--pre-seconds",
        min=0.1,
        help="Seconds before initial peak to prioritize neutral search.",
    ),
    onset_alpha: float = typer.Option(
        0.2,
        "--onset-alpha",
        min=0.01,
        max=0.95,
        help="Onset threshold ratio relative to amplitude.",
    ),
    offset_alpha: float = typer.Option(
        0.2,
        "--offset-alpha",
        min=0.01,
        max=0.95,
        help="Offset threshold ratio relative to amplitude.",
    ),
) -> None:
    params = SegmentParams(
        median_window=median_window,
        moving_average_window=moving_average_window,
        pre_seconds=pre_seconds,
        onset_alpha=onset_alpha,
        offset_alpha=offset_alpha,
    )
    try:
        result = run_segmentation(
            video_path=video,
            task=task.value,
            landmarks_path=landmarks,
            artifact_root=artifact_root,
            params=params,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="--landmarks") from exc
    except ValueError as exc:
        message = str(exc)
        if "task" in message:
            raise typer.BadParameter(message, param_hint="--task") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved signals CSV to {result['signals_csv_path']}")
    console.print(f"Saved segment JSON to {result['segment_json_path']}")
    console.print(f"Saved signal plot to {result['plot_path']}")


@align_app.command("run")
def align_run(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    landmarks: Path | None = typer.Option(
        None,
        "--landmarks",
        help="Optional landmarks.npz path. Defaults to artifacts/<video_stem>/landmarks.npz.",
    ),
    segment: Path | None = typer.Option(
        None,
        "--segment",
        help="Optional segment.json path. Defaults to artifacts/<video_stem>/segment.json.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    scale_z: bool = typer.Option(
        False,
        "--scale-z/--keep-z",
        help="Scale z with similarity scale. Default keeps original z values.",
    ),
    scale_outlier_factor: float = typer.Option(
        3.0,
        "--scale-outlier-factor",
        min=1.01,
        help="Flag frames with scale > (median_scale * factor).",
    ),
) -> None:
    try:
        result = run_alignment(
            video_path=video,
            landmarks_path=landmarks,
            segment_path=segment,
            artifact_root=artifact_root,
            scale_z=scale_z,
            scale_outlier_factor=scale_outlier_factor,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Landmarks file not found" in message:
            raise typer.BadParameter(message, param_hint="--landmarks") from exc
        if "Segment file not found" in message:
            raise typer.BadParameter(message, param_hint="--segment") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except ValueError as exc:
        message = str(exc)
        if "scale_outlier_factor" in message:
            raise typer.BadParameter(message, param_hint="--scale-outlier-factor") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved aligned landmarks to {result['aligned_npz_path']}")
    console.print(f"Saved alignment metadata to {result['alignment_json_path']}")


@align_app.command("viz")
def align_viz(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    landmarks: Path | None = typer.Option(
        None,
        "--landmarks",
        help="Optional landmarks.npz path. Defaults to artifacts/<video_stem>/landmarks.npz.",
    ),
    segment: Path | None = typer.Option(
        None,
        "--segment",
        help="Optional segment.json path. Defaults to artifacts/<video_stem>/segment.json.",
    ),
    aligned: Path | None = typer.Option(
        None,
        "--aligned",
        help="Optional landmarks_aligned.npz path. Defaults to artifacts/<video_stem>/landmarks_aligned.npz.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    max_frames: int = typer.Option(
        300,
        "--max-frames",
        min=1,
        help="Maximum frames to render into the overlay video.",
    ),
    stride: int = typer.Option(
        2,
        "--stride",
        min=1,
        help="Frame stride for overlay video generation.",
    ),
    n_samples: int = typer.Option(
        15,
        "--n-samples",
        min=1,
        help="Number of frames sampled for alignment_check overlay plot.",
    ),
) -> None:
    try:
        result = run_alignment_visualization(
            video_path=video,
            artifact_root=artifact_root,
            landmarks_path=landmarks,
            segment_path=segment,
            aligned_path=aligned,
            max_frames=max_frames,
            stride=stride,
            n_samples=n_samples,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Landmarks file not found" in message:
            raise typer.BadParameter(message, param_hint="--landmarks") from exc
        if "Segment file not found" in message:
            raise typer.BadParameter(message, param_hint="--segment") from exc
        if "Aligned landmarks file not found" in message:
            raise typer.BadParameter(message, param_hint="--aligned") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except ValueError as exc:
        message = str(exc)
        if "max_frames" in message:
            raise typer.BadParameter(message, param_hint="--max-frames") from exc
        if "stride" in message:
            raise typer.BadParameter(message, param_hint="--stride") from exc
        if "n_samples" in message:
            raise typer.BadParameter(message, param_hint="--n-samples") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved alignment check plot to {result['alignment_check_path']}")
    console.print(f"Saved trajectory plot to {result['trajectory_plot_path']}")
    console.print(f"Saved alignment overlay video to {result['overlay_path']}")


@metrics_app.command("run")
def metrics_run(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    task: TaskType = typer.Option(
        ...,
        "--task",
        "-t",
        help="Target movement task: smile, brow, eyeclose.",
    ),
    aligned: Path | None = typer.Option(
        None,
        "--aligned",
        help="Optional landmarks_aligned.npz path. Defaults to artifacts/<video_stem>/landmarks_aligned.npz.",
    ),
    segment: Path | None = typer.Option(
        None,
        "--segment",
        help="Optional segment.json path. Defaults to artifacts/<video_stem>/segment.json.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    rois: str = typer.Option(
        "all",
        "--rois",
        help="ROI selection: all or comma-separated values (mouth,eye,brow).",
    ),
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="Normalize displacement by neutral interocular distance.",
    ),
) -> None:
    try:
        result = run_metrics(
            video_path=video,
            task=task.value,
            artifact_root=artifact_root,
            aligned_path=aligned,
            segment_path=segment,
            rois=rois,
            normalize=normalize,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Aligned landmarks file not found" in message:
            raise typer.BadParameter(message, param_hint="--aligned") from exc
        if "Segment file not found" in message:
            raise typer.BadParameter(message, param_hint="--segment") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except ValueError as exc:
        message = str(exc)
        if "roi" in message or "rois" in message:
            raise typer.BadParameter(message, param_hint="--rois") from exc
        if "normalize" in message or "interocular" in message:
            raise typer.BadParameter(message, param_hint="--normalize") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved plots to {result['plots_dir']}")
    console.print(f"Saved timeseries CSV to {result['timeseries_csv_path']}")
    console.print(f"Saved metrics CSV to {result['metrics_csv_path']}")
    console.print(f"Saved metrics JSON to {result['metrics_json_path']}")


@viewer_app.command("generate")
def viewer_generate(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Input video path used for artifact lookup.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        help="Root directory for artifact outputs.",
    ),
    landmarks: Path | None = typer.Option(
        None,
        "--landmarks",
        help="Optional landmarks.npz path. Defaults to artifacts/<video_stem>/landmarks.npz.",
    ),
    segment: Path | None = typer.Option(
        None,
        "--segment",
        help="Optional segment.json path. Defaults to artifacts/<video_stem>/segment.json.",
    ),
) -> None:
    try:
        result = generate_motion_viewer(
            video_path=video,
            artifact_root=artifact_root,
            landmarks_path=landmarks,
            segment_path=segment,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Landmarks file not found" in message:
            raise typer.BadParameter(message, param_hint="--landmarks") from exc
        if "Segment file not found" in message:
            raise typer.BadParameter(message, param_hint="--segment") from exc
        raise typer.BadParameter(message, param_hint="--video") from exc
    except ImportError as exc:
        raise typer.BadParameter(str(exc), param_hint="mediapipe") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--video") from exc

    console.print(f"Saved 3D motion viewer to {result['html_path']}")
    console.print("Open in browser...")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
