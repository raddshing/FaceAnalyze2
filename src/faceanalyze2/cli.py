from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from faceanalyze2.analysis.alignment import run_alignment
from faceanalyze2.analysis.alignment_viz import run_alignment_visualization
from faceanalyze2.analysis.segmentation import SegmentParams, run_segmentation
from faceanalyze2.config import RunConfig, TaskType
from faceanalyze2.landmarks.mediapipe_face_landmarker import extract_face_landmarks_from_video
from faceanalyze2.io.video_reader import probe_video, save_frame_png

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FaceAnalyze2 CLI scaffold for facial palsy analysis workflows.",
)
video_app = typer.Typer(help="Video I/O utilities.")
landmarks_app = typer.Typer(help="Landmark extraction utilities.")
segment_app = typer.Typer(help="Signal segmentation utilities.")
align_app = typer.Typer(help="Landmark alignment utilities.")
app.add_typer(video_app, name="video")
app.add_typer(landmarks_app, name="landmarks")
app.add_typer(segment_app, name="segment")
app.add_typer(align_app, name="align")
console = Console()


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


def run() -> None:
    app()


if __name__ == "__main__":
    run()
