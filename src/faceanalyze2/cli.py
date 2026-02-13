from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from faceanalyze2.config import RunConfig, TaskType
from faceanalyze2.io.video_reader import probe_video, save_frame_png

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FaceAnalyze2 CLI scaffold for facial palsy analysis workflows.",
)
video_app = typer.Typer(help="Video I/O utilities.")
app.add_typer(video_app, name="video")
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


def run() -> None:
    app()


if __name__ == "__main__":
    run()
