from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from faceanalyze2.config import RunConfig, TaskType

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FaceAnalyze2 CLI scaffold for facial palsy analysis workflows.",
)
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


def run() -> None:
    app()


if __name__ == "__main__":
    run()
