"""Gradio demo app package for local presentations."""


def create_demo_app():
    from faceanalyze2.demo.gradio_app import create_demo_app as _create_demo_app

    return _create_demo_app()


def main() -> None:
    from faceanalyze2.demo.gradio_app import main as _main

    _main()


__all__ = ["create_demo_app", "main"]
