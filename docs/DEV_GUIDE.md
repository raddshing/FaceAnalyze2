# Developer Guide

## Environment

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

## Common commands

```bash
python -m faceanalyze2 --help
pytest
ruff check .
```

## Notes

- M0 CLI is a stub and does not run MediaPipe.
- Keep patient/sensitive files out of git.
