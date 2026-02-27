"""Utilities for converting base64-encoded PNG strings to QPixmap."""

from __future__ import annotations

import base64

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QPixmap


def base64_png_to_pixmap(b64_str: str) -> QPixmap:
    """Decode a base64-encoded PNG string and return a QPixmap."""
    raw = base64.b64decode(b64_str)
    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(raw), "PNG")
    return pixmap
