# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FaceAnalyze2 desktop application (onedir mode)."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(SPECPATH).parent.resolve()
ENTRY_POINT = str(PROJECT_ROOT / "src" / "faceanalyze2" / "desktop" / "app.py")

# ---------------------------------------------------------------------------
# MediaPipe: bundle entire package to avoid FileNotFoundError at runtime
# ---------------------------------------------------------------------------
import mediapipe  # noqa: E402

mediapipe_path = mediapipe.__path__[0]

# ---------------------------------------------------------------------------
# Conda DLLs not found automatically by PyInstaller
# ---------------------------------------------------------------------------
CONDA_LIB_BIN = Path(sys.prefix) / "Library" / "bin"

_conda_dlls = [
    "ffi.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll",
    "libexpat.dll",
]

extra_binaries = []
for dll_name in _conda_dlls:
    dll_path = CONDA_LIB_BIN / dll_name
    if dll_path.exists():
        extra_binaries.append((str(dll_path), "."))

# ---------------------------------------------------------------------------
# PySide6 VC runtime override: PySide6 ships newer MSVC runtimes than conda.
# If conda's older DLLs land in _internal/ root, they shadow PySide6's copies
# and cause "DLL load failed: specified procedure not found" on QtCore import.
# Fix: place PySide6's VC runtimes at the root so they take priority.
# ---------------------------------------------------------------------------
import PySide6  # noqa: E402

_pyside6_dir = Path(PySide6.__path__[0])
_vc_dlls = [
    "MSVCP140.dll",
    "MSVCP140_1.dll",
    "MSVCP140_2.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
]
for dll_name in _vc_dlls:
    dll_path = _pyside6_dir / dll_name
    if dll_path.exists():
        extra_binaries.append((str(dll_path), "."))

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
hiddenimports = [
    # MediaPipe
    "mediapipe",
    "mediapipe.python",
    "mediapipe.python.solutions",
    # OpenCV
    "cv2",
    # NumPy
    "numpy",
    # Matplotlib – Agg backend only
    "matplotlib",
    "matplotlib.backends.backend_agg",
    # PySide6
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # Project modules
    "faceanalyze2",
    "faceanalyze2.desktop",
    "faceanalyze2.desktop.app",
    "faceanalyze2.desktop.main_window",
    "faceanalyze2.desktop.input_panel",
    "faceanalyze2.desktop.results_panel",
    "faceanalyze2.desktop.pipeline_worker",
    "faceanalyze2.desktop.image_utils",
    "faceanalyze2.runtime_paths",
    "faceanalyze2.config",
    "faceanalyze2.api.dynamic_analysis",
    "faceanalyze2.analysis",
    "faceanalyze2.landmarks",
    "faceanalyze2.viz",
    "faceanalyze2.viz.motion_viewer",
]

# Collect all mediapipe submodules automatically
hiddenimports += collect_submodules("mediapipe")

# ---------------------------------------------------------------------------
# Excludes – trim bundle size
# ---------------------------------------------------------------------------
excludes = [
    "gradio",
    "pandas",
    "tkinter",
    "_tkinter",
    "test",
    # NOTE: do NOT exclude "unittest" -- pyparsing.testing imports it at module
    # level and it is reachable via mediapipe -> matplotlib -> pyparsing chain.
    "xmlrpc",
    "pydoc",
    "doctest",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
    "setuptools",
    "pip",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [ENTRY_POINT],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=extra_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Remove conda's ICU DLLs: conda ships ICU 58 with versioned exports
# (e.g. ucnv_open_58) but PySide6's Qt6Core.dll expects the Windows system
# ICU with unversioned exports (e.g. ucnv_open). If conda's ICU lands in
# _internal/ it shadows the system DLL and causes "DLL load failed:
# specified procedure not found" on QtCore import.
# ---------------------------------------------------------------------------
_icu_exclude = {"icuuc.dll", "icudt58.dll", "icuin.dll", "icudt.dll"}
a.binaries = [(name, path, typ) for name, path, typ in a.binaries
              if name.lower() not in _icu_exclude]

# Append full mediapipe data tree
from PyInstaller.building.datastruct import Tree  # noqa: E402

a.datas += Tree(mediapipe_path, prefix="mediapipe")

# ---------------------------------------------------------------------------
# PYZ (bytecode archive)
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir mode
    name="FaceAnalyze2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app – no console window
)

# ---------------------------------------------------------------------------
# COLLECT (onedir bundle)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceAnalyze2",
)
