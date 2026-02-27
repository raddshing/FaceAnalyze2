#Requires -Version 5.1
<#
.SYNOPSIS
    Build FaceAnalyze2 standalone exe using PyInstaller (onedir mode).

.DESCRIPTION
    1. Runs PyInstaller with packaging/faceanalyze2.spec
    2. Copies models/ to dist/FaceAnalyze2/
    3. Creates config/ directory
    4. Generates README.txt

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$SpecFile    = Join-Path $ProjectRoot "packaging\faceanalyze2.spec"
$DistDir     = Join-Path $ProjectRoot "dist\FaceAnalyze2"

Write-Host "=== FaceAnalyze2 Build ===" -ForegroundColor Cyan
Write-Host "Project root : $ProjectRoot"
Write-Host "Spec file    : $SpecFile"
Write-Host ""

# ------------------------------------------------------------------
# 1. Run PyInstaller
# ------------------------------------------------------------------
Write-Host "[1/4] Running PyInstaller..." -ForegroundColor Yellow
pyinstaller --noconfirm --clean $SpecFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Host "[1/4] PyInstaller completed." -ForegroundColor Green

# ------------------------------------------------------------------
# 2. Copy models/
# ------------------------------------------------------------------
Write-Host "[2/4] Copying models..." -ForegroundColor Yellow
$ModelsSrc = Join-Path $ProjectRoot "models"
$ModelsDst = Join-Path $DistDir "models"

if (Test-Path $ModelsSrc) {
    if (Test-Path $ModelsDst) { Remove-Item -Recurse -Force $ModelsDst }
    Copy-Item -Recurse $ModelsSrc $ModelsDst
    Write-Host "  Copied: $ModelsSrc -> $ModelsDst" -ForegroundColor Green
} else {
    Write-Warning "models/ directory not found at $ModelsSrc"
}

# ------------------------------------------------------------------
# 3. Create config/
# ------------------------------------------------------------------
Write-Host "[3/4] Creating config directory..." -ForegroundColor Yellow
$ConfigDir = Join-Path $DistDir "config"
if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir | Out-Null
}
Write-Host "  Created: $ConfigDir" -ForegroundColor Green

# ------------------------------------------------------------------
# 4. Generate README.txt
# ------------------------------------------------------------------
Write-Host "[4/4] Generating README.txt..." -ForegroundColor Yellow
$ReadmePath = Join-Path $DistDir "README.txt"
@"
FaceAnalyze2 - Facial Palsy Analysis Tool
==========================================

How to run:
  Double-click FaceAnalyze2.exe

Directory structure:
  FaceAnalyze2.exe   - Main application
  _internal/         - Application runtime files (do not modify)
  models/            - ML model files (required)
  config/            - Settings (auto-generated)
  artifacts/         - Analysis results (auto-generated)

Requirements:
  - Windows 10 or later (64-bit)
  - No additional software installation required

Usage:
  1. Launch FaceAnalyze2.exe
  2. Select a video file or drag-and-drop
  3. Choose motion type and click Run
  4. View analysis results in the right panel

Troubleshooting:
  - If the app does not start, ensure models/face_landmarker.task exists.
  - Analysis artifacts are saved in the artifacts/ folder next to the exe.
  - Settings are stored in config/ as INI files (portable).
"@ | Set-Content -Path $ReadmePath -Encoding UTF8
Write-Host "  Created: $ReadmePath" -ForegroundColor Green

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Cyan
Write-Host "Output: $DistDir"
Write-Host ""

# Verify exe exists
$ExePath = Join-Path $DistDir "FaceAnalyze2.exe"
if (Test-Path $ExePath) {
    $size = (Get-Item $ExePath).Length / 1MB
    Write-Host "FaceAnalyze2.exe size: $([math]::Round($size, 1)) MB" -ForegroundColor Green
} else {
    Write-Warning "FaceAnalyze2.exe not found at expected location!"
}
