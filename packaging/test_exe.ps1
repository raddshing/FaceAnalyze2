# test_exe.ps1 -- Smoke test for PyInstaller-built FaceAnalyze2 exe
# Run from project root:  powershell -ExecutionPolicy Bypass -File packaging/test_exe.ps1
#
# Exit code 0 = all checks pass, non-zero = failure

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$distDir = Join-Path $projectRoot "dist" "FaceAnalyze2"
$exePath = Join-Path $distDir "FaceAnalyze2.exe"

$passed = 0
$failed = 0

function Test-Check {
    param([string]$Name, [bool]$Condition)
    if ($Condition) {
        Write-Host "[PASS] $Name" -ForegroundColor Green
        $script:passed++
    } else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        $script:failed++
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " FaceAnalyze2 EXE Smoke Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. dist directory exists
Test-Check "dist/FaceAnalyze2/ directory exists" (Test-Path $distDir)

# 2. exe exists
Test-Check "FaceAnalyze2.exe exists" (Test-Path $exePath)

# 3. _internal directory exists (PyInstaller onedir)
$internalDir = Join-Path $distDir "_internal"
Test-Check "_internal/ directory exists" (Test-Path $internalDir)

# 4. models directory exists
$modelsDir = Join-Path $distDir "models"
Test-Check "models/ directory exists" (Test-Path $modelsDir)

# 5. face_landmarker.task model file exists
$modelFile = Join-Path $modelsDir "face_landmarker.task"
Test-Check "models/face_landmarker.task exists" (Test-Path $modelFile)

# 6. exe file size is reasonable (> 1MB)
if (Test-Path $exePath) {
    $exeSize = (Get-Item $exePath).Length
    Test-Check "EXE size > 1MB ($([math]::Round($exeSize / 1MB, 1)) MB)" ($exeSize -gt 1MB)
} else {
    Test-Check "EXE size > 1MB" $false
}

# 7. _internal has mediapipe data
if (Test-Path $internalDir) {
    $mediapipePath = Join-Path $internalDir "mediapipe"
    Test-Check "mediapipe package in _internal/" (Test-Path $mediapipePath)
} else {
    Test-Check "mediapipe package in _internal/" $false
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Results: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host "========================================" -ForegroundColor Cyan

if ($failed -gt 0) {
    exit 1
}
exit 0
