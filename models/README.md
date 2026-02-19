# Model Files

이 디렉터리는 런타임 모델 파일을 로컬에 두는 위치입니다.

## Required file
- `face_landmarker.task`

## Download (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "models/face_landmarker.task"
```

## Policy
- 모델 파일(`*.task`)은 git에 커밋하지 않습니다.
- `models/README.md`만 버전 관리 대상입니다.

