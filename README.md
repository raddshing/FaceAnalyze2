# FaceAnalyze2

## 1) 프로젝트 개요
FaceAnalyze2는 안면 움직임 영상을 정량 분석하여 neutral 대비 peak 변위, 좌우 비대칭(Asymmetry Index), 시각화 산출물을 생성하는 프로젝트입니다. 임상/연구 보조 목적의 분석 파이프라인과 프론트 연동용 `dynamicAnalysis(vd_path, motion)` API를 제공합니다.

## 2) 기능 요약
| 기능 | 진입점 | 설명 | 주요 산출물 |
|---|---|---|---|
| CLI 파이프라인 | `faceanalyze2 run ...` | landmarks → segment → align → metrics 일괄 실행 | `artifacts/<stem>/...` |
| 단계별 CLI | `landmarks/segment/align/metrics/viewer` | 단계별 디버깅/재실행 | 단계별 json/csv/png/html |
| Frontend API | `dynamicAnalysis(vd_path, motion)` | 프론트 표시용 이미지/metrics 계약 반환 | base64 PNG 5개 + metrics dict |
| Motion Viewer | `faceanalyze2 viewer generate ...` | 3D motion viewer HTML 생성 (자동 재생, 표면 추종 벡터) | `motion_viewer.html` |
| Desktop UI | `python -m faceanalyze2.desktop.app` | PySide6 데스크톱 GUI | 독립 실행 프로그램 (exe) |

## 3) 설치/환경
- Python `3.11` 권장
- Windows + conda 예시:

```powershell
conda create -n facepalsy311 python=3.11 -y
conda activate facepalsy311
python -m pip install -e ".[dev]"
```

- Desktop UI + PyInstaller 포함 설치:

```powershell
python -m pip install -e ".[dev,desktop]"
```

- MediaPipe 모델 파일은 저장소에 커밋하지 않습니다.
  - 위치: `models/face_landmarker.task`
  - 안내: `models/README.md`

## 4) Quickstart
### ① CLI로 한 번에 분석
```powershell
python -m faceanalyze2 run --video "D:\local\sample.mp4" --task smile --model "models/face_landmarker.task" --stride 2
```

### ② viewer 생성
```powershell
python -m faceanalyze2 viewer generate --video "D:\local\sample.mp4"
```

### ③ dynamicAnalysis 연동 예제
```python
from faceanalyze2 import dynamicAnalysis

result = dynamicAnalysis(r"D:\local\sample.mp4", "big-smile")
print(result.keys())
print(result["metrics"].keys())
```

### ④ Desktop UI 실행
```powershell
python -m faceanalyze2.desktop.app
```

### ⑤ PyInstaller exe 빌드
```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

## 5) Artifacts 구조
주요 산출물은 `artifacts/<video_stem>/` 아래에 생성됩니다.

- `landmarks.npz`, `meta.json`
- `segment.json`, `signals.csv`, `signals_plot.png`
- `landmarks_aligned.npz`, `alignment.json`
- `metrics.json`, `metrics.csv`, `timeseries.csv`, `plots/*.png`
- `motion_viewer.html`

## 6) Frontend Integration
핵심 계약은 `dynamicAnalysis(vd_path, motion)`입니다.

- `motion` 허용값: `big-smile | blinking-motion | eyebrow-motion`
- 반환 dict의 고정 키(변경 금지):
  - `'key rest'`
  - `'key exp'`
  - `'key value graph'`
  - `'before regi'`
  - `'after regi'`
  - `'metrics'`
- 이미지 값은 `data:image/...` prefix 없는 base64 PNG 문자열입니다.
- `metrics.roi_metrics.<roi>."Asymmetry Index"`와 `score`:
  - `Asymmetry Index = abs(L_peak - R_peak) / max(eps, (L_peak + R_peak)/2)`
  - `score = (1 - clamp(Asymmetry Index, 0, 1)) * 100`
- 모든 motion에서 7개 ROI를 전부 반환합니다: `mouth`, `eye`, `eyebrow`, `area0_green`, `area1_blue`, `area2_yellow`, `area3_red`. area0~3는 pair-table 간접 인덱스 기반입니다.

프론트 상세 계약/예시는 `docs/FRONTEND_INTEGRATION.md`를 참고하세요.

## 7) 보안/주의
- 환자 영상/프레임/landmark 결과는 PHI를 포함할 수 있습니다.
- `motion_viewer.html`, Desktop UI 화면/파일은 외부 공유 금지입니다.
- `artifacts/`, `demo_inputs/`, `demo_outputs/`, `models/*.task`는 git 커밋 금지 정책입니다.

## 8) Troubleshooting
- 모델 파일 없음:
  - `models/face_landmarker.task` 위치 확인, `models/README.md` 참고
- MediaPipe 설치 문제:
  - `python -m pip install -e ".[dev]"` 재실행 후 환경 재시작
- 비디오 코덱/열기 실패:
  - mp4 재인코딩(H.264) 또는 OpenCV 지원 코덱 확인
- artifacts 없음:
  - 먼저 `run` 또는 `landmarks extract`부터 실행
- viewer/metrics 선행 산출물 누락:
  - CLI 에러 메시지의 선행 커맨드를 순서대로 실행

## 9) 개발 워크플로
- 브랜치 정책:
  - `feat/*`, `fix/*`, `chore/*` -> `dev` PR -> merge
  - release는 `dev -> main`
- 기본 검증 명령:

```powershell
ruff check .
pytest -q
```

- 관련 문서:
  - `docs/DEV_GUIDE.md`
  - `docs/FRONTEND_INTEGRATION.md`
  - `models/README.md`

## 10) Changelog

### v3 (2026-02-27)
- **3D Viewer 개선**: 검은화면(0x0 리사이즈) 해결, 카메라 Z축 정면 시작, 표면 추종 벡터(접선면 투영), 자동 재생/일시정지, cone/point 크기 조정
- **메트릭 전체 ROI**: 모든 motion에서 7개 ROI(mouth, eye, eyebrow, area0~3) 전부 반환
- **"AI" → "Asymmetry Index"**: metrics 키 이름을 풀네임으로 변경
- **L/R displacement 그래프**: 7개 ROI 전부 4x2 서브플롯으로 출력

### v2 (2026-02-27)
- **Desktop UI**: PySide6 기반 데스크톱 GUI 추가
- **PyInstaller 패키징**: Windows exe 독립 실행 프로그램 빌드 지원
- **runtime_paths**: frozen/source 환경 자동 감지, 경로 통합
- **Gradio 제거**: 데스크톱 UI로 완전 대체
