# FaceAnalyze2

## 1) 프로젝트 개요
FaceAnalyze2는 안면 움직임 영상을 정량 분석하여 neutral 대비 peak 변위, 좌우 비대칭(AI), 시각화 산출물을 생성하는 프로젝트입니다. 임상/연구 보조 목적의 분석 파이프라인과 프론트 연동용 `dynamicAnalysis(vd_path, motion)` API를 제공합니다.

## 2) 기능 요약
| 기능 | 진입점 | 설명 | 주요 산출물 |
|---|---|---|---|
| CLI 파이프라인 | `faceanalyze2 run ...` | landmarks → segment → align → metrics 일괄 실행 | `artifacts/<stem>/...` |
| 단계별 CLI | `landmarks/segment/align/metrics/viewer` | 단계별 디버깅/재실행 | 단계별 json/csv/png/html |
| Frontend API | `dynamicAnalysis(vd_path, motion)` | 프론트 표시용 이미지/metrics 계약 반환 | base64 PNG 5개 + metrics dict |
| Motion Viewer | `faceanalyze2 viewer generate ...` | 3D motion viewer HTML 생성 | `motion_viewer.html` |
| Gradio Demo | `python -m faceanalyze2.demo.gradio_app` | 데모 시연용 로컬 UI | 로컬 브라우저 데모 |

## 3) 설치/환경
- Python `3.11` 권장
- Windows + conda 예시:

```powershell
conda create -n facepalsy311 python=3.11 -y
conda activate facepalsy311
python -m pip install -e ".[dev]"
```

- Gradio demo 포함 설치:

```powershell
python -m pip install -e ".[dev,demo]"
```

- MediaPipe 모델 파일은 별도로 다운로드가 필요합니다.
  - 위치: `models/face_landmarker.task`
  - 다운로드 안내: [`models/README.md`](models/README.md)

## 4) Quickstart
### ① CLI로 한 번에 분석
```powershell
python -m faceanalyze2 run --video "D:\local\sample.mp4" --task smile --model "models/face_landmarker.task" --stride 2
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--video` | 분석할 입력 영상 경로 (.mp4) | (필수) |
| `--task` | 분석할 안면 움직임: `smile`, `brow`, `eyeclose` | (필수) |
| `--model` | MediaPipe Face Landmarker 모델 파일 경로 | `models/face_landmarker.task` |
| `--stride` | 프레임 샘플링 간격 (2 = 2프레임당 1프레임 처리) | `1` |


### ② viewer 생성
```powershell
python -m faceanalyze2 viewer generate --video "D:\local\sample.mp4"
```

### ③ Python API 호출 예제 (dynamicAnalysis 연동 예제)
```python
from faceanalyze2 import dynamicAnalysis

result = dynamicAnalysis(r"D:\local\sample.mp4", "big-smile")
print(result.keys())
print(result["metrics"].keys())
```

### ④ Gradio demo 실행
```powershell
python -m faceanalyze2.demo.gradio_app
```

## 5) Artifacts 구조
주요 산출물은 `artifacts/<video_stem>/` 아래에 생성됩니다.

| 단계 | 산출물 | 설명 |
|---|---|---|
| landmarks | `landmarks.npz` | 478개 얼굴 랜드마크 좌표 (T×478×3) |
| landmarks | `meta.json` | 영상 메타데이터 (fps, width, height 등) |
| segment | `segment.json` | neutral/peak/onset/offset 프레임 인덱스 |
| segment | `signals.csv` | 프레임별 task 신호 (raw + smoothed) |
| segment | `signals_plot.png` | 신호 시각화 및 구간 표시 그래프 |
| align | `landmarks_aligned.npz` | neutral 기준 정합된 랜드마크 좌표 (pixel) |
| align | `alignment.json` | 정합 파라미터 및 품질 통계 |
| metrics | `metrics.json` | ROI별 좌/우 변위, 비대칭 지수(AI), score |
| metrics | `metrics.csv` | metrics 요약 (CSV 형식) |
| metrics | `timeseries.csv` | 프레임별 ROI 좌/우 변위 시계열 |
| metrics | `plots/*.png` | ROI별 좌/우 displacement 그래프 |
| viewer | `motion_viewer.html` | 3D/2D 인터랙티브 모션 뷰어 |

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
- `metrics.roi_metrics.<roi>.AI`와 `score`:
  - `AI = abs(L_peak - R_peak) / max(eps, (L_peak + R_peak)/2)`
  - `score = (1 - clamp(AI, 0, 1)) * 100`
- `big-smile`은 mouth + area0~3 ROI를 포함하며, area0~3는 pair-table 간접 인덱스 기반입니다.

프론트 상세 계약/예시는 `docs/FRONTEND_INTEGRATION.md`를 참고하세요.

## 7) 보안/주의
- 환자 영상/프레임/landmark 결과는 PHI를 포함할 수 있습니다.
- `motion_viewer.html`, Gradio demo 화면/파일은 외부 공유 금지입니다.
- Gradio는 로컬 전용이며 `share=True`를 사용하지 않습니다.
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
  - `docs/DEMO_GRADIO.md`
  - `models/README.md`

