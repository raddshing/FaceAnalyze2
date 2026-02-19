# Frontend Integration Guide

## Scope
이 문서는 프론트엔드 팀이 `dynamicAnalysis(vd_path, motion)`를 바로 연동하기 위한 계약 문서입니다.

## API Entry Point
```python
from faceanalyze2 import dynamicAnalysis
result = dynamicAnalysis(vd_path, motion)
```

## Input Contract
- `vd_path`: 분석 대상 mp4 경로 (문자열 또는 `Path`)
- `motion`: 아래 3개 중 하나
  - `big-smile`
  - `blinking-motion`
  - `eyebrow-motion`

내부적으로 motion은 CLI task로 매핑됩니다.
- `big-smile -> smile`
- `blinking-motion -> eyeclose`
- `eyebrow-motion -> brow`

## Output Contract (Stable)
반환 dict는 아래 키를 반드시 포함합니다.  
키 이름(공백 포함)은 프론트 계약이므로 변경하지 않습니다.

- `'key rest'`: base64 PNG 문자열
- `'key exp'`: base64 PNG 문자열
- `'key value graph'`: base64 PNG 문자열
- `'before regi'`: base64 PNG 문자열
- `'after regi'`: base64 PNG 문자열
- `'metrics'`: dict

이미지 문자열은 `data:image/png;base64,` prefix가 없는 raw base64입니다.

## Optional Extension Keys
현재/향후 구현에서 아래 키가 추가될 수 있습니다(없을 수도 있음).
- `roi_plots`: `{roi_name: base64_png}`
- `viewer_html_path`: 로컬 html 파일 경로
- `timeseries_csv`: base64 csv payload
- `timeseries_csv_path`: 로컬 csv 파일 경로

프론트는 필수 키 6개만 강제 처리하고, 나머지는 존재할 때만 렌더링하는 방식이 안전합니다.

## `metrics` Object
예시 구조:

```json
{
  "motion": "big-smile",
  "source": "landmarks_aligned.npz",
  "neutral_idx": 10,
  "peak_idx": 58,
  "neutral_frame_idx": 120,
  "peak_frame_idx": 168,
  "normalize": true,
  "interocular_px": 124.3,
  "roi_metrics": {
    "mouth": {
      "L_peak": 0.32,
      "R_peak": 0.21,
      "AI": 0.41,
      "score": 59.0
    }
  }
}
```

계산 규칙:
- `AI = abs(L_peak - R_peak) / max(eps, (L_peak + R_peak)/2)`
- `score = (1 - clamp(AI, 0, 1)) * 100`

`big-smile`에서는 mouth + area0~3 ROI가 포함됩니다.

## ROI Notes (Pair Indirect Indices)
area0~3 및 mouth/eye/eyebrow ROI는 MediaPipe landmark 번호 직접 리스트가 아니라, pair-table 간접 인덱스입니다.

- Pair table: `PAIR_L`, `PAIR_R` (각 220)
- ROI 리스트 값 범위: `0..219`
- 변환 함수:
  - `resolve_pair_indices(pair_idx_list) -> (L_landmarks, R_landmarks)`

즉 ROI 리스트는 먼저 pair index를 landmark index로 해석한 뒤 계산에 사용됩니다.

## Frontend Rendering Example (JS)
```js
function toDataUri(base64Png) {
  return `data:image/png;base64,${base64Png}`;
}

const images = {
  rest: toDataUri(result["key rest"]),
  exp: toDataUri(result["key exp"]),
  graph: toDataUri(result["key value graph"]),
  before: toDataUri(result["before regi"]),
  after: toDataUri(result["after regi"]),
};

const metricsRows = Object.entries(result.metrics.roi_metrics).map(([roi, v]) => ({
  roi,
  L_peak: v.L_peak,
  R_peak: v.R_peak,
  AI: v.AI,
  score: v.score,
}));
```

## Error Handling Guidance
- 잘못된 `motion`: `ValueError` 발생
- 선행 artifacts 누락:
  - landmarks/segment/meta 관련 파일 누락 시 예외 발생
- 프론트 권장 처리:
  - 사용자에게 짧은 오류 메시지 노출
  - 상세 로그는 collapse/console로 분리

## Security / PHI
- 반환 이미지와 viewer/demo 화면은 환자 얼굴 프레임이 포함될 수 있습니다.
- 외부 공유 금지 정책을 UI와 운영 문서에 명시해야 합니다.
- Gradio demo는 로컬 실행만 허용(`share=True` 금지).
