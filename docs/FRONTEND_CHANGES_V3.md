# FaceAnalyze2 v3 — 프론트엔드 변경사항 전달 문서

> **작성일**: 2026-02-27
> **작성자**: 헙헙이 (Agent-GitHub)
> **대상**: 프론트엔드 팀

---

## ⚠️ Breaking Changes 요약

| # | 변경 내용 | 영향도 | 프론트엔드 수정 필요 |
|---|----------|--------|-------------------|
| 1 | `"AI"` → `"Asymmetry Index"` 키 이름 변경 | **높음** | JSON 파싱 코드 수정 |
| 2 | 모든 동작에서 7개 ROI 반환 | **중간** | ROI 하드코딩 제거 또는 확장 |
| 3 | L/R Displacement 그래프 변경 | **낮음** | 이미지 표시 크기 조정 검토 |

---

## 1. 메트릭 키 이름 변경: `"AI"` → `"Asymmetry Index"`

### 변경 이유
- 사용자/의료진 피드백: "AI"가 "Artificial Intelligence"로 오해될 수 있음
- 전체 이름 "Asymmetry Index"로 명확화

### 이전 (v2) JSON 구조

```json
{
  "metrics": {
    "roi_metrics": {
      "mouth": {
        "L_peak": 12.34,
        "R_peak": 10.56,
        "AI": 0.078,
        "score": 85.0
      }
    }
  }
}
```

### 이후 (v3) JSON 구조

```json
{
  "metrics": {
    "roi_metrics": {
      "mouth": {
        "L_peak": 12.34,
        "R_peak": 10.56,
        "Asymmetry Index": 0.078,
        "score": 85.0
      }
    }
  }
}
```

### 프론트엔드 수정 포인트

`"AI"` 키로 접근하는 모든 코드를 `"Asymmetry Index"`로 변경:

```javascript
// Before (v2)
const ai = roiMetrics[roi].AI;

// After (v3)
const ai = roiMetrics[roi]["Asymmetry Index"];
```

> **검색 키워드**: 코드베이스에서 `.AI`, `["AI"]`, `['AI']`를 검색하여 모든 참조를 수정하세요.

---

## 2. 모든 동작에서 7개 ROI 반환

### 변경 이유
- 기존에는 동작별로 관련 ROI만 반환하여 데이터가 불완전했음
- 의료진이 모든 영역의 비대칭 지수를 비교하길 원함

### 7개 ROI 목록

| ROI 이름 | 설명 |
|----------|------|
| `mouth` | 입 영역 |
| `eye` | 눈 영역 |
| `eyebrow` | 눈썹 영역 |
| `area0_green` | 영역 0 (초록) |
| `area1_blue` | 영역 1 (파랑) |
| `area2_yellow` | 영역 2 (노랑) |
| `area3_red` | 영역 3 (빨강) |

### 이전 (v2) — 동작별로 다른 ROI

```
big-smile 응답:
  roi_metrics: { mouth, area0_green, area1_blue, area2_yellow, area3_red }  // 5개

blinking-motion 응답:
  roi_metrics: { eye }  // 1개

eyebrow-motion 응답:
  roi_metrics: { eyebrow }  // 1개
```

### 이후 (v3) — 모든 동작에서 동일한 7개 ROI

```
big-smile 응답:
  roi_metrics: { mouth, eye, eyebrow, area0_green, area1_blue, area2_yellow, area3_red }  // 7개

blinking-motion 응답:
  roi_metrics: { mouth, eye, eyebrow, area0_green, area1_blue, area2_yellow, area3_red }  // 7개

eyebrow-motion 응답:
  roi_metrics: { mouth, eye, eyebrow, area0_green, area1_blue, area2_yellow, area3_red }  // 7개
```

### 프론트엔드 수정 포인트

1. **동작별 ROI 하드코딩 제거**: 동작 타입에 따라 특정 ROI만 표시하는 로직이 있다면 수정

```javascript
// Before (v2) — 동작별 ROI 목록 하드코딩
const ROI_MAP = {
  "big-smile": ["mouth", "area0_green", "area1_blue", "area2_yellow", "area3_red"],
  "blinking-motion": ["eye"],
  "eyebrow-motion": ["eyebrow"],
};

// After (v3) — 응답의 roi_metrics 키를 동적으로 사용
const rois = Object.keys(result.metrics.roi_metrics);
```

2. **테이블/UI 레이아웃**: 항상 7개 ROI가 표시되므로 레이아웃 확인 필요

---

## 3. L/R Displacement 그래프 변경

### 변경 내용
- **이전**: 동작의 primary ROI 1개에 대한 L/R displacement 그래프
- **이후**: 7개 ROI 전부에 대한 4x2 서브플롯 그래프

### 영향
- `result["key value graph"]`의 base64 PNG 이미지 크기 증가 (해상도 상승)
- 기존과 동일하게 base64 PNG 문자열이므로 포맷 변경은 없음

### 프론트엔드 수정 포인트
- 그래프 이미지 표시 영역의 크기/비율 조정이 필요할 수 있음
- 서브플롯이 4x2 배치이므로 가로가 넓은 레이아웃 권장

---

## 전체 `dynamicAnalysis()` 응답 구조 (v3)

```json
{
  "key rest": "<base64 PNG — 안정 프레임>",
  "key exp": "<base64 PNG — 최대 표정 프레임>",
  "key value graph": "<base64 PNG — 7개 ROI L/R displacement 4x2 서브플롯>",
  "before regi": "<base64 PNG — 정렬 전 랜드마크>",
  "after regi": "<base64 PNG — 정렬 후 랜드마크>",
  "metrics": {
    "motion": "big-smile",
    "source": "landmarks_aligned.npz",
    "neutral_idx": 0,
    "peak_idx": 45,
    "neutral_frame_idx": 0,
    "peak_frame_idx": 45,
    "normalize": true,
    "interocular_px": 120.5,
    "roi_metrics": {
      "mouth": {
        "L_peak": 12.34,
        "R_peak": 10.56,
        "Asymmetry Index": 0.078,
        "score": 85.0
      },
      "eye": {
        "L_peak": 5.12,
        "R_peak": 4.89,
        "Asymmetry Index": 0.023,
        "score": 95.0
      },
      "eyebrow": {
        "L_peak": 3.45,
        "R_peak": 3.21,
        "Asymmetry Index": 0.036,
        "score": 92.0
      },
      "area0_green": { "L_peak": 0.0, "R_peak": 0.0, "Asymmetry Index": 0.0, "score": 100.0 },
      "area1_blue": { "L_peak": 0.0, "R_peak": 0.0, "Asymmetry Index": 0.0, "score": 100.0 },
      "area2_yellow": { "L_peak": 0.0, "R_peak": 0.0, "Asymmetry Index": 0.0, "score": 100.0 },
      "area3_red": { "L_peak": 0.0, "R_peak": 0.0, "Asymmetry Index": 0.0, "score": 100.0 }
    }
  }
}
```

---

## 3D Viewer 변경사항 (참고)

프론트엔드 API 파싱에는 영향 없으나, 3D Viewer HTML을 직접 임베딩하는 경우 참고:

| # | 변경 | 설명 |
|---|------|------|
| 1 | 검은화면 해결 | 2D→3D 모드 전환 시 renderer 0x0 resize 문제 수정 |
| 2 | 기본값 조정 | pointSize=2.8, coneScale=2.5 |
| 3 | 카메라 시작 | Z축 정면에서 시작 |
| 4 | 벡터 표면 추종 | z=0 flattening → 접선면(tangent plane) 투영. UI 라벨 "Surface tangent vectors" |

---

## 체크리스트

프론트엔드 팀에서 확인해야 할 항목:

- [ ] `"AI"` → `"Asymmetry Index"` 키 변경 반영
- [ ] 동작별 ROI 하드코딩 제거 (7개 ROI 동적 처리)
- [ ] 그래프 이미지 표시 영역 크기 검토
- [ ] 전체 통합 테스트 실행
