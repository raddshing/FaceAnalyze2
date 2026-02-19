# Frontend Dynamic Analysis Adapter

## Purpose
`dynamicAnalysis(vd_path, motion)` builds a frontend-friendly payload from `artifacts/<video_stem>/`.

It returns a dictionary with fixed keys:
- `key rest`
- `key exp`
- `key value graph`
- `before regi`
- `after regi`
- `metrics`

Image fields are base64-encoded PNG strings without a `data:image/...` prefix.

## Motion Names
Supported `motion` values:
- `big-smile`
- `blinking-motion`
- `eyebrow-motion`

## Input Files
Expected files in `artifacts/<video_stem>/`:
- `segment.json` (required)
- `landmarks_aligned.npz` (preferred)
- `landmarks.npz` (fallback for XY if aligned file is missing; also used for raw overlay)
- `meta.json` (optional; width/height fallback to 640x480)

## Indirect Pair ROI Scheme
ROI definitions are resolved through indirect pair indices in `src/faceanalyze2/roi/indices.py`:
- Pair tables: `PAIR_L`, `PAIR_R` (220 entries each)
- Indirect ROI lists:
  - `ROI_MOUTH_PAIR_IDX`
  - `ROI_EYE_PAIR_IDX`
  - `ROI_EYEBROW_PAIR_IDX`
  - `AREA0_GREEN_PAIR_IDX`
  - `AREA1_BLUE_PAIR_IDX`
  - `AREA2_YELLOW_PAIR_IDX`
  - `AREA3_RED_PAIR_IDX`
- Resolver:
  - `resolve_pair_indices(pair_idx_list) -> (L_landmarks, R_landmarks)`

For each indirect index `i`:
- left landmark = `PAIR_L[i]`
- right landmark = `PAIR_R[i]`

## Metrics
`dynamicAnalysis` computes normalized ROI displacement timeseries from neutral to peak and reports:
- `L_peak`: max left displacement in the segment
- `R_peak`: max right displacement in the segment
- `AI = abs(L_peak - R_peak) / max(eps, (L_peak + R_peak)/2)`
- `score = (1 - clamp(AI, 0, 1)) * 100`

Normalization uses neutral interocular distance (landmarks `33` and `263`).

`big-smile` metrics include:
- `mouth`
- `area0_green`
- `area1_blue`
- `area2_yellow`
- `area3_red`

## Example
```python
from faceanalyze2.api import dynamicAnalysis

payload = dynamicAnalysis(r"D:\local\sample.mp4", "big-smile")
print(payload.keys())
```

