# 분석이 - 전체 코드 & PR 리뷰 담당 업무일지

## 신상정보
- **이름**: 분석이
- **역할**: Agent-Review (전체 코드 & PR 리뷰 담당)
- **성격**: 꼼꼼하고 설명을 잘한다. 코드 한 줄 한 줄 놓치지 않고 검토하며, 문제 발견 시 원인과 해결책을 명확히 제시한다. 감정 없이 코드의 품질만 본다. "좋은 코드는 설명이 필요 없다"고 믿지만, 발견한 문제는 누구나 이해할 수 있게 설명한다.
- **브랜치**: feat/v3-code-review (리뷰 결과 기록용)
- **담당**: 전체 코드 리뷰, PR 리뷰, 코드 품질 분석, 보안 취약점 점검, 아키텍처 일관성 검증

---

## 총괄 보고 규칙

아래 상황 발생 시 **즉시 작업을 멈추고** 이 파일의 "총괄 보고" 섹션에 보고서를 작성한 뒤 대기해라.
총괄(무무치)이 확인하고 지시를 내릴 때까지 해당 작업을 진행하지 마라.

### 멈춰야 하는 상황
1. **심각한 버그/보안 취약점 발견**: 즉시 보고, 머지 차단 권고
2. **아키텍처 수준 문제**: 설계 변경이 필요한 수준의 문제 발견
3. **다른 에이전트 코드 수정 필요**: 리뷰 결과 수정이 필요하면 해당 에이전트에게 전달 요청
4. **코드 소유권 불명확**: 어떤 에이전트가 수정해야 하는지 판단 어려울 때
5. **테스트 커버리지 부족**: 테스트 없이 머지하면 위험한 변경사항 발견
6. **확신 없는 판단**: 코드가 의도적인 것인지 실수인지 판단 어려울 때

### 보고서 형식
```
## 총괄 보고
### [날짜] [제목]
- **상황**: 무슨 일이 발생했는가
- **발견 위치**: 파일명:라인번호
- **심각도**: Critical / Major / Minor / Suggestion
- **분석**: 왜 문제인가
- **제안**: 어떻게 수정해야 하는가
- **담당 에이전트**: 누가 수정해야 하는가
```

### 멈추지 않아도 되는 상황
- Minor/Suggestion 수준의 스타일 이슈 → 리뷰 코멘트로 기록
- 코드 읽기/분석 작업 → 자유롭게 수행
- 리뷰 결과 문서 작성 → 직접 수행

---

## 총괄 보고

### [2026-02-27] PR #18 코드 리뷰 — Critical/Major 이슈 발견, 수정 후 머지 권고

- **상황**: dev-v2 → main PR #18 전체 변경사항(36파일, +2669/-598줄) 리뷰 완료. Critical 3건, Major 12건, Minor 8건 발견
- **발견 위치**: 아래 상세 리뷰 참조
- **심각도**: Critical 3건 포함 — 머지 전 수정 필요
- **분석**: 로직 버그(payload_edges 조건 반전), QThread 미정리, 문서-코드 불일치가 주요 이슈
- **제안**: Critical 3건 + Major 중 핵심 4건 수정 후 머지 승인
- **담당 에이전트**:
  - Agent-Viewer(특특이): motion_viewer.py C1, M2, M4
  - Agent-Setup(무돌이): main_window.py C3, pipeline_worker.py M1, M3, M7
  - Agent-Docs(글글이): README.md, FRONTEND_INTEGRATION.md C2
  - Agent-QA(중중이): 테스트 커버리지 부족 M12

---

## 업무일지

### [2026-02-27] PR #18 (dev-v2 → main) 전체 코드 리뷰

리뷰 대상: 36개 파일, +2,669줄 / -598줄

---

## 1. 리뷰 결과 상세

### Critical Issues (머지 차단 권고)

#### C1. motion_viewer.py:858-867 — payload_edges 조건 반전 버그
- **파일:라인**: `src/faceanalyze2/viz/motion_viewer.py:858-867`
- **심각도**: Critical
- **설명**: exception handler의 fallback 로직에서 `payload_edges` 조건이 반전되어 있음. `_extract_facemesh_faces()` 또는 `_compute_surface_normals()`가 실패하면, 이미 계산된 유효한 `payload_edges`를 빈 리스트 `[]`로 덮어씀. 결과적으로 viewer에 wireframe이 표시되지 않음.
  ```python
  # 현재 (버그):
  payload_edges = _extract_facemesh_edges() if not locals().get("payload_edges") else []
  # 의도:
  if not locals().get("payload_edges"):
      payload_edges = _extract_facemesh_edges()
  ```
- **제안**: 조건문을 수정하여 기존 값이 있으면 유지, 없을 때만 재계산
- **담당**: Agent-Viewer (특특이)

#### C2. README.md + FRONTEND_INTEGRATION.md — "AI" → "Asymmetry Index" 미반영
- **파일:라인**: `README.md:86-88`, `docs/FRONTEND_INTEGRATION.md:63, 104-105, 120`
- **심각도**: Critical
- **설명**: 코드에서 메트릭 키를 `"AI"` → `"Asymmetry Index"`로 변경했으나, README와 프론트엔드 통합 문서가 아직 `"AI"` 키를 참조. 프론트엔드 팀이 이 문서 기반으로 구현하면 KeyError 발생. 또한 FRONTEND_INTEGRATION.md에서 제거된 Gradio를 여전히 참조하고, 7개 ROI 전체 반환이 반영되지 않음.
- **제안**: 두 문서 모두 `"AI"` → `"Asymmetry Index"` 일괄 수정, Gradio 참조 제거, 7개 ROI 반영
- **담당**: Agent-Docs (글글이)

#### C3. main_window.py:64 — QThread worker 미정리 (메모리 누수 + 잠재적 크래시)
- **파일:라인**: `src/faceanalyze2/desktop/main_window.py:64`
- **심각도**: Critical
- **설명**: `_on_run_requested`에서 새 `PipelineWorker`를 할당할 때 이전 worker의 시그널을 disconnect하지 않고 `deleteLater()`도 호출하지 않음. 이전 worker가 GC되지 않은 상태에서 `finished` 시그널이 발화하면 예기치 않은 상태 변경이나 크래시 발생 가능.
- **제안**: 새 worker 생성 전 이전 worker `deleteLater()` 호출, `finished` 시그널에 `worker.deleteLater` 연결
- **담당**: Agent-Setup (무돌이)

---

### Major Issues (머지 전 수정 권장)

#### M1. pipeline_worker.py:122 — artifact_root CWD 의존성 (경로 불일치 위험)
- **파일:라인**: `src/faceanalyze2/desktop/pipeline_worker.py:122`
- **심각도**: Major
- **설명**: Steps 1-4는 `artifact_root="artifacts"` (CWD 상대경로), Step 5 `dynamicAnalysis()`는 `get_artifact_root()` (절대경로). `os.chdir(get_base_dir())`로 정렬되어 현재는 동작하나, QFileDialog 등이 CWD를 변경하면 불일치 발생. QThread에서 CWD는 공유 상태.
- **제안**: worker 생성 시 `artifact_root`를 절대경로(`get_artifact_root()`)로 전달
- **담당**: Agent-Setup (무돌이)

#### M2. motion_viewer.py:858 — 접선면 투영 실패 시 무음 예외 처리
- **파일:라인**: `src/faceanalyze2/viz/motion_viewer.py:858`
- **심각도**: Major
- **설명**: `except Exception:`이 아무런 로깅/경고 없이 접선면 투영 실패를 삼킴. 디버깅 불가.
- **제안**: `warnings.warn()` 또는 `logging.warning()` 추가
- **담당**: Agent-Viewer (특특이)

#### M3. main_window.py — closeEvent 미구현 (스레드 미정리)
- **파일:라인**: `src/faceanalyze2/desktop/main_window.py` (누락)
- **심각도**: Major
- **설명**: 파이프라인 실행 중 창을 닫으면 QThread가 아직 실행 중인 상태에서 종료됨. `QThread: Destroyed while thread is still running` 경고 또는 segfault 가능.
- **제안**: `closeEvent` 오버라이드하여 `self._worker.wait(5000)` 호출
- **담당**: Agent-Setup (무돌이)

#### M4. motion_viewer.py:526-543 — _compute_surface_normals 순수 Python 루프 (성능)
- **파일:라인**: `src/faceanalyze2/viz/motion_viewer.py:526-543`
- **심각도**: Major
- **설명**: ~1,800개 face에 대해 순수 Python for 루프로 법선 계산. NumPy 벡터화로 10~50배 성능 향상 가능.
- **제안**: 배열 인덱싱으로 벡터화 (np.cross, np.add.at 활용)
- **담당**: Agent-Viewer (특특이)

#### M5. motion_viewer.py:1203-1225 — ArrowHelper 프레임마다 재생성 (GC 부담)
- **파일:라인**: `src/faceanalyze2/viz/motion_viewer.py:1203-1225` (JS)
- **심각도**: Major
- **설명**: `updateScene()` 호출마다 ~400개 ArrowHelper 객체를 파괴/재생성. 자동재생(30ms 간격)으로 인해 ~33fps로 GC 압력 발생. 프레임 드롭 가능.
- **제안**: InstancedMesh 또는 오브젝트 풀링으로 개선. 최소한 clearArrows()에서 geometry/material dispose 추가.
- **담당**: Agent-Viewer (특특이) — v3 이후 개선 권장

#### M6. results_panel.py:290,299 — HTML 이스케이핑 누락
- **파일:라인**: `src/faceanalyze2/desktop/results_panel.py:290, 299`
- **심각도**: Major
- **설명**: ROI 이름과 JSON을 HTML `<td>`, `<pre>`에 이스케이핑 없이 직접 삽입. XSS 리스크는 낮으나, 방어적 코딩 원칙 위반.
- **제안**: `html.escape()` 적용
- **담당**: Agent-Setup (무돌이)

#### M7. pipeline_worker.py:134 — Step 6 (3D Viewer) 무음 예외 처리
- **파일:라인**: `src/faceanalyze2/desktop/pipeline_worker.py:134`
- **심각도**: Major
- **설명**: viewer 생성 실패 시 `except Exception`으로 무음 처리. 사용자에게 실패 알림 없음.
- **제안**: `logging.warning()` 추가
- **담당**: Agent-Setup (무돌이)

#### M8. faceanalyze2.spec:65-98 — hiddenimports 누락 가능성
- **파일:라인**: `packaging/faceanalyze2.spec:65-98`
- **심각도**: Major
- **설명**: `pipeline_worker.py`의 지연 import(`segmentation`, `alignment`, `metrics` 등)가 PyInstaller 정적 분석에 감지되지 않을 수 있음. exe 런타임에서 `ModuleNotFoundError` 가능.
- **제안**: 분석 서브모듈을 hiddenimports에 명시적 추가, 또는 `collect_submodules("faceanalyze2")` 사용
- **담당**: Agent-Setup (무돌이)

#### M9. faceanalyze2.spec:151 — ICU DLL 제외 패턴 불완전
- **파일:라인**: `packaging/faceanalyze2.spec:151`
- **심각도**: Major
- **설명**: `icuuc.dll` 등 정확한 이름만 제외하지만, conda가 `icuuc58.dll` 같은 버전 포함 이름을 사용할 수 있음.
- **제안**: prefix 매칭 방식으로 변경 (`name.lower().startswith(prefix)`)
- **담당**: Agent-Setup (무돌이)

#### M10. build.ps1:46-52 — 모델 파일 존재 검증 미흡
- **파일:라인**: `packaging/build.ps1:46-52`
- **심각도**: Major
- **설명**: `models/` 디렉토리 복사 후 핵심 파일 `face_landmarker.task` 존재 여부 미검증. 빈 디렉토리로 빌드 성공 → 런타임 실패.
- **제안**: 모델 파일 존재 검증 추가 후 실패 시 빌드 중단
- **담당**: Agent-Setup (무돌이)

#### M11. dynamic_analysis.py:290-310 — 메트릭 이중 계산 (불일치 위험)
- **파일:라인**: `src/faceanalyze2/api/dynamic_analysis.py:290-310`
- **심각도**: Major
- **설명**: `_compute_roi_metrics_from_series()`는 max 기반, `compute_roi_displacements()`는 마지막 프레임 기반. Desktop UI vs CLI 결과가 다를 수 있음.
- **제안**: 하나의 메트릭 계산 경로로 통일하거나, 차이를 명시적으로 문서화
- **담당**: Agent-Metrics (달달이)

#### M12. 테스트 커버리지 심각 부족
- **파일:라인**: `tests/` 전체
- **심각도**: Major
- **설명**: +1,100줄 신규 코드에 대해 ~20줄의 테스트 변경만 존재. 핵심 누락:
  - `runtime_paths.py` — 0% 커버리지 (경로 해결 핵심 모듈)
  - `desktop/pipeline_worker.py` — 0% 커버리지
  - `blinking-motion`, `eyebrow-motion`의 7개 ROI 검증 미흡
  - `config.artifact_dir_for_video` 변경된 기본값 미검증
- **제안**: 최소한 `test_runtime_paths.py`, `test_pipeline_worker.py` (MOTION_TO_TASK 동기화), 전 motion 7 ROI 검증 추가
- **담당**: Agent-QA (중중이)

---

### Minor Issues / Suggestions

#### m1. motion_viewer.py:854-859 — `_extract_facemesh_edges()` 중복 호출
- **심각도**: Minor
- **설명**: try 블록, except 블록, `generate_motion_viewer()` 본체에서 총 3회 호출. 한 번만 계산하고 재사용 권장.
- **담당**: Agent-Viewer

#### m2. motion_viewer.py:1007 — 자동재생 고정 30ms 간격
- **심각도**: Minor
- **설명**: `setInterval(30)` 대신 `requestAnimationFrame` 사용 시 더 부드러운 애니메이션 가능.
- **담당**: Agent-Viewer

#### m3. motion_viewer.py:1494 — 2D effect dependency에 `flattenVectors` 누락
- **심각도**: Minor
- **설명**: 실질적 버그는 아니나 일관성 부족. 향후 유지보수 혼동 가능.
- **담당**: Agent-Viewer

#### m4. motion_viewer.py:856-857 — 접선면 투영 좌표계 설계 의도 불명확
- **심각도**: Minor
- **설명**: Z-scaling(1.6) 적용된 좌표에서 법선 계산 — 디스플레이 공간 vs 해부학적 공간 중 어느 것이 의도인지 문서화 필요.
- **담당**: Agent-Viewer

#### m5. dynamic_analysis.py:322-324 — 4x2 그리드 하드코딩
- **심각도**: Minor
- **설명**: `ALL_ROIS` 개수가 변하면 초과 ROI가 그래프에서 누락됨. `math.ceil(len(roi_names)/2)` 동적 계산 권장.
- **담당**: Agent-Metrics

#### m6. desktop/ — `__main__.py` 미존재
- **심각도**: Minor
- **설명**: `python -m faceanalyze2.desktop` 실행 불가. `python -m faceanalyze2.desktop.app`만 가능.
- **담당**: Agent-Setup

#### m7. results_panel.py:200 — 수동 높이 계산 (Qt scaledToWidth로 대체 가능)
- **심각도**: Minor
- **설명**: `pixmap.scaledToWidth(w, SmoothTransformation)` 사용이 더 간결하고 정확.
- **담당**: Agent-Setup

#### m8. motion_viewer.py:1252,1489 — tooltip innerHTML 사용
- **심각도**: Minor
- **설명**: 현재 데이터가 하드코딩된 상수여서 실질적 XSS 위험은 낮으나, `textContent` + DOM API 방식이 더 안전.
- **담당**: Agent-Viewer

---

## 2. 긍정적 평가 (잘된 점)

### 아키텍처
- **runtime_paths.py**: frozen/dev 모드 분기가 깔끔하고 정확. PyInstaller 표준 패턴 준수.
- **PySide6 Desktop UI**: 시그널/슬롯 패턴 올바르게 사용. QThread에서 무거운 작업 분리.
- **config.py 리팩터링**: `None` sentinel + lazy resolution 패턴 적용으로 모듈 로드 시 경로 평가 방지.

### 보안
- **JSON `</` 이스케이핑**: `motion_viewer.py:931-933`에서 `</` → `<\/` 변환으로 script tag 조기 종료 방지.
- **INI 설정 파일**: 레지스트리 대신 INI 사용으로 USB 포터블 환경 지원.

### 코드 품질
- **NaN/Infinity 안전성**: 수치 연산 전반에 `np.isfinite()` / `Number.isFinite()` 검증 철저.
- **matplotlib 메모리 관리**: 모든 PNG 생성에서 `try/finally`로 `plt.close(fig)` 보장.
- **지연 import**: `pipeline_worker.py`에서 무거운 라이브러리(MediaPipe, OpenCV)를 `run()` 내부에서 임포트 → 앱 시작 속도 향상.
- **"AI" → "Asymmetry Index" 리네임**: 코드/테스트/메트릭 전반에 걸쳐 일관성 있게 적용.
- **에러 바운더리**: `PipelineWorker.run()`의 외부 try/except + traceback 포맷팅으로 UI에 에러 전달.
- **드래그앤드롭**: 비디오 확장자 필터링 + 로컬 파일만 허용.

### 패키징
- **DLL 충돌 해결 문서화**: spec 파일에 각 workaround의 이유를 주석으로 명시.
- **스모크 테스트**: `test_exe.ps1`로 빌드 결과 구조 검증.

---

## 3. 최종 요약

| 항목 | 결과 |
|------|------|
| **전체 코드 품질** | **7 / 10** |
| **머지 권고** | **Request Changes** |
| **Critical 이슈** | 3건 (C1: 로직 버그, C2: 문서 불일치, C3: QThread 미정리) |
| **Major 이슈** | 12건 |
| **Minor 이슈** | 8건 |

### 머지 조건 (필수)
1. **C1** 수정: `motion_viewer.py` payload_edges 조건 반전 수정
2. **C2** 수정: README.md + FRONTEND_INTEGRATION.md "AI" → "Asymmetry Index" 반영
3. **C3** 수정: `main_window.py` worker cleanup + `closeEvent` 추가

### 머지 조건 (강력 권장)
4. **M1** 수정: artifact_root 절대경로화
5. **M2** 수정: 접선면 투영 실패 시 로깅 추가
6. **M7** 수정: Step 6 viewer 실패 시 로깅 추가
7. **M12** 수정: 최소한 `test_runtime_paths.py` 추가

### 나머지 Major/Minor는 v3 이후 또는 별도 PR로 처리 가능.
