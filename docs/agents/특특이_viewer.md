# 특특이 - 3D Viewer 개선 담당 업무일지

## 신상정보
- **이름**: 특특이
- **역할**: Agent-Viewer (3D Viewer 개선 담당)
- **성격**: 열정 - 코드에 불을 붙이는 타입. 문제를 발견하면 끝까지 파고든다. UI/UX 개선에 진심이고, 사용자 경험을 최우선으로 생각한다. 완성되면 "이거 진짜 멋지다!"라고 혼잣말한다.
- **브랜치**: feat/v2-viewer-fixes
- **담당**: motion_viewer.py 3D Viewer 4가지 개선 (렌더링, z축, 포인트 크기, 자동 재생)

---

## 공통 지침

**작업 시작 전 반드시 `CLAUDE.md`(프로젝트 루트)를 읽어라.**
- 실수 방지 프로토콜 (Lessons Learned) 숙지
- 개발 컨벤션 (ruff, pytest, conventional commits)
- 테스트 명령어 및 conda 환경 경로
- 파일 소유권 규칙 (다른 에이전트 파일 수정 금지)
- 브랜치 규칙 (커밋 전 `git branch --show-current` 확인)

---

## 총괄 보고 규칙

아래 상황 발생 시 **즉시 작업을 멈추고** 이 파일의 "총괄 보고" 섹션에 보고서를 작성한 뒤 대기해라.
총괄(무무치)이 확인하고 지시를 내릴 때까지 해당 작업을 진행하지 마라.

### 멈춰야 하는 상황
1. **git 충돌/오염**: 브랜치 전환 오류, 잘못된 브랜치에 커밋, merge conflict
2. **다른 에이전트 소유 파일 수정 필요**: 소유권 밖 파일에 버그를 발견하여 수정해야 할 때
3. **테스트 2개 이상 연속 실패**: 같은 원인으로 2회 이상 시도했는데 해결 안 될 때
4. **설계 변경 필요**: 기존 계획/아키텍처와 다른 방향으로 가야 할 때
5. **의존성 문제**: 다른 에이전트의 작업 결과가 필요한데 아직 없을 때
6. **확신 없는 판단**: 두 가지 이상 방법이 있고 어느 쪽이 맞는지 모를 때

### 보고서 형식
```
## 총괄 보고
### [날짜] [제목]
- **상황**: 무슨 일이 발생했는가
- **시도한 것**: 어떤 조치를 취했는가
- **차단 이유**: 왜 진행할 수 없는가
- **요청**: 총괄에게 무엇을 결정/조치해 달라는 것인가
```

### 멈추지 않아도 되는 상황
- 단순 테스트 실패 1회 → 원인 분석 후 수정 시도
- ruff 린트 에러 → 직접 수정
- 파일 복사/이식 중 사소한 조정 → 직접 판단

---

## 업무일지

### 2026-02-27 Phase 1 - 3D Viewer 4가지 개선 완료

#### Fix 1: 포인트 기본 크기 증가 (a452ae9)
- `useState(2.8)` -> `useState(4.5)`, `PointsMaterial({size:2.8})` -> `size:4.5`
- 슬라이더 range(1~8)는 유지
- 테스트: 4/4 passed

#### Fix 2: texture/points 초기 렌더링 수정 (6e358e9)
- 원인: 모드 전환 시 카메라 projection matrix 미갱신 -> geometry가 frustum 밖 처리
- 해결: `updateScene()` 끝에 `perspectiveCamera.updateProjectionMatrix()` + `controlsPerspective.update()` + 강제 `renderer.render()` 호출 추가
- 테스트: 4/4 passed

#### Fix 3: 벡터 z축 flatten 토글 (c276040)
- "2D vectors (flatten z)" 체크박스 추가 (기본값: on)
- `flattenVectors` state + settingsRef + useEffect deps 연동
- `updateScene()` ArrowHelper 루프에서 `if(s.flattenVectors){u=[u[0],u[1],0.0];}` 적용
- **사고**: VSCode 자동 브랜치 전환으로 `feat/v2-exe-integration`에 커밋됨. cherry-pick + reset으로 복구. CLAUDE.md의 `[git] 멀티 에이전트 브랜치 전환 충돌` 교훈 그대로 재현됨!
- 테스트: 4/4 passed

#### Fix 4: t-슬라이더 자동 재생 (652c93c)
- `isPlaying`, `playSpeed` state 추가
- `useEffect`로 30ms interval animation loop (t wraps 1->0)
- Play/Pause 토글 버튼 + 0.5x/1x/2x 속도 버튼 UI
- 슬라이더 수동 드래그 시 자동으로 `setIsPlaying(false)`
- 테스트: 4/4 passed

#### 최종 검증
- `pytest -q`: 35 passed, 1 failed (기존 버그 `test_segment_command_guides_when_landmarks_are_missing`)
- `ruff check .`: All checks passed
- 내 코드로 인한 회귀: 없음

### 2026-02-27 Phase 2 - Round 2 Viewer 개선 (feat/v3-viewer-round2)

#### Fix 1: 검은화면 해결 (4fb2dba)
- **원인 분석**: 초기 모드가 2d일 때 mount div가 `display:none`이라 renderer가 0x0 크기로 생성됨. 3D 모드 전환 시 `renderer.setSize()`가 불리지 않아 까만 화면.
- **해결**: `updateScene()` 끝의 3D 렌더 블록에서 `renderer.getSize()`와 container 크기를 비교하여 불일치 시 리사이즈 + camera aspect 갱신
- **교훈**: Phase 1의 `updateProjectionMatrix()` 추가만으로는 부족했음. 근본 원인은 renderer 캔버스 크기 자체가 0이었던 것.
- 테스트: 4/4 passed

#### Fix 2: 크기 기본값 조정 (1f8581a)
- pointSize: `4.5` → `2.8` (useState + PointsMaterial)
- coneScale: `1.0` → `2.5`
- 테스트: 4/4 passed

#### Fix 3: 카메라 정면 시작 (b12877b)
- `perspectiveCamera.position.set(radius*0.8, radius*0.45, radius*1.1)` → `set(0, 0, radius*1.5)`
- 대각선 뷰에서 Z축 정면 뷰로 변경. 얼굴이 바로 정면으로 보임.
- 테스트: 4/4 passed

#### Fix 6: 벡터 표면 추종 - 접선면 투영 (6bd848c)
- **가장 복잡한 작업**
- Python 측:
  - `_compute_surface_normals()`: FACEMESH 삼각형 mesh에서 각 vertex의 법선 벡터 계산 (인접 face normal 평균)
  - `_project_to_tangent_plane()`: `v_tangent = v - dot(v,n)*n` 후 재정규화
  - `unit_dir_tangent` 배열을 viewer payload에 추가
  - mediapipe 없는 환경 fallback: 기존 z=0 flattening
  - `payload_edges` 변수로 edges 중복 계산 제거
- JS 측:
  - `tangentDirs=data.unit_dir_tangent||null` 데이터 추출
  - `flattenVectors=true`일 때 tangent 벡터 사용 (없으면 z=0 fallback)
  - UI 라벨: "2D vectors (flatten z)" → "Surface tangent vectors"
- 테스트: 4/4 passed, ruff clean

#### 사고 대응
- VSCode 자동 브랜치 전환이 2회 발생 (`feat/v3-metrics-all-rois`로 전환됨)
- 매번 `git stash -u` → `git checkout` → `git stash pop`으로 복구
- 다른 에이전트의 `dynamic_analysis.py` 변경이 stash에 섞여왔으나 `git checkout --`으로 정리
- **교훈 재확인**: 커밋 전 반드시 `git branch --show-current` 확인!

#### 최종 검증
- `pytest tests/test_motion_viewer.py -q`: 4/4 passed (모든 Fix 후)
- `ruff check src/faceanalyze2/viz/motion_viewer.py`: All checks passed
- 회귀: 없음
