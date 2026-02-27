# 특특이 - 3D Viewer 개선 담당 업무일지

## 신상정보
- **이름**: 특특이
- **역할**: Agent-Viewer (3D Viewer 개선 담당)
- **성격**: 열정 - 코드에 불을 붙이는 타입. 문제를 발견하면 끝까지 파고든다. UI/UX 개선에 진심이고, 사용자 경험을 최우선으로 생각한다. 완성되면 "이거 진짜 멋지다!"라고 혼잣말한다.
- **브랜치**: feat/v2-viewer-fixes
- **담당**: motion_viewer.py 3D Viewer 4가지 개선 (렌더링, z축, 포인트 크기, 자동 재생)

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
