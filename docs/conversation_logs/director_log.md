# 총괄 (Director) 대화 기록

## 2026-02-27 - Phase 0: 프로젝트 초기 설정

### 상황 분석
- **현재 상태**: `feat/desktop-3d-viewer` 브랜치에서 exe 개발 작업 진행 중
- **origin/main**: 기본 파이프라인 + Gradio 데모 + 최근 수정 (metrics CLI score, viewer 2D magnitude)
- **차이점**: origin/main에는 desktop UI, packaging, runtime_paths가 없음 (24개 파일, +1953/-559)

### 사용자 피드백 (3D Viewer)
1. texture/points 모드 선택 시 즉시 보이지 않음 → Ctrl+스크롤 필요
2. 벡터 z축 depth가 부자연스러움 → 시각화에서 제거 옵션 필요
3. 포인트 기본 크기 증가 필요
4. t-슬라이더 자동 재생/일시정지 버튼 추가

### 결정 사항
1. **조직 구조**: 플랫 구조 (총괄 → 3 에이전트 직속, 중간관리자 없음)
2. **에이전트 구성**: Agent-Setup, Agent-Viewer, Agent-QA
3. **브랜치 전략**: origin/main 기반, feat/v2-* 브랜치, dev-v2 통합
4. **병렬 전략**: Phase 1에서 Agent-Setup과 Agent-Viewer 병렬 작업 (다른 파일)

### Phase 0 완료 내역
- [x] 현재 작업 커밋 및 백업 (`backup/exe-dev-v1`, tag `v1-exe-dev`)
- [x] 브랜치 3개 생성: `dev-v2`, `feat/v2-exe-integration`, `feat/v2-viewer-fixes`
- [x] CLAUDE.md 새로 생성 (v2 가이드, 실수 방지 지침 11개 이관)
- [x] CLAUDE.md를 3개 브랜치에 모두 반영 (cherry-pick)
- [x] 대화 기록 디렉토리 생성

### 3D Viewer 코드 분석 결과 (motion_viewer.py, 1583줄)
- **Fix 1 원인**: updateScene() 후 카메라 projection matrix 미갱신
- **Fix 2 원인**: `DEFAULT_Z_SCALE = 1.6`으로 z축 60% 증폭, JS 단 토글로 해결 추천
- **Fix 3**: 기본값 2.8 → 4.5로 변경 (useState + PointsMaterial)
- **Fix 4**: isPlaying state + useEffect animation loop + Play/Pause UI

### 다음 단계
- Phase 1 시작: Agent-Setup과 Agent-Viewer에게 프롬프트 전달
- 병렬 작업 시작
