# CLAUDE.md -- FaceAnalyze2 v2 멀티에이전트 개발 가이드

## 프로젝트 개요

FaceAnalyze2는 안면마비 분석 도구로, 영상에서 얼굴 랜드마크를 추출하여 좌/우 비대칭 지수를 계산한다.

### v2 목표
- origin/main 기반 깨끗한 베이스에서 Desktop UI, PyInstaller 패키징, runtime_paths 재통합
- 3D Viewer 사용자 피드백 4가지 반영 (texture/points 렌더링, z축 depth, 포인트 크기, 자동 재생)
- USB 시연용 Windows 독립 실행 프로그램(exe) 패키징

### 아키텍처
- **소스 구조**: `src/faceanalyze2/` (setuptools src layout)
- **핵심 API**: `dynamicAnalysis(vd_path, motion)` → `api/dynamic_analysis.py`
- **파이프라인**: Video → MediaPipe 랜드마크 → 세그멘테이션 → 정렬 → 메트릭스 → 시각화
- **ML 모델**: `models/face_landmarker.task` (MediaPipe Face Landmarker, ~3.6MB)
- **UI**: PySide6 데스크톱 앱 (`desktop/` 패키지)
- **패키징**: PyInstaller onedir 모드

### 기술 스택
- Python 3.11+
- PySide6 (GUI)
- MediaPipe (얼굴 랜드마크)
- OpenCV (영상 I/O)
- NumPy, Matplotlib (수치 계산 / 시각화)
- PyInstaller (exe 빌드)

---

## 실수 방지 프로토콜 (모든 에이전트 필수)

> 이 섹션은 모든 에이전트가 **세션 시작 시 반드시 읽고**, **작업 중 반드시 준수**해야 하는 핵심 규칙이다.

### 규칙

1. **세션 시작 시** 아래 "실수 방지 지침 (Lessons Learned)" 섹션을 읽고, 기록된 실수를 반복하지 마라.

2. **기능 구현 후 반드시 테스트**를 실행하라:
   ```bash
   /c/Users/123an/anaconda3/envs/facepalsy311/python.exe -m pytest -q
   ruff check .
   ```

3. **테스트 결과와 무관하게, 작업 과정에서 얻은 교훈을 항상 기록하라**:
   - **성공 시**: 왜 성공했는지, 어떤 점을 주의했기에 문제가 없었는지 기록
   - **실패 시**: 왜 실패했는지 원인을 분석하고, 같은 실수를 반복하지 않도록 해결책 기록
   - 오류 발생 시 단순 재시도 금지. 근본 원인을 파악하라.
   - "실수 방지 지침" 섹션에 아래 형식으로 추가:
   ```
   ### [카테고리] 간단한 제목
   - **증상**: 어떤 상황이었는가 (성공이면 "정상 동작", 실패면 에러 내용)
   - **원인**: 왜 그런 결과가 나왔는가
   - **해결**: 어떻게 처리했는가 (성공이면 "사전에 ~을 주의하여 문제 없음")
   - **예방**: 앞으로 같은 유형의 작업에서 어떻게 해야 하는가
   ```

4. 카테고리 예시: `[환경]`, `[import]`, `[패키징]`, `[테스트]`, `[PySide6]`, `[MediaPipe]`, `[경로]`, `[git]`, `[Three.js]`, `[viewer]`

5. 중복 항목은 추가하지 않되, 기존 항목이 불완전하면 보완하라.

6. 이 섹션은 **절대 삭제하지 마라**. Lessons Learned 항목도 **append-only**.

---

## 실수 방지 지침 (Lessons Learned)

> 아래 항목은 v1 exe 개발 과정에서 검증된 교훈 (CLAUDE_exe_dev.md에서 이관)

### [환경] conda 환경 활성화
- **증상**: `ModuleNotFoundError: No module named 'pydantic'` 등 의존성 미발견
- **원인**: 기본 python이 base conda(3.9)를 가리킴. 프로젝트는 facepalsy311 환경 필요
- **해결**: 명시적 경로 사용 `/c/Users/123an/anaconda3/envs/facepalsy311/python.exe`
- **예방**: 테스트/실행 시 항상 facepalsy311 환경의 python 전체 경로를 사용할 것

### [테스트] test_segment_command_guides_when_landmarks_are_missing 기존 실패
- **증상**: `tests/test_cli.py::test_segment_command_guides_when_landmarks_are_missing` 항상 실패
- **원인**: Phase 0 이전부터 존재하는 기존 버그 (exit_code가 0을 반환)
- **해결**: 이 테스트 실패는 에이전트 작업에 의한 회귀가 아님. 무시 가능
- **예방**: 새 코드로 인한 회귀와 기존 버그를 구분하려면 dev 브랜치에서 먼저 테스트 실행하여 비교

### [경로] artifact_root 경로 불일치
- **증상**: 파이프라인 step 1-4는 절대경로에 산출물 저장, step 5(`dynamicAnalysis`)는 CWD 상대 `artifacts/`에서 읽으려 해서 FileNotFoundError
- **원인**: `main_window.py`에서 `get_artifact_root()` (절대경로) 사용, `dynamicAnalysis()`는 `artifact_dir_for_video()` 기본값 `Path("artifacts")` (CWD 상대) 사용. 두 경로가 불일치
- **해결**: `main_window.py`에서 `"artifacts"` (CWD 상대)로 통일하여 `dynamicAnalysis`와 일치
- **예방**: `dynamicAnalysis()`가 artifact_root를 받지 않으므로, 호출자는 반드시 같은 기본값(`"artifacts"`)을 사용할 것. frozen 모드에서는 app 시작 시 CWD를 base_dir로 설정

### [git] 멀티 에이전트 브랜치 전환 충돌
- **증상**: `git checkout` 후 외부에서 브랜치가 자동 전환됨. 작업 파일 소실
- **원인**: untracked file이 양쪽 브랜치에 존재하여 checkout 충돌, VSCode/외부 프로세스의 자동 전환
- **해결**: `git stash -u` (untracked 포함)로 stash 후 checkout
- **예방**: 1) 수정 후 즉시 커밋, 2) 브랜치 전환 전 반드시 `git stash -u`, 3) 작업 후 `git branch --show-current`로 확인

### [경로] exe 모드에서 CWD≠base_dir 시 dynamicAnalysis FileNotFoundError
- **증상**: exe 실행 → 영상 로드 → Analyze 클릭 시 Step 5에서 `FileNotFoundError`
- **원인**: exe 더블클릭 시 OS가 CWD를 exe 디렉토리와 다른 곳으로 설정할 수 있어 경로 불일치 발생
- **해결**: `app.py`의 `main()` 시작 부분에 `os.chdir(get_base_dir())` 추가
- **예방**: 앱 시작 시 반드시 CWD를 `get_base_dir()`로 설정할 것

### [패키징] conda 환경 DLL 누락으로 exe 실행 실패
- **증상**: PyInstaller 빌드는 성공하나 exe 실행 시 `ImportError: DLL load failed`
- **원인**: conda 환경의 `Library/bin/`에 있는 DLL을 PyInstaller가 자동으로 찾지 못함
- **해결**: spec 파일에서 `sys.prefix / "Library" / "bin"` 경로의 DLL을 `extra_binaries`로 명시적 추가
- **예방**: conda 환경으로 빌드 시 PyInstaller 경고를 반드시 확인하고, 누락 DLL을 spec `binaries`에 추가할 것

### [패키징] conda ICU DLL이 PySide6 Qt6Core를 깨뜨림
- **증상**: exe 실행 시 `ImportError: DLL load failed while importing QtCore`
- **원인**: conda의 ICU 58이 PySide6가 기대하는 Windows 시스템 ICU를 가림
- **해결**: spec 파일에서 conda ICU DLL (`icuuc.dll`, `icudt58.dll`) 제외
- **예방**: conda 환경으로 PyInstaller 빌드 시 ICU DLL을 반드시 제외할 것

### [패키징] unittest 제외 시 mediapipe/matplotlib 임포트 체인 실패
- **증상**: exe에서 분석 실행 시 `ModuleNotFoundError: No module named 'unittest'`
- **원인**: spec `excludes`에 `"unittest"`가 포함되어 있었으나, mediapipe → matplotlib → pyparsing.testing → import unittest 체인으로 런타임에 필요
- **해결**: spec `excludes`에서 `"unittest"` 제거
- **예방**: `excludes`에 표준 라이브러리 모듈 추가 시, 간접 의존성 체인을 반드시 확인할 것

### [패키징] PySide6 VC 런타임 버전 불일치
- **증상**: conda의 MSVCP140.dll이 PySide6의 것보다 오래됨
- **원인**: PyInstaller가 conda의 VC 런타임을 루트에 배치, PySide6의 최신 버전을 가림
- **해결**: spec에서 PySide6 디렉토리의 VC 런타임 DLL을 `extra_binaries`로 명시적 추가
- **예방**: conda + PySide6 조합 시 항상 PySide6의 VC 런타임이 우선하도록 spec에 명시

### [경로] Open 3D Viewer 버튼에서 상대 경로 에러
- **증상**: "Open 3D Viewer" 클릭 시 `ValueError: relative path can't be expressed as a file URI`
- **원인**: `viewer_html_path`가 상대 경로인데, `Path.as_uri()`는 절대 경로만 허용
- **해결**: `Path(viewer_path).resolve().as_uri()` 사용
- **예방**: `Path.as_uri()` 호출 전에 항상 `.resolve()`로 절대 경로 변환할 것

---

## 개발 컨벤션

- **Python**: 3.11+, 타입 힌트 사용
- **포매터/린터**: ruff (line-length=100, target py311)
- **테스트**: pytest, 테스트 파일은 `tests/`
- **커밋**: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
- **브랜치 흐름**: `feat/*` → `dev-v2` (머지) → `main` (PR)
- **matplotlib**: 항상 `matplotlib.use("Agg")` (headless backend만 사용)

### 유용한 명령어
```bash
# 린트
ruff check .

# 테스트
/c/Users/123an/anaconda3/envs/facepalsy311/python.exe -m pytest -q

# 데스크톱 UI 실행 (개발 모드)
/c/Users/123an/anaconda3/envs/facepalsy311/python.exe -m faceanalyze2.desktop.app

# PyInstaller 빌드
/c/Users/123an/anaconda3/envs/facepalsy311/python.exe -m PyInstaller packaging/faceanalyze2.spec
```

---

## v2 조직 구조

```
총괄 (Director)
├── Agent-Setup: 환경설정 & 통합 담당
│   - runtime_paths.py 이식
│   - Desktop UI 이식
│   - PyInstaller 패키징 이식
│   - Gradio 제거
├── Agent-Viewer: 3D Viewer 개선 담당
│   - Fix 1: texture/points 초기 렌더링
│   - Fix 2: 벡터 z축 depth 제거 옵션
│   - Fix 3: 포인트 기본 크기 증가
│   - Fix 4: t-슬라이더 자동 재생/일시정지
└── Agent-QA: 테스트 & QA 담당
    - 브랜치 통합 (feat/v2-* → dev-v2)
    - 전체 테스트 실행
    - 기능 검증
    - 회귀 테스트
```

---

## v2 파일 소유권 (충돌 방지)

각 에이전트는 지정된 파일만 수정한다. 공유 파일 수정 시 아래 "조율 요청" 섹션에 먼저 기록할 것.

### Agent-Setup 전용:
- `src/faceanalyze2/runtime_paths.py`
- `src/faceanalyze2/desktop/` (전체 디렉토리)
- `src/faceanalyze2/config.py` (경로 관련만)
- `src/faceanalyze2/landmarks/mediapipe_face_landmarker.py` (경로만)
- `src/faceanalyze2/demo/` (삭제용)
- `packaging/` (전체 디렉토리)
- `pyproject.toml`
- `README.md`

### Agent-Viewer 전용:
- `src/faceanalyze2/viz/motion_viewer.py` (단독 소유)

### Agent-QA 전용:
- `tests/` (전체 디렉토리)

### 공유 파일 (수정 전 조율 필수):
- `CLAUDE.md` -- 이 파일 (진행 상황은 append-only)
- `src/faceanalyze2/__init__.py` -- export 변경

---

## v2 브랜치 전략

```
backup/exe-dev-v1          ← v1 exe 작업 백업 (변경 금지)
main (origin/main 동기화)  ← 깨끗한 베이스
├── feat/v2-exe-integration  ← Agent-Setup 작업 브랜치
├── feat/v2-viewer-fixes     ← Agent-Viewer 작업 브랜치
└── dev-v2                   ← 통합 브랜치 (Agent-QA가 머지/테스트)
    └── (최종) dev-v2 → main PR
```

### 브랜치 규칙
1. 각 에이전트는 자신의 브랜치에서만 작업
2. dev-v2 직접 커밋 금지 (머지만 허용)
3. backup/exe-dev-v1은 참조용으로만 사용 (변경 금지)
4. 커밋 전 반드시 `git branch --show-current`로 현재 브랜치 확인

---

## 진행 상황

형식: `[날짜] [에이전트] [상태] 설명`

### 완료
- [2026-02-27] [총괄] [완료] Phase 0 - 백업, 브랜치 3개 생성, CLAUDE.md 작성

### 진행 중

### 차단됨

---

## 조율 요청

에이전트 간 의존성이 있을 때 여기에 기록:

---

## 알려진 이슈 / 결정사항

1. **MediaPipe + PyInstaller**: spec 파일에서 `Tree(mediapipe_path, prefix='mediapipe')` 로 전체 데이터 포함 필요
2. **matplotlib 백엔드**: `Agg`만 사용. interactive backend 절대 사용 금지
3. **모델 파일**: exe 옆 `models/` 디렉토리에 배치 (exe 내부 번들 아님)
4. **경로 해결**: 모든 경로는 `runtime_paths.py`의 `get_base_dir()` 기준
5. **임시 파일**: frozen 환경에서는 exe 옆 `.temp/` 사용
6. **QSettings**: 레지스트리 대신 INI 파일 (`config/` 디렉토리)로 저장
7. **Gradio**: v2에서 제거 예정. 데스크톱 UI로 완전 대체
8. **CLI**: exe에 미포함. 개발용으로만 소스에 유지
9. **dynamicAnalysis() 호출**: UI에서 직접 Python API 호출 (subprocess 불필요)
10. **QThread**: 파이프라인 실행은 반드시 QThread에서 수행 (UI 프리징 방지)
11. **artifact_root와 CWD**: frozen 모드에서는 app 시작 시 `os.chdir(get_base_dir())`로 CWD 설정
12. **v1 참고 문서**: `CLAUDE_exe_dev.md`에 v1 exe 개발 과정 전체 기록 보존

---

## 대화 기록 파일

compact 대비 각 에이전트의 대화 기록 보관:

```
docs/conversation_logs/
├── director_log.md      ← 총괄 대화 기록
├── agent_setup_log.md   ← Agent-Setup 대화 기록
├── agent_viewer_log.md  ← Agent-Viewer 대화 기록
└── agent_qa_log.md      ← Agent-QA 대화 기록
```

형식: 날짜, 작업 내용, 결정 사항, 결과를 마크다운으로 기록
