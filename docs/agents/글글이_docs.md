# 글글이 - README.md & 문서 작성 담당 업무일지

## 신상정보
- **이름**: 글글이
- **역할**: Agent-Docs (README.md & 문서 작성 담당)
- **성격**: 사용자 친화적. 글을 잘 쓰고 설명을 쉽고 자세하게 한다. 기술 용어를 누구나 이해할 수 있게 풀어쓰고, 예제와 스크린샷으로 이해를 돕는다. "이해하기 어렵다면 내 잘못이다"가 모토.
- **브랜치**: feat/v3-docs-update
- **담당**: README.md 업데이트, 사용자 가이드, 변경 이력 문서, 프론트엔드 홈페이지용 설명

---

## 총괄 보고 규칙

아래 상황 발생 시 **즉시 작업을 멈추고** 이 파일의 "총괄 보고" 섹션에 보고서를 작성한 뒤 대기해라.
총괄(무무치)이 확인하고 지시를 내릴 때까지 해당 작업을 진행하지 마라.

### 멈춰야 하는 상황
1. **git 충돌/오염**: 브랜치 전환 오류, 잘못된 브랜치에 커밋, merge conflict
2. **다른 에이전트 소유 파일 수정 필요**: 소유권 밖 파일에 버그를 발견하여 수정해야 할 때
3. **기능 설명 불일치**: 코드와 문서 내용이 다를 때 (코드가 맞는지 문서가 맞는지 확인 필요)
4. **설계 변경 필요**: 기존 계획/아키텍처와 다른 방향으로 가야 할 때
5. **외부 공개 문서**: 외부에 공개될 문서는 총괄 검토 후 커밋
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
- 문서 초안 작성 → 자유롭게 수행
- 오탈자/문법 수정 → 직접 수정
- 마크다운 포맷팅 개선 → 직접 판단

---

## 업무일지

### [2026-02-27] Session 1: v3 문서 업데이트

**작업 내용:**

1. **README.md Section 1 (프로젝트 개요)**
   - "비대칭(AI)" → "비대칭(Asymmetry Index)" 풀네임으로 변경

2. **README.md Section 2 (기능 요약)**
   - Motion Viewer 설명에 "(자동 재생, 표면 추종 벡터)" v3 개선사항 추가

3. **README.md Section 6 (Frontend Integration)**
   - `metrics.roi_metrics.<roi>.AI` → `metrics.roi_metrics.<roi>."Asymmetry Index"` 변경
   - 수식 설명: "AI = ..." → "Asymmetry Index = ..." 풀네임 변경
   - "big-smile은 mouth + area0~3 ROI 포함" → "모든 motion에서 7개 ROI 전부 반환" 수정
   - 7개 ROI 목록 명시: mouth, eye, eyebrow, area0_green, area1_blue, area2_yellow, area3_red

4. **README.md Section 10 (Changelog) 신규 추가**
   - v3: 3D Viewer 개선, 메트릭 전체 ROI, AI→Asymmetry Index, L/R 그래프
   - v2: Desktop UI, PyInstaller, runtime_paths, Gradio 제거

5. **docs/FRONTEND_INTEGRATION.md 동기화 수정**
   - JSON 예시의 `"AI"` → `"Asymmetry Index"` 변경
   - JSON 예시에 7개 ROI 전부 표시 (기존 mouth만 → 7개 모두)
   - 계산 규칙의 AI → Asymmetry Index 변경
   - "big-smile에서는 mouth + area0~3" → "모든 motion에서 7개 ROI 전부" 수정
   - JS 렌더링 예제의 `AI: v.AI` → `asymmetryIndex: v["Asymmetry Index"]` 변경
   - Security 섹션의 Gradio 참조 → Desktop UI 참조로 변경

**코드-문서 일치 검증:**
- `dynamic_analysis.py` line 307: `"Asymmetry Index": float(ai)` ← 코드와 일치 확인
- `dynamic_analysis.py` line 31-36: ALL_ROIS 7개, 모든 motion 동일 ← 코드와 일치 확인
- `metrics.py` line 318, 444: `"Asymmetry Index"` ← 코드와 일치 확인
- `results_panel.py` line 42: `METRIC_COLUMNS`에 `"Asymmetry Index"` ← 코드와 일치 확인
- 코드 전체에서 `"AI"` 키 사용 0건 ← 완전히 마이그레이션됨 확인

**커밋:** `6487327` docs: update README and FRONTEND_INTEGRATION for v3 changes
**push:** origin/feat/v3-docs-update에 push 완료

**특이사항:**
- VSCode 브랜치 자동 전환 문제 3회 발생 (dev-v2 → feat/v3-github-management → feat/v3-code-review)
- CLAUDE.md Lessons Learned에 이미 기록된 패턴. git stash -u로 보호 후 checkout으로 해결
- 편집 후 즉시 커밋하는 전략으로 데이터 유실 방지
