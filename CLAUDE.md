# CourtVision — 프로젝트 인수인계 (Claude용)

> 다른 PC에서 새 세션을 열었을 때 맥락을 잇기 위한 문서. 마지막 정리: 2026-09-22.
> 사용자는 한국어를 쓰고, 응답도 한국어로 한다.

## 1. 프로젝트 한 줄
테니스 경기 영상을 올리면 선수를 추적해 이동 경로·이동거리·속도·코트 히트맵을 보여주는 **웹 서비스** (FastAPI + 단일 HTML/JS 프론트 + SQLite).
수업 과제(크롤링 → 서비스화 → 발표)와 취업 포트폴리오를 겸한다. 사용자는 테니스를 "배우고 싶은 운동"으로 관심을 가진 입장이다(**실제 선수/아마추어 경력을 가정해서 쓰지 말 것**).

## 2. 현재 상태 (사실만)
| 항목 | 상태 |
|---|---|
| 영상 업로드 → 코트 4점 보정 → 선수 검출·추적 → 이동거리/속도/궤적/히트맵 | 로컬에서 동작 |
| 대회 일정 크롤링 (KATO, KTA) | 동작, SQLite 저장, **하루 1회, 매번 무작위 시각에 수집**(패턴 예측 어렵게, 2026-09-23 변경). 서버 기동 시 마지막 수집이 24시간 넘었으면 바로 보충. 서버가 꺼져 있으면 수집 안 함. Docker는 시계가 UTC일 수 있어 `TZ=Asia/Seoul` 설정·확인 필요(미확인) |
| UI: 홈 / 영상 분석 / 대회 일정 / 마이페이지, 다크(기본)·라이트 | 구현됨 |
| 로그인, 분석 저장, 마이페이지 | 구현됨. **API 수준만 확인.** 실제 영상 분석 후 자동 저장·상세 보기·삭제·라이트 테마·모바일은 브라우저에서 미검증 |
| Docker | `Dockerfile`, `docker-compose.yml`, `.dockerignore`만 작성. **빌드한 적 없음** |
| 공 검출 | 별도 실험(서비스 미연결). **검출까지만, 추적 없음** |
| INT8/OpenVINO | 실험 완료(선수 검출 모델). 서비스 미연결 |
| 이동거리·속도의 실측 검증 | **못 함** |

## 3. 사용 모델
- 선수 검출: YOLO11n(COCO 사전학습, person만), 직접 학습 안 함. imgsz 960. **서비스에서 사용**
- 선수 추적: BoT-SORT(Ultralytics 내장, 기본 설정). **서비스에서 사용**
- 좌표 변환: OpenCV Homography (학습 아님). 사용자가 코트 4점(가까운 서비스라인 좌우 끝 + 네트 아래 싱글 사이드라인 바닥 2점)을 클릭
- 공 검출: YOLO11n 파인튜닝. **채택 모델은 v2** (`models/tennis_ball_v2.pt`, 배경 이미지 571장 추가)
- INT8: NNCF 사후 양자화 + OpenVINO (재학습 없음)

## 4. 실험 결과 요약 (자세한 것은 results/*.md)
공 검출, video9 기준(20px 이내 적중 / 공 없는 프레임 오검출), conf 0.35:
- v1 92.9% / 37.6% → **v2 95.3% / 15.6% (채택)** → v3(공개 데이터 추가) 92.9% / 21.8% (악화) → v3a(배경 보정) 99.4% / 31.6% (가설 기각)
- 한계: 검증 영상 1개, 학습 1회, video9는 best epoch 선택에도 써서 완전한 시험 데이터가 아님

INT8 (선수 검출, video9 300프레임, CPU Xeon W-2223, 입력 544x960):
- PyTorch FP32 72.2ms / OpenVINO FP32 85.7ms / **INT8 50.0ms**, 크기 10.9MB → 3.5MB
- FP32 대비 검출 재현율 97.8%, 정밀도 91.5%, 상위 2트랙 300·298프레임 유지
- 1차 측정은 입력 크기 불일치로 폐기(기록: `results/int8_run1_square960_log.txt`). 측정 1회, 회차 간 편차 큼
- **서비스 지표(이동거리·속도) 전후 비교 완료(2026-09-23).** video9 전체로 파이프라인(추적→Homography→통계) 실행: 총 이동거리 33.83→33.99m(+0.5%), 평균속도 3.41→3.40(-0.3%), 최고속도 14.60→13.43m/s(-8%, 순간값이라 민감). 핵심 지표(이동거리·평균속도)는 경량화 후에도 거의 그대로. 코트 4점은 근사 좌표(양쪽에 동일 적용). 로그: `results/int8_service_compare_log.txt`

## 5. 파일 구조
```
webapp/
  backend/  main.py(API·정적파일) pipeline.py(YOLO·BoT-SORT·Homography·히트맵)
            accounts.py(회원·세션·분석기록) crawler/(kta, kato, store, run_crawl)
            requirements.txt(버전 고정)
  frontend/ index.html(전체 UI 한 파일) assets/(hero, login_side, empty_history, filming_guide, og, sample_heatmap) calib_example*.jpg
src/        실험·검증 스크립트 (공 학습/평가, INT8 export·benchmark 등)
results/    실험 기록 md
models/     tennis_ball_v*.pt, int8/(OpenVINO 모델)
data/       (용량 큼, 옮기지 않음) courtvision.db 는 대회 데이터
```
서버 실행: `cd webapp/backend` 후 `uvicorn main:app --port 8000` (세션이 메모리라 워커 1개). 환경변수: `COURTVISION_DB`, `UPLOAD_DIR`, `RESULTS_DIR`, `YOLO_IMGSZ`, `COOKIE_SECURE`(HTTPS 배포 시 1).

## 6. UI 작업 시 (노트북에서 주로 할 일)
- 파일: `webapp/frontend/index.html` (CSS 변수 기반 테마, 해시 라우팅 `#home #analyze #tournaments #mypage`)
- **디자인 리뉴얼 진행 중 (2026-09-22 시작).** 시안: `design_ref/*.webp`(home, match_analysis, my_matches, tournaments, about). 시안의 이름은 CourtSight이지만 **화면 이름은 CourtVision 유지**로 결정
  - 다크 = 검정 + 골드(#d8b07a), 라이트 = 흰색/연한 청회색 + 파랑(#1f5fd0), 주요 버튼은 다크에서 골드·라이트에서 네이비. 제목은 Playfair Display(세리프), 본문 Noto Sans KR, 손글씨 액센트 Caveat(Google Fonts, 인터넷 필요)
  - 색·사진은 모두 `:root` CSS 변수(`--accent`, `--hero-img` 등)로 테마별로 정의. 사진은 `assets/photos/*.webp`, 로고는 `assets/brand/`
  - 단계: ① 색 토큰·헤더·푸터·**홈(완료)** → ② **영상 분석(완료, 화면 확인만 함: 실제 업로드→분석 흐름은 새 화면에서 아직 시험 안 함)** → ③ **대회 일정(완료: 출처 탭·부 필터·접수중 필터·다가오는 카드 4개·월별 표, 필터 조합 동작 확인)** → ④ **마이페이지(완료: 로그아웃 안내 / 필터·요약·촬영일별 카드 / 상세. 화면은 임시 데이터로만 확인, 실제 DB의 삭제 버튼 동작은 새 화면에서 미시험)** → ⑤ **About(완료: How It Works·Under the Hood·Experiments in Numbers(실제 실험 수치·자동으로 대회 건수 fetch)·Where We Stand(된 것/실험만 한 것/안 된 것 명시). 메뉴에 '소개'로 추가됨)**
  - 영상 분석 화면: 4단계 스테퍼(`setStep`), 업로드 카드에 촬영 조건 안내 + `assets/filming_guide.jpg`, 코트 보정은 좌(영상)·우(안내) 2단, 결과는 `statsTable()`이 선수별 타일로 렌더(마이페이지 상세도 공용). 기존 요소 ID는 전부 유지
  - 브라우저는 index.html을 캐시하므로 수정 후 확인할 때는 새로고침 필요. 패치 전 백업: `design_ref/index_before_analyze.html`
  - **영상 분석을 '스텝 위저드'(step wizard)로 변경(2026-09-22): 한 번에 한 단계 패널만 보임**(`showStepPanel`), 완료한 단계는 `#step-summaries`에 한 줄 요약 + '수정' 버튼으로 접힘, 상단 스테퍼의 완료된 번호를 클릭해도 그 단계로 돌아감(`goToStep`). 이전에는 4단계 카드가 전부 쌓여 스크롤이 길었음(사용자 피드백으로 변경)
  - 시안에 있는 것 중 **넣지 않는다**: 가짜 통계 숫자(10K+ 경기, 95% 만족도 등), 미구현 기능(Ball Tracking, Shot Analysis, 스코어, Key Moments), 실제 대회 로고/상표
  - 패치 전 원본: `design_ref/index_before_redesign.html`
- 외부 텍스트(크롤링)는 반드시 `textContent`로 넣는다(XSS 방지). https 링크만 허용
- 정적 미리보기만 하려면 frontend 폴더에서 `python -m http.server` (API가 없어 대회/로그인 영역은 비어 있음). 전체 확인은 서버 실행 필요(`main.py`가 시작 시 YOLO를 불러오므로 requirements 설치 필요)
- 모바일 폭은 아직 눈으로 확인하지 못했다. 라이트 테마는 홈·영상 분석·대회 일정·마이페이지 모두 확인함
- **디자인 리뉴얼 5개 화면 모두 완료(2026-09-22).** 실제 흐름(회원가입→업로드→보정→분석→저장→마이페이지 보기·삭제) 새 화면에서 재시험 완료, 정상 동작
- 영상 분석 1단계 업로드 UI 개선(2026-09-22): 파일 선택 버튼이 안 보인다는 피드백으로 커스텀 `.file-btn`(라벨+숨긴 input)로 교체, 업로드 버튼과 한 줄 배치, 선택 파일명 표시(`#file-name`). 상단에 고정(position:fixed)되어 있던 영상 미리보기(`#video-bar`)가 불편하다는 피드백으로 고정을 풀고 업로드 카드 아래 일반 카드로 변경, 페이지 패딩 보정 코드(`paddingTop` 핵)도 제거
- 촬영일(`#rec-date`, `<input type=date>`) 자동화 브라우저에서 네이티브 달력 팝업 검증 불가(포커스·값 설정은 정상, 클릭해도 팝업 렌더 확인 안 됨) — 실사용자가 실제 클릭해서 달력이 뜨는지 확인 필요
- **모바일 폭 버그 발견·수정(2026-09-22): `<head>`에 `<meta name="viewport">`가 아예 없었음.** 그래서 실제 휴대폰에서 레이아웃 뷰포트가 기본값(약 980px)으로 처리되어 900px 기준 모바일 CSS가 전혀 적용되지 않았음(메뉴가 빈 자리로 사라짐 등). 메타 태그 추가로 해결
- 위 문제와 함께, 좁은 화면에서 `.pills` 내비게이션을 숨기기만 하고 대체 UI가 없던 문제도 수정: 헤더에 햄버거 버튼(`#nav-toggle`)과 드롭다운 `#mobile-menu`(메뉴 5개 + 로그인/회원가입/로그아웃, `mm-` 접두사 id로 데스크톱과 별도 엘리먼트) 추가. `refreshUser()`가 두 세트를 함께 갱신, `showPage()`가 이동 시 메뉴 자동 닫음
- 사진 원칙: 사람이 크게 나오는 AI 사진은 팔·손 왜곡이 있을 수 있어 히어로에 쓰지 않는다(`serve_overhead`는 팔이 꺾여 교체함). 라이트에서 공이 큰 사진은 히어로 하나만

## 7. 남은 일 (우선순위)
1. ~~마이페이지 전체 흐름을 실제 영상으로 시험~~ **완료(2026-09-22).** 새 화면(스텝 위저드)에서 회원가입→업로드→보정→분석→저장→마이페이지 보기·삭제까지 재시험, 정상 동작
2. ~~Docker Desktop 설치 후 `docker compose up --build`~~ **완료(2026-09-22).** Docker Desktop per-user 설치, 이미지 빌드·컨테이너 기동·`localhost:8000` 응답까지 확인함(로그: 컨테이너 안 크롤러도 정상 동작, 대회 99건 자동 수집). Dockerfile/compose는 수정 없이 그대로 성공. 컨테이너는 로컬에서만 접속 가능(진짜 원격 배포 아님) — **다음 목표: 실제 서버(예: Render/Railway/Fly.io)에 올려 외부에서 접속 가능하게 하는 완전 배포**
   - 이 세션에서 겪은 함정(다음에 또 나올 수 있음): Docker 설치 직후 이 세션(터미널)이 새 PATH를 못 읽어 `docker`, `docker-credential-desktop` 명령을 못 찾음 → 전체 경로(`%LOCALAPPDATA%\Programs\DockerDesktopesourcesin`)를 그 호출 안에서 `$env:PATH`에 직접 추가해서 해결. Windows PowerShell 세션은 매 호출마다 환경이 초기화되므로 매번 같이 설정해야 함
3. ~~INT8 전후 이동거리·최고속도 비교~~ **완료(2026-09-23).** 자세한 내용은 4절 참고
4. PPT·대본·시연 녹화·예상 질문 (SCQA 스토리, 발표 10분 이상, 발표일 미정)
5. GitHub (아래 규칙 참고)

## 8. 지켜야 할 규칙
- **git은 사용자가 저장소 URL을 주고 승인하기 전까지 절대 건드리지 않는다** (init/commit/push 금지). 공개 전에 크롤러 User-Agent의 개인 이메일(`src/probe_*.py`, `webapp/backend/crawler/kato.py`, `kta.py`) 제거, `.gitignore`(data/, runs/, models/*.pt, uploads, results, *.db) 추가
- **발표·문서의 주장은 검증된 것만.** 쓸 수 있는 문구: "YOLO·BoT-SORT 기반 선수 추적 웹 서비스(로컬 동작), 공 검출 파인튜닝·평가, INT8 경량화 실험". 쓰면 안 되는 것: 공 추적, 배포 완료, 정확도/이동거리 실측 검증, 실시간
- 크롤링 데이터는 사실 정보 요약 + 원본 링크만. 장비 크롤링은 약관/robots 확인 전에는 하지 않는다
- 학습·측정은 결과를 과장하지 않고 한계까지 기록한다. 비교 실험은 조건을 먼저 맞춘다
- 코드 수정 방식: 사용자가 "해줘"라고 한 변경만 직접 수정한다. 파일 전체를 붙여넣은 경우에는 부분 패치가 아니라 **전체 교체본**을 준다. 요청하지 않은 내용(목업, 대시보드 등)을 추가하지 않는다. 용어가 어렵다고 하면 쉬운 말과 비유로 먼저 설명한다
- 사용자가 모른다고 한 부분은 채워 주되, 면접용 답변은 실제 대화 기록에 근거한 사실만으로 쓴다. 사용자의 역할(요구사항 정의, 결과 검토, 검증 질문)을 부풀리지 않는다. AI와 협업했다는 사실을 숨기지 않는다
- 환경: Windows, Anaconda env `courtvision`(Python 3.11), RTX 3080(데스크톱만). Ultralytics runs 폴더는 `D:\JEN\VISION_AI\hyrox\runs\detect\runs\...`로 리다이렉트됨
