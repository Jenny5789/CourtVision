import os
import time
import uuid
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import accounts
import pipeline
from crawler import run_crawl, store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def _next_random_time(now):
    """매일 같은 시각이 아니라, 하루 중 무작위 시각에 수집되도록(패턴을 예측하기 어렵게)"""
    import random
    return (now + timedelta(days=1)).replace(
        hour=random.randint(0, 23), minute=random.randint(0, 59), second=0, microsecond=0
    )


def _crawler_loop():
    """하루 1번, 매번 다른 무작위 시각에 수집. 서버가 꺼져 있다가 켜졌을 때 마지막 수집이 24시간 넘었으면 바로 1번 보충"""
    try:
        run_crawl.refresh_if_stale(max_age_hours=24)
    except Exception as e:
        print(f"[crawler] 수집 실패(다음 예약 시각에 재시도): {e}")
    while True:
        now = datetime.now()
        nxt = _next_random_time(now)
        print(f"[crawler] 다음 수집 예정 시각: {nxt.isoformat(timespec='minutes')}")
        time.sleep((nxt - now).total_seconds())
        try:
            run_crawl.refresh_if_stale(max_age_hours=20)   # 서버를 방금 켜서 이미 수집했다면 건너뜀
        except Exception as e:
            print(f"[crawler] 수집 실패(다음 예약 시각에 재시도): {e}")


@asynccontextmanager
async def lifespan(app):
    accounts.init()
    threading.Thread(target=_crawler_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

sessions = {}

COOKIE_NAME = "cv_session"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"  # HTTPS로 배포할 때 1


def current_user(request: Request):
    return accounts.user_from_token(request.cookies.get(COOKIE_NAME))


def require_user(request: Request):
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def _set_login_cookie(response: Response, user_id: int):
    response.set_cookie(COOKIE_NAME, accounts.create_session(user_id),
                        max_age=accounts.SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=COOKIE_SECURE)


class SignupRequest(BaseModel):
    email: str
    nickname: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/signup")
async def signup(req: SignupRequest, response: Response):
    try:
        user_id = accounts.create_user(req.email, req.nickname, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _set_login_cookie(response, user_id)
    return {"email": req.email.strip().lower(), "nickname": req.nickname.strip()}


@app.post("/api/login")
async def login(req: LoginRequest, response: Response):
    user = accounts.authenticate(req.email, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
    _set_login_cookie(response, user["id"])
    return {"email": user["email"], "nickname": user["nickname"]}


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    accounts.delete_session(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request):
    user = current_user(request)
    return {"user": {"email": user["email"], "nickname": user["nickname"]} if user else None}


@app.get("/api/analyses")
async def list_my_analyses(request: Request):
    user = require_user(request)
    return {"items": accounts.list_analyses(user["id"])}


@app.delete("/api/analyses/{analysis_id}")
async def delete_my_analysis(analysis_id: int, request: Request):
    user = require_user(request)
    if not accounts.delete_analysis(user["id"], analysis_id):
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return {"ok": True}


@app.get("/api/tournaments")
async def get_tournaments(upcoming: bool = True, source: str | None = None,
                          division: str | None = None, limit: int = 100):
    items = store.query_tournaments(upcoming_only=upcoming, source=source,
                                    division=division, limit=min(limit, 300))
    return {"count": len(items), "last_crawled": store.latest_seen(), "items": items}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_dir = os.path.join(RESULTS_DIR, video_id)
    frame_path = pipeline.save_first_frame(video_path, output_dir)

    sessions[video_id] = {
        "video_path": video_path,
        "output_dir": output_dir,
    }

    return {
        "video_id": video_id,
        "frame_url": f"/results/{video_id}/first_frame.jpg",
        "video_url": f"/uploads/{video_id}.mp4",
    }


class SetFrameRequest(BaseModel):
    video_id: str
    time_sec: float


@app.post("/api/set_frame")
async def set_frame(req: SetFrameRequest):
    session = sessions.get(req.video_id)
    if session is None:
        raise HTTPException(status_code=404, detail="video_id를 찾을 수 없습니다.")

    pipeline.save_first_frame(session["video_path"], session["output_dir"], req.time_sec)
    session["frame_time"] = req.time_sec

    return {"frame_url": f"/results/{req.video_id}/first_frame.jpg?t={int(req.time_sec*1000)}"}


class CalibrateRequest(BaseModel):
    video_id: str
    corners: list[list[int]]  # [[x,y] x4] 순서: 가까운 서비스라인 왼쪽 끝, 오른쪽 끝, 네트 아래 오른쪽, 네트 아래 왼쪽


@app.post("/api/calibrate")
async def calibrate(req: CalibrateRequest):
    session = sessions.get(req.video_id)
    if session is None:
        raise HTTPException(status_code=404, detail="video_id를 찾을 수 없습니다.")

    if len(req.corners) != 4:
        raise HTTPException(status_code=400, detail="코트 모서리 4개를 모두 지정해주세요.")

    H = pipeline.compute_homography(req.corners)

    if "full_tracks" not in session:
        full_tracks, fps = pipeline.track_full_video(session["video_path"])
        session["full_tracks"] = full_tracks
        session["fps"] = fps

    frame_time = session.get("frame_time", 0.0)
    court_polygon = pipeline.court_polygon_pixels(H)
    candidates, player1_id = pipeline.get_candidates_at_time(
        session["full_tracks"], session["fps"], court_polygon, frame_time
    )

    overlay = pipeline.court_overlay_segments(H)

    if player1_id is None:
        raise HTTPException(status_code=400, detail="코트 안에서 선수를 찾지 못했습니다. 4점을 다시 확인해주세요.")

    session["H"] = H
    session["candidates"] = candidates
    session["player1_id"] = player1_id

    return {"candidates": candidates, "player1_id": player1_id, "overlay": overlay}


class AnalyzeRequest(BaseModel):
    video_id: str
    click_x: int | None = None
    click_y: int | None = None
    recorded_date: str | None = None   # 촬영일(YYYY-MM-DD). 로그인 상태에서 값이 있으면 기록으로 저장
    title: str = ""


@app.post("/api/analyze")
async def analyze_video(req: AnalyzeRequest, request: Request):
    session = sessions.get(req.video_id)
    if session is None or "H" not in session:
        raise HTTPException(status_code=404, detail="먼저 코트 보정을 완료해주세요.")

    target_ids = {session["player1_id"]: "player1"}

    if req.click_x is not None and req.click_y is not None:
        player2_id = pipeline.pick_player2(session["candidates"], req.click_x, req.click_y)
        if player2_id is not None and player2_id != session["player1_id"]:
            target_ids[player2_id] = "player2"

    result = pipeline.run_full_analysis(
        session["full_tracks"], session["fps"], target_ids, session["H"],
        session["video_path"], session["output_dir"]
    )

    saved_id = None
    user = current_user(request)
    if user is not None and req.recorded_date:
        try:
            saved_id = accounts.save_analysis(user["id"], req.video_id, req.recorded_date,
                                              req.title, result["stats"])
        except ValueError:
            raise HTTPException(status_code=400, detail="촬영일 형식이 올바르지 않습니다.")

    return {
        "stats": result["stats"],
        "trajectory_url": f"/results/{req.video_id}/trajectory.jpg",
        "heatmap_url": f"/results/{req.video_id}/heatmap.png",
        "saved_id": saved_id,
    }


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")