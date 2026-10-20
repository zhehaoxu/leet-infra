"""FastAPI entry: problem APIs, judging, and static hosting of the built frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import judge, problems, progress

app = FastAPI(title="leet-infra", docs_url=None, redoc_url=None)

# CORS only matters for the vite dev server (:5173); prod is single-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Submission(BaseModel):
    code: str


@app.get("/api/problems")
def list_problems():
    solved = progress.solved_map()
    return [p.summary(p.slug in solved) for p in problems.load_problems()]


@app.get("/api/problems/{slug}")
def get_problem(slug: str):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail="题目不存在")
    return {
        **problem.summary(progress.is_solved(slug)),
        "description": problem.description(),
        "starter_code": problem.starter(),
    }


@app.post("/api/problems/{slug}/run")
def run_problem(slug: str, submission: Submission):
    """运行：只执行样例用例，不记录进度。"""
    return _evaluate(slug, submission, mode="run")


@app.post("/api/problems/{slug}/submit")
def submit_problem(slug: str, submission: Submission):
    """提交：全量评测，全部通过时记录进度。"""
    return _evaluate(slug, submission, mode="submit")


def _evaluate(slug: str, submission: Submission, mode: str):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail="题目不存在")
    result = judge.run_judgement(submission.code, problem.test_cases(), mode=mode)
    if mode == "submit" and result["accepted"]:
        progress.mark_solved(slug)
    return result


@app.get("/api/problems/{slug}/solution")
def get_solution(slug: str):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail="题目不存在")
    if not progress.is_solved(slug):
        raise HTTPException(status_code=403, detail="通过全部测试用例后可查看参考答案")
    return {"slug": slug, "solution_code": problem.solution()}


@app.get("/api/progress")
def get_progress():
    all_problems = problems.load_problems()
    solved = progress.solved_map()
    return {
        "total": len(all_problems),
        "solved": sum(1 for p in all_problems if p.slug in solved),
        "solved_at": solved,
    }


# --- static frontend (production build) ------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
