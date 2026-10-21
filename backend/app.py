"""FastAPI entry: problem APIs, judging, and static hosting of the built frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import collections, judge, problems, progress

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


def _summary_with_collections(problem, solved: bool) -> dict:
    """题目 summary 附加所属合集信息，供前端渲染合集徽章。"""
    return {
        **problem.summary(solved),
        "collections": collections.collections_of(problem.slug),
    }


@app.get("/api/problems")
def list_problems():
    solved = progress.solved_map()
    return [_summary_with_collections(p, p.slug in solved) for p in problems.load_problems()]


@app.get("/api/problems/{slug}")
def get_problem(slug: str):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail="题目不存在")
    return {
        **_summary_with_collections(problem, progress.is_solved(slug)),
        "description": problem.description(),
        "starter_code": problem.starter(),
    }


@app.get("/api/collections")
def list_collections():
    solved = progress.solved_map()
    return [
        {
            "slug": c["slug"],
            "name": c["name"],
            "description": c["description"],
            "problem_count": len(c["problems"]),
            "solved_count": sum(1 for s in c["problems"] if s in solved),
        }
        for c in collections.all_collections()
    ]


@app.get("/api/collections/{slug}")
def get_collection_detail(slug: str):
    collection = collections.get_collection(slug)
    if collection is None:
        raise HTTPException(status_code=404, detail="合集不存在")
    solved = progress.solved_map()
    items = []
    for problem_slug in collection["problems"]:
        problem = problems.get_problem(problem_slug)
        if problem is not None:
            items.append(_summary_with_collections(problem, problem_slug in solved))
    return {
        "slug": collection["slug"],
        "name": collection["name"],
        "description": collection["description"],
        "problems": items,
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


@app.post("/api/judge/cancel")
def cancel_judge():
    """取消当前正在运行的评测（死循环/过长计算的逃生门）。"""
    return {"cancelled": judge.cancel_running()}


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


@app.post("/api/progress/reset")
def reset_progress():
    """重置全部解题进度（二刷场景），返回重置后的进度。"""
    progress.reset()
    return get_progress()


# --- static frontend (production build) ------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
