"""FastAPI entry: problem APIs, judging, and static hosting of the built frontend."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import benchmarks, collections, judge, problems, progress
from .i18n import tr

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
    variant: str = "pytorch"
    lang: str = "zh"


def _summary_with_collections(problem, solved: bool, lang: str) -> dict:
    """题目 summary 附加所属合集信息，供前端渲染合集徽章。"""
    return {
        **problem.summary(solved, lang),
        "collections": collections.collections_of(problem.slug, lang),
    }


@app.get("/api/problems")
def list_problems(lang: str = Query(default="zh")):
    solved = progress.solved_map()
    return [
        _summary_with_collections(p, p.slug in solved, lang) for p in problems.all_problems()
    ]


@app.get("/api/problems/{slug}")
def get_problem(slug: str, variant: str | None = Query(default=None), lang: str = Query(default="zh")):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail=tr(lang, "problem_not_found"))
    active = problem.resolve_variant(variant)
    return {
        **_summary_with_collections(problem, progress.is_solved(slug), lang),
        "description": problem.description(lang),
        "notes": problem.notes(lang),
        "starter_code": problem.starter(active),
        "active_variant": active,
    }


@app.get("/api/collections")
def list_collections(lang: str = Query(default="zh")):
    solved = progress.solved_map()
    return [
        {
            "slug": c["slug"],
            "name": collections.localized_name(c, lang),
            "description": collections.localized_description(c, lang),
            "problem_count": len(c["problems"]),
            "solved_count": sum(1 for s in c["problems"] if s in solved),
        }
        for c in collections.all_collections()
    ]


@app.get("/api/collections/{slug}")
def get_collection_detail(slug: str, lang: str = Query(default="zh")):
    collection = collections.get_collection(slug)
    if collection is None:
        raise HTTPException(status_code=404, detail=tr(lang, "collection_not_found"))
    solved = progress.solved_map()
    items = []
    for problem_slug in collection["problems"]:
        problem = problems.get_problem(problem_slug)
        if problem is not None:
            items.append(_summary_with_collections(problem, problem_slug in solved, lang))
    return {
        "slug": collection["slug"],
        "name": collections.localized_name(collection, lang),
        "description": collections.localized_description(collection, lang),
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
        raise HTTPException(status_code=404, detail=tr(submission.lang, "problem_not_found"))
    variant = problem.resolve_variant(submission.variant)
    result = judge.run_judgement(
        submission.code, problem.test_cases(variant), mode=mode, lang=submission.lang
    )
    if mode == "submit" and result["accepted"]:
        progress.mark_solved(slug)
        if benchmarks.get_benchmarks(slug, variant) is None:
            _spawn_benchmark(problem, variant)  # 首次 AC：后台测参考解基准
    bench = benchmarks.get_benchmarks(slug, variant)
    if bench:
        result["benchmark"] = bench
    result["variant"] = variant
    return result


def _spawn_benchmark(problem, variant: str) -> None:
    """后台用指定 variant 的参考解全量跑一遍记录用例耗时；失败静默，不阻塞用户请求。"""

    def _measure() -> None:
        try:
            res = judge.run_judgement(problem.solution(variant), problem.test_cases(variant), mode="submit")
            if res["accepted"]:
                benchmarks.record(problem.slug, variant, res)
        except Exception:
            pass  # 基准仅用于展示，失败不影响主流程

    threading.Thread(target=_measure, daemon=True, name=f"benchmark-{problem.slug}-{variant}").start()


@app.post("/api/judge/cancel")
def cancel_judge():
    """取消当前正在运行的评测（死循环/过长计算的逃生门）。"""
    return {"cancelled": judge.cancel_running()}


@app.get("/api/problems/{slug}/solution")
def get_solution(
    slug: str,
    variant: str | None = Query(default=None),
    lang: str = Query(default="zh"),
):
    problem = problems.get_problem(slug)
    if problem is None:
        raise HTTPException(status_code=404, detail=tr(lang, "problem_not_found"))
    if not progress.is_solved(slug):
        raise HTTPException(status_code=403, detail=tr(lang, "solution_locked"))
    active = problem.resolve_variant(variant)
    return {"slug": slug, "variant": active, "solution_code": problem.solution(active)}


@app.get("/api/progress")
def get_progress():
    all_problems = problems.all_problems()
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


@app.post("/api/problems/{slug}/unmark")
def unmark_problem(slug: str, lang: str = Query(default="zh")):
    """取消单题已解记录（重刷粒度），幂等；参考答案随之重新锁定。"""
    if problems.get_problem(slug) is None:
        raise HTTPException(status_code=404, detail=tr(lang, "problem_not_found"))
    return {"slug": slug, "removed": progress.unmark(slug)}


@app.get("/api/progress/export")
def export_progress():
    """下载当前进度文件，与 data/progress.json 同构，换机/重装可恢复。"""
    payload = json.dumps(progress.export_store(), ensure_ascii=False, indent=2)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="leet-infra-progress.json"'},
    )


@app.post("/api/progress/import")
async def import_progress(req: Request, lang: str = Query(default="zh")):
    """全量恢复进度（覆盖当前记录）；不在当前题库内的 slug 被过滤，防止脏数据。"""
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail=tr(lang, "import_json_invalid")) from None
    solved = payload.get("solved") if isinstance(payload, dict) else None
    if not isinstance(solved, dict):
        raise HTTPException(status_code=400, detail=tr(lang, "import_format_invalid"))
    known = {p.slug for p in problems.all_problems()}
    valid = {s: ts for s, ts in solved.items() if s in known and isinstance(ts, str)}
    progress.replace_all(valid)
    return {"imported": len(valid), "skipped": len(solved) - len(valid)}


# --- static frontend (production build) ------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
