#!/usr/bin/env bash
# 一键启动开发环境：FastAPI 后端 (:8000) + Vite 前端 (:5173)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -x .venv-backend/bin/python ] || { echo "请先运行 scripts/setup_eval_env.sh"; exit 1; }
[ -d frontend/node_modules ] || { echo "请先运行: cd frontend && npm install"; exit 1; }

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT

echo "=> 后端 http://127.0.0.1:8000"
.venv-backend/bin/uvicorn backend.app:app --port 8000 --reload &

echo "=> 前端 http://localhost:5173"
(cd frontend && npm run dev) &

wait
