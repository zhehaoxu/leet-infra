#!/usr/bin/env bash
# leet-infra 开发模式：后端热重载 + 前端热更新（不同于 start.sh 的生产单进程托管）
# 用法: ./scripts/dev.sh
#   后端 http://127.0.0.1:8000（uvicorn --reload，改后端代码自动重启）
#   前端 http://127.0.0.1:5173（Vite dev，HMR；/api 代理到后端 8000）→ 浏览器开这个
# 题库文件（problems/**）本就按 mtime 热加载，改题面/notes.md 无需重启；此脚本主要解决
# 「改了后端代码不生效」和「改前端要手动 build + 硬刷新」两个开发期痛点。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8000}"
[ -x "$ROOT/.venv-backend/bin/python" ] || { echo "缺少 .venv-backend，请先运行 scripts/setup_eval_env.sh"; exit 1; }

echo "==> 启动后端（热重载）http://127.0.0.1:${API_PORT}"
cd "$ROOT"
"$ROOT/.venv-backend/bin/python" -m uvicorn backend.app:app \
  --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir backend &
BACK_PID=$!
trap 'kill "$BACK_PID" 2>/dev/null || true' EXIT INT TERM

echo "==> 启动前端 Vite（热更新）http://127.0.0.1:5173  （/api 代理 → ${API_PORT}）"
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "   首次运行：安装前端依赖"
  npm install
fi
npm run dev
