#!/usr/bin/env bash
# leet-infra 一键生产模式：构建前端 → 由 FastAPI 同源托管并启动服务
# 用法: ./scripts/start.sh [端口]   （默认 8000）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8000}"

echo "==> 构建前端"
cd "$ROOT/frontend"
npm run build

echo "==> 启动服务 http://127.0.0.1:${PORT}"
cd "$ROOT"
exec "$ROOT/.venv-backend/bin/python" -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT"
