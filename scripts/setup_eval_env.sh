#!/usr/bin/env bash
# 创建 leet-infra 专用评测环境（.venv-eval）与后端环境（.venv-backend）。
# 依赖：uv（https://docs.astral.sh/uv/）。若 GitHub 下载 Python 失败，
# 会自动回退到本机已有的 conda/homebrew CPython 3.10+ 解释器。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 若 PyPI 官方源下载超时，可改用国内镜像：
#   export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

if ! command -v uv >/dev/null 2>&1; then
  echo "错误：需要先安装 uv（brew install uv 或 curl -LsSf https://astral.sh/uv/install.sh | sh）" >&2
  exit 1
fi

pick_python() {
  # 优先让 uv 自带管理版本；失败则找本机 3.10~3.13 解释器
  for cand in \
    "$HOME/anaconda3/envs/llm/bin/python" \
    "$HOME/anaconda3/bin/python" \
    /opt/homebrew/bin/python3 \
    python3
  do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      local ver
      ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0.0)"
      local major="${ver%%.*}" minor="${ver##*.}"
      if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

create_venv() {
  local name="$1"
  if [ -x "$name/bin/python" ]; then
    echo "=> $name 已存在，跳过创建"
    return 0
  fi
  echo "=> 创建 $name"
  if ! uv venv "$name" --python 3.12 2>/dev/null; then
    local base
    base="$(pick_python)" || { echo "错误：找不到合适的 Python 3.10-3.13 解释器"; exit 1; }
    echo "   （uv 托管下载失败，回退到 $base）"
    uv venv "$name" --python "$base"
  fi
}

create_venv .venv-eval
create_venv .venv-backend

echo "=> 安装评测环境依赖（torch/numpy，首次约 100+MB）"
uv pip install -p .venv-eval torch numpy

echo "=> 安装后端依赖"
uv pip install -p .venv-backend fastapi "uvicorn[standard]" pytest httpx

echo "=> 冒烟验证：评测环境内 import torch"
.venv-eval/bin/python -c "import torch; print('torch', torch.__version__, 'OK')"

echo "完成。EVAL_PYTHON 默认指向 $ROOT/.venv-eval/bin/python"
