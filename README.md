# leet-infra

本地单用户的 AI Infra 刷题网站（类 LeetCode / LeetGPU），第一阶段覆盖 **PyTorch** 题库。
在浏览器里写代码，点击「运行」只跑样例用例，点击「提交」才全量评测；代码在你本机的专用 Python 虚拟环境中以子进程真实执行，按测试用例逐项判定通过与否。

## 架构

```
浏览器 (Vue 3 + Vite + Monaco Editor, :5173)
   │  POST /api/problems/{slug}/run  { code }
   ▼
FastAPI 后端 (:8000)
   │  写入临时目录 → subprocess 调用评测环境 python（30s 超时）
   ▼
专用评测 venv (.venv-eval, Python + PyTorch, CPU)
   runner.py 逐用例执行 → 返回 { passed, total, cases[], stdout, error, elapsed_ms }
```

- 单用户本地工具：无登录、无数据库，进度存于 `data/progress.json`
- 评测环境独立于你日常的 conda/venv，互不影响
- Triton 题库为第二阶段规划（当前 Mac 无官方 Triton 支持，方案待定）

## 目录结构

```
backend/          FastAPI：题目 API、评测引擎、进度
problems/pytorch/ 题库（每题: meta.json / description.md / starter.py / test_cases.py / solution.py）
frontend/         Vue 3 + Monaco 前端
scripts/          setup_eval_env.sh / dev.sh
data/             本地进度（自动生成）
```

## 快速开始

前置：[uv](https://docs.astral.sh/uv/)（`brew install uv`）与 Node.js 18+。

```bash
# 1. 创建两个虚拟环境并安装依赖（评测环境含 torch，约 100+MB）
bash scripts/setup_eval_env.sh

# 2. 前端依赖
cd frontend && npm install && cd ..

# 3. 一键开发（后端 :8000 + 前端 :5173）
bash scripts/dev.sh
```

打开 http://localhost:5173 即可开始刷题。

生产模式（单端口）：

```bash
cd frontend && npm run build && cd ..
.venv-backend/bin/uvicorn backend.app:app --port 8000
# 打开 http://127.0.0.1:8000
```

## 测试判定规则

- 每题含 4~7 个独立测试用例（`check_*` 函数）；其中标有 `check_xxx.example = True` 的是样例用例
- **运行**（`POST /run`）只执行样例用例，不记录进度；若题库未标记任何样例则回退为全部
- **提交**（`POST /submit`）执行全部用例，全部通过即 **Accepted**，记录进度并解锁参考答案
- 运行耗时仅展示，不设性能门槛
- 评测固定使用 CPU、`torch.manual_seed(42)`，结果可复现；单次运行超时 30s

## 添加新题目

在 `problems/pytorch/` 下新建目录 `<slug>/`，提供 5 个文件（参考现有题目）。
要求实现 `def solve(...)` 签名，测试用例写成 `def check_xxx(solve): assert ...`。
无需重启后端：题面/代码按请求读盘，题目索引按目录 mtime 热加载（增删题目、修改 meta.json 立即生效）。

## 后端测试

```bash
.venv-backend/bin/pytest backend/tests -v
```
