# 项目合集（Project Collections）设计规范

> 用途：在首页新增「项目」标签页，承载**可跟着做的里程碑式项目合集**（如「从零训练一个小语言模型」「实现一个简单 Agent」）。创建任何新项目前，按本文件执行。
>
> **2026-11-05 修订**：项目题**不进** `problems/pytorch/`，全部放 `problems/projects/`；项目与项目进度**单独记录**，不混入 `collections.json` / 全库 order 池 / `data/progress.json`（详见 §1）。

## 0. 核心原则（先读这个）

1. **项目不是新的判题形态**。项目 = 现有无状态题之上的一层**组织结构**：把一道道题串成里程碑链，复用现有判题机与 30s 预算。
2. **拒绝有状态判题**。任何一步的判题不得依赖用户自己前序步骤写的代码——判题机只保存/回放官方脚手架，不回放用户历史代码（状态爆炸 + 防作弊崩坏）。状态传递一律走**脚手架链**（见 §2）。
3. **确定性红线与全库一致**：模型/LLM 生成本身不出，一切生成文本、概率、logprob、奖励、隐态**全部显式入参**（fixture）；判题环境只有 torch+numpy，无 GPU、无网络、无 librosa/transformers 等重依赖。
4. **每一步都要能独立验收**：脚手架内置前序参考实现，用户可从任意一步开始；任何一步挂了不污染后续。
5. **项目与普通题库物理隔离、记录隔离**：项目题有自己的根目录、自己的注册表、自己的编号（step）、自己的进度台账、自己的护栏。普通题库的扫描、order 去重、integrity 统计**都不覆盖** `problems/projects/`。

## 1. 目录结构与「单独记录」

### 1.1 目录树（项目题放 `problems/projects/`）

```
problems/
├── collections.json                 # 仅登记普通合集，项目一律不写进这里
├── pytorch/                         # 普通题（809+ 题），不动
│   └── <slug>/{meta.json, description.md, description.en.md, notes.md, notes.en.md,
│               variants/pytorch/{solution.py, starter.py, test_cases.py}}
└── projects/                        # 项目独立根目录
    ├── projects.json                # ← 总注册表（单独记录，见 1.2）
    └── <project-slug>/
        ├── project.json             # ← 单项目台账（单独记录，见 1.3）
        ├── architecture.md          # 开篇页：架构图 + 里程碑依赖图 + 每步「为什么存在」
        ├── overview.md              # 终局解锁的总览页（全程数据流拼图）
        ├── scaffold/                # 冻结的前序官方参考模块（见 §2）
        │   ├── __init__.py
        │   └── <module>.py
        └── steps/
            └── <NN>-<step-slug>/    # NN 为两位 step 号，如 04-causal-attention
                ├── meta.json
                ├── description.md / description.en.md
                ├── notes.md / notes.en.md
                └── variants/pytorch/{solution.py, starter.py, test_cases.py}
```

- 步骤目录内部**沿用普通题的五件套结构**（含 `variants/pytorch/` 三件套），判题机可直接复用，只需把「根目录 + 步骤路径」换成 projects 侧（见 §6.4）。
- `NN` 前缀保证目录按里程碑顺序自然排序；目录名与 `meta.json.step` 必须一致。
- 双语文件命名与普通题完全一致，`scripts/check_bilingual.py` 需扩展扫描 `problems/projects/`。

### 1.2 总注册表 `problems/projects/projects.json`

单独记录**所有项目**的展示级信息，前端「项目」tab 只读这一个文件：

```json
[
  {
    "slug": "build-a-simple-agent",
    "name": "实现一个简单的 Agent",
    "name_en": "Build a Simple Agent",
    "summary": "从零搭一个 ReAct 循环：工具 schema、记忆账本、重试熔断到多轮验收",
    "summary_en": "...",
    "difficulty": "Medium",
    "status": "draft",              // draft | live
    "step_count": 8,
    "display_order": 1
  }
]
```

### 1.3 单项目台账 `<project-slug>/project.json`

单独记录**一个项目**的里程碑台账——每步的编号、依赖、验收层、脚手架引用、状态都在这里，是「项目」这一层的唯一事实来源：

```json
{
  "slug": "build-a-simple-agent",
  "name": "实现一个简单的 Agent",
  "name_en": "Build a Simple Agent",
  "status": "draft",
  "architecture_page": "architecture.md",
  "overview_page": "overview.md",
  "framework": "pytorch",
  "steps": [
    {
      "step": 1,
      "slug": "01-react-loop",
      "title": "ReAct 循环状态机",
      "title_en": "ReAct Loop State Machine",
      "why": "让 Agent 有可观测的一轮循环边界",
      "depends_on": [],
      "verification": ["L1", "L2"],
      "scaffold_imports": [],
      "final_step": false,
      "status": "draft"
    },
    {
      "step": 8,
      "slug": "08-end-to-end-dialog",
      "title": "多轮脚本化对话终局验收",
      "depends_on": [6, 7],
      "verification": ["L1", "L2", "L4"],
      "scaffold_imports": ["react_loop", "tool_schema", "memory_ledger"],
      "final_step": true,
      "status": "draft"
    }
  ]
}
```

- `project.json` 与 `projects.json` 的 slug/name 必须一致，且 `step_count` = `steps` 长度；由护栏强制（§6.5）。
- 步骤的展示顺序取 `steps` 数组顺序，不取全库 `order`。
- **项目步骤 `meta.json` 不写 `order` 字段**（不占全库 order 编号池），字段为：
  ```json
  {
    "title": "注意力与因果掩码",
    "title_en": "Causal Attention Mask",
    "difficulty": "Medium",
    "tags": ["project", "attention"],
    "variants": {"pytorch": {"label": "PyTorch", "available": true}},
    "project": "build-a-simple-agent",
    "step": 4,
    "step_title": "注意力与因果掩码",
    "depends_on": [3],
    "verification": ["L1", "L2"],
    "scaffold_imports": ["tokenizer", "embed"],
    "final_step": false
  }
  ```
  `tags` 首项固定 `project`，便于前端过滤时与普通题区分；`variants` 仍保留（判题机依赖）。
- `depends_on` 仅用于展示推荐路径，不强锁（§7）。
- 终局步骤额外 `"final_step": true`；L4 复盘题作为终局项目的一个独立步骤（`NN` 最大、`verification: ["L4"]`），或挂在终局之后由 `project.json` 标 `"review_step": N`。

### 1.4 进度单独记录

- 项目进度**不写入** `data/progress.json`（那是普通题的单用户进度），单独放 `data/project_progress.json`：
  ```json
  {
    "build-a-simple-agent": {"completed_steps": [1, 2, 3], "unlocked_overview": false}
  }
  ```
- `backend/progress.py` 增加一组 project 侧读写函数与 `/api/projects/<slug>/progress` 路由，不与现有进度 API 混用。
- 判题通过与进度写入解耦：判题结果照常返回，`pass` 后由前端调用 project progress 接口记录；服务端不因进度记录失败影响判题结果。

### 1.5 后端路径解析

`backend/problems.py` 现有扫描只认 `problems/pytorch/<slug>/`。项目侧新增独立加载器：
- `GET /api/projects` → 读 `projects.json`；`GET /api/projects/<slug>` → 读 `project.json` + 各步 meta。
- 提交判题时按 `project` + `step` 定位 `problems/projects/<project>/steps/<NN>-<slug>/variants/<fw>/`，**拼接路径必须校验不越出 `problems/projects/` 根**（防目录穿越）。
- 脚手架模块在判题沙箱内以只读方式注入 `scaffold/` 命名空间。

## 2. 脚手架链（状态传递方案）

- 每步的 `starter.py` 中，**前序所有步骤的官方参考实现已内置**（作为冻结模块直接 import，来自 `scaffold/`，源头是前序步骤的 `solution.py`）。
- 用户只写当前步骤缺失的那一块；判题输入 = 官方脚手架 + 用户填空，仍然无状态。
- 命名与隔离纪律：
  - 内置参考模块统一放 `problems/projects/<slug>/scaffold/`（如 `scaffold.tokenizer`），与用户当前步代码物理隔离，杜绝用户「顺手改了前序实现导致对拍错位」。
  - `project.json` 的 `scaffold_imports` 与 `meta.json` 的同名字段一致，且必须与 `depends_on` 传递闭包对应（护栏校验）。
  - 脚手架内模块**禁止有隐藏状态**（全局变量、缓存、随机数），全部显式传参——否则状态会沿链泄漏。
  - 修改任一前序步骤的 `solution.py` 时，必须同步重新生成 `scaffold/`，并在 `project.json` 里记录一次版本标记（避免旧脚手架与新参考解漂移）。
- 终局步骤可额外支持「提交整包」模式：下载完整项目骨架、跑用户自己串联的全链路（仅限终局一步，且断言集合与分步验收相同；初期可不做，见 §7 待决策）。

## 3. 每步验证——四层验证栈

按步骤性质选用，一题可以组合多层；所选层写入 `project.json` 与 `meta.json` 的 `verification` 字段：

| 层 | 适用 | 做法 | 例 |
|---|---|---|---|
| **L1 单元对拍** | 局部算子/纯函数（tokenize、attention 输出、梯度值） | 与普通题完全一致：oracle 逐位/容差对拍 | 分词 round-trip、causal mask 输出 |
| **L2 性质不变量** | 大块代码无法只考精确值时 | 考性质而非数值：`decode(tokenize(x))==x`、梯度范数有限、参数仍在合法域 | encode/decode 往返、更新后权重有限性 |
| **L3 固定种子集成验收** | 训练类步骤 | **禁止判「训练收敛」**。判：固定初始化 + 固定 batch 跑 N 步后，loss 落进 `[lo, hi]` 带 + 参数快照逐元素容差比对。容差兜底 BLAS 差异；规模必须 tiny（≤2 层、≤256 维、几百 token），守住 30s | 50 步后 loss 3.1→2.4 带 |
| **L4 设计复盘** | 终局之后 | 选择题/排序题考整体认知（「为什么 encoder-only 不需要 cross-attention」），确定性可判 | 架构决策、数据流排序 |

L3 细则：
- 种子、初始化、数据全部显式固化在题面/fixture 中，禁止题内随机。
- loss 带与快照容差必须先用本地 venv 实测出带宽（同版本 torch CPU 基本可复现），再留 ±裕量。
- 超时红线：单步判题 ≤30s，tiny 模型 + 短训练步数先行实测。

## 4. 里程碑叙事（整体认知怎么建立）

- **项目开篇页** `architecture.md`：architecture map + 里程碑依赖图，每步标注「为什么存在」（如：手写反向传播是为了理解 autograd 替你做了什么）。
- **每步 notes 固定三段式**：`已完成 → 本步引入 → 下一步拿它做什么`（数据流视角，不得写训练日志式流水账）。
- **终局步骤通过后解锁「项目总览」** `overview.md`：把全程数据流（token→embedding→logits→loss→grad→update 或 用户输入→循环→工具调用→答案）拼成一张完整图，作为用户的「认知资产」；解锁状态记在 `data/project_progress.json`。
- L4 复盘题挂在终局之后，作为项目收尾（普通题形态，先不做页面内嵌交互，见 §7）。

## 5. Agent 类项目的确定性技巧

- **fixture LLM**：模型回复全部作为显式入参/fixture 给定；用户写的是 ReAct 循环状态机、工具 schema 校验、记忆账本、重试/退避、熔断等纯逻辑。
- 终局验收 = 一段多轮脚本化对话：断言每轮的工具调用序列、断点、最终答案。
- 与已有题的同构关系：`agent-react-loop`、`ag-tool-loop-breaker` 等题的判定哲学直接升级为叙事步骤，**机制可复用、题面需重写为叙事口径**（不得直接复制题面造成重复感；项目步骤单独记录，不占普通题库编号）。

## 6. 落地清单（每建一个项目必走）

1. 在 `problems/projects/<slug>/` 下建 `project.json`、`architecture.md`、`overview.md`、`scaffold/`、`steps/`；**不要**在 `problems/pytorch/` 建目录，**不要**改 `collections.json`。
2. 在 `problems/projects/projects.json` 登记项目总表条目（slug、双语名、步骤数、status、display_order）。
3. 每步用 `scripts/add_problem.py --project <slug> --step N` 生成 `steps/<NN>-<slug>/` 五件套；`meta.json` 填 `project/step/depends_on/verification/scaffold_imports`，**不填 `order`**。
4. 每步照常走题库五件套 + `--selftest --mode submit` 全绿 + oracle 对拍；`check_bilingual.py` 扩展扫描 projects 目录。
5. 后端：加 `/api/projects` 加载器与 project 侧判题路径解析（含目录穿越校验）、`data/project_progress.json` 进度读写；护栏**新建独立文件** `backend/tests/test_project_integrity.py`（不塞进 `test_problem_integrity.py`）：
   - `projects.json` 与每个 `project.json` 的 slug/name/step_count 一致，slug 无重复；
   - 每步 `step` 从 1 连续、`NN` 目录前缀与 `step` 一致、`depends_on` 均指向更小的 step、`final_step` 唯一且为最大 step；
   - `scaffold_imports` 引用的模块在 `scaffold/` 中存在、且与 `depends_on` 闭包一致；
   - 步骤 `meta.json` **不得**含 `order`；`tags` 首项为 `project`；五件套文件齐全；
   - `architecture.md` / `overview.md` 存在且双语齐备。
6. 前端：新 tab 读 `projects.json` + `project.json`，项目卡片 + 步骤进度环（进度读 `data/project_progress.json`，依赖不锁但显示推荐路径）。
7. 双语校验照常；notes 三段式纳入 review。

## 7. 已定与待决策

**已定**：
- 无状态判题 + 脚手架链（§0.2 / §2）。
- 四层验证栈（§3）；训练步骤禁判收敛，只判固定种子带（L3）。
- 开篇页 + 三段式 notes + 终局总览页（§4）。
- Agent 项目用 fixture LLM（§5）。
- **项目题独立放 `problems/projects/`、独立记录（§1）**：不进 `collections.json`、不占全库 order 池、进度与护栏各自独立。

**待决策**（实现前拍板）：
1. 步骤依赖是否强锁——当前倾向：**不锁**，显示推荐路径，照顾直取某步的老手。
2. 终局是否支持「上传整包、跑用户全链路」重判题模式——当前倾向：**初期不做**，先纯脚手架填空。
3. L4 复盘题形态——当前倾向：**普通题挂在项目尾部**，不做页面内嵌非判题交互。

## 8. 试点建议

- 先做 **「实现一个简单 Agent」**（fixture LLM，风险最低，复用 agentic 机制最多）。
- 再做 **「从零训练一个小语言模型」**（验证 L3 固定种子带的判题体验与超时控制）。
- 第三个候选方向：**从零实现 RAG / 推理服务账本**（复用 retrieval / serving 机制，验证项目层对非训练类叙事的适配）。

## 9. 与既有纪律的关系

- **order 池**：全库普通题 order 编号池照常（立项前实扫、动笔前复查并行会话）；**项目步骤不参与该池**，只用项目内 `step` 序号，因此 `test_problem_integrity.py` 的 order 唯一性断言不覆盖 projects。
- **查重纪律照常**：项目步骤题与 `problems/pytorch/` 普通题互相查重（尤其注意与同名机制的普通题撞题面）；项目之间也互相查重。
- **进度隔离**：`data/progress.json`（普通题）与 `data/project_progress.json`（项目）互不读写。
- **双语校验**：checker 扫描范围扩展，但判定标准一致。
- 修改代码后不自动 commit，等用户确认（全库协作约定）。
