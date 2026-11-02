# notes — ag-rewoo-plan-ledger (order 721)

## 回源口径

- 官方仓 billxbf/ReWOO（论文同款）；ReWOO 算法主类在 **`algos/PWS.py`**（algos/ 下没有 rewoo.py；另有 react.py/notool.py 对照组）。核心三段：`_parse_plans` / `_parse_planner_evidences` / `_get_worker_evidences` + `run()` 的 log/cost 组装。
- 价格表 `utils/util.py::get_token_unit_price`：text-davinci-003=0.00002、gpt-3.5-turbo=0.000002、gpt-4=0.00003、LLAMA_WEIGHTS（tloen/alpaca-lora-7b、rewoo/planner_7B）=0.0，未知名 `ValueError("Model not found")`。工具 token 单价官方**固定用 davinci-003 价**（不随 planner/solver 模型变）。Google 每次查询 0.01。

## 关账点（官方 quirk 全录）

1. **多位下标失效**：`if len(e) == 3` —— `#E12 = Google[q]` 被记 `No evidence found`，原工具调用丢弃；但该 key 仍进 planner_evidences，后续 `#E\d+` 引用会替换成 `[No evidence found]`（"No evidence found" 字符串本身 pass-through 进 worker_evidences）。
2. **证据行解析**：`line[1]=='E' and line[2].isdigit()`（`#e1`/`#-1` 不识别）；`split("=", 1)` 两侧 strip；**重复证据行后者覆盖**（dict 覆盖、首次插入位置保序）。
3. **split("[",1) + [:-1] 盲删**：`Google[foo][bar` → tool_input `foo][ba`；`=` 右侧先整体 strip，故「工具名前导空格」quirk 不存在，但 `Google [x]`（方括号前空格）→ tool=`Google ` → 未知工具。
4. **替换语义**：`for var in re.findall(r"#E\d+", tool_input)`（按原文出现序、含重复）→ `if var in worker_evidences`（**worker 证据表**，非 planner 表）→ `tool_input.replace(var, "[" + ev + "]")` **全量替换**——插入文本新出现的占位符被后续 var 的 replace 命中（`#E1=Quote #E2 line`、`#E3=LLM[#E1 and #E2]` → `[Quote [cats] line] and [cats]`）；前向引用原样保留且计入 LLM token。
5. **worker_log 按下标配对**：`plans[i]` ↔ `#E{i+1}`（非按证据存在性）；证据多于 Plan 照算不进 log；缺失即崩（官方 KeyError，本题 ValueError 记差异）。
6. **token 账**：LLM `len(tool_input + evidence) // 4`（整除、算的是**替换后**调用串）；Google 计次数；Calculator 的 token 账依赖 langchain `LLMMathChain` prompt 模板常量——**不入题**（tools 里出现 Calculator 按普通工具处理，差异记档）。
7. **steps = len(plans) + 1**（+1 是 Solver 调用）。

## 官方不建模

- 真实 LLM/Google/Calculator 调用、planner/solver 的 prompt 模板与 fewshot（planner_log/solver_log 只剩 prompt+输出拼接，无独立信息）、wall_time、`WORKER_REGISTRY` 的具体工具实现（输出显式入参）。

## 出题过程教训

- **`=` 右侧先整体 strip**：设计「工具名带前导空格」用例时先被 strip 吃掉——quirk 必须用 `Google [x]`（左括号前空格）触发；写锚点前逐字符核对官方 strip 顺序。
- **Plan 行与证据按下标配对**：早期锚点给 3 条 Plan 只配 2 条证据直接 ValueError——构造随机用例时 plans 数与 `#E{i+1}` 存在性独立，ValueError 对称性（solve/oracle 同错）要单独验证。
- **变异 `len(a+b)//4 → len(a)+len(b)//4` 不是等价变异**（优先级：除法只作用 len(b)），实际 CAUGHT；真等价需加括号 `(len(a)+len(b))//4`（SURVIVED 实测确认记档）。
- 变异 22 条全 CAUGHT（log-use-planner 首轮 PATTERN NOT FOUND 因引号/括号匹配面写错、unknown-model-free 首轮被「缺 worker 输出」的 ValueError 抢先——两处修正后补测 CAUGHT）。
