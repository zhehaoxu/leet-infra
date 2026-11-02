# notes（ag-got-operations-ledger）

## 回源

- 官方仓 spcl/graph-of-thoughts（ETH Zurich，BSD），核心在 `graph_of_thoughts/operations/operations.py`（900 行）+ `thought.py`（117 行）+ `graph_of_operations.py`。README 展示的示例在 `examples/`（sorting/set_intersection/keyword_counting/doc_merge）。
- 论文 arXiv 2308.09687（Besta et al., "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"）。框架把推理过程建模为任意 DAG 上的操作图；本题取**线性操作链**（`append_operation` 语义），多前驱汇聚（`add_operation`）不入题——线性链已覆盖全部单操作语义，图拓扑调度（can_be_executed 的 DAG 就绪判定）留作后续候选。

## 关键官方 quirk（全部逐行核实）

1. **改进产物重置**（最大陷阱）：官方 `ValidateAndImprove` 每次改进后 `current_thought = Thought({**current_thought.state, **state_update})`——**全新 Thought**，score 复位 0.0、scored/valid/validated/solved/compared 全部复位。`Improve`/`Aggregate`/`Generate` 的产物同理。只有 `from_thought` 路径（Score/Selector/KeepBestN/KeepValid/GroundTruth 与 validate 的初始 attempt）复制标志。第一版 solution 把 score 带过改进链——被锚点抓出后修正。
2. **zip 截断**：`Score._execute` combined 分支 `for thought, score in zip(previous_thoughts, scores)`——parser 返回的 scores 更短时多出的 thought 静默消失（无日志无报错）。individual 分支逐 thought 调 scoring_function。
3. **尝试计数**：`while True` 内先验证、append、再判 `if not improve or valid or current_try >= num_tries: break`，改进后 `current_try += 1`。恒定无效 → 验证 num_tries+1 次（最后一轮验证发生在最后一次改进之后）。`current_try` 从 0 起。
4. **Aggregate 的排序合并**：`for thought in sorted(previous_thoughts, key=lambda t: t.score): base_state = {**base_state, **thought.state}`——升序 stable；**score 高的后合并**、dict merge 后者胜；未打分 thought 用缺省 0.0 参与排序（官方不检查 scored！）。官方注释 "applied in order of score"。
5. **KeepBestN 的 fallback**：`try: sorted(...) except: 降级为只保留 isinstance(i.score, float) 的 thoughts`。裸 `except:`——sorted 抛 TypeError（如 score=None 混入）即触发。**int score 不是 float**（`isinstance(1, float) is False`），fallback 时 int-scored thought 被丢弃——用例 `scores=[None, 2.5, 1, 3.5]` + `n=2` 同时钉死 None 丢弃与 int 丢弃。
6. **KeepValid 放行**：`if not thought.validated or thought.valid`——未验证 thought 直接保留（官方仅日志警告 "has unvalidated thoughts"）。
7. **GroundTruth 吞异常**：`try: new_thought.solved = evaluator(state) except: new_thought.solved = False`——裸 except，异常时 solved=False 且 compared_to_ground_truth=True（setter 无条件置位）。
8. **Generate 空转**：`if len(previous_thoughts) == 0 and len(self.predecessors) > 0: return`——thoughts 保持 []，后续操作级联空转；超产只 log warning 不截断。无前驱时 `previous_thoughts = [Thought(state=kwargs)]`（kwargs 即整题 base_state）。
9. **前驱断言 vs 空 thoughts**：官方 assert 检查 `len(self.predecessors) > 0`（操作图层面），线性链中除首个操作外恒真——**零 thoughts 不报错**（Aggregate 甚至显式 `if len(previous_thoughts)==0: return`）。第一版 solution 用 `if not prev: raise` 把两者混淆，random 对拍抓出（generate zip 截断产生零 thoughts → aggregate 误报 ValueError）。

## 出题工程

- 判题环境无官方依赖（graph_of_thoughts 包依赖 langchain/openai）——纯标量化无障碍：LLM 调用全部显式入参化（scores/valids/updates/parsed/results 序列按消费序）。
- 线性链的"父节点数"陷阱：首个 generate 无前驱 → `Thought(kwargs)` 单父 → parsed 多槽被 zip 截断。测试里要单父多子须把多条 state 塞进 parsed[0]。第一版锚点全按"槽位数=thought 数"构造，双层 zip 截断后全挂——修正为单槽多 state。
- base_state 污染：首个 generate 的父态是 base_state，合并后 state 含 base 键——锚点若用 `_compare(..., {"p": 1})` 则所有期望 state 都要带 `p` 键，干脆传 None。
- 随机对拍对称化：随机链可触发 IndexError（selector 越界）/TypeError（aggregate 遇 None score 排序）/ValueError——两侧实现行为一致，只要求同抛或同过（按异常类型名对称）。
- **变异 20 条全 CAUGHT + 2 条等价**：`selector 不复制 thought`（`prev[i]` 直引 vs `_from_thought`）与 `aggregate dict 包装写法`——官方语义下 state 共享引用且无原地变异，两种写法观测等价，记 notes。
- 红线：prompter/parser 的 prompt 模板与解析正则不入题（纯字符串工程）；GraphOfOperations 的 roots/leaves 维护（add_operation 的拓扑簿记）不入题——线性链下退化为恒等。

## 后续候选

- 多前驱汇聚 + DAG 就绪调度（can_be_executed + controller 的执行循环）可出 Hard 题：操作图账（拓扑就绪 + 多前驱 thoughts 拼接序 = predecessors 列表序）。
