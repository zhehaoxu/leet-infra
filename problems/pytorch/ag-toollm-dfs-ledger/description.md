# ToolLLM DFS 决策树账（ag-toollm-dfs-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：728

把 ToolLLM（arXiv **2307.16789**）的「基于深度优先搜索的决策树」**DFSDT** 搜索无状态化成一棵可复算的树账：给定 LLM 响应队列、工具执行结果表与两两比对票，复算整棵树的展开序、剪枝/放弃/终局记账、回溯倒计数与答案挑选。回源 [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) master（原 X-PLUG/ToolLLM 已迁此仓）：`toolbench/inference/Algorithms/DFS.py`（`DFS_tree_search.DFS`）、`Tree/Tree.py`（`make_finish` / `compute_weight`）、`LLM_rank/rank_candidate.py`（`rank2_subfix` / `rank2symmetry` / `sum_based_rankn`）。

## 你要实现的

`solve(responses, tool_table, pair_votes, single_chain_max_step=5, tree_beam_size=2, max_query_count=20, answer=1, with_filter=True, give_up_seed=42)`：返回树的展开/剪枝/终局账与最终答案账。

- `responses`：LLM 响应队列，**一次 LLM 查询消费一条**（先序消费）。每项 `{"content": str|None, "function_call": {"name","arguments"}|None, "error_code": int, "tokens": int}`。
- `tool_table`：`{(action_name, action_input): (observation, status)}`；status：0 成功 / 1 幻觉 API 名 / 3 给出最终答案 / 4 放弃。
- `pair_votes`：`{(i, j): [vote_ij, vote_ji]}`（i<j，i/j 为候选在 `next_tree_split_nodes` 里的生成序）；vote 是 `rank2_subfix` 的 LLM 输出串。

## 账点（源码逐行核对）

1. **进入即判三态**：`DFS(node)` 先 `expand_num = now_expand_num++`，再判 `depth >= single_chain_max_step or pruned or is_terminal`。顺序决定优先级：**终局优先于深度**（depth 到顶但 status 3 仍记 terminal），**深度到顶则剪**（`pruned = True`，只有 `observation_code == 4` 才落入 give up 并返回 2，否则返回 1）。
2. **回溯倒计数**：终局返回 `final_answer_back_length = 2`、放弃返回 `prune_back_length = 2`、普通剪枝返回 1；上层拿到 `result > 1` 就 `return result - 1` —— 倒计数决定「往回退几格」。
3. **DFSDT（`with_filter=False`）**：生成一个子链（Thought → 可选 Action + Action Input）后**立即递归**；递归返回 1 表示不必回退，循环继续生成本层下一个候选；`len(terminal_nodes) >= answer` 时直接返回 10000 提前退出。
4. **normal DFS（`with_filter=True`）**：先把 `tree_beam_size` 个子链攒完，候选 >1 时用 LLM 两两比对：每对**双向各问一次**（`rank2symmetry`，`score[1-bigger1] += 1`、`score[bigger2] += 1`），和分多者得 1 分、平局各 0.5 分（`sum_based_rankn`）；再按 `prior_score` **稳定降序**排序（平分保生成序）后逐个展开。**比对查询计入 `query_count`**（每对 2 次）。
5. **票的判定只看末字符**：`rank2_subfix` 用 `output["content"].strip().lower()[-1] == "a"` 判「第一个候选更好」——串里写什么都没用，只认末字符。
6. **展开循环的回退**：`result > 1`（子树出终局或放弃）→ 对当前节点 `make_finish(2)` 并 `return result - 1`，**同层剩余候选不再展开**（即便 `answer` 未满也一样）。
7. **`make_finish(inter_val)`**：标记 `finished` 并向上传播，每遇一个 `Action Input` 型节点（**含自身，root 也是 Action Input**）递减一次，`inter_val < 0` 即停——`make_finish(2)` 到第二个 Action Input 就断，第三个 Action Input 及其上层不标。
8. **diversity 候选**：节点已有子节点时，沿每个子节点的 `children[0]` 链下探到**第一个** `Action Input`（遇 terminal 或叶子即停），把 `{name: Action 描述, arguments: AI 描述, function_output: observation, value: compute_weight()}` 喂回 prompt；master 的 `compute_weight()` **恒返回 0.0**（ToT 的 Elo 残迹）。候选列表为空则**不喂**。
9. **预算**：`query_count >= max_query_count` 在计数之后、建节点之前判定——超预算的那条响应**整个丢弃**（节点不建），返回 100000 逐层 -1 上卷。
10. **答案挑选**：`terminal_nodes` 按 append 序取**第一个未被剪**的终局，`final_answer` 是该节点 `description`（Action Input 节点即 action_input 串）；无终局则从 `give_up_nodes` 里 `randint(0, len-1)` 随机挑一个（本题以 `give_up_seed` 显式化）。
11. **域约束（ValueError）**：响应队列耗尽；`tool_table` 缺 `(name, args)`；`pair_votes` 缺键或票串为空；**Action Input 节点上收到无 function_call 的响应**（官方此处 `new_message["function_call"]` KeyError）；`function_call` 缺 `name`/`arguments`。
12. **status 三态副作用**：4 → 节点剪枝（进入即 give up）；3 → `is_terminal` + `make_finish(2)`；1（幻觉 API 名）**不动树**，只把响应里的 name 改写成 `invalid_hallucination_function_name`（消息层，不入账）。

## 样例

```python
responses = [{"content": "先试 A", "function_call": {"name": "get", "arguments": "A"}},
             {"content": "再试 B", "function_call": {"name": "final", "arguments": "X"}}]
tool_table = {("get", "A"): ("okA", 0), ("final", "X"): ("ansX", 3)}
led = solve(responses, tool_table, {(0, 1): ["b", "a"]},
            tree_beam_size=2, single_chain_max_step=3, answer=1)
# 票末字符 'b'/'a' → 候选 1 全胜 → 先展开第二棵子树（node 6）
# expand_order == [0, 6]；terminal_nodes == [6]；finish_type == "give_answer"
# query_count == 4（2 次生成 + 2 次比对）；answer=1 → return_value == 10000
```

## 计分

12 个检查点：DFSDT 单链倒计数、深度上限与终局优先、放弃三来源（工具 status 4 / LLM error_code 4 / 普通剪枝）、normal DFS 两两比对与稳定排序（含三候选 0.5 平分、票只看末字符）、展开循环回退与 `make_finish` 跨度、双终局登记序、预算丢弃、diversity 候选（含空列表不喂、value 恒 0）、默认参数、域约束，以及 150 组随机脚本对拍。
