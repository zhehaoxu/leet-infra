# notes（ag-toollm-dfs-ledger）

## 回源

- 论文 arXiv **2307.16789**（ICLR'24 spotlight）App A.4 原话：「Classical DFS algorithms generate multiple child nodes at each step, then sort all the child nodes, and select the highest-scoring node for expansion... we find empirically that in most cases, the nodes ranked highest are often the node generated at first. Therefore, we skip the sorting process of child nodes and choose a pre-order traversal」——这就是代码里 `with_filter=False`（DFSDT）与 `with_filter=True`（normal DFS）的分野。
- **todos 原候选「pass rate 回传」系误记，已纠正**：论文的 pass rate 是 ToolEval 的评估指标（PASS rate / win rate），与树搜索内部毫无关系；代码里唯一像「回传」的是 `compute_weight()`，而它在 master 里**恒返回 0.0**（ToT 的 Elo 残迹，Elo 更新只在 `Algorithms/ToT.py`，本仓未收该题）。真正需要「回传」的是**回溯倒计数**返回值（2/2/1 与 result-1）。
- 代码位置：原 X-PLUG/ToolLLM 仓已 404（API 查 `full_name` 为空），现为 **OpenBMB/ToolBench** master。DFS 于 2023-07-27「update new version」进入 `toolbench/inference/Algorithms/DFS.py`；`Tree/Tree.py` 提供 `make_finish` / `compute_weight` / `get_depth`；`LLM_rank/rank_candidate.py` 提供 `rank2_subfix` / `rank2symmetry` / `sum_based_rankn`。
- **抓码方式**：WebFetch 抓 raw 会吞缩进（本题多处缩进即语义），且 raw.githubusercontent 的 curl 反复 SIGKILL；改走 GitHub blob API（`/git/blobs/<sha>` + base64）拿原字节，才钉住 `delete_former_diversity_message = True` 在 `if len(js_list) > 0:` **之内**（若误认为在 children 级，会写出「diversity 候选为空也把上一条消息标 invalid」的错误口径）。

## 官方 quirk（逐条核实）

1. **进入即判的顺序就是优先级**：`depth >= max_step or pruned or is_terminal` 作为一个条件，进入后先看 `is_terminal`（终局 → status=1、入 terminal、返回 2），否则 `pruned = True`，再看 `observation_code == 4`（give up → 返回 2）或返回 1 —— **终局优先于深度上限**（depth 到顶但 status 3 仍记 terminal，见 `check_depth_cap`）。
2. **回溯是倒计数**：终局 `final_answer_back_length = 2`、放弃 `prune_back_length = 2`、普通剪枝 1；上层 `result > 1` → `return result - 1`。所以「2」意味着「再往回退一格」。
3. **answer 提前退出走 10000**（不是倒计数），且**同层剩余候选不再展开**：展开循环里 `if len(terminal_node) >= answer: return 10000` 在 `elif result > 1` 之前，于是第一棵子树出终局后，同层兄弟连 `make_finish` 都不做就直接返回——`answer=2` 想凑满必须跨层（`check_two_terminals_and_finish_span` 用两层结构凑出 `[9, 15]`）。论文说「找 n 个答案即退」，代码实际语义比论文描述的更早收手。
4. **normal DFS 的排序**：每对**双向各问一次**（`rank2symmetry` → `score[1 - bigger1] += 1`、`score[bigger2] += 1`），和分高者 +1、平局各 +0.5（`sum_based_rankn`）；`zip_value.sort(key=prior_score, reverse=True)` 是**稳定**降序（平分保生成序）。比对查询（每对 2 次）计入 `query_count`，也会把预算推过线。
5. **票只看末字符**：`rank2_subfix` 判 `output["content"].strip().lower()[-1] == "a"`——`"A is better"` 末字符是 `r`，会被判成「第一个候选输」。空串官方 `[-1]` IndexError，本题收敛成 ValueError。
6. **`make_finish` 的三个 quirk**：递减发生在标记**自身之后**（自身是 Action Input 就立刻 -1）；`inter_val >= 0` 才继续向上（`-1` 即停）；**root 也是 Action Input**，所以 `make_finish(2)` 在浅链上会一路标到 root，在三层深链上只标到第二个 Action Input（`[3..9]`，0/1/2 不标）。
7. **diversity 候选**：沿每个子节点的 `children[0]` 链下探到第一个 `Action Input`（`while not is_terminal and node_type != "Action Input" and children`），`name` 取 Action 的 `description`、`arguments` 取 AI 的 `description`、`function_output` 取 `observation`、`value` 取 `compute_weight()` **恒 0.0**；`js_list` 为空则整条 diversity 消息都不拼（`delete_former_diversity_message` 也跟着为 False）。
8. **预算判定在建点之前**：`query_count += 1` 之后立刻 `if query_count >= max_query_count: return 100000`——超预算的那条响应**连 Thought 节点都不建**（`check_query_budget` 断言 `node_types` 只有 4 个）。
9. **status 三态**：`4` → `pruned = True`（进入即 give up）；`3` → `is_terminal = True` **且** `make_finish(2)`（两处置位）；`1`（幻觉 API 名）**不动树**，只把 `new_message["function_call"]["name"]` 改成 `invalid_hallucination_function_name`（仅消息层，本题不入账，记在此处）。
10. **Action Input 节点上收到无 function_call 的响应会炸**：官方 `temp_now_node.messages.append(new_message)` 后紧接 `new_message["function_call"]["name"]` → KeyError；本题收敛为 ValueError。
11. **空响应（content 与 function_call 皆 None）落在 Thought 上会自递归**：`temp_now_node` 没变，`self.DFS(temp_now_node, ...)` 就是 `DFS(now_node)` 自己——`expand_num` 被重复赋值、预算一路烧到 `max_query_count` 才停（`check_query_budget` 里 `expand_order == [0, 1, 1]`、`return_value == 99998`）。
12. **give up 的随机挑选**：`random.randint(0, len(give_up_node) - 1)` 用的是全局 `random`；本题把它显式化为 `give_up_seed`（`random.Random(seed).randint`），语义等价且可复现。

## 出题工程

- 两处期望值初版写错，都是「想当然」：① `answer=1` 时第一棵子树出终局就返回 10000，同层候选不再展开（`expand_order` 是 `[0, 6]` 不是 `[0, 6, 3]`）；② **深度只在递归进入节点时判**，链内建点不看深度（`max_step=2` 时 depth 2 的 Action 照样建出来，被剪的是 depth 3 的 AI）。教训：所有断言先跑 trace 打印 ledger 再钉。
- **变异第一轮 6 条 SURVIVED**，全部靠补场景解决：双终局（terminal 登记序）、三层深链（`make_finish` 跨度）、三候选 0.5 平分、末字符票、默认参数。其中 0.5 平分原本**不可观测**（只影响相对序，而平局两侧同加 0.5 不改变序），因此新增账本字段 `prior_scores` 把排序分导出——`rank2_subfix`/`sum_based_rankn` 的分数口径才真正可判。
- 3 条等价变异（有证明，记此不设检查点）：`eq-min-commute`（`(lambda v: (v[0], v[1]))(...)` 直通是纯 no-op）；`eq-expand-num-as-len`（赋值点 `append` 尚未发生，`len(expand_order) == now_expand_num`）；`eq-sort-desc-vs-neg`（`sorted(key=-s)` 与 `sorted(key=s, reverse=True)` 在无 NaN 时给出同一置换——`reverse=True` 仍稳定）。
- 最终 37 条实质变异全 CAUGHT：回溯长度/预算/提前退出常量、进入即判的四条分支、`make_finish` 三条、排序四条（含不稳定 tie-break）、倒计数两处、diversity 四条、expand 记账两条、预算/工具表 key、默认参数两条。
- 红线：真实 API 执行器、LLM 打分与 token 计数（rank tokens 记 0）、真实 `io_state`（`check_success` 收敛成 status 3）、消息层（diversity 消息内容、`sift_first_invalid_message`、hallucination rename）均不入题。

## 后续候选

- ToT 的 Elo 回传（`Algorithms/ToT.py`：成对比较 + 期望更新 + 剪枝）可作续题，但需与 `ag-tot-bfs-ledger`（order 720）的 BFS 价值头划界，且要回源 Elo 更新公式。
- Self-RAG 反思 token 账（2310.11511）为下一题：检索触发门 + segment 级 ISREL/ISSUP/ISUSE 概率加权（权重入参），查重 `rs-prm-best-of-n`。
