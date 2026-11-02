# notes（ag-selfrag-reflection-ledger）

## 回源

- 论文 arXiv **2310.11511**（Self-RAG，ICLR'24 oral）§4 推理：反思 token（Retrieve / IsREL / ISSUP / ISUSE）由模型自己生成，推理期用它们的**概率**做两件事——① 自适应检索门，② 多证据重排（segment 级加权分）。论文口径是「概率」，但**代码里门的实现用的是 log-prob 直接相除**，见下条。
- 代码：`AkariAsai/self-rag` main 的 `retrieval_lm/run_short_form.py::call_model_rerank_w_scores_batch`（门 + 三个打分 + 聚合一体）、`postprocess_answer_option_conditioned`；`retrieval_lm/utils.py` 的 `retrieval_tokens_names` / `rel_tokens_names` / `ground_tokens_names` / `utility_tokens_names` / `control_tokens` 与 `postprocess`。抓码走 GitHub blob API 拿原字节（raw 的 curl 偶发 SIGKILL）。
- CLI 默认值：`--threshold None`、`--w_rel 1.0`、`--w_sup 1.0`、`--w_use 1.0`（**函数签名默认 0.5**，argparse 覆盖成 1.0——本题按签名默认 0.5，并把三个权重都做成显式入参）、`--use_groundness` / `--use_utility` / `--use_seqscore` 均默认关、`--ndocs 10`、`--mode default`（**不在 choices 里**，等价于 adaptive）。

## 官方 quirk（逐条核实）

1. **检索门方向是反的**：`score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold`——log 概率为负，负数相除把大小关系翻过来。想检索（`lp[Retrieval]` 大、接近 0）时比值反而小；不想检索时比值接近 1 → 触发。用 exp 后的概率比才是论文意图。这是本题头号考点（`check_retrieval_gate` 两个方向都钉）。
2. **门的第三态**：`ret_tokens` 含 `[Continue to Use Evidence]`，但比值只用两条 token；该 token 只进 `score_dict` 不参与判定。
3. **死代码**：`if id not in pred_log_probs[0]: score_dict[tok] = -100` 之后紧接着 `prob = pred_log_probs[0][id]`（**无 else**），缺键先写兜底再当下标 → KeyError；兜底永远救不了。本题收敛成 ValueError。
4. **缺失 token 的兜底是 exp(-100) 而不是 0**：`prob = ... if id in ... else -100` 然后 `np.exp(float(prob))` → 3.72e-44（次正规）。两个 relevance token 都缺 → 归一化恰 0.5；该位置三个 ground token 都缺 → `1/3 + 0.5·1/3 = 0.5`。若把兜底写成 0 会变成 0/0（ZeroDivisionError 或 NaN）——变异 `m-missing-zero` 正是这样被抓。
5. **ground 只读第一个 ground token 位置**（`break`），utility **不 break**（收全部再取 `[0]`，结果等价、纯性能）。两者都只在「该 token 出现过」时才计算，否则 0.0（不是 0.5）。
6. **utility 权重表带负号**：`[-1, -0.5, 0, 0.5, 1]`——偏好 `[Utility:1]` 得负分，`final` 可被压成负数（`w_use=1.0` 时）。
7. **ground 与 utility 是可开关的**：`use_groundness`/`use_utility` 默认 False → `grd_tokens=None` → 对应分恒 0.0，即便证据里出现该 token。
8. **seq score 是长度归一的 exp**：`exp(cumulative_logprob / max(len(token_ids), 1))`；默认 `use_seqscore=False` 不参与。
9. **`results["no_retrieval"]` 没有 score**：它只存 `pred`。两条聚合路径都按 key 跳过它，所以 `do_retrieve=True` 时第一次生成的答案被丢弃；而 `len(results)==1` 分支返回的又是**第二次**生成（变量 `pred` 被重赋）——`results` 里存的与返回的不是同一段文本。
10. **两套 postprocess 不对称**：`utils.postprocess` 删 14 控制 token + `</s>` 后剥**一个**前导空格（空串直接返回 ""）；`postprocess_answer_option_conditioned` 删控制 token + `</s>` + `\n` + `<|endoftext|>`，**不剥前导空格**。closed 路径用后者、open 路径干脆不处理（直接返回原文）。
11. **closed 聚合按答案文本累加**：`answer2score.setdefault(answer, 0); answer2score[answer] += score`，所以同文本证据分数相加；`sorted(..., reverse=True)` 稳定 → 平局取先出现的答案。open 路径 `path2score` 取最高分路径，`sorted(...)[0][0]` 平局取先出现的 `retrieval_0`。
12. **`overall_scores` 是死数据**：函数里逐证据算出的 `relevance_score_dict` / `grd_score_dict` / `ut_score_dict` 只塞进局部 `overall_scores`，**从不返回**（`results` 只存 pred/score/ctx），且 `"ut_score_dict": utility_score` 还把 dict 写成了标量。本题把这些中间账做成 `evidence_scores` 暴露出来，正是为了让打分口径可判。

## 出题工程

- 反向门是**回源发现的**（论文写概率、代码写 log-prob 相除），先按代码实现再验证：`-2.0 / (-2.1) = 0.952` 与 `-0.1 / (-2.1) = 0.0476` 两个方向都钉进 `check_retrieval_gate`。
- 第一版把「空 token 序列」当合法输入（`max(len,1)` 归一），实测 `lps[0]` 越界 → 官方恒有第 0 个位置，改成域约束 ValueError（`tokens`/`lps` 不得为空），并把 `max(...,1)` 的守卫记成**受约束保护因而等价**的变异（`m-seq-no-max`）。
- 首轮 6 条 SURVIVED 的处理：`RET_TOKENS[:2]`（第三个 token 只算分不判定，等价）、`rel` 分母显式两项和（等价）、`seq` 的 `max(...,1)`（受约束保护，等价）、`no_retrieval` 的 score 值（两条聚合路径都跳过、单结果分支不下发 score，等价）、两条我写坏的空变异（删掉）；另补 `check_default_flags` 抓 `use_utility` 默认值。
- 浮点/精确锚点：`exp(-100)` 是次正规数，两次相等 → 比值恰 0.5 / 0.5（`ground` 1/3+1/6）。这类「缺失即相等」的构造比 `1e-9` 容差更稳，能把「兜底写成 0」的变异打出来。
- 最终 34 条实质变异全 CAUGHT + 6 条等价（有证明记此）。等价清单：① `RET_TOKENS` 第三个 token 只被算分（`score_dict` 多一个未被读取的键）；② relevance 分母写成两项显式和（与 `sum(dict.values())` 同序同值）；③ `max(len(tokens),1)`（空序列已被域约束拒绝）；④ `results["no_retrieval"]["score"]` 的取值（任何路径都不读）；⑤ `ratio` 分母两项交换（IEEE 加法可交换）；⑥ `final` 的左结合写法 `(a+b)+c` 与 `a+b+c` 是同一棵语法树。
- 红线：真实模型前向、tokenizer、vLLM 采样、数据集与 `acc`/`match` 指标不入题；检索器（Contriever/FAISS）与 `ctxs` 内容不入题；`overall_scores` 的内部 dict 只以 `evidence_scores` 的标量形式暴露。

## 后续候选

- Search-R1 检索 token 掩码账（2503.09516，order 730 起）：多轮 `<search>/<information>/<answer>` 交错序列的 loss mask 构造与轮次解析（纯序列账），立项前与 `rl-systems` 查重。
- Self-RAG 的 tree-decoding 分支（`max_depth` / `beam_width`，`run_short_form.py` 里 `[Continue to Use Evidence]` 的多段续写）可拆续题，但需先回源 `use_seqscore` 在长答侧的调度。
