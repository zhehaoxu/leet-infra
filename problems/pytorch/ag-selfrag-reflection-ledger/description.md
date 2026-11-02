# Self-RAG 反思 token 打分账（ag-selfrag-reflection-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：729

把 Self-RAG（arXiv **2310.11511**）推理期的「自适应检索门 + 证据重排聚合」无状态化：给定无检索生成的反思 token logprob、每条证据的生成 token 序列与逐位置 logprob、以及打分开关，复算 `do_retrieve` 判定、segment 级 relevance/ground/utility 加权分与最终答案挑选。回源 [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) main：`retrieval_lm/run_short_form.py::call_model_rerank_w_scores_batch`、`postprocess_answer_option_conditioned`，`retrieval_lm/utils.py` 的反思 token 名单与 `postprocess`。

## 你要实现的

`solve(no_retr, evidences, second_text, mode="adaptive_retrieval", threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False, use_groundness=False, use_utility=False, closed=False)`：返回检索判定账、逐证据打分校验账与答案聚合账。

- `no_retr`：无检索那次生成 `{"text", "lp0"}`（`lp0` 是**第 0 个生成位置**的 `{token: logprob}`）。
- `evidences`：`[{"text", "tokens", "lps", "cumulative_logprob"}]`，每条证据一段检索增强生成；`tokens` 与 `lps` 等长。
- `second_text`：`do_retrieve` 为 False 时**第二次**生成（prompt 追加 `"[No Retrieval]"`）的文本。

## 账点（源码逐行核对）

1. **检索门是「原始 log-prob 相除」，不是概率比**：`do_retrieve = lp[Retrieval] / (lp[Retrieval] + lp[No Retrieval]) > threshold`。log 概率是负数，负数相除**把方向翻过来**：越不想要检索（`lp[Retrieval]` 越小），比值越接近 1，反而越触发——例如 `lp[No Retrieval] = -0.1, lp[Retrieval] = -2.0` 时比值 = 0.952 > 0.5 触发，而 exp 后的真实概率比只有 0.11。门槛**严格大于**（恰等 0.5 不触发）。
2. **门有三条短路与一条退化**：`mode="always_retrieve"` 恒真、`"no_retrieval"` 恒假（都不看门、也不下发 `retrieval_ratio`）；`threshold=None`（CLI 默认）退化成**字符串子串判定** `"[Retrieval]" in text`；argparse 的默认值 `"default"` 不在 choices 里，走的是 adaptive 分支。
3. **`[Continue to Use Evidence]` 只被算分、不进门**：`ret_tokens` 有三个 token，比值只用其中两个。
4. **缺失反思 token 分两种兜底**：门里 `if id not in logprobs: score[tok] = -100` 是**死代码**（下一行无条件下标取值，缺键当场 KeyError）；relevance/ground/utility 用 `exp(float(prob))`，缺失 token 兜底 `exp(-100)` ≈ 3.7e-44（**次正规数，不是 0**）——两个 relevance token 都缺时归一化恰为 0.5。
5. **relevance**：`P([Relevant]) / (P([Relevant]) + P([Irrelevant]))`。
6. **ground**：只在**第一个** ground token 出现的位置读 logprob（`break`），`P(Fully)/Σ + 0.5·P(Partially)/Σ`（第三个 token 权重 0）；token 从未出现则 0.0（不是 0.5）。
7. **utility**：同样取首个出现的 `[Utility:k]` 位置，权重表 **[-1, -0.5, 0, 0.5, 1]** —— 偏好 `[Utility:1]` 会得到负分。收集位置时官方**不 break**（收全再取首个，结果等价）。
8. **final**：`w_rel·rel + w_sup·ground + w_use·utility`（默认 1.0/1.0/**0.5**），`use_seqscore=True` 再加 `exp(cumulative_logprob / max(len(tokens), 1))`。`use_groundness`/`use_utility` 默认**关**（对应 CLI 开关），关时即使证据里出现对应 token 也不计分。
9. **答案聚合两口径**：`len(results) == 1` 时直接 `postprocess(pred)` 返回（`pred` 是第二次生成，`results["no_retrieval"]` 里存的却是第一次）；否则 `closed=True` 按**答案文本**累加分数再取最高（同文本合并、平局取先出现者），`closed=False` 取最高分路径并返回**未 postprocess 的原文**。
10. **无检索结果永远不参与聚合**：`results["no_retrieval"]` 没有 score，两条聚合路径都按 key 跳过——所以 `do_retrieve=True` 时第一次无检索的答案被丢弃。
11. **两个 postprocess 不对称**：`postprocess` 删 14 个控制 token + `</s>`、再剥**一个**前导空格；`postprocess_answer_option_conditioned` 删控制 token + `</s>` + `\n` + `<|endoftext|>`、但**不剥前导空格**。
12. **域约束（ValueError）**：`threshold` 路径下 `lp0` 缺反思 token（官方 KeyError）；`tokens` 与 `lps` 长度不一致或为空（官方 `lps[0]` 越界）；`do_retrieve` 为真却没有证据（官方 UnboundLocalError / 聚合空列表 IndexError）。

## 样例

```python
lp0 = {"[No Retrieval]": -0.5, "[Retrieval]": -1.5,
       "[Continue to Use Evidence]": -2.0}
ev = [{"text": "paris [Relevant]", "tokens": ["paris"],
       "lps": [{"[Relevant]": -0.357, "[Irrelevant]": -1.204}],
       "cumulative_logprob": -1.0},
      {"text": "lyon [Relevant]", "tokens": ["lyon"],
       "lps": [{"[Relevant]": -1.609, "[Irrelevant]": -0.223}],
       "cumulative_logprob": -1.0}]
led = solve({"text": "answer? [No Retrieval]", "lp0": lp0}, ev, "no",
            threshold=0.5)
# 门：-1.5 / (-1.5 + -0.5) = 0.75 > 0.5 → 检索
# best_path == "retrieval_0"；answer == "paris [Relevant]"（原文不清理）
```

## 计分

12 个检查点：反向检索门与严格大于、字符串退化门与 mode 短路、第二次生成与 `postprocess` 单空格剥离、relevance 归一与 `exp(-100)` 次正规兜底、ground 首位置与 0.5 权重、utility 负权重、seqscore 长度归一、默认开关、closed/open 两种聚合（同文本合并、平局稳定）、单结果分支、域约束，以及 120 组随机反思表对拍。
