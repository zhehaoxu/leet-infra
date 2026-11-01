# notes — eval-judge-swap-consistency（order 684）

## 回源

- `lm-sys/FastChat` `fastchat/llm_judge/common.py`（run_judge_pair / play_a_match_pair）+ `show_result.py`（display_result_pairwise），GitHub API contents + base64。
- 官方账点逐条核对：
  - **判决解析**：`if "[[A]]" in judgment: A elif "[[B]]": B elif "[[C]]": tie else: error` —— **标记优先级**（A 恒先查），与标记在文本中的位置无关；大小写敏感。
  - **swap 两局**：`g1_map = {"A": "model_1", "B": "model_2"}`、`g2_map = {"A": "model_2", "B": "model_1"}`；`g1_map.get(w, w)` 使 tie/error 原样穿透。
  - **聚合条件**（show_result）：`if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]: 双方 tie` —— g1 平局或两局不一致都算平。位置偏差（swap 后仍判 A 位）恰好落进"两局各判一方"→ 平局。
  - **error 行级剔除**：`df_all[(g1_winner != "error") & (g2_winner != "error")]` —— 先剔 error 再谈 tie，因此「g1 tie + g2 error → 整场剔除」。
  - **胜率**：`win_rate_adjusted = (win + 0.5*tie)/(win+loss+tie)`；注释原话 "each tie counts as 0.5 win + 0.5 loss"。
  - 参考实现里 g1/g2 结果直接以 `g1_winner`/`g2_winner` 双字段存一行（single 模式两字段同值）——出题沿用此输出形态。
- MT-Bench 论文（arXiv 2306.05685）§position bias：GPT-4 判决有显著位置偏差，swap 一致性是对策；论文还报告了 agreement 指标（compute_agreement.py），未出题。

## 出题抽象

- 不真的调裁判 LLM：g1/g2 判决文本直接给出，考解析→映射→结算→聚合的全链路确定账。
- single 模式（one_score + TIE_DELTA=0.1）与两分数格式（[[a,b]] + regex backup）**未出题**——留作后续候选（notes 记账）。
- stats 覆盖 models 全集（缺席者全 0 行）是题目规定，非官方行为（官方 DataFrame 只含出现过的模型）——已在题面钉死，避免歧义。

## 账点（测试钉死）

1. **swap 一致算真胜**（check_consistent_win）：g1 A→alpha、g2 B→alpha（B 位 = answer_1）→ 双 alpha。
2. **位置偏差不一致 → 平局**（check_position_bias_disagreement）：swap 后仍 [[A]] → 两局各判一方。
3. **g1 平局即整场平**（check_any_tie_is_tie），g2 的胜负被丢弃。
4. **error 整场剔除**（check_error_dropped；变异 error-as-tie 抓）。
5. **error 优先于 tie**（check_error_beats_tie）：g1 tie + g2 error → 剔除而非平局。
6. **标记优先级 quirk**（check_mark_priority_quirk；变异 b-first 抓）：同文本双标记时先查的赢。
7. **大小写敏感**（check_case_sensitive_and_c_tie；变异 case-insensitive 抓）；[[C]] 平局（变异 c-as-b 抓）。
8. **多场聚合 + 半胜口径**（check_aggregation_over_matches；变异 adj-no-half / denom-short 抓）。
9. **缺席模型全 0 行、分母 0 → 0.0**（check_both_error_and_absent_models；变异 absent-skip / half-zero-to-one 抓）。
10. **负方向 swap**（check_loss_from_swapped_positions）：g1 B→model_2 且 g2 A→model_2 → model_1 记 loss。

## 变异记录

12/12 全 CAUGHT，无等价变异：`swap-forget`、`g1-map-flip`、`tie-strict`（or → and）、`b-first`、`error-as-tie`、`adj-no-half`（半胜 → 全胜）、`denom-short`、`c-as-b`、`no-mark-tie`、`case-insensitive`、`absent-skip`、`half-zero-to-one`。

## 手算锚点踩坑

- oracle parse 初版写成「标记在文本中的位置优先」（min(hits)）——官方是标记优先级；对拍失败暴露。
- 聚合用例 match 12 的 g2_judgment 原写 "whatever"（无标记）→ error 剔除优先于 tie，整场消失——手算漏了 error 优先级；反而促成独立用例 check_error_beats_tie，这是本批最隐蔽的账点。

## 与相邻题划界

- `eval-ifeval-rules`（order 665）：指令遵循的规则判别器（确定性规则，无裁判模型）。
- 本题：LLM-as-judge 的**评测协议账**（位置偏差对策与聚合），不含判决模型调用本身。
- 后续候选：single 模式打分（TIE_DELTA + score=-1 剔除）、[[a,b]] 双分数 regex backup、compute_agreement 的 kappa 账。
