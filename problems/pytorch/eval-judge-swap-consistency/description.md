# LLM-as-judge 位置偏差账：swap 两局、不一致即平局与胜率聚合

用 LLM 给两个模型的回答打 pairwise 分，裁判会偏向放在 A 位的回答（位置偏差）。MT-Bench 的官方实现（FastChat `fastchat/llm_judge`）的对策是**每个问题打两局**：g1 按 (answer_1, answer_2) 摆位、g2 交换摆位；两局结论一致才算真胜负，不一致按平局计。本题实现这条全链路账：判决文本解析 → swap 摆位映射 → 场次结算 → 多场聚合。

## 判决解析（"[[_A_]]" 输出格式）

对每局裁判文本**按顺序**检查（官方 if/elif 链，与标记在文本中的位置无关）：

1. 包含 `[[A]]` → A 方胜
2. 否则包含 `[[B]]` → B 方胜
3. 否则包含 `[[C]]` → 平局
4. 否则 → error（解析失败）

匹配大小写敏感（`[[a]]` 不算标记）。同一文本含多个标记时，**先检查的标记赢**——这是官方实现的顺序 quirk。

## 摆位映射（swap 账）

- g1：A → model_1，B → model_2
- g2（交换过摆位）：A → model_2，B → model_1
- tie / error 不映射，原样保留

## 场次结算（逐场，官方 show_result 的条件）

1. g1 或 g2 **任一局 error → 整场剔除**（outcome = "error"，不计入任何统计）。注意优先级：error 剔除先于平局判定——g1 平局但 g2 解析失败时整场仍被剔除。
2. g1 为平局，**或**两局映射后不一致（位置偏差的典型形态：swap 后裁判仍判 A 位，两局各判一方）→ 平局（两个模型各记 tie 一次）。
3. 两局一致且非平局 → 胜负分明（胜者记 win、负者记 loss）。

## 统计与胜率

```
win_rate_adjusted = (win + 0.5 * tie) / (win + loss + tie)
```

每场平局算半胜半负；分母为 0 的模型 win_rate_adjusted = 0.0。stats 覆盖 `models` 列出的全部模型（没出现的模型全 0 行）。

## 输入 / 输出

```python
solve(matches, models)
# matches: [{"question_id", "model_1", "model_2", "g1_judgment", "g2_judgment"}, ...]
# 返回:
{
  "matches": [  # 与输入等长、同序
    {"question_id", "g1_winner", "g2_winner", "outcome"}
    # g1_winner/g2_winner 为映射后的 "model_1"/"model_2"/"tie"/"error"
  ],
  "stats": {model: {"win", "loss", "tie", "win_rate_adjusted"}}
}
```

## 例子

```
输入: matches = [{"question_id": 1, "model_1": "alpha", "model_2": "beta",
                  "g1_judgment": "x [[A]] y",        # A 位 = alpha 胜
                  "g2_judgment": "y [[A]] x"}],      # A 位 = beta 胜
      models = ["alpha", "beta"]
输出: matches[0] = {"g1_winner": "alpha", "g2_winner": "beta", "outcome": "tie"}
      （位置偏差：swap 后裁判仍判 A 位 → 两局各判一方 → 平局）
      stats = {"alpha": {"win":0,"loss":0,"tie":1,"win_rate_adjusted":0.5}, ...}
```
