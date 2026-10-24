# Beam Search：一步扩展、打分与回退

## 题目描述

**Beam Search** 在每步维护 $B$ 条候选序列（beam）。解码一步时：每条 beam 都会尝试接上词表中每个 token，得到 $B\times V$ 个候选；每个候选的累计分数 = 该 beam 已有的累计对数概率 + 新 token 的对数概率；再从 $B\times V$ 里挑出分数最高的 $B$ 个作为下一步的 beam。

关键在于**回退（backtracking）**：被选中的 top-$k$ 是"展平"后的全局下标，必须还原出它来自哪条父 beam（`parent = flat // V`）以及接的是哪个 token（`token = flat % V`），才能把正确的历史序列搬过来续写。这一步决定了 beam search 不是贪心，而是"保留多条路径按需回退"。

## 规则

给定 `input_ids` $(B, L)$（当前 $B$ 条 beam 的 token 序列）、`beam_scores` $(B,)$（各自累计 log-prob）、`next_log_probs` $(B, V)$（每条 beam 下一位的 log 概率）、`num_beams` $k$：

1. 候选打分 $S[b, t] = $`beam_scores[b]` $+$ `next_log_probs[b][t]`，shape $(B, V)$
2. 按行优先**展平**为 $B\cdot V$，下标 $f$ 对应 $b = f // V$、$t = f \bmod V$
3. 取分数最高的 $k$ 个（`torch.topk`）；**并列时保留展平下标更小者**（topk 对相等的值按索引升序返回）
4. 对每个选中项还原 `parent = f // V`、`token = f % V`
5. 新序列 = `input_ids[parent]` 末尾接上 `token`；新分数 = 该候选的 $S$ 值

返回 `(new_input_ids, new_scores)`：`new_input_ids` 为 $(k, L+1)$ 的 `torch.long`，`new_scores` 为 $(k,)$ 的 float，**按分数降序**排列。

## 示例

```
B=3, L=2, V=4, k=2
input_ids   = [[1,2],[1,3],[4,5]]
beam_scores = [-0.1, -0.2, -1.5]
next_log_probs = log([[0.1,0.4,0.3,0.2],
                      [0.5,0.1,0.2,0.2],
                      [0.2,0.3,0.4,0.1]])

候选 S = beam_scores[:,None] + next_log_probs，展平后取 top-2：
  best:  parent=1, token=0 → seq [1,3,0], score -0.893147
  2nd:   parent=0, token=1 → seq [1,2,1], score -1.016291

new_input_ids = [[1,3,0],[1,2,1]]
new_scores    = [-0.893147, -1.016291]
```

## 约束

- CPU float32；`next_log_probs` 已是 log 概率（可直接相加）。
- `torch.topk` 在分数相等时按展平下标升序返回。
- 返回二元组 `(new_input_ids, new_scores)`。
