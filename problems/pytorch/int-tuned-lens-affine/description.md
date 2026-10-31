# Tuned Lens 仿射探针账：层间 logits 映射与 KL 收敛账

## 题目描述

Tuned Lens 用每层一个仿射探针把中间层隐态投到词表空间，回答「如果 LM head 接在这一层会预测什么」。本题把探针账拆成四项：逐层 logits、对 final 预测的交叉熵、与 final 分布的 KL 散度、argmax 一致性——外加探针矩阵的秩账。

## 规则

实现 `solve(hidden_states, probe_w, probe_b, final_logits, temperature=1.0)`，返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `layer_logits` | `[L, B, V]` | `logits_l = W_l @ h_l + b_l` |
| `layer_ce` | `[L, B]` | 每层 log-probs 对 `target = argmax(final_logits)` 的交叉熵 |
| `layer_kl` | `[L, B]` | `KL(softmax(final/T) ‖ softmax(layer_logits/T))` 逐样本求和 |
| `argmax_match` | `[L, B]` | 层 logits 的 argmax 是否命中 final 的 argmax |
| `probe_rank` | `[L]` | 每个探针矩阵的秩 |

全部在 float64 下计算；`temperature ≤ 0` 抛 `ValueError`。

## 示例

```
V=2，probe 两行同向 + 偏置使层 logits = [1,1]（均匀），final=[-1,3]（命中 1）
layer_ce = -ln(0.5) = ln2
layer_kl = KL(softmax([-1,3]) ‖ uniform) = ln2 + Σ p·ln p（手算可验）
```

## 约束

- `L ≤ 4`、`V ≤ 12`、`D ≤ 8` 小规模，oracle 纯 Python 手写 softmax/KL
- probe = LM head 时最后一层 KL 严格为 0、更早层 > 0
