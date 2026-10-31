# Logit lens：final LayerNorm + unembed 的层间读出账

## 题意

Logit lens（nostalgebraist 提出）把中间层隐态直接当作「如果现在就输出」的读出：过 **final LayerNorm** 再乘 **unembedding 矩阵**得到该层的 logits。本题把读出做成账本：给定逐层（最后位置）隐态 `hiddens [L, D]`、final LN 的 `w/b`、unembed `[D, V]`，输出每层 logits、argmax（并列取最小下标）、与最后一层预测的一致性 `agree`、最后一层 top token 在每层 logits 中的排名 `rank_of_final`（1-based，严格大于计数）与每层 top logit。

## 求解

```
每层 l：
  mu  = mean(h)                       # D 维
  var = mean((h - mu)²)               # 有偏方差（除 D）
  n   = (h - mu)/sqrt(var + eps)·w + b     # eps 钉死 1e-5
  logits[l] = n @ unembed
argmax：最大值并列取最小下标（手写 min-index，不依赖 torch.argmax 的并列行为）
final_argmax = argmax[L-1]；agree[l] = (argmax[l] == final_argmax)
rank_of_final[l] = 1 + #{v : logits[l][v] > logits[l][final_argmax]}   # 终层恒 1
```

关键点：LayerNorm 是 logit lens 定义的一部分（`apply_norm=False` 是退化口径，用于对照）；eps 不是可忽略的小量（normed = ±2/√(4+1e-5) ≠ ±2）；`rank_of_final` 用**严格大于**计数，并列不影响。

## 数据规模

- 1 ≤ L ≤ 8，1 ≤ D ≤ 64，1 ≤ V ≤ 128。
- 形状不匹配、空输入、eps ≤ 0、apply_norm 非 bool → `ValueError`。
