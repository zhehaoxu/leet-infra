# 解题说明 · ObservedAttention

## 与 H2O 的差别

H2O 用**累计 attention**（每个 key 从被生成到当前步收到的 attention 总和）；kvpress 的 ObservedAttention 除以 `L - j`（能看到它的 query 数）变成 **平均值**——避免早期 key 因"存活时间久"而天然高分。

## 三步的实际形态

```python
scores = attn.sum(2)                     # (B, Hq, K) — 对 query 求和
n_in_sum = torch.arange(L, 0, -1)         # 每 key 被 (L - j) 个 query 看到
scores = scores / n_in_sum                # 平均
scores = scores.view(B, H_kv, groups, L).mean(2)  # kv-group 平均
```

## kv-group 平均

GQA 里多个 Q-head 共享一个 KV-head。`attn` 是 `num_heads` 张量，`keys` 是 `num_key_value_heads` 张量。要给 KV 头打分就要**把组内 Q-head 分数平均**。题目里 `attn` 直接给到 `Hq`，`groups = Hq // H`。

## 分母为何是 L - j

位置 $j$ 是 key（列），只有 $i \geq j$ 的 query（行）才能看到它——因果掩码下 $i < j$ 处为 0。所以对 attn 求和后再除以"有效 query 数"就是平均。
