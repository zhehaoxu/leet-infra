# 解题说明 · TOVA

## 为什么只看最后一个 query

生成时，"下一步最可能引用的历史信息"就是最后一个 query 的注意力分布。TOVA 假设这个分布在解码期相对稳定，就用它一次性给全部 key 打分——比 SnapKV 更省（不看 attn 矩阵的内部、只用一行）。

## 跨 head 平均是"全局共享"

TOVA 的原始论文强调：**分数在 head 之间共享**。这样 topk 时每个头挑到同一批位置——保留了"所有头一致重要"的 token。kvpress 里 `scores = attn.mean(dim=head)` 后 `repeat(num_kv_heads, 1)` 就是把这份共享分数复制给每个 KV 头。

这与 AdaKV 的**跨头 flatten 挑全局 topk**不同：AdaKV 会让不同头保不同数量、TOVA 让每个头保一样多（且是同一批位置）。

## 尾部 pad

kvpress 用 `F.pad(scores, (0,1), value=max+1)` 把最后一个 token（`attn[..., -1:, -1:]`，即自注意力位）以最高分保留。

## 与 ObservedAttention 的关系

ObservedAttention 用**所有 query 对每个 key 的平均**，TOVA 用**最后一个 query**——两者是"历史注意力 vs 未来注意力"的两个极端。
