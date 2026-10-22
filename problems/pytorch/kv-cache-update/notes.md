# KV-Cache 更新详解

## 背景：自回归推理的加速核心

生成时逐 token 解码。若每步都重算整段的 $K,V$，复杂度 $O(t^2)$ 且大量重复。KV-Cache 把历史 token 的 $K,V$ 缓存下来，每步只算**新 token** 的 $K,V$ 并**追加**到缓存尾部，再对全序列做注意力。这样把「重复计算」换成「增量存储」，是 LLM 推理引擎（vLLM、continuous batching）显存管理的主体。

## 本关操作

沿**序列维**（通常 `dim=-2`，形状 $(B,h,L,d_k)$ 里 $L$ 那一轴）把新来的 `new_k` 拼到 `cache_k`：

$$ \text{cache}_k \leftarrow \text{concat}([\text{cache}_k,\ \text{new}_k],\ \dim=L) $$

$K,V$ 要**成对、同维**更新，长度始终一致。

## 坑

1. **拼接维度是序列轴，不是特征轴**：`dim=-2`（$L$），错用 `dim=-1`（$d_k$）会把向量拼宽、彻底破坏注意力。
2. **首步 cache 为空/None**：没有缓存时直接返回 `new_k,new_v`（本解处理了 `None` 与 `numel()==0` 两种「空」）。别对空张量 `cat` 报维度错。
3. K 和 V 可能一个已有一个没有（边缘测试），要各自判空。

## 进阶（视野）

真实推理里 cache 预分配 + 原地写入（避免每步 cat 重新分配），并有 **PagedAttention** 分块管理显存、缓存复用。prefill（一次性算整段入缓存）与 decode（逐步追加）两阶段。本关是这套机制的最小语义内核。
