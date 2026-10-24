# KV Cache 显存估算

## 题目描述

自回归解码会把每一层已算过的 Key/Value 缓存下来（KV cache），避免重复计算。长上下文推理时
**KV cache 的显存常常比模型权重还大**。本题按配置算出 KV cache 占用的字节数，并体现
MQA / GQA 通过共享 KV 头把这块显存压下去的效果。

## 规则

- 实现 `solve(layers, heads, head_dim, seq, batch, dtype_bytes=2, kv_heads=None)`，返回 KV cache 字节数（整数）。
- 缓存里同时存 **K 和 V** 两份，尺寸随 层数 × KV头数 × 每头维度 × 序列长度 × batch × 每元素字节数 增长：
  $$ \text{bytes} = \underbrace{2}_{K,V}\cdot \text{layers}\cdot \text{kv\_heads}\cdot \text{head\_dim}\cdot \text{seq}\cdot \text{batch}\cdot \text{dtype\_bytes} $$
- `kv_heads` 决定缓存里存几组 K/V：
  - `None`（默认，MHA）：每个查询头都有自己的 KV，`kv_heads = heads`；
  - `kv_heads=1`（MQA）：所有查询头共享一组 KV，缓存缩小到 $1/\text{heads}$；
  - `1 < \text{kv\_heads} < \text{heads}`（GQA）：分组共享，缩小到 $\text{heads}/\text{kv\_heads}$。
- `seq` 是**已缓存的 token 数**（生成长度），不是 prompt 之外的增量。
- `head_dim` 直接给定（不要求等于 `hidden/heads`）。

## 示例

```
输入：layers=32, heads=32, head_dim=128, seq=2048, batch=1, dtype_bytes=2
  MHA (kv_heads=None):  2*32*32*128*2048*1*2 = 1,073,741,824  （1 GiB）
  MQA (kv_heads=1):     2*32*1 *128*2048*1*2 =    33,554,432  （32 MiB）
  GQA (kv_heads=8):     2*32*8 *128*2048*1*2 =   268,435,456  （256 MiB）
solve(32,32,128,2048,1,2)         == 1073741824
solve(32,32,128,2048,1,2, kv_heads=1) == 33554432
```

## 约束

- `layers, heads, head_dim, batch` 为正整数；`seq ≥ 0`
- `dtype_bytes` ∈ {1, 2, 4}；`kv_heads` 为 `None` 或 $1\le \text{kv\_heads}\le \text{heads}$ 的整数
- 返回**整数**字节数（Python int）
