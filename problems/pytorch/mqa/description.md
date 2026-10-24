# 多查询注意力（Multi-Query Attention, MQA）

## 题目描述

标准多头注意力（MHA）里每个 query 头都有自己配对的一套 key/value，KV 缓存随头数线性膨胀。
**多查询注意力（MQA）** 让**所有 query 头共享同一组 key/value 头**（KV 头数 = 1），大幅压缩
解码期的 KV 缓存与显存带宽，是推理提速的经典手段（MHA 与 MQA 的折中即 GQA）。

## 规则

- 实现 `solve(q, k, v, num_heads)`。
  - `q` 形状 `(B, L_q, hidden)`，会切成 `num_heads` 个查询头，每头维度 $d_h = \text{hidden}/\text{num\_heads}$。
  - `k`、`v` 形状 `(B, L_k, d_h)`——**只有一个** KV 头，被所有查询头共享。
- 把该单 KV 头**广播（expand）**到 `num_heads` 份，逐头做缩放点积注意力：
  $$ \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)V $$
  再把各头输出按头序拼回 `(B, L_q, hidden)`。
- 无因果掩码、无 dropout。结果须与把单 KV 头 expand 后的
  `F.scaled_dot_product_attention`（或 `enable_gqa=True`）一致。

## 示例

```
输入：num_heads=2 → head_dim=2
     q = [[[1, 0, 2, 1]]]                形状 (1,1,4)  head0=[1,0], head1=[2,1]
     k = [[[0.5, 1], [1, 0]]]            形状 (1,2,2)  单个共享 KV 头, L_k=2
     v = [[[1, 2], [0, 1]]]              形状 (1,2,2)
   两个 query 头各自对同一组 K/V 做注意力后拼接
输出：[[[0.412521, 1.412521, 0.5, 1.5]]]  形状 (1,1,4)
```

## 约束

- float32 输入；`hidden` 能被 `num_heads` 整除
- KV 头数固定为 1（MQA 的定义）；`k`/`v` 的最后维必须等于 `head_dim`
- 不使用 `F.scaled_dot_product_attention` / `nn.MultiheadAttention`；需手写广播 + 缩放点积
