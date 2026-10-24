# MQA 详解

## 三种注意力头共享方式

| 方案 | KV 头数 | KV 缓存 | 表达力 |
|------|---------|---------|--------|
| MHA  | = num_heads | 最大 | 最强 |
| GQA  | 1 < g < num_heads（分组共享） | 中 | 强 |
| **MQA** | **1**（全部共享） | 最小 | 略降 |

MQA 出自 Shazeer（2019）「Multi-Query Attention」。解码时每步要为所有历史 token 存 K/V，
MHA 下这块缓存正比于头数，是长上下文推理的显存/带宽瓶颈。MQA 把 K/V 投影降到单头，
缓存缩小到 $1/\text{num\_heads}$，显著加快自回归解码，代价是质量略降。

## 实现要点：广播而非复制

单 KV 头形状 `(B, L_k, 1, d_h)`，要参与 `num_heads` 个查询头的注意力，用
`.expand(B, num_heads, L_k, d_h)` **广播**（不占额外内存），而不是 `repeat` 真复制。
之后每个头独立算 $\text{softmax}(Q_h K^{\top}/\sqrt{d_h})V$，最后按头拼接回 `(B, L_q, hidden)`。

## 易错点

- **head_dim 一致性**：`q` 切头后的 $d_h=\text{hidden}/\text{num\_heads}$ 必须等于 `k`/`v` 的最后维，
  否则 `@` 形状对不上（这是本题 dump 阶段曾踩的坑：k 写成单元素维导致维度不符）。
- **缩放用 $\sqrt{d_h}$**，不是 $\sqrt{\text{hidden}}$。
- **reshape vs view**：`transpose(1,2)` 后张量非连续，拼回 `(B,L_q,hidden)` 用 `.reshape`
  （或先 `.contiguous().view`），直接 `.view` 会报错。
- 头序拼接：`(B, L_q, num_heads, d_h).reshape(B, L_q, hidden)` 保证头按序、维按序。

## 与 SDPA 的关系

PyTorch 的 `F.scaled_dot_product_attention(..., enable_gqa=True)` 原生支持「KV 头少于 Q 头」的
分组/多查询共享。本参考答案手写 expand，与该接口在 KV 头数=1 时结果一致（已对拍误差 ~1e-7）。
