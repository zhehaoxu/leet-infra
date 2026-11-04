# PEFT：MiSS 分片共享扩展与前向合并

## 题目描述

**MiSS**（Matrix Shard Sharing, arXiv:2409.15371，ICLR 2026，官方仓 `Joluck/MiSS`，2024.11 已进入 HuggingFace PEFT 的 `tuners/miss`）从 LoRA 的收敛根源下手：\(B\) 与 \(A\) 双矩阵联合优化是慢收敛的主因。MiSS 抛弃双矩阵结构，只留**一个零初始化小矩阵** `miss_block`，用「分片共享」复制机制把它扩展成低秩更新——参数量 \(O(d\cdot r)\)，训练起点 \(\Delta W=0\)，且初始化零开销。

本题复刻 PEFT `MissLayer` 的三形态确定性账：

- **miss（默认 expand）**：`miss_block (r, out)`。前向把输入按 \(r\) 分片、跨片求和聚合成 \(r\) 维（不整除则右侧零 pad），再乘 `miss_block`；merge 时 \(\Delta W\) 的每个输入分片位置都是完整的 `miss_block`ᵀ——两种形态数学等价（\(\Delta W\cdot x = B\cdot\sum_\text{块}x\)）。
- **mini**：`miss_block (r, mini_r)`，先沿输出维 repeat `out//mini_r` 次扩展成 \((r, out)\)，之后与 miss 同账。
- **bat**：`miss_block (out//r, r, r)`，基座权重按 \((out/r, r, in/r, r)\) 分块，第 \(j\) 输出块走 \(W_\text{块}\,@\,B_j + B_j\)——加性项破除 Bone 方法各分片更新共线的缺陷。

## 规则

- `mode="split_dims"`：返回 `{"n_blocks", "pad_size", "miss_block", "zero_init"}`；\(r\le0\) 或 \(r>in\) 抛 `ValueError`；bat 要求 \(in,out\) 均被 \(r\) 整除、mini 要求 \(out\) 被 `mini_r` 整除（违者抛 `ValueError`）。
- `mode="shard_sum"`：右侧零 pad 到 `n_blocks*r`，reshape \((\dots,n_\text{blocks},r)\) 后**跨片求和**（`sum(dim=-2)`），输出 \((\dots,r)\)。
- `mode="forward_delta"`：mini 先 repeat 扩展，再 `shard_sum(x) @ miss_block`；mini 缺 `out_features` 抛 `ValueError`。
- `mode="expand_delta"`：merge 的 \(\Delta W\) 账——\(\Delta W = \text{tile}(B^\top,(1,in//r))\)，形状 \((out,in)\)。
- `mode="bat_delta"`：\(\Delta W\) 的 \((j,i)\) 块 \(= W_{0}^{(j,i)}\,B_j+B_j\)。
- 全程 float64；dropout=0（Identity 分支不设参）。

## 示例

\(in=8, out=6, r=4\)（2 个输入分片）：

```
split_dims -> {n_blocks: 2, pad_size: 0, miss_block: (4, 6), zero_init: True}
shard_sum(x)  # (2,3,8) -> (2,3,4)
forward_delta = shard_sum(x) @ B   # (2,3,6)
expand_delta  = tile(B.T, (1, 2))  # (6, 8)，且 x @ dW.T == forward_delta
```

\(in=7, r=4\) 时 pad 到 8（`pad_size=1`）再聚合。

## 约束

- `x` 支持任意前置维，最后一维为 \(d_\text{in}\)；bat 的 \(\Delta W\) 需要完整基座 `w0` 入参。
- mini 的 `forward_delta`/`expand_delta` 必须显式提供 `out_features`。
- `expand_delta` 结果必须与 `forward_delta` 数学等价（测试里互证）。
