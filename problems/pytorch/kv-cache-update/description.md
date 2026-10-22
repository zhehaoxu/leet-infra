# KV Cache 追加（KV Cache Update）

## 题目描述

自回归解码时，为避免每步都重算历史 token 的 Key/Value，会把它们缓存在 KV Cache 中。每生成一个新 token，只需把它的 K、V **沿序列维追加**到缓存末尾。

$$ K_{\text{cache}} \leftarrow \text{Concat}(K_{\text{cache}},\ K_{\text{new}})\ \text{along seq dim} $$

## 规则

- 实现 `solve(cache_k, cache_v, new_k, new_v, dim=-2)`，返回二元组 `(k, v)`。
- 追加沿**序列维** `dim=-2`（形状 `(B, H, L, dh)` 中的 $L$）。
- **空缓存**：`cache_k`（或 `cache_v`）为 `None` 或元素数为 0 的张量时，直接返回对应的 `new_k` / `new_v`（长度 0 的空张量与 `None` 等价）。
- **不得原地修改**传入的缓存张量：`torch.cat` 返回新张量，旧缓存内容保持不变（`check_input_not_mutated`）。

## 示例

```
输入：cache_k=[[[1],[2]]], cache_v=[[[10],[20]]], new_k=[[[3]]], new_v=[[[30]]]
沿 dim=-2 追加
输出：k=[[[1,2,3]]] 形状(1,3,1),  v=[[[10,20,30]]]
      即 k=[[[1.0],[2.0],[3.0]]], v=[[[10.0],[20.0],[30.0]]]
```

## 约束

- 各步序列长度可不同（增量解码）；除 `dim` 外其余维度必须匹配
- 与 `torch.cat([cache, new], dim=-2)` 一致
