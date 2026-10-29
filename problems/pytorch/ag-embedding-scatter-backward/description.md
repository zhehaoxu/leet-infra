# Embedding 反向：重复 id 累加、padding 行与稀疏梯度

## 题目描述

`F.embedding` 的前向只是查表，反向却是一次**散点累加**（scatter-add）：同一个 token id 在一个 batch 里出现多次，它的梯度必须把多份 `grad_out` 相加，而不是后写覆盖前写。再叠上三个工程口径——`padding_idx` 那行永远不该收到梯度、`scale_grad_by_freq` 要按词频摊薄、`sparse=True` 要返回真正合并过的 COO 稀疏张量——这道题就从「查表」变成了「稀疏归约」。本题给你 `grad_out, idx, num_embeddings` 及三个开关，返回形状 `(num_embeddings, dim)` 的梯度。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 输出第 \(r\) 行 = 所有满足 `idx[i] == r` 的 `grad_out[i]` 之和；未命中的行全为 0。
- `padding_idx` 指定的行**整行置 0**（前向它是常量），且这一条优先于其它规则。
- `scale_grad_by_freq=True` 时，每个 token 的梯度先除以该 id 在 `idx` 中的出现次数，再参与累加。
- `sparse=True` 时返回 `torch.sparse_coo_tensor` 且必须已 `coalesce()`，`to_dense()` 与稠密路径逐位一致。
- `idx` 越界、`grad_out` 非二维、行数与 `idx` 不符、`num_embeddings <= 0` 一律抛 ValueError；返回 dtype 与 `grad_out` 一致。

## 示例

4 行词表、2 维、`idx=[1,1,0]`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
grad_out = [[2, 4], [6, 8], [10, 12]]    idx = [1, 1, 0]
dense    = [[10, 12],   # 第 0 行来自第 3 个 token
            [ 8, 12],   # 第 1 行 = [2,4] + [6,8]（累加，不是覆盖）
            [ 0,  0], [ 0,  0]]
freq     = [[10, 12], [4, 6], [0, 0], [0, 0]]   # scale_grad_by_freq=True：id 1 出现 2 次
```

## 约束

- 内部 float64 累计后折回 `grad_out.dtype`；与 `F.embedding` + `torch.autograd.grad` 对拍，float64 atol=1e-9
- `idx` 为一维 int64、`n` 可达数千；稀疏路径要用 `values().shape[0] <= n` 验证没有退化成稠密
