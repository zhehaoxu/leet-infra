# SFT 数据：因果 LM 移位标签

## 题目描述

自回归（causal LM）训练的目标是「用前缀预测**下一个** token」：位置 \(i\) 的 logits 应当去拟合位置 \(i+1\) 的 token。于是把 `labels` 相对 `input_ids` **左移一位**，再叠加「只在被监督位算 loss」的掩码，其余填 `ignore_index`（`F.cross_entropy` 默认 `-100`）。这一步的 off-by-one、末位无后继、别跨 pad/段预测，是 SFT/预训练数据管线下最经典的坑。

## 规则

实现 `solve(input_ids, supervised_mask, ignore_index=-100)`，返回 int64、形状 \((B, L)\) 的 `labels`。

- `input_ids`：int64 \((B, L)\)。
- `supervised_mask`：\((B, L)\)，逐位置标明该 **token 是否应作为预测目标**（\(>0\) 视为监督，等价于 241 题产出的掩码）。
- 对每个位置 \(i\)：若存在**同一行内**的后继 \(i+1<L\) 且 `supervised_mask[b, i+1] > 0`，则 `labels[b, i] = input_ids[b, i+1]`（预测下一个被监督 token）；否则 `labels[b, i] = ignore_index`。
- 特别地，每行最后一列 \(i=L-1\) 没有行内后继，恒为 `ignore_index`（不跨到 batch 下一行）。

## 示例

\(B=1\)，`input_ids=[10,11,12,13,14]`，`supervised_mask=[0,0,1,1,0]`（只有下标 2、3 是监督目标）：

```
labels:
 i=0 -> 下一 idx1 mask=0 -> -100
 i=1 -> 下一 idx2 mask=1 -> input_ids[2]=12
 i=2 -> 下一 idx3 mask=1 -> input_ids[3]=13
 i=3 -> 下一 idx4 mask=0 -> -100
 i=4 -> 末列无后继      -> -100
=> [-100, 12, 13, -100, -100]
```

## 约束

- CPU int64 输入输出，形状严格 \((B, L)\)。
- 「预测目标」的判定用 `supervised_mask[b, i+1]`（看**后继位**），而非 `i` 本身。
- 每行最后一列恒 `ignore_index`；绝不把下一行首 token 当作本行末位的后继。
- `ignore_index` 默认 `-100`，按传入值填。
