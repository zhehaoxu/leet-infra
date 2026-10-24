# 数据整理：批 Padding 与 attention_mask / labels

## 题目描述

一个 batch 里各条样本 token 数不同，喂进模型前必须**右侧补齐**到同一长度，并产出配套的 `attention_mask`（别让模型看 pad）与 `labels`（别让 loss 算到 pad 上）。这就是 DataLoader 的 `collate_fn` / HF `DataCollator` 每天干的活。本题手写这个整理函数。

## 规则

实现 `solve(seqs, pad_token_id, ignore_index=-100)`：

- `seqs`：`list[list[int]]`，每条是一串 token id（长度可不同，均 ≥ 1）。
- `pad_token_id`：用来补齐的 id。
- 批内最大长度 `L = max(len(s))`，所有序列**右侧**补 `pad_token_id` 到 `L`。
- 返回一个 **dict**，含三个 `torch.long` 张量，形状都是 `[B, L]`：
  - `input_ids`：补齐后的 token。
  - `attention_mask`：真实 token 位置为 `1`，pad 位置为 `0`。
  - `labels`：与 `input_ids` 相同，但 **pad 位置填 `ignore_index`**（默认 `-100`，即 `F.cross_entropy` 的 `ignore_index`，这些位置不贡献损失）。

> 本题只做右侧补齐、不截断。`B`、`L` 由输入决定。

## 示例

```
seqs = [[1,2,3],[4],[5,6]], pad_token_id = 0

input_ids     = [[1,2,3],
                 [4,0,0],
                 [5,6,0]]
attention_mask= [[1,1,1],
                 [1,0,0],
                 [1,1,0]]
labels        = [[1,2,3],
                 [4,-100,-100],
                 [5,6,-100]]
```

## 约束

- 三个输出张量 dtype 均为 `torch.int64`（`torch.long`），形状 `[B, L]`
- 不修改传入的 `seqs`
- pad 位置：`attention_mask=0` **且** `labels=ignore_index`——两处都要对，是最常见的漏点
- 纯确定性整数运算，无随机
