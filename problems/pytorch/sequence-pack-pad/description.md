# 数据整理：序列打包成定长块（pack）

## 题目描述

逐条 padding（`data-collator`）会浪费大量算力——短样本被填成一堆 pad。现代 LLM 训练改用**序列打包（sequence packing）**：把许多短序列首尾相接拼成一条长流，再切成固定长度 `block_size` 的块，只有最后一块可能含 pad。代价是要额外产出 `position_ids`（每条子序列的位置**各自从 0 重新计**）与 `segment_ids`（标记每个 token 属于第几条子序列），好让注意力**不跨子序列**。本题实现这个打包函数。

## 规则

实现 `solve(seqs, block_size, pad_token_id)`：

- `seqs`：`list[list[int]]`，每条 ≥ 1。按**给定顺序**拼接。
- 把所有 token 连成一条流（子序列之间不插 pad）。设真实 token 总数 `total`，块数 `num_blocks = ceil(total / block_size)`。
- 在流末尾补 `num_blocks * block_size - total` 个 `pad_token_id`（pad 的 `position=0`、`segment=-1`）。
- 把补齐后的流**按行优先** reshape 成 `[num_blocks, block_size]`。
- 返回 **dict**，四个 `torch.long` 张量，形状都 `[num_blocks, block_size]`：
  - `input_ids`：打包 + 补 pad 的 token。
  - `position_ids`：每个 token 在其**原子序列**内的下标（每条子序列都从 0 起）；pad 处为 `0`。
  - `segment_ids`：每个 token 所属子序列的序号（0、1、2…，跨块**连续**）；pad 处为 `-1`。
  - `attention_mask`：真实 token 为 `1`、pad 为 `0`。

> 关键点：一条子序列**可以跨块**，其 `position_ids` 越过块边界仍继续递增；块边界不重置位置，只有「新子序列开始」才重置。

## 示例

`seqs = [[10,11,12],[20],[30,31]]`, `block_size = 4`, `pad_token_id = 0`（total=6 → 2 块）：

```
input_ids     = [[10,11,12,20], [30,31, 0, 0]]
position_ids  = [[ 0, 1, 2, 0], [ 0, 1, 0, 0]]   # doc0:0,1,2 | doc1:0 | doc2:0,1 | pad:0,0
segment_ids   = [[ 0, 0, 0, 1], [ 2, 2,-1,-1]]   # 每条子序列一个 id，pad=-1
attention_mask= [[ 1, 1, 1, 1], [ 1, 1, 0, 0]]
```

`seqs = [[5],[6,7]]`, `block_size = 3`（total=3 → 1 块，无 pad）：

```
input_ids=[[5,6,7]]  position_ids=[[0,0,1]]  segment_ids=[[0,1,1]]  attention_mask=[[1,1,1]]
```

## 约束

- `block_size ≥ 1`；四个输出 dtype 均 `torch.int64`，形状 `[num_blocks, block_size]`
- 子序列按输入顺序拼接、不修改 `seqs`；pad 判定唯一来源是 `segment_ids == -1`
- 纯确定性整数运算、无随机
