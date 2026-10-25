# DistributedSampler 的单机索引账

## 题目描述

多卡训练时每个 rank 只吃 \(1/\text{world}\) 的数据，但每个 epoch 的洗牌必须全局一致、样本数必须被整除——这三本账（epoch 决定置换、补齐重复、rank 切片偏移）都由 `DistributedSampler` 一处管。本题在单机上显式传 `num_replicas`/`rank`，无需进程组，把它的索引分配彻底核对一遍。

## 规则

实现 `solve()`，按 starter docstring 的固定构造（\(N=7\)、`num_replicas=3`、`seed=0`）产出五份账：`num_samples`/`total_size`、epoch0/epoch1 的可重放置换切片、同 epoch 重放一致性、`drop_last=True` 的截尾账、`shuffle=False` 的静态切片账。

采样器语义可独立核对：置换种子 \(=\texttt{seed}+\texttt{epoch}\)；补齐 \(\texttt{total\_size}=\lceil N/R\rceil\cdot R\)，重复样本取置换开头若干项；rank \(r\) 拿 \(\texttt{full}[r::R]\)。

## 示例

```
输入：无
输出：
{
 "num_samples": 3, "total_size": 9,
 "epoch0_ranks": [[4, 3, 1], [0, 2, 4], [5, 6, 0]],
 "epoch1_ranks": [[0, 3, 4], [6, 5, 0], [1, 2, 6]],
 "epoch0_again": [[4, 3, 1], [0, 2, 4], [5, 6, 0]],
 "drop_last_epoch0_ranks": [[4, 3], [0, 2], [5, 6]],
 "no_shuffle_ranks": [[0, 3, 6], [1, 4, 0], [2, 5, 1]]
}
```

推演 epoch0：置换 \([4,0,5,3,2,6,1]\)，补齐两位重复 \([4,0]\) 得完整序列 \([4,0,5,3,2,6,1,4,0]\)；rank1 拿位置 1,4,7 → \([0,2,4]\)。

## 约束

- 返回 list 均为 Python `int` 的列表（`list(sampler)` 即得）；全部精确对拍。
- 不得依赖全局 RNG：采样器内部用自己的 `torch.Generator`，重复运行结果必须逐位一致。
