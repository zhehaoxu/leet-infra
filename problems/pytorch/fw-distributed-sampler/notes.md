# 详解：DistributedSampler 的单机索引账

## 出处

`torch.utils.data.distributed.DistributedSampler`（PyTorch 文档与源码 `torch/utils/data/distributed.py`）：shuffle 分支每次 `__iter__` 新建 `torch.Generator()` 并 `manual_seed(self.seed + self.epoch)`，故 epoch 是洗牌轮次的唯一变量（文档强烈建议在 epoch 边界调 `set_epoch`，否则每轮拿到相同置换）；`num_samples = ceil(N/R)`（drop_last=False）或 `floor(N/R)`（True）；补齐用**重复置换开头**的样本（源码 `indices += indices[:padding_size]`，padding 超过 N 时改为整段复制），不是随机补；切片是 `indices[rank : total_size : num_replicas]`。`num_replicas`/`rank` 显式传入时不要求初始化进程组，可单机模拟。

## 核心与易错点

1. **不调 `set_epoch` 就不洗牌**：置换种子只由 `seed+epoch` 决定，默认 epoch=0——epoch0 与 epoch0_again 逐位一致、与 epoch1 完全不同，是本题最重要的账。
2. **重复样本落在哪些 rank**：\(N=7, R=3\) 补 \(9-7=2\) 个（perm 开头两项 4、0）。stride 切片后补齐位（下标 7、8）落到 rank1、rank2——所以 rank1 的 `[0,2,4]` 里藏着重复的 4。每个样本出现次数 \(\ge 1\)、总数恰 9。
3. **`drop_last=True` 是截尾不是补齐**：`total_size = floor(N/R)*R = 6`，perm 尾部一项（样本 1）整个 epoch 不被任何 rank 看到——这是分布式训练「每 epoch 实际样本数略少」的出处。
4. **shuffle=False 时 epoch 无效**：序列恒为 `range(7)` 补齐，set_epoch(3) 也拿 \([0,3,6]/[1,4,0]/[2,5,1]\)。
5. **每 rank 长度恒为 `num_samples`**：补齐保证均分；源码里长度不符直接 `assert`。

## 对拍口径

solution 走 `DistributedSampler.__iter__`；判题 `_oracle` 不实例化采样器：只借 `torch.Generator + torch.randperm` 复现同一置换流（torch 置换算法无法用 numpy 复现，这是唯一需要共享的框架原语），补齐/截尾/stride 切片全部纯 Python 列表运算重写。另设 epoch 重放一致、drop_last 截尾重建、no_shuffle 静态账等断言。

## 复杂度

\(O(\text{epoch 数}\times R\times N)\) 的索引搬运，微秒级。
