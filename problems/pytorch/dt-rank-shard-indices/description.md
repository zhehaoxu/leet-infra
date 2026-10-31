# 数据分片：rank 切分、补齐与覆盖账本

## 任务

实现 `solve`：给一条已经排好序的全局样本索引序列，按 PyTorch `DistributedSampler` 的口径把它切成当前 rank 的那一份，并算出每个样本在**整个 world** 里被取了几次。返回分片索引、每 rank 步数、全局总槽数、覆盖账。

DDP 训练里最容易答错的一段不是通信，而是「$N$ 个样本分给 $R$ 个 rank 除不尽怎么办」。工业实现有两条截然相反的路线：`drop_last=True` 砍尾巴（各 rank 步数少一层，样本一次不多读），`drop_last=False` 从头补齐（每个样本至少读一次，代价是有些样本被重复读）。补出来的槽还会**改变哪个样本落到哪个 rank**，所以「重复的是头几个样本」不是随手写的。本题把这三件事（步数公式、补齐代数、跨步切片）做成纯函数，并额外要求算出覆盖账——它是排查「某个 rank 少跑一步」「验证集被重复计入」时真正会去看的数字。shuffle 本身（`torch.Generator().manual_seed(seed + epoch)` 配 `randperm`）不在本题内：`order` 作为显式入参给出，判题不碰随机数。

## 输入

- `order`：CPU int64 `[N]`，$N\ge 1$，内容必须是 $0..N-1$ 的一个**排列**（每个下标恰好出现一次）。
- `num_replicas`：Python `int`（`bool` 不算），$\ge 1$。
- `rank`：Python `int`（`bool` 不算），$0\le \mathit{rank}<\mathit{num\_replicas}$。
- `drop_last`：Python `bool`。

不修改输入，非法输入抛 `ValueError`。

## 精确规则

记 $R=\mathit{num\_replicas}$、$N=|\mathit{order}|$。

**步数**：

$$
\mathit{num\_samples}=
\begin{cases}
\lfloor N/R\rfloor,& \mathit{drop\_last}\ \text{为真且}\ N\bmod R\neq 0,\\
\lceil N/R\rceil,& \text{否则},
\end{cases}
\qquad \mathit{total\_size}=\mathit{num\_samples}\cdot R.
$$

两个分支的差别只在「除不尽」那一次；$N\bmod R=0$ 时两条公式重合，$\mathit{total\_size}=N$。$N<R$ 且 $\mathit{drop\_last}$ 为真时 $\mathit{num\_samples}=0$，该 rank 一步都不跑。

**effective 序列**（补齐或砍尾，二选一）：

$$
\mathit{padding}=\mathit{total\_size}-N,\qquad
\mathit{effective}=
\begin{cases}
\mathit{order}[:\mathit{total\_size}],& \mathit{drop\_last},\\
\mathit{order}\,+\,\mathit{order}[:\mathit{padding}],& \mathit{drop\_last}=\mathit{False},\ \mathit{padding}\le N,\\
\mathit{order}\,+\,(\mathit{order}\cdot\lceil \mathit{padding}/N\rceil)[:\mathit{padding}],& \mathit{drop\_last}=\mathit{False},\ \mathit{padding}>N.
\end{cases}
$$

第三种情形只在 $N<R$ 且不砍尾时出现（补齐量比整个数据集还长），此时同一个样本会被抄好几遍。$\mathit{effective}$ 的长度恒为 $\mathit{total\_size}$。

**切片**：本 rank 拿到的是跨步切片

$$
\mathit{shard}=\mathit{effective}[\mathit{rank}:\mathit{total\_size}:R],
$$

含左不含右、起点是 $\mathit{rank}$、步长 $R$，长度恰为 $\mathit{num\_samples}$。

**覆盖账**：$\mathit{coverage}[i]$ 是样本 $i$ 在 $\mathit{effective}$（也就是**所有** rank 加起来）中出现的次数，长度 $N$。它满足 $\sum_i\mathit{coverage}[i]=\mathit{total\_size}$；不砍尾时每项 $\ge 1$，砍尾时 $\mathit{effective}$ 里没出现的下标计 0。

## 返回

`(shard, num_samples, total_size, coverage)`：`shard` 是 int64 `[num_samples]`，`num_samples` 与 `total_size` 是 Python `int`，`coverage` 是 int64 `[N]`。全部逐位比较。

## 示例

$N=7$、$R=3$、`order = [6, 2, 0, 4, 1, 3, 5]`：

```python
# drop_last = False
# num_samples = ceil(7/3) = 3, total_size = 9, padding = 2
# effective = [6, 2, 0, 4, 1, 3, 5, 6, 2]      (= order + order[:2])
# rank 0 -> [6, 4, 5]   rank 1 -> [2, 1, 6]   rank 2 -> [0, 3, 2]
# coverage = [1, 1, 2, 1, 1, 1, 2]             (sum = 9)

# drop_last = True
# num_samples = floor(7/3) = 2, total_size = 6
# effective = [6, 2, 0, 4, 1, 3]               (= order[:6])
# rank 0 -> [6, 4]      rank 1 -> [2, 1]       rank 2 -> [0, 3]
# coverage = [1, 1, 1, 1, 1, 0, 1]             (sample 5 is dropped, sum = 6)
```

对照读：样本 6 排在 effective 的第 0 位和第 7 位，$0\bmod 3=0$、$7\bmod 3=1$，所以它**同时**出现在 rank 0 和 rank 1 的分片里（就是 `rank 1 -> [2, 1, 6]` 末尾那个 6）；同理样本 2 占第 1 与第 8 位，落在 rank 1 和 rank 2。把三个 rank 的分片按「rank 0 第 0 个、rank 1 第 0 个、rank 2 第 0 个、rank 0 第 1 个……」的顺序交织起来，正好还原 `effective`——这是跨步切片的定义性质，也是判题里最好用的一条不变量。
