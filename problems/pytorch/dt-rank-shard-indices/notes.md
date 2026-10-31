# 参考思路：先定步数，再拼 effective，最后跨步切片

## 实现

顺序很重要，三步之间不能互相「补偿」。

1. **先算 $\mathit{num\_samples}$ 与 $\mathit{total\_size}$**。分支条件是「`drop_last` 为真**且** $N\bmod R\neq 0$」，不是「`drop_last` 为真」。整除时两条公式重合，所以写成 `if drop_last and size % R: floor else: ceil` 与上游那句 `math.ceil((len(dataset) - num_replicas) / num_replicas)` 逐情形等价；这里用整除表达，不引入浮点，也就不会在 $N$ 很大时因为 `ceil` 的浮点实现出现偏差。
2. **再拼 `effective`**。不砍尾时补的是 `order` 的**头** `padding` 个元素，`padding > N`（只在 $N<R$ 时出现）就把整个 `order` 重复 $\lceil \mathit{padding}/N\rceil$ 遍再截断。这段等价于按取模下标取值：$\mathit{effective}[k]=\mathit{order}[k\bmod N]$。参考解写成显式列表拼接、oracle 写成取模，两条结构无关的路径必须撞在一起。
3. **最后跨步切片** $\mathit{effective}[\mathit{rank}:\mathit{total\_size}:R]$。长度天然是 $\mathit{num\_samples}$，不需要再切一次。

$\mathit{coverage}$ 是在 `effective` 上数出来的全局账，不是当前 rank 的账，所以它必须在切片**之前**、对整条补齐/砍尾后的序列统计。

上游同一段逻辑（逐字摘录，仅去掉类型注释）：

```text
if self.drop_last and len(self.dataset) % self.num_replicas != 0:
    self.num_samples = math.ceil(
        (len(self.dataset) - self.num_replicas) / self.num_replicas
    )
else:
    self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
self.total_size = self.num_samples * self.num_replicas

padding_size = self.total_size - len(indices)
if padding_size <= len(indices):
    indices += indices[:padding_size]
else:
    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
# drop_last branch instead does: indices = indices[: self.total_size]

indices = indices[self.rank : self.total_size : self.num_replicas]
```

示例 $\mathit{order}=[6,2,0,4,1,3,5]$、$R=3$ 的逐格推演（`drop_last=False`）：

```python
# num_samples = ceil(7/3) = 3, total_size = 9, padding = 2
# effective  = [6, 2, 0, 4, 1, 3, 5, 6, 2]
# positions   0  1  2  3  4  5  6  7  8
# rank 0 takes positions 0,3,6 -> [6, 4, 5]
# rank 1 takes positions 1,4,7 -> [2, 1, 6]
# rank 2 takes positions 2,5,8 -> [0, 3, 2]
# coverage by value: 0->1 1->1 2->2 3->1 4->1 5->1 6->2   sum = 9
```

## 正确性检验

`order` 是显式入参且必须是 $0..N-1$ 的排列，判题里没有任何随机数进入返回值的对拍；`_case` 里的 `torch.randperm` 只是**生成**排列的工具，种子写死。八条 `check_*`：

- `check_example`：题面两个分支的四元组逐位比对，三个 rank 都查。
- `check_matches_oracle`：25 组 $(N,R,\mathit{rank},\mathit{drop\_last})$ 组合 × 3 个排列，覆盖 $N=1$、$N<R$、$N=R$、$R=1$、$N\bmod R\in\{0,\dots\}$ 各档。
- `check_shards_interleave_to_effective`：**所有** rank 的分片按「每步依次取各 rank 的第 $k$ 个」交织回去，必须精确还原 `effective`；再用 `merged.count(value)` 独立复核 `coverage`。这条同时锁死「切片是跨步的」和「coverage 是全局账」——写成连续块的话交织结果与取模序列对不上。
- `check_padding_and_truncation_edges`：$\mathit{padding}=0$ 时两条分支必须给出完全相同的四元组；$\mathit{padding}=N$（$N=5,R=10$，恰好把整份复制一遍）；$N<R$ 且不砍尾但 $\mathit{padding}<N$（$N=3,R=5$，`coverage=[2,2,1]`，`rank 0 -> [1]`、`rank 4 -> [0]`）；$\mathit{padding}>N$ 的「整份重复」分支由 $N=1,R=4$（`coverage=[4]`）与 `check_matches_oracle` 里的 $N=2,R=5$ 覆盖；$N<R$ 砍尾时 `num_samples=0`、`shard` 是空的 int64 张量而不是 `None`；再加 $N=1$、$R=1$ 两个极端口。
- `check_duplicated_samples_are_the_head`：用 `coverage` 反证「被读两次的恰是 `order` 的头 `padding` 个」，砍尾时则恰是尾部若干样本记 0、其余记 1，且总和等于 `total_size`。
- `check_identity_order_is_strided`：$\mathit{order}=\mathit{arange}$ 时跨步切片与连续块的区别最明显——$N=12,R=4$ 每个 rank 拿到 $\mathit{range}(\mathit{rank},12,4)$；$N=10,R=4$ 得到 `[[0,4,8],[1,5,9],[2,6,0],[3,7,1]]`，多出来的两个重复槽落在 rank 2 与 rank 1，而不是「最后一个 rank 多拿一个」。
- `check_invalid`：17 个非法入参（非张量、2-D、空、int32、float、非排列、越界、负下标、$R\le 0$、`float`/`bool` 型参数、`rank == R`、`drop_last` 非 bool）都要 `ValueError`；另加 4 个合法边界必须放行。
- `check_maximum_shape`：$512\times 7$ 与 $257\times 64$，两个 `rank`、两种 `drop_last`。

变异测试放了 17 个 mutant：连续块切片、`effective[rank:size:R]`、起点偏一格、步长取 $\max(1,N-\mathit{rank})$、补尾不补头、`padding > N` 时不重复、恒用 `floor`、砍尾但步数仍向上取整、`coverage` 只数 `shard`、`coverage` 恒 1、`coverage` 数 `order`、`total_size` 回 `N`、空分片返回 `None`、`coverage` 转 int32，以及三条放宽校验（不查排列、`drop_last` 收 int、`rank` 允许等于 $R$）——**16 个被抓到**。唯一幸存的是「`drop_last` 时无条件用 `floor`，否则 `ceil`」：$N\bmod R=0$ 时 `floor` 与 `ceil` 相同，它与参考解在全部输入上逐位一致，属于真等价变异，判题无法也没有必要区分。

## 范围与来源

逻辑取自本地 `.venv-eval` 内 torch 的 `torch/utils/data/distributed.py`（`DistributedSampler.__init__` 的步数计算与 `__iter__` 的补齐、砍尾、子采样，含两处 `AssertionError` 自检），上面围栏是逐字摘录。以下几点是**题面自定**，上游没有对应实现，不做归属声明：

- `coverage` 这个返回值（上游只有 `indices` 序列本身，不产出按样本下标聚合的账）。
- 把 shuffle 外置成显式 `order` 入参并要求它是排列。上游是 `g.manual_seed(self.seed + self.epoch)` 配 `torch.randperm`，那部分不在本题内，本题也不校验任何种子级复现。
- 非法入参统一抛 `ValueError`（上游对 `rank` 越界抛 `ValueError`，对长度不符抛 `AssertionError`）。
- 不含 `IterableDataset` 分支、不含 `set_epoch`、不要求返回可迭代对象而是直接返回张量。

与已登记的 `fw-distributed-sampler`（order 253）的分工：那一题不调用算法本身，而是对**真实的** `DistributedSampler` 做黑盒对账（固定 $N=7$、`seed=0`，考 epoch 置换的可重现性与重放一致性）；本题反过来，考的是你能否不靠库、在任意 $(N,R,\mathit{rank},\mathit{drop\_last})$ 上把索引分配写成纯函数，并算出库本身不提供的覆盖账。
