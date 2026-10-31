# 稀疏化：Wanda 激活加权幅值剪枝

## 题目描述

只看权重大小会误判：某个权重虽然很小，但如果它连接的输入通道**激活很大**，乘出来的输出并不小；反过来，一个中等权重若长期乘在接近 0 的激活上，删掉它几乎无损。Wanda（*A Simple and Effective Pruning Approach for Large Language Models*, 2306.11695）因此把重要性改成**权重幅值 × 输入通道激活的 RMS**：

\[
m_{ij} = |W_{ij}|\cdot \sqrt{\frac{1}{N}\sum_{t=1}^{N} X_{t,j}^{2}}
\]

其中 \(X\in\mathbb{R}^{N\times d_{\text{in}}}\) 是校准集过这一层时展平的输入激活，\(N\) 是 token 数。每一**输出行**独立地在行内挑 \(m\) 最小的若干列置零——行与行之间不共享阈值，也不需要 Hessian。上游 `lib/prune.py::prune_wanda` 的写法是：

```python
W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
```

`scaler_row` 由 `lib/layerwrapper.py::WrappedGPT.add_batch` 逐 batch 增量累积 `norm(inp, p=2, dim=1) ** 2 / nsamples`，因此不开方、不除 \(N\) 的差异只影响统一缩放，不影响行内排序；本题把它写成「逐列均方 → 开方」的等价口径。

三个容易写错的细节：

1. **k 是截断而不是四舍五入**：上游 `int(W_metric.shape[1] * sparsity)`，`d_in=7`、`sparsity=0.5` 时剪 3 个而不是 4 个；
2. **同分取更小的列下标**：上游用 `torch.sort(..., stable=True)` 升序取前 k 个，并列时列索引小者先被剪；
3. **结构化模式只认 0.5**：`prune_n != 0` 时按每 `prune_m` 个连续列分块、块内取最小的 `prune_n` 个；`main.py` 对 `2:4`/`4:8` 断言 `sparsity == 0.5`，本题照抄这条约束，并要求 `in_features` 能被 `prune_m` 整除（上游对残缺块直接 `topk`，行为不完整）。

## 任务

实现 `solve(weight, activations, sparsity, prune_n=0, prune_m=0)`：

- `weight`：`(out_features, in_features)`，float32/float64；
- `activations`：`(N, in_features)`，同 dtype，`N >= 1`；
- `sparsity`：`[0, 1]` 内的数值（bool 不是合法标量）；
- `prune_n`、`prune_m`：非负整数；`prune_n > 0` 时要求 `0 < prune_n < prune_m`、`in_features % prune_m == 0` 且 `sparsity == 0.5`，否则抛 `ValueError`。

计算 `metric = |weight| * sqrt(mean_N(activations**2))`（内部 float64）。剪枝规则：

- `prune_n == 0`：每行按 metric **升序稳定排序**，剪前 `int(in_features * sparsity)` 个；
- `prune_n > 0`：每行在每个长度为 `prune_m` 的连续列块内，剪 metric 最小的 `prune_n` 个。

同分一律取更小的列下标。返回 4 键字典：

- `metric`：`(out, in)`，按输入 dtype；
- `mask`：`(out, in)` bool，**True 表示被剪**；
- `pruned_weight`：`(out, in)`，被剪位置为 0、其余保持原值，按输入 dtype；
- `sparsity`：python float，实际稀疏率 `mask.sum() / numel`。

禁止使用 `torch.nn.utils.prune`；不得就地修改入参。

## 示例

```
weight      = [[ 1.0, -5.0, 3.0, -2.0],
               [ 0.5,  9.0,-2.0, -8.0]]
activations = [[ 2.0,  1.0, 0.0,  1.0],
               [ 0.0,  3.0, 0.0,  0.0]]
sparsity    = 0.5            # k = int(4 * 0.5) = 2

逐列均方 = [2, 5, 0, 0.5]     逐列 RMS = [1.4142, 2.2361, 0, 0.7071]
metric   = [[1.4142, 11.1803, 0, 1.4142],
            [0.7071, 20.1246, 0, 5.6569]]
```

- 行 0：最小的是第 2 列（0），其次第 0 列与第 3 列**并列** `1.41421356…`，小下标 0 先剪 → `mask=[T,F,T,F]`；
- 行 1：最小的是第 2 列（0），其次第 0 列（0.7071）→ 同样 `[T,F,T,F]`。

```
pruned_weight = [[0.0, -5.0, 0.0, -2.0],
                 [0.0,  9.0, 0.0, -8.0]]
sparsity      = 0.5
```

注意第 2 列权重是 3 和 -2，看上去不算小，但它的输入激活恒为 0，RMS 为 0，所以两行都优先剪掉它；这正是 Wanda 与纯幅值剪枝的区别。

## 约束

- **激活按列聚合**：RMS 沿 token 维求均值再开方，长度等于 `in_features`，广播到每一行；不是按行聚合，也不是逐元素乘激活。
- **截断取整**：`int(in_features * sparsity)`，`7×0.5 → 3`。换成 `round` 会多剪一个，和上游不一致。
- **同分取小下标**：上游 stable sort 的确定性来自升序排序本身；本题把它钉成硬约定，测试用「全 1 权重 + 全 1 激活」造全并列。
- **N:M 约束**：`sparsity` 必须恰为 0.5；块内以 metric 为准，块与块之间不互相竞争。
- **只改掩码位置**：未剪元素保持原值与符号；`metric` 与 `pruned_weight` 返回输入 dtype，内部 float64。
