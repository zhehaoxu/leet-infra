# 解题说明 · Wanda 激活加权幅值剪枝

## 一句话

重要性 \(=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}\)，每个输出行独立剪最小的 `int(in·s)` 个；N:M 时块内竞争。

## 为什么乘激活 RMS

把一层输出写成 \(y_i=\sum_j W_{ij}x_j\)。删掉一项 \(W_{ij}x_j\) 带来的扰动量级是 \(|W_{ij}|\cdot|x_j|\)，所以「重要不重要」天然是权重与**该输入通道典型幅度**的乘积。校准集上 \(x_j\) 的典型幅度用 RMS 表达，就得到上游的 `W_metric`。这解释了为什么「权重 3 但激活恒 0」的列会被优先剪：它对任何输入都没有贡献。上游代码里 `scaler_row` 累积的是平方和均值（未开方），`W_metric` 里再统一开方；本题直接给「均方 → 开方」。

## 三个口径

1. **k 截断**。`int(7*0.5)=3`。上游就是 `int(shape[1]*sparsity)`，这里不能顺手用 `round`——已有题 `sparsity-rowwise-prune` 用的是 `round`，两题正好形成对照。
2. **同分取小下标**。上游 `stable=True` 的升序排序在并列时保持原列序，取前 k 个即小下标先剪。本题把它写成显式约定，并用全 1 输入验证。
3. **N:M 只认 0.5**。`prune_n>0` 时按 `prune_m` 列分块，块内取最小的 `prune_n` 个；`main.py` 对 `2:4`/`4:8` 断言 `sparsity_ratio == 0.5`。本题额外要求 `in_features % prune_m == 0`，因为上游对尾部残缺块直接 `topk(prune_n)`，块宽不足时会退化。

## 判分点

- `metric` 是**逐列**广播，不是逐行；形状与 `weight` 相同；
- `mask` 的语义是 True=剪，和上游 `W_mask` 一致；
- `pruned_weight` 只把被剪位置置 0，保留元素连符号一起不动；
- 实际稀疏率由掩码真实统计，N:M 模式下它恒为 0.5（结构化）或恰为 `int(in·s)/in`（非结构化）；
- 输入不被就地修改。

## 与相邻题目的边界

- `sparsity-magnitude-prune`：全局阈值、只看 \(|W|\)；本题逐行、乘激活 RMS。
- `sparsity-rowwise-prune`：逐行但只看 \(|W|\)，且用 `round`；本题多激活项，且用截断。
- `sparsity-nm-2-4`：只考 2:4 的块内取大；本题的 N:M 分支以 Wanda metric 排序，并额外支持 4:8。
- `sparsity-sparsegpt-reconstruct`：用 Hessian 逆把误差分摊到未剪列；Wanda 完全不做补偿，剪完就结束。

## 出处

`locuslab/wanda`（arXiv 2306.11695）：

- `lib/prune.py::prune_wanda`：`W_metric = torch.abs(weight) * torch.sqrt(scaler_row.reshape((1,-1)))`；`sort(..., stable=True)`；`int(W_metric.shape[1] * args.sparsity_ratio)`；N:M 分支 `topk(tmp, prune_n, dim=1, largest=False)`；
- `lib/layerwrapper.py::WrappedGPT.add_batch`：`scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / nsamples`；
- `main.py`：`assert sparsity_ratio == 0.5`（结构化模式）。

**本题自己的补充**：并行度与显式入参（直接给 `activations`，不跑模型）；同分取小下标写成硬约定；`in_features % prune_m` 校验；`sparsity` 的实际值由掩码统计返回。
