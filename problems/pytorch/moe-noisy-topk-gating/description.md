# 显式噪声 Top-k 门控

## 任务

实现 `solve`：在路由 logits 上加入**调用方给定的**高斯噪声，按加噪后的分数做 top-k，再只对选中分数做 softmax。返回稀疏门控矩阵、选中索引和加噪后的完整 logits。

这是 Shazeer 稀疏门控中 noisy top-k 的受限数值练习。噪声不从全局随机数生成器采样；训练/推理的差异只体现为是否使用噪声项。

## 输入

- `x`：CPU float64 `[T,D]`，每行是一个 token。
- `gate_w`：CPU float64 `[E,D]`；门控 logits 为 `x @ gate_w.T`。
- `noise_w`：CPU float64 `[E,D]`；噪声尺度为 `softplus(x @ noise_w.T)`。
- `noise`：CPU float64 `[T,E]`，已给定的标准正态噪声样本，绝对值不超过 4。
- `k`：Python `int`，`1 <= k <= E`。必须是真正的 `int`，`bool` 和 `2.0` 不合法，否则抛出 `ValueError`。
- `training`：Python `bool`，默认 `True`。为 `False` 时完全不使用 `noise` 和 `noise_w`。
- `T,D,E=1..8`；`x` 绝对值不超过 3，权重绝对值不超过 2。其余输入合法，允许非连续张量，不修改输入。

`softplus(z)=log(1+e^z)`，必须按此式理解；不要用 `|z|` 或 `relu(z)` 替代。

## 精确规则

门控 logits 与加噪分数：

$$
z=xW_g^{\mathsf T},\qquad
\tilde z=z+\epsilon\odot\operatorname{softplus}(xW_n^{\mathsf T})\quad(\text{training}).
$$

推理时 $\tilde z=z$。按 $\tilde z$ **降序**选前 k 个专家；完全相等时编号较小者优先。

$$
(\tilde z_{(1)},\ldots,\tilde z_{(k)})=\operatorname{topk}(\tilde z,k),\qquad
a_j=\frac{e^{\tilde z_{(j)}}}{\sum_{\ell=1}^{k}e^{\tilde z_{(\ell)}}}.
$$

门控矩阵 `gates` 形状 `[T,E]`：选中位置填 $a_j$，其余位置**严格为 0**。归一化只在 k 个选中项内进行，不包含未选专家。返回的 `indices` 按降序分数排列。

## 返回

返回三项 `tuple`：`(gates, indices, effective_logits)`。

- `gates`：float64 `[T,E]`，每行和为 1；
- `indices`：int64 `[T,k]`，降序选择顺序；
- `effective_logits`：float64 `[T,E]`，训练时为 $\tilde z$，推理时为 $z$。
- 浮点结果按 `rtol=2e-11, atol=2e-11` 比较。

## 示例

```python
x = [[1.0]]
gate_w = [[1.0], [0.9]]
noise_w = [[0.0], [0.0]]
noise = [[0.0, 1.0]]
k = 1
training = True
```

两个专家的噪声尺度都为 `softplus(0)=ln 2`，因此：

```python
indices = [[1]]
effective_logits = [[1.0, 0.9 + ln 2]]
gates = [[0.0, 1.0]]
```

噪声把原本落后的专家 1 推到前面。令 `noise=[[0.0,0.0]]` 则选专家 0，`gates=[[1.0,0.0]]`。令 `training=False` 时，即使把 `noise` 改成任意值，结果也与去噪时相同。
