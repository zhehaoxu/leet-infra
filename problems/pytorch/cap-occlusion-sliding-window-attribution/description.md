# Captum Occlusion 滑动窗口扰动归因账

## 背景

`captum.attr.Occlusion` 是一种**黑盒扰动归因**（perturbation-based attribution）方法：不像 Integrated Gradients 那样沿路径积分，也不像 Layer-wise Relevance Propagation 那样按规则逐层传播，而是把输入切成一块块「滑动窗口」，逐个把窗口内的特征替换成 baseline，再看模型输出的变化量。

它的核心账本由四部分组成：

1. **网格枚举**：窗口在每一维上平移多少次、总共枚举多少个窗口、每个窗口的偏移量是多少。
2. **窗口掩码与覆盖计数**：每个窗口实际覆盖了哪些特征位置、每个特征位置被多少个窗口覆盖。
3. **差分账**：原始输出、每个窗口遮挡后的输出、以及两者的差。
4. **归因账**：把每个窗口的差累加到它覆盖的位置上，再除以覆盖次数得到最终归因。

本题要求你用纯 Python 重建这一整套账本，并处理官方实现里若干**不看源码就会写错**的细节。

## 口径约定

- 所有输入都是**精确有理数**（内部按 `fractions.Fraction` 计算），只在最终输出时转成 `float`。
- 前向模型是一个显式 MLP：`layers` 是层列表，每层是 `{"W": [[...]], "b": [...]}`。`W` 的行数 = 输出维，列数 = 输入维。**层与层之间加 ReLU，最后一层不加 ReLU**。
- 输入 `x` 可以是嵌套列表（按 `shape` 的形状），内部展平成一维后参与计算。
- `shape` 是**非 batch 维**的形状，每个维度必须为正整数。
- `window_shape`（滑动窗口形状）必须与 `shape` **同维数**，每维为正整数，且**不得大于**对应的输入维。
- `strides` 可以是：
  - `None` → 所有维度的 stride 视为 `1`；
  - 一个**标量整数** → 该值适用于所有维度；
  - 一个与 `shape` **同维数**的列表，每维一个正整数。
- `strides` 的约束（与官方一致）：**某维的 stride 不得大于该维的窗口长度**，除非该维的窗口恰好覆盖整个输入维（即 `shape[d] == window_shape[d]`）。
- `baseline` 可以是一个标量（广播到所有位置），或一个与 `shape` 同形 / 与展平后等长的嵌套或一维列表。
- `target` 是输出维的索引，采用 **Python/PyTorch 负索引语义**：`target=-1` 选最后一个输出，`target=-n` 选第一个；越界（`target >= n` 或 `target < -n`）报错。

## mode 1：`occlusion_grid` —— 网格枚举账

```python
solve("occlusion_grid", shape=[5], window_shape=[3], strides=[1])
```

出参：

```json
{
  "shift_counts": [3],
  "num_windows": 3,
  "offsets":  [[0], [1], [2]],
  "extents":   [[[0, 3]], [[1, 4]], [[2, 5]]]
}
```

- `shift_counts[d]`：第 `d` 维的平移次数，等于 `ceil((shape[d] - window_shape[d]) / strides[d]) + 1`。
- `num_windows`：所有维平移次数的乘积。
- `offsets[k]`：第 `k` 个窗口在每一维上的起始偏移（**每一维是一个列表**）。
- `extents[k]`：第 `k` 个窗口在每一维上的**实际覆盖区间** `[start, stop)`（**嵌套两层**：先按维，再按区间端点）。

### ★ 枚举顺序：base-conversion，dim0 是最低位

官方 `_occlusion_mask` 的枚举写法是：

```python
for idx in range(num_windows):
    remaining = idx
    current_index = []
    for i, shift_count in enumerate(shift_counts):
        current_index.append((remaining % shift_count) * strides[i])
        remaining = remaining // shift_count
```

也就是说 **第 0 维变化最快**（它是最低位）。以 `shape=[3,4]`、`window_shape=[2,2]`、`strides=[1,2]` 为例，`shift_counts = [2, 2]`，枚举顺序是：

```
(0,0) → (2,0) → (0,2) → (2,2)
```

而不是行优先的 `(0,0) → (0,2) → (2,0) → (2,2)`。

### ★ 窗口截断：末窗会被裁掉

某维的窗口偏移为 `start`、窗口长度为 `w` 时，若 `start + w > shape[d]`，官方用 `torch.nn.functional.pad` 的**负 padding** 实现截断 —— 右侧 padding 为负数时 `F.pad` 会把窗口尾部裁掉。等价于把覆盖区间的右端压到 `shape[d]`。

以 `shape=[6]`、`window_shape=[4]`、`strides=[3]` 为例：

- `shift_counts = ceil((6-4)/3)+1 = 1+1 = 2`
- 窗口 0：偏移 0 → 覆盖 `[0, 4)`
- 窗口 1：偏移 3 → `3+4 = 7 > 6` → 截断为 `[3, 6)`

## mode 2：`occlusion_mask` —— 掩码与覆盖计数账

```python
solve("occlusion_mask", shape=[3, 4], window_shape=[2, 2], strides=[1, 2])
```

出参：

```json
{
  "masks":   [[0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], ...],
  "coverage": [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]
}
```

- `masks[k]`：第 `k` 个窗口的 0/1 掩码，**按行优先（row-major）展平**，长度 `prod(shape)`。
- `coverage[i]`：第 `i` 个特征位置被多少个窗口覆盖（**窗口数**，不是覆盖次数的总和）。

多维掩码的展平顺序是行优先：最后一维变化最快。

## mode 3：`occlusion_diff` —— 差分账

```python
solve("occlusion_diff",
      layers=[{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [0.5]}],
      x=[1.0, 2.0, 3.0, 4.0, 5.0],
      baseline=0.0,
      shape=[5], window_shape=[3], strides=[1], target=0)
```

出参：

```json
{
  "initial_eval": 55.5,
  "window_evals": [41.5, 26.5, 5.5],
  "eval_diffs":   [14.0, 29.0, 50.0],
  "total_attrib": [14.0, 43.0, 93.0, 79.0, 50.0]
}
```

- `initial_eval`：原始输入的前向输出（取 `target` 那一维）。
- `window_evals[k]`：把第 `k` 个窗口内的特征替换成 baseline 之后的前向输出。
- `eval_diffs[k]`：`initial_eval - window_evals[k]`。
- `total_attrib[i]`：**未归一化**的累加 —— 位置 `i` 被所有覆盖它的窗口的 `eval_diffs` 加起来。

遮挡的实现是逐位置的：`ablated[i] = x[i] * (1 - mask[i]) + baseline[i] * mask[i]`，即**只有被窗口覆盖的位置**才替换成 baseline。

## mode 4：`occlusion_attribution` —— 完整归因账

```python
solve("occlusion_attribution",
      layers=[{"W": [[1.0, 2.0, 3.0, 4.0, 5.0]], "b": [0.5]}],
      x=[1.0, 2.0, 3.0, 4.0, 5.0],
      baseline=0.0,
      shape=[5], window_shape=[3], strides=[1], target=0)
```

出参：

```json
{
  "total_attrib": [14.0, 43.0, 93.0, 79.0, 50.0],
  "weights":     [1, 2, 3, 2, 1],
  "attribution": [14.0, 21.5, 31.0, 39.5, 50.0]
}
```

### ★★ 重叠区归因取平均（本题最核心的判别点）

`attribution[i] = total_attrib[i] / weights[i]`，其中 `weights[i]` 是位置 `i` 被覆盖的**窗口数**。

这与 `FeatureAblation`（`use_weights=False`）**行为不同**：后者把每个窗口的差直接累加，不除以覆盖次数。

以上面的例子为例，位置 2 被 3 个窗口覆盖（`weights[2] = 3`），累加值 `93.0`，归因 `93.0 / 3 = 31.0`。

### ★ float32 舍入

官方的 `format_result` 最后一步是：

```python
single_attrib = single_attrib.float() / weight
```

也就是**先把 float64 的累加值降级到 float32，再在 float32 下做除法**。本题的输出必须复现这个舍入：当 `total_attrib[i] / weights[i]` 不是二进制有限小数时（例如 `1/3`），期望值是 float32 舍入后的结果 `0.3333333432674407958984375`，而不是 float64 的 `0.3333333333333333`。

## 常见错误清单

1. **枚举顺序写成行优先**。官方是 base-conversion，dim0 是最低位。
2. **忘了末窗截断**。`F.pad` 的负 padding 会把窗口尾部裁掉，`extents` 的右端要压到 `shape[d]`。
3. **重叠区没有取平均**（`attribution` 直接等于 `total_attrib`），或者除以了窗口总数而不是覆盖该位置的窗口数。
4. **归因结果直接输出 float64 除法结果**，没有复现官方的 float32 降级。
5. **`diff` 的符号搞反**。`eval_diffs[k] = initial_eval - window_evals[k]`，遮挡会让输出**下降**时差为正。
6. **baseline 应用到了窗口之外的位置**。只有被覆盖的位置才替换。
7. **ReLU 加在了最后一层**。官方等价的前向是层间加 ReLU、最后一层不加。
8. **忘了 stride 的约束**：某维窗口未恰好覆盖该维时，stride 不得大于窗口长度。
9. **标量 `strides` 没有广播到所有维度**（1-D 下看不出来，2-D 才会暴露）。
10. **负 `target` 直接报错**。官方走 PyTorch 的负索引语义，`-1` 是最后一个输出。
11. **多维掩码用列优先展平**。掩码是行优先展平。
12. **`shift_counts` 写成 `ceil(...)` 而忘了 `+1`**。

## 自检

```python
solve("occlusion_grid", shape=[6], window_shape=[4], strides=[3])
```

应得到 `shift_counts = [2]`、`num_windows = 2`、`offsets = [[0], [3]]`、`extents = [[[0, 4]], [[3, 6]]]`。

```python
solve("occlusion_mask", shape=[5], window_shape=[3], strides=[1])
```

应得到 3 个掩码，覆盖计数为 `[1, 2, 3, 2, 1]`。
