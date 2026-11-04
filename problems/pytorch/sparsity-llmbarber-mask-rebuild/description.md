# LLM-Barber 稀疏掩码重建

## 题意

一次性剪枝（one-shot pruning）后的稀疏模型精度掉了，LLM-Barber 的补救思路是：**别急着回稠密再重剪，而是在现有稀疏模式上做「交换」**——把当前 mask 中被剪掉但很重要的权重恢复（growth），同时把存活但很不重要的权重剪掉（prune），在保持稀疏度不变的前提下重建 mask。

给你三组输入：

- `weights`：稠密参考权重（dict[str, 2 维张量]，多个矩阵如 attention 的 q/o、MLP 的 fc1/fc2）；
- `ref_weights`：已剪枝的稀疏权重（**0 的位置就是当前 mask 的被剪位**）；
- `scalers`：每个权重的 grad² Fisher 滑动平均量（上游通过 block 前向 MSE 反传采集，本题已显式给出，不在判题域）。

## 你要实现的三步

### 1. 度量（metric 三选）

先用 `combined` 权重算重要性度量：

- `retain_reconstruction=True` 时 `combined = where(mask, weights, ref_weights)`（被剪位回填稠密值，存活位保留重构后的稀疏值）；
- `retain_reconstruction=False` 时 `combined = weights.clone()`（整体取稠密权重）。

| metric | 公式 | 含义 |
|--------|------|------|
| `WD` | \|combined\| · √scaler | 幅度 × 重要性 |
| `W` | \|combined\| | 纯幅度 |
| `D` | √scaler | 纯 Fisher 重要性 |

### 2. find_threshold 双阈值

把被剪位的 metric 叫 growth 集合、存活位的叫 prune 集合：

- growth 降序排序、prune 升序排序，取前 `min_len = min(两侧元素数)` 项**按位配对**；
- `recons_num = Σ(growth[i] − prune[i] > 0)`（严格大于，差恰为 0 不计数）；
- `idx = int(recons_num × threshold)`；
- `growth_threshold = growth_sort[idx]`，若 `idx == growth 元素数` 越界则取 `growth_sort[-1]`（末元素）；
- `prune_threshold = prune_sort[idx]`，若 `idx == prune 元素数` 越界则取 `prune_sort[-1]`；
- 任一侧为空时返回 `(+inf, −inf)`——此时两个比较都不会命中，mask 保持不动。

直觉：recons_num 是「交换后收益为正的配对数」，threshold 控制交换多少比例。

### 3. 交换与重建

按 `granularity` 确定 threshold 的作用域：

- `block`：所有矩阵的 growth/prune 集合**拼接**后算一对全局阈值；
- `layer`：逐矩阵一对阈值；
- `output1`：逐输出行一对阈值；
- `input1`：逐输入列一对阈值。

然后对每个矩阵（block 粒度用全局阈值）：

- `growth_indices = metric > growth_threshold`（严格大于）；
- `prune_indices = metric < prune_threshold`（严格小于）；
- `mask[growth_indices] = False`（恢复）→ `mask[prune_indices] = True`（剪掉）；
- `rebuilt = where(mask, 0, combined)`，返回整体稀疏度 `Σmask / 元素数`。

## 输入约束

- `weights` / `ref_weights` / `scalers` 均为非空 dict，键一致且为非空字符串，值为形状一致的 2 维浮点张量；
- `scalers` 非负（Fisher 量是 grad² 的滑动平均）；
- `metric ∈ {WD, W, D}`；`granularity ∈ {block, layer, output1, input1}`；
- `threshold` 是 [0, 1] 内的有限数值（bool 不算数）。

返回 `dict(masks={name: bool 张量}, rebuilt={name: 权重}, sparsity=float, thresholds={name: (growth, prune)})`。

## 提示

- 交换的顺序有讲究：先恢复后剪掉，重叠位置（同时命中两个集合）最终是被剪掉——升序/降序配对下 growth 高位与 prune 低位不会同时命中同一位置，但语义上要先写恢复再写剪除。
- 官方 N:M 结构化分支（逐行 find_threshold + 块内交错 topk）不在本题范围。
- Fisher 采集（block 前向 MSE 反传的 grad² 滑动平均）不在判题域，scaler 直接作为输入。
