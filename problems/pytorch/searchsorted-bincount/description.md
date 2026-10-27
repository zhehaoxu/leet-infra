# searchsorted 分桶与 bincount 直方图

## 题目描述

「把连续值落到离散桶里，再数每个桶几个」是直方图化的一对操作：先对每个元素查桶（`searchsorted`），再对桶下标计数（`bincount`）。它是反 CDF 采样（给定均匀随机数 \(u\)，在累积分布上找第一个 \(\ge u\) 的位置，见 `top-p-sampling`、`rl-per-priority-replay`）、MoE 专家负载计数（`moe-switch-aux-loss`）、码本最近邻编码（`quant-nf4-quantile-grid` 的 16 段分位码本）里共同的底层原语。

本题把两步串成一个函数，难点全在**边界的闭合方向**与**桶数固定**这两件事上：

- `right=False` 时桶下标是「严格小于 \(x\) 的 boundary 个数」，即最小 \(i\) 使 \(\text{boundaries}[i] \ge x\)；
- `right=True` 时是「小于等于 \(x\) 的 boundary 个数」，即最小 \(i\) 使 \(\text{boundaries}[i] > x\)。

两者只在 \(x\) **恰好等于**某个 boundary（或 boundary 有重复值）时可分辨——而题面示例刻意同时包含这两种情形。重复 boundary 会让某些桶恒为空，但 `counts` 的长度**必须**是 \(M+1\)：空桶也要占一位，这是与裸 `torch.bincount`（会截掉尾部空桶）最实质的差异。

CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。`boundaries` 是已按升序排列的一维张量（允许重复值、允许 \(M=0\)），`x` 任意形状、与 `boundaries` 同 dtype（int64 或 float32/float64 均可），`right` 默认 False，`weights` 默认 None。
- **查桶**：返回与 `x` 同形状的 int64 张量 `idx`，逐元素取值 \(i\in[0,M]\)，定义即上面两条计数口径。等价的区间读法：`right=False` 时桶 \(i\) 收 \((\text{boundaries}[i-1],\ \text{boundaries}[i]]\)（右闭），`right=True` 时收 \([\text{boundaries}[i-1],\ \text{boundaries}[i])\)（左闭）；starter docstring 里「左闭右开」一句是 `right=True` 的读法，`right=False` 时闭合方向翻转，桶 0/桶 \(M\) 两个无界外侧段也同步跟着翻（\((-\infty,\text{boundaries}[0]]\) 与 \((\text{boundaries}[M-1],+\infty)\)）。boundary 有重复时中间会出现恒空桶，一律以 `idx` 的计数定义为准。
- **计数**：`counts` 是长度**恰为 \(M+1\)** 的一维张量，`counts[i]` = `idx` 中等于 \(i\) 的元素个数（不给 `weights`）或这些元素的权重之和（给 `weights`）。`idx` 先展平再计数，因此多维 `x` 只影响 `idx` 的形状。
- **dtype**：`idx` 恒 int64；不给权重时 `counts` 恒 int64；给权重时 `counts` 为浮点且 dtype 跟随 `weights`（float32 → float32、float64 → float64）。判题只投浮点 `weights`；顺带一提原生 `torch.bincount` 对整型 `weights` 会静默升到 float64，与本题「dtype 跟随 weights」的约定不同，别照抄。
- **恒等式**（判题用作探针）：未加权时 `counts.sum() == x.numel()`；加权时 `counts.sum() == weights.sum()`。
- **空输入**：`x` 为空张量时 `idx` 为空、`counts` 是长度 \(M+1\) 的全 0；`boundaries` 为空（\(M=0\)）时所有元素都落进唯一的桶 0，`idx` 全 0、`counts` 长度 1。两者都**不**抛错。
- **错误约定**：`boundaries` 不是一维抛 RuntimeError；`weights` 形状与 `x` 不符抛 ValueError。`boundaries` 保证升序，实现无需处理乱序输入。

## 示例

`boundaries = [1.0, 3.0, 3.0, 7.0]`（\(M=4\)，`counts` 长度 5）、`x = [0.0, 3.0, 5.0, 9.0]`：

**right=False**（默认）：

| x | 严格小于它的 boundary | idx |
|---|---|---|
| 0.0 | 无 | 0 |
| 3.0 | `[1]` | 1 |
| 5.0 | `[1,3,3]` | 3 |
| 9.0 | 全部 4 个 | 4 |

`idx = [0, 1, 3, 4]`，`counts = [1, 1, 0, 1, 1]`（桶 2 的区间在 `right=False` 下退化为空集 \((3,3]\)，恒为 0）。

**right=True**：`x=3.0` 时小于等于它的有 `[1,3,3]` 三个，其余元素不变，于是 `idx = [0, 3, 3, 4]`、`counts = [1, 0, 0, 2, 1]`。

两支持别只在 `x=3.0` 这一个元素上（`x=5.0`、`9.0`、`0.0` 都远离重复段），它从桶 1（\((1,3]\)）挪到桶 3（\([3,7)\)）；而由重复段 \([3,3]\) 造成的**恒空桶**始终是下标 2，两种设置下 `counts[2]` 都为 0，`counts` 长度也都仍是 5。这正是本题两条核心口径的最小判别式。

加权口径示例：`boundaries=[1,3,7]`、`x=[0,2,2,5,9]`、`weights=[1,2,3,4,5]`，未加权 `idx=[0,1,1,2,3]`、`counts=[1,2,1,1]`；加权 `counts=[1.0, 5.0, 4.0, 5.0]`（桶 1 收 \(2{+}3\)，桶 2 只收 4）。

## 约束

- 返回二元组 `(idx, counts)`：`idx` 与 `x` 同形状、int64；`counts` 一维、长度 \(M+1\)、int64 或跟随 `weights` 的浮点 dtype。
- 判题以 `torch.equal`（下标与整型计数）和 \(10^{-6}\)（浮点计数）为准，实现路径不限：二分查找、把 `boundaries` 与 `x` 广播比较后求和、或 `index_add_` 累加计数都可以。
- 不使用随机数；对未触及的桶必须输出显式的 0 而不是省略。
