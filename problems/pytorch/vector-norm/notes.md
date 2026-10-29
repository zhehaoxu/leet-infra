# 详解：向量范数 vector_norm（ord 全族与多轴归约）

## 出处

`torch.linalg.vector_norm`（1.9 起的新接口）取代了旧的 `torch.norm(x, p, dim)`，但旧接口那套「ord 随维度解释」的历史包袱被完整继承下来：同一个 `ord=1` 在向量上是绝对值和，在矩阵上是最大列和。面试里问「`torch.norm` 的 `p=0` 和 `p=-1` 分别是什么」，答错的比例非常高。工程侧的直接使用点：`torch.nn.utils.clip_grad_norm_` 的范数口径、KV cache 里按通道 L2 打分（`kv-knorm-importance`）、以及各类 normalization 的分母。

## 口径全表与稳定式推导

记段内绝对值向量为 $a_i = |x_i|$，$m=\max_i a_i$。

| ord | 定义 | 退化行为 |
| --- | --- | --- |
| `None` / 2 | $(\sum a_i^2)^{1/2}$ | 全零 → 0 |
| 1 | $\sum a_i$ | 全零 → 0 |
| -1 | $(\sum a_i^{-1})^{-1}$ | 含 0 → 0 |
| 0 | $\#\{a_i \ne 0\}$ | 全零 → 0 |
| `inf` | $\max a_i$ | 全零 → 0 |
| `-inf` | $\min a_i$ | 含 0 → 0 |
| 有限 $p\notin\{0,\pm1,2\}$ | $(\sum a_i^p)^{1/p}$ | 全零 → 0 |

$-1$ 那一行是本题最反直觉处：它跟 1-范数不是一对，而是幂次公式在 $p=-1$ 处的取值，$\big(\sum 1/a_i\big)^{-1}$ 恰是调和平均的 $n$ 倍缩放；一旦出现 $a_i=0$，$1/a_i=+\infty$，倒数即 0。参考答案不需要为它写特例：稳定式里 $(a_i/m)^{-1}=+\infty$，$\infty^{1/p}=0$（$p<0$），乘上 $m$ 仍是 0，与 torch 逐点一致。

提最大值的恒等式对任意非零 $p$ 都成立：

$$\Big(\sum_i a_i^p\Big)^{1/p} = \Big(m^p \sum_i (a_i/m)^p\Big)^{1/p} = m\Big(\sum_i (a_i/m)^p\Big)^{1/p}$$

因为 $a_i/m \in [0,1]$，$(a_i/m)^p$ 在 $p>0$ 时不溢出、在 $p<0$ 时至多是 $+\infty$，永远不会出现 `inf − inf`。全零段必须单独挡掉：$m=0$ 时 $a_i/m$ 是 $0/0$，所以参考答案用 `torch.where(m > 0, ...)` 把除数换成 1、把被除数换成 0，最后乘 $m=0$ 得 0。

## 核心与易错点

1. **`ord=0` 不能套幂次公式**：$(\sum a^0)^{1/0}$ 在数学上无意义，它数的是非零个数，必须走 `(absx != 0).sum()` 这条分支。
2. **`"inf"` 字符串与 `float("inf")` 要同待**：`torch.linalg.vector_norm` 只接 Number，本题接口额外接受字符串，判题两种写法都会试。
3. **多轴归约后的形状**：`dim=(0,2)` 时输出把这两根轴都消掉，`keepdim=True` 则都留成长度 1，位置仍在原轴上——所以 `keepdim` 的实现只能是 `squeeze(dim=axes)`，不能用 `reshape` 猜顺序。
4. **`dim=None, keepdim=True` 是本题扩展**：torch 在这种情况下根本不看 `keepdim`（返回 0 维），本题要求返回形状全 1 的张量，判题单独断言 `(1,1,1)`。
5. **dtype 必须回投 float32**：中间全 float64，若忘记 `.to(x.dtype)`，`got.dtype == x.dtype` 断言会挂。

## 与邻近题的区别

- `matrix-norm`（本批 order359）：`1/-1/inf` 在矩阵上是「先求和再取极值」，本题是「直接求和」，两题各考一侧，专门用来逼出口径差异。
- `lp-pool-unpool`：那是沿指定轴做 $L^p$ 池化 $(\frac1n\sum|x|^p)^{1/p}$（带 $1/n$ 均值因子），本题没有归一化因子。
- `distributed-clip-grad-norm`：算全局梯度范数做裁剪的**账**，本题算范数**本身**。
- `rmsnorm` / `layernorm-forward`：用到了 2-范数/RMS，但考的是归一化仿射变换。

## 对拍口径与复杂度

- oracle 直接是 `torch.linalg.vector_norm`，覆盖 11 个 `ord` × 4 个 `dim` 的笛卡儿积，另含 `ord=-1` 调和口径、多轴/keepdim 形状、$10^3$ 与 $10^{-8}$ 量级的溢出探针、全零段与 6 个非法参数组合。
- 复杂度 $O(N)$（$N$ 为参与归约的元素数），float64 下多一次除法与幂运算，判题规模毫秒级。
