# 详解：logsumexp（减最大值稳定式与 ±inf 退化段）

## 出处

「减去最大值再取指数」是 log-sum-exp 技巧的标准形式，`scipy.special.logsumexp` 就是一个独立实现的原语；PyTorch 把它内联在 `F.log_softmax` 与 `F.binary_cross_entropy_with_logits`（后者用的是同族的 `log1p(exp(-|x|))` 写法）里。FlashAttention 的 tile 累加显式保存 LSE（$\log\sum e$）作为运行 softmax 的分母，因此「给定 O/LSE/dO 反推梯度」这类题（`flash-attention-tile` 的反向侧）都要求 LSE 数值与整段 logsumexp 一致——本题就是那个原语本身。

## 稳定式与非有限分支

设段内 $m=\max_i x_i$。恒等式

$$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$$

成立的条件是 $m$ 有限。$x_i - m \le 0$ 保证 $e^{x_i-m}\in(0,1]$，求和落在 $[1, n]$，取 log 落在 $[0,\log n]$，全段不溢不欠。三个非有限情形必须各自单独取值：

| 段内情况 | $m$ | 朴素稳定式 | 正确结果 |
| --- | --- | --- | --- |
| 存在 $+\infty$ | $+\infty$ | $+\infty - (+\infty) = \mathrm{NaN}$ | $+\infty$ |
| 全为 $-\infty$ | $-\infty$ | $-\infty - (-\infty) = \mathrm{NaN}$ | $-\infty$ |
| 含 NaN | NaN | NaN | NaN（传播） |

参考答案把四步压成三次 `where`，不做 Python 层的段级 if：

```
m  = xd.amax(dim=axes, keepdim=True)
fm = torch.isfinite(m)                       # 只有 m 有限才走稳定式
sm = torch.where(fm, m, torch.zeros_like(m)) # 非有限段用 0 占位，避免 inf-inf
s  = (xd - sm).exp().sum(dim=axes, keepdim=True)
out = torch.where(fm, sm + s.log(), m)       # 直接取 m（±inf 或 NaN）
```

关键在最后一步 `torch.where(fm, ..., m)`：`m` 本身就是正确答案（$+\infty$、$-\infty$、NaN 三种都对），因此不需要为每种退化各写一支。占位值取 0 也非随意——它让 `xd - sm` 在非有限段保持有限或 $\pm\infty$，那些段的 `s` 之后会被整支丢弃。

**为什么下溢也要管**：$x\sim-10^3$ 时 $e^{x}$ 在 float32/float64 下都是 0，朴素式得 $\log 0 = -\infty$；稳定式把最大项抬到 $e^0=1$，其余项即使仍下溢为 0 也只丢掉「可忽略的贡献」，主项 $m$ 保住。判题用 $[1000,1002,999]$ 与 $[-1000,-1002,-999]$ 两段同时验证，并先断言朴素式在这两段确实非有限（防止实现「抄对了但没起作用」）。

## 核心与易错点

1. **`dim=None` 是扩展口径**：`torch.logsumexp` 无此写法，判题用 `logsumexp(x.reshape(-1), 0)` 作参照。
2. **`keepdim=True` + `dim=None`** 要求全轴保留为 1（`(1,1)`），实现里等价于把 axes 取成 `tuple(range(x.dim()))` 后走同一条 `squeeze` 判定。
3. **`amax` 后再 `.squeeze(dim=axes)`**：axes 必须排序去重，否则 `squeeze` 报错或压错轴。
4. **float64 中间量**：`exp` 与 `log` 在 float32 下相对误差会放大，$m$ 很大时（$10^3$ 段）容差只有 1e-4 可容，直接 float32 计算容易压线。
5. **别用 `logsumexp` 实现 `log_softmax`**：本题口径只交标量归一化项；`log-softmax` 题逐元素返回 $x - \text{LSE}$，两者由 `check_log_softmax_consistency` 互相验证。

## 与邻近题的区别

- `log-softmax` / `softmax`：前向归一化输出，本题只交分母那一项。
- `cross-entropy-loss` / `bce-with-logits`：内部用到 logsumexp（或等价的 softplus 稳定式），考的是损失与 reduction 口径。
- `flash-attention-tile`：在线 softmax 的 LSE 累加与分块重算，本题是单段一次性归约。
- `jsd-loss` / `kl-divergence`：散度类，输入已是概率/对数概率。

## 对拍口径与复杂度

- oracle 是 `torch.logsumexp`（`dim=None` 用展平写法作参照），覆盖 float32 随机、×100 放大、三维输入、`keepdim` 两态、±inf/NaN 退化段与 5 个非法 `dim`。
- 复杂度 $O(N)$，两次归约（`amax`、`sum`）+ 一次逐元素 `exp`；判题规模毫秒级。
