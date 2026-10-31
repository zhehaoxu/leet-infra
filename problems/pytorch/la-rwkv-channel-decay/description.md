# RWKV v6 通道衰减：per-token w/u 的 WKV 递推账

## 题意

RWKV v6（x060）把 v4/v5 的固定 `time_decay` 换成了**逐 token、逐通道的数据依赖衰减**，并把 bonus（`time_faaaa`）也变成逐通道量。本题转写官方 `wkv6` CUDA kernel 的 CPU 账本版：给定 receptance `r`、key `k`、value `v`（均 `[T, C]`）、per-token 衰减 `w`（`[T, C]`）、per-channel bonus `u`（`[C]`）与可选初始状态 `(a, b, p)`（均 `[C]`），逐 token 递推每个通道的 log-space 状态三元组 `(aa, bb, pp)`，输出乘 receptance 前后的结果与终态。

## 求解

每个通道独立维护数值稳定的三元组（分子 `aa`、分母 `bb`、指数上界 `pp`），全程 float32：

**输出分支**（bonus 与旧指数比）：

```
p_out = max(pp, u)
e1 = exp(pp - p_out);  e2 = exp(u - p_out)
wkv = (e1·aa + e2·v) / (e1·bb + e2)
```

**更新分支**（衰减加在**旧指数** pp 上，再与当前 k 比）：

```
ww = pp + w
p   = max(ww, k)
e1  = exp(ww - p);  e2 = exp(k - p)
aa  = e1·aa + e2·v
bb  = e1·bb + e2
pp  = p
```

最终输出 `y = r * wkv`（receptance 原样乘回，本题不做 sigmoid——那是模型侧的门控）。

关键结构点：

1. **w 作用在旧指数上**：`ww = pp + w` 中 `pp` 是本 token 输出分支所用的旧指数，而不是输出分支产生的 `p_out`。这是 v6 相对常见复现错误的最大坑。
2. **bonus 只进输出分支**：`u` 参与的是「读旧状态 + 当前 v」的插值；更新分支完全不含 `u`。
3. **初始状态**：缺省 `aa = bb = 0`、`pp = -1e38`（官方 kernel 的 -inf 代理常量）。零状态下 t=0 的 `wkv = v` 恒成立；且 `pp + w` 对 `-1e38` 数值吸收，p_trace[0] = k0。
4. **通道独立**：C 个通道互不影响，可逐通道并行；`p_trace` 逐 token 记录指数上界，便于观察衰减算术。
5. 分段喂入与整段一次喂入结果一致（递推因果性），终态可作为断点续算的状态。

## 数据规模

- 1 ≤ T ≤ 32，1 ≤ C ≤ 16；张量均为 float32。
- 输入不符合形状、空序列、state 键不齐 → `ValueError`。
