# Mamba selective scan：ZOH 离散递推与门控跳连账

## 题目描述

Mamba（S6）的核心是把状态空间模型的 A/B/C 全部变成输入依赖（selective），离散化后每步只是一个逐元素线性递推。官方 `selective_scan_ref` 是这条递推的权威参考实现。本题要求把它的算术账逐项转写：ZOH 指数、`ΔB·u` 聚合、B/C 的静态与变体两分支、D 跳连、z 门控与 `delta_bias + softplus` 预处理。

## 规则

实现 `solve(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)`，转写 `selective_scan_ref`（实数 A），返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `out` | `[B, D, L]` | 与 `u` 同 dtype |
| `last_state` | `[B, D, N]` | 终态 `x` |

递推（全部 `.float()` 化后计算，复数 A 抛 `ValueError`）：

```
deltaA   = exp(delta ⊗ A)                      # 'bdl,dn->bdln'
deltaB_u = delta ⊗ B ⊗ u                       # 静态 B 'bdl,dn,bdl->bdln'
                                               # 变体 B 'bdl,bnl,bdl->bdln'
x = 0
for i in range(L):
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
    y_i = x·C  (静态 'bdn,dn->bd' / 变体 'bdn,bn->bd' 取 C[:, :, i])
out = y；D 非 None 时 out += u * D[:, None]
z 非 None 时 out = out * silu(z)
```

`delta` 预处理顺序：先加 `delta_bias` 再 `softplus`。`B`/`C` 各自独立支持静态 `[D,N]` 与变体 `[B,N,L]` 两分支。

## 示例

```
D=1, N=1, L=2, A=-1, B=C=1, delta=[0, ln2], u=[1,2]
步1: deltaA=1, ΔBu=0 → x=0, y=0
步2: deltaA=exp(-ln2)=1/2, ΔBu=ln2·1·2 → x=2ln2, y=2ln2
out=[0, 2ln2]，last_state=2ln2
```

## 约束

- `A` 负实数（`exp` 稳定）；`D/N/L ≤ 3×3×6` 级小规模，oracle 纯 Python 五重循环
- 分块扫描的边界 carry（把 L 切段、段间传终态）与整条递推数学恒等
