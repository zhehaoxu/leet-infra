# Gated DeltaNet 递推账：门控衰减与 delta rule 状态更新

## 题目描述

Gated DeltaNet 把线性注意力的状态更新改成「带门控衰减的 delta rule」：状态矩阵每步先乘 `exp(g_t)` 衰减，再用 `β_t` 把「v 减去状态对 k 的响应」写入外积。这是 2412.06464 的核心递推，fla-org 官方仓的 `naive_recurrent_gated_delta_rule` 是其逐 token 参考实现。本题要求逐行转写这条递推并让每个旋钮（`g`/`β`/`scale`/初态）的行为可验证。

## 规则

实现 `solve(q, k, v, beta, g, scale=None, initial_state=None)`（输入 `[B,T,H,·]`），返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `o` | `[B,T,H,V]` | 逐步输出（float32） |
| `final_state` | `[B,H,K,V]` | 终态 |

递推（先 `transpose(1,2)` 与 float32 化；`scale` 缺省 `1/sqrt(K)` 且**先乘 q**）：

```
h = initial_state（缺省 0）
for i in range(T):
    h = h * exp(g_i)                                # 门控衰减（data-dependent，逐头）
    v' = (v_i − h·k_i) * β_i                        # delta 修正
    h = h + k_i ⊗ v'                                # 外积写入
    o_i = (q_i * scale) · h
```

## 示例（K=V=1 标量手算）

```
h0=4, k=1, v=3, β=0.5, g=-0.1, scale=1
h ← 4·e^(-0.1) = 3.6398
v' ← (3 − 3.6398)·0.5 = −0.3199
h ← 3.6398 − 0.3199 = 3.3199
o ← q·h = 2·3.3199 = 6.6398
```

## 约束

- `g ≤ 0`（log 衰减）；`T ≤ 6`、`K,V ≤ 4` 小规模，oracle 纯 Python 四重循环
- 衰减必须在 delta 修正**之前**作用（`g→−∞` 时状态清零、修正只剩 `β·v`）
