# 解题说明 · Typical Acceptance

## 用阈值近似概率比

修改版拒绝采样要求"知道 draft 分布 $q$"：接受概率 $\min(1,p/q)$ 和残差 $\mathrm{norm}(\max(p-q,0))$ 都得同时握住 $p,q$ 两列。Typical acceptance 的观察是——在低温度、draft/target 接近时，绝大多数 draft token 落在目标分布的**典型集**里（$p(x)$ 显著大于 0）。于是干脆丢掉 $q$，用一个固定阈值 $\varepsilon$：只要 $p_t(\text{draft}[t])\ge\varepsilon$ 就接受，否则就地从 $p_t$ 重采一个并刹车。少了除法与残差归一化，验证几乎零成本。

## 与拒绝采样、Medusa 的对照

| 验证方式 | 需要 q | 拒绝时采样 | 结构 |
|---|---|---|---|
| 修改版 RS (155) | 是 | $\mathrm{norm}(\max(p-q,0))$ | 单序列 |
| **Typical (本题, 178)** | **否** | **直接 $p_t$** | **单序列** |
| Medusa (175) | 否 | — | 候选树取最长 |

三者都用"接受前缀 + 一个 bonus/替换"的范式；本题是其中最单线的一支：逐位置判定、遇拒即停。

## CDF 反查的边界

采样统一用 $\mathrm{cdf}(\text{probs},u)=\min\{k:\sum_{j\le k}\text{probs}[j]>u\}$（严格 $>$）。实现里就是累加到第一个"累积和严格大于 $u$"的下标。接受判据用 $\ge$（含等号）——这两个不等号方向必须和 oracle 完全一致，否则在 $p=\varepsilon$ 或 $u$ 落在累积点上会分叉。随机对拍里对 `u_new` 特意取到接近 1 又 `<0.98`，避开浮点累积恰好等于 1 的边界。

## 边界自检

- `check_example`：整条 draft 全接受 → 追加 `P[γ]` 上的 bonus。
- `check_reject_first`：首位置 $p<\varepsilon$ → 只输出替换 token，长度 1。
- `check_reject_mid`：中途被拒 → 输出=已接受前缀 + 替换 token，不再验证其后 draft。
- `check_matches_oracle`：随机 $P/\text{draft}/\varepsilon/u$，与独立纯 Python 实现对拍。
