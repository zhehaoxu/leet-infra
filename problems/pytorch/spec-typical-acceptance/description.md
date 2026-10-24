# Typical Acceptance：按目标概率阈值的投机验证

## 题目描述

标准投机解码的拒绝采样要用 draft 分布 $q$ 算接受概率 $\min(1,p/q)$，还要在拒绝时从残差 $\mathrm{norm}(\max(p-q,0))$ 重采样——需要维护两份分布。**Typical acceptance**（Leviathan et al., 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.2）给出一个更简单的近似：只用目标分布 $p$，把 draft token $x$ 在**典型集**内（即 $p(x)$ 不小于阈值 $\varepsilon$）当作可接受。它对单条 draft 序列逐 token 判定，一旦某个 token 被判拒绝就直接从当前的目标分布 $p_t$ 采样一个 token 顶上并停止。

## 规则

入参：
- `P`：形状 $(\gamma+1, V)$ 的目标概率，`P[t]` 是位置 $t$ 的分布；最后一行 `P[γ]` 供 bonus 采样。
- `draft`：长度 $\gamma$ 的投机 token id 序列。
- `eps`：接受阈值 $\varepsilon$。
- `u_new`：$[0,1)$ 均匀数，用于拒绝位置的替换采样与全接受时的 bonus 采样（CDF 反查）。

对 $t=0,1,\dots,\gamma-1$：

1. 取 $x=\text{draft}[t]$。若 $P[t][x]\ge \varepsilon$，**接受** $x$，追加到输出，继续下一位。
2. 否则**拒绝**：从 $P[t]$ 按 $\mathrm{cdf}(\cdot, u_{\text{new}})$ 采样一个 token 追加，**立即停止**。

若 $\gamma$ 个 draft token 全部被接受，则从 $P[\gamma]$ 按 $\mathrm{cdf}(\cdot, u_{\text{new}})$ 采样一个 bonus token 追加。

返回 `torch.long` 一维张量（已接受前缀 + 拒绝位的替换 token，或全部 + bonus）。

其中 $\mathrm{cdf}(\text{probs}, u)$：返回**最小** $k$ 使 $\sum_{j\le k}\text{probs}[j] > u$。

- 与修改版拒绝采样（[arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.1 / [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)）不同，typical acceptance **不需要** draft 分布 $q$，拒绝时直接采样 $p_t$ 而非残差分布——是"近似无损"的简化验证。
- 与采样合集里的 `typical-sampling`（按熵/困惑度截断的 top-p 型采样）不是一回事：这里是投机解码的 draft 验证判据。

## 示例

```
P = [[0.9, 0.05, 0.05],
     [0.1, 0.2,  0.7 ],
     [0.3, 0.3,  0.4 ],
     [0.2, 0.6,  0.2 ]]      # gamma=3, V=3
draft=[0,0,2], eps=0.1, u_new=0.4

t0: P[0][0]=0.9 >=0.1 接受
t1: P[1][0]=0.1 >=0.1 接受
t2: P[2][2]=0.4 >=0.1 接受
全部接受 -> bonus = cdf(P[3]=[0.2,0.6,0.2], 0.4)：cum=[0.2,0.8,..] -> 索引 1

输出 = [0, 0, 2, 1]
```

## 约束

- CPU float32；返回 `torch.long` 一维张量，用 `torch.equal` 比较。
- 接受判据为 $P[t][x]\ge\varepsilon$（含等号）。
- 拒绝即从 $P[t]$ 采样并停止，不继续验证后续 draft token。
- 采样边界一律取**最小** $k$ 使累积和严格 $>u$，与独立 oracle 一致。
