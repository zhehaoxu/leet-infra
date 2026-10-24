# 解题说明 · Speculative Decoding 拒绝采样

## 为什么分布等价

标准采样要从 target $p$ 抽 token。这里改成先抽 $x_t\sim q$，再以 $\min(1, p/q)$ 接受。若接受就直接用 $x_t$；若拒绝就从 $(p-q)_+$ 重抽。对任意 token $v$：

$$P(\text{最终}=v) = \underbrace{\min(p_v,q_v)}_{\text{直接接受}} + \underbrace{(1-\sum_u \min(p_u,q_u))\cdot \frac{(p_v-q_v)_+}{\sum_w (p_w-q_w)_+}}_{\text{拒绝后重采}}$$

利用恒等式 $\sum(p-q)_+ = \sum(p-q)_- = 1-\sum\min(p,q)$，可化简回 $p_v$。这正是"拒绝采样修正"能保证与 target 逐 token 采样**完全同分布**的关键。

## bonus token 的由来

target 模型验证 draft 的 $\gamma$ 个 token 时，它的单次前向其实同时给出了 $\gamma+1$ 个位置的分布（最后一位是"若前 $\gamma$ 全对，下一位该是什么"）。所以全接受时免费多产出 1 个 token，这是加速比的来源。

## CDF 反查的确定性

题面要求用"最小 $k$ 使 $\mathrm{cumsum}[k]\ge u$"而非 `torch.multinomial`，是为了让给定 $u$ 后采样结果**可复现**、oracle 能用纯 python 累加循环独立对拍。注意边界：`torch.cumsum` 末位可能因浮点略小于 1，$u\in[0,1)$ 时 `k` 不会越界到 $V$。
