# Speculative Decoding：推测解码的接受/拒绝准则

## 题目描述

**Speculative Decoding**（Leviathan et al. 2022）用一个廉价的小 draft 模型一次猜出 $\gamma$ 个 token，再用大 target 模型**一次前向并行验证**这 $\gamma$ 个猜测。为保证最终采样分布严格等于 target 分布 $p$，采用**拒绝采样**修正：对每个 draft token $x_t$，以概率 $\min\!\big(1, \frac{p(x_t)}{q(x_t)}\big)$ 接受它（$q$ 为 draft 分布）；一旦某步拒绝，就丢弃该步及其后的所有猜测，改从**修正分布** $\mathrm{normalize}(\max(0,\,p-q))$ 重新采样一个 token 收尾。若 $\gamma$ 个全部接受，则额外从 target 的第 $\gamma$ 步分布免费多取一个 **bonus token**（因为 target 本就算了 $\gamma+1$ 个位置）。

这样每个解码步最多产出 $\gamma+1$ 个 token，与逐 token 采样**分布完全等价**，却把 target 的串行前向次数大幅降低。

## 规则

给定 `draft_probs` $(\gamma, V)$、`target_probs` $(\gamma+1, V)$、`draft_tokens` $(\gamma,)$、`u_accept` $(\gamma,)$、标量 `u_resample`：

1. 顺序遍历 $i = 0 \ldots \gamma-1$，取 draft 猜测 $x_t = $`draft_tokens[i]`：
   - 接受概率 $\mathrm{ratio} = \dfrac{p_i[x_t]}{q_i[x_t]}$；若 $q_i[x_t]=0$ 视为 $\mathrm{ratio}=+\infty$
   - 若 `u_accept[i]` $< \min(1, \mathrm{ratio})$：**接受**，输出 $x_t$，进入 $i+1$
   - 否则：**拒绝**，令修正分布 $\pi = \mathrm{normalize}\big(\max(0,\; p_i - q_i)\big)$（逐元素、先 clamp 到 $\ge 0$ 再除以其和），从 $\pi$ 采样一个 token 追加输出，**立即停止**（后续 draft 全部丢弃）
2. 若循环结束仍未拒绝（$\gamma$ 个全接受）：从 `target_probs[γ]`（bonus 步分布）采样一个 token 追加输出
3. **采样算子**（给定分布 $\pi$ 与均匀数 $u$）：返回**最小**的下标 $k$ 使 $\sum_{j \le k} \pi[j] \ge u$（即 CDF 反查、浮点比较用 `>=`）

返回一维 `torch.long` 张量：本次解码步输出的 token id 序列（长度 $\le \gamma+1$）。

## 示例

```
γ=3, V=4
draft_probs  = [[0.5,0.3,0.1,0.1],
                [0.2,0.5,0.2,0.1],
                [0.1,0.1,0.6,0.2]]
target_probs = [[0.3,0.4,0.2,0.1],
                [0.4,0.2,0.3,0.1],
                [0.1,0.1,0.6,0.2],
                [0.25,0.25,0.25,0.25]]
draft_tokens = [0, 1, 2]
u_accept     = [0.5, 0.7, 0.9]
u_resample   = 0.5

i=0: x_t=0, ratio=0.3/0.5=0.6, u=0.5<0.6 → 接受 tok0
i=1: x_t=1, ratio=0.2/0.5=0.4, u=0.7≥0.4 → 拒绝
     修正 = clamp([0.4-0.2, 0.2-0.5, 0.3-0.2, 0.1-0.1],0)=[0.2,0,0.1,0]
     normalize → [0.6667,0,0.3333,0]，CDF=[0.6667,0.6667,1.0,1.0]
     u=0.5 → 最小 k 使 CDF≥0.5 → k=0
输出 = [0, 0]   （tok0 接受，随后拒绝位重采样为 0，i=2 丢弃）
```

## 约束

- CPU float32；所有分布每行和为 1。
- 采样下标用 CDF 反查（`>=` 比较），不使用 `torch.multinomial`（保证确定性）。
- 输出为 `torch.long` 一维张量。
