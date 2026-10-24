# Sequoia：多分支采样与验证（Algorithm 2）

## 题目描述

推测解码要在**同一个位置**上验证 draft 提出的多个候选分支。经典拒绝采样把 draft 摊平成单序列，无法利用一个位置上的多分支信息。**Sequoia**（Chen et al., 2024, [arXiv:2402.12374](https://arxiv.org/abs/2402.12374)）的采样/验证算法（其 Algorithm 2，也与 SpecInfer [arXiv:2305.09781](https://arxiv.org/abs/2305.09781) 相关）在一个位置上依次取出 $k$ 个 draft 分支，逐个用残差分布 $R$ 与 draft 分布 $D$ 做接受判定；被拒的 token 会从 $D$ 中剔除、并把 $R$ 收缩为 $\mathrm{norm}(\max(R-D,0))$，从而保证最终输出的 token 严格服从目标分布，同时具备"最优传输"等鲁棒性。

本题让你实现这个单位置的多分支验证循环。随机性（采样分支、接受硬币、回退采样）全部以显式均匀数入参注入，保证确定性。

## 规则

入参：
- `P`：目标分布，长度 $V$。
- `Q`：draft 分布，长度 $V$。
- `k`：本位置最多验证的分支数。
- `v_list`：长度 $k$ 的均匀数，第 $i$ 个用于从当前 $D$ 反查采样 $s_i$。
- `r_list`：长度 $k$ 的均匀数，第 $i$ 个是分支 $s_i$ 的接受硬币。
- `u_final`：回退时从 $R$ 采样所用的均匀数。

初始化 $R=\mathrm{norm}(P)$，$D=\mathrm{norm}(Q)$，已拒集合 $S=\varnothing$。对 $i=0,1,\dots,k-1$：

1. $s_i=\mathrm{cdf}(D, v_i)$（取最小 $j$ 使 $\sum_{t\le j}D[t] > v_i$）。
2. **接受判定**：若 $D[s_i] > 0$ 且 $r_i < R[s_i]/D[s_i]$，则**接受** $s_i$，返回 $(s_i, \text{accepted})$。
3. **否则拒绝**并更新：
   - $R \leftarrow \mathrm{norm}(\max(R-D, 0))$（逐元素取非负再归一化）；
   - $D[s_i] \leftarrow 0$，$S \leftarrow S \cup \{s_i\}$；
   - 若此刻 $\sum D = 0$（draft 全被拒完），令 $D[t]=0\ (t\in S)$ 否则 $1$（在非拒集合上重设均匀）；
   - $D \leftarrow \mathrm{norm}(D)$。

其中 $\mathrm{norm}(x)$：除以元素和；若和为 $0$ 则返回均匀分布。

$k$ 个分支全部被拒后，返回回退 token $x=\mathrm{cdf}(R, u_{\text{final}})$，标记 residual。

返回 `torch.long` 张量 `[token, reason]`，`reason`：`0`=接受、`1`=回退。

## 示例

```
P=[0.1,0.2,0.3,0.4], Q=[0.4,0.3,0.2,0.1], k=2
v_list=[0.5,0.9], r_list=[0.9,0.5], u_final=0.3

i=0: s0 = cdf(Q,0.5)=1（cum .4→.7）;  r0=0.9 >= R[1]/D[1]=0.2/0.3=0.667  -> 拒绝
     R = norm([0,0,0.1,0.3]) = [0,0,0.25,0.75]
     D[1]=0 -> norm([0.4,0,0.2,0.1]) = [0.571,0,0.286,0.143];  S={1}
i=1: s1 = cdf(D,0.9)=3;  r1=0.5 < R[3]/D[3]=0.75/0.143=5.25  -> 接受 s1=3

输出 = [3, 0]   # 接受，token=3
```

## 约束

- CPU float32；返回形状 `(2,)` 的 `torch.long`，用 `torch.equal` 比较。
- 采样与接受均按上面的顺序、边界（严格 $<$）执行，保证与独立 oracle 逐步一致。
- $\mathrm{cdf}$ 取**最小** $j$ 使累积和 $> u$；$\mathrm{norm}$ 在零向量时退化为均匀。
