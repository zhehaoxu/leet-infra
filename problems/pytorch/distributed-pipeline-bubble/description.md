# 流水线并行的气泡率（GPipe）

## 题目描述

流水线并行（Pipeline Parallelism）把模型按层切成 $p$ 段、分给 $p$ 张卡，一个 batch 再拆成 $m$ 个
**micro-batch** 依次流过各段来填满流水线。但热身和收尾阶段总有卡闲着——这就是**气泡（bubble）**。
气泡占比直接决定流水线并行的效率，是面试里衡量「懂不懂 GPipe/1F1B」的标尺。本题算它（纯计算）。

## 模型（GPipe：全部前向再全部反向）

把一个 micro-batch 流过**一段**记作 1 个时间单位。$m$ 个 micro-batch、$p$ 段流水线：

- **makespan（总墙钟时间，单位）**：流水线从第一段开工到最后一段完工
  $$ T = m + p - 1 $$
- **有效工作（每段）**：每段真正忙的时间 $= m$（处理 $m$ 个 micro-batch）
- **利用率**：$\text{util} = \dfrac{m}{m+p-1}$
- **气泡率**：空闲占比
  $$ \text{bubble} = \frac{T - m}{T} = \frac{p-1}{m+p-1} = 1 - \text{util} $$

（对称 fwd/bwd 时 fwd、bwd 时间在比值里约掉，故只与 $m,p$ 有关——这正是 GPipe 经典结论。）

## 规则

实现 `solve(n_stages, n_microbatches)`（记 $p=$`n_stages`，$m=$`n_microbatches`），返回三元组
`(bubble, util, makespan)`：`bubble`、`util` 为 float，`makespan` 为 float（$=m+p-1$）。

## 示例

```
p=4, m=8:  makespan=11, util=8/11=0.727272…, bubble=3/11=0.272727…
→ (0.2727272727, 0.7272727273, 11.0)
```

## 约束

- `1 ≤ n_stages, n_microbatches ≤ 4096`
- $p=1$：无流水线，bubble $=0$、util $=1$、makespan $=m$
- $m=1$：无法重叠，bubble $=(p-1)/p$、util $=1/p$、makespan $=p$
- 返回 `(float, float, float)`，相对误差 $10^{-6}$ 内判等；须满足 `bubble + util == 1`
- 只算 GPipe 气泡，不涉及 1F1B 的具体调度或显存
