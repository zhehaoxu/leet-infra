# 调度器：线性 warmup + 余弦退火学习率

## 题目描述

训练大模型或视觉模型时，学习率（\(lr\)）几乎从不是常数：开头几步要**线性 warmup**（用小 \(lr\) 平滑起步、避免早期梯度冲击二阶矩估计），随后按**余弦退火**（cosine annealing）把 \(lr\) 从峰值一路降到接近下界。这套「warmup + cosine」几乎是 Transformer 训练的默认配方。

本题要求你**继承 `torch.optim.lr_scheduler.LRScheduler`** 实现一个调度器：它在构造时把优化器的 \(lr\) 设为第 \(0\) 步的值，每次调用 `step()` 前进到下一步并写回新的 \(lr\)。评测会反复读取 `param_groups[0]["lr"]` 并与参考序列比对。

## 规则

- 工厂函数 `solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.0)`：内部用 `torch.optim.SGD(params, lr=base_lr)` 建一个优化器，返回继承 `LRScheduler` 的调度器实例；`base_lr` 是余弦段的峰值（上界），`min_lr` 是退火下界。
- 记当前步为 \(t\)（\(t=0,1,2,\dots\)，构造完成即第 \(0\) 步）。\(lr(t)\) 的分段定义：
  - **warmup 段**（\(0\le t<\) `warmup_steps`）：
    \[lr(t)=base\_lr\cdot\frac{t+1}{warmup\_steps}\]
    即第 \(0\) 步为 \(base\_lr/\text{warmup}\)，最后一步（\(t=\text{warmup}-1\)）恰好达到 \(base\_lr\)。
  - **余弦段**（\(t\ge\) `warmup_steps`）：令 \(\text{denom}=\text{total\_steps}-\text{warmup\_steps}\)，
    \[\text{progress}=\operatorname{clamp}\!\left(\frac{t-\text{warmup\_steps}}{\text{denom}},\,0,\,1\right),\qquad
    lr(t)=min\_lr+(base\_lr-min\_lr)\cdot\tfrac12\bigl(1+\cos(\pi\cdot\text{progress})\bigr)\]
    若 \(\text{denom}\le 0\)（即 `total_steps<=warmup_steps`）则取 \(\text{progress}=1\)，\(lr=\text{min\_lr}\)。
- 调度器**构造后**优化器的 \(lr\) 必须等于 \(lr(0)\)；每调用一次 `step()`，\(t\) 加 \(1\) 并把 \(lr(t)\) 写回 `param_groups[0]["lr"]`。若 `warmup_steps=0` 则直接全程余弦（此时 \(lr(0)=base\_lr\)）。
- 禁止调用现成的 `LinearLR`/`CosineAnnealingLR`/`LambdaLR` 等，必须自己按上式计算并写回。

## 示例

`base_lr=0.1`、`warmup_steps=3`、`total_steps=10`、`min_lr=0.001`。读取第 \(0\) 到第 \(9\) 步的 \(lr\)：

```
[0.033333, 0.066667, 0.100000, 0.100000, 0.095098, 0.081363, 0.061515, 0.039485, 0.019637, 0.005902]
```

（第 2 步 warmup 末达到峰值 0.1；第 3 步余弦段起点仍是 0.1——这是 warmup 与 cosine 在峰值处的正常衔接。）

## 约束

- 参数 \(t\) 从 \(0\) 起；`step()` 每次令 \(t\leftarrow t+1\)。
- 须与参考实现逐步一致（浮点比对 `atol=1e-6, rtol=1e-6`，对拍口径见 notes.md）。
- 支持任意 `params`（≥1 个张量），只对第一个 param_group 的 \(lr\) 负责。
