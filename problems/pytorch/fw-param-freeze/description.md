# 参数冻结、换 head 与优化器分组

## 题目描述

微调的标准配方：骨干全冻结、扔掉旧 head、接上新 head、只把可训参数交给优化器。每一步都在框架层产生可观测的账：参数量变化、`requires_grad` 标志、优化器状态条目、冻结参数在 `step()` 后必须逐位不变。本题走一遍完整流程并汇总这些账。

## 规则

实现 `solve()` 并按固定流程返回行为账目 dict。模型结构、显式权重生成公式 \(\texttt{\_explicit(shape,start)}=(\texttt{arange}(n)+\texttt{start})/100\)、8 步流程与返回格式，全部以 starter 的 docstring 为唯一完整信息源。要点：

1. 冻结用 `requires_grad_(False)`；换 head 后新 `nn.Linear` 默认可训，且新旧维度不同（\(6\to4\) 换成 \(6\to5\)），总参数量随之变化。
2. 优化器只收 `p.requires_grad` 为真的参数；`SGD(lr=0.1, momentum=0.9)` 跑 3 步，loss 为 MSE \(\frac{1}{N}\sum (f(x)-y)^2\)。
3. 冻结参数在 3 步后必须与步前 clone **逐位相等**（`torch.equal`）；且 backward 后其 `.grad` 应始终为 `None`（autograd 根本不会为它求梯度）。
4. `len(opt.state)` 只统计真正建了动量缓冲区的参数个数。

## 示例

```
输入：无
输出（节选自参考解完整运行）：
{
 "total_before": 124, "total_after": 131,
 "trainable_before": 0, "trainable_after": 35,
 "flags": [["encoder.0.weight", false], ["encoder.0.bias", false],
           ["encoder.2.weight", false], ["encoder.2.bias", false],
           ["head.weight", true], ["head.bias", true]],
 "losses": [6.816761016845703, 11.316238403320312, 1.1022732257843018],
 "final_head_weight_flat": [0.002984001999720931, 0.013380030170083046,
   0.02377604879438877, 0.03417206183075905, 0.04456809163093567,
   0.054964106529951096, ...],
 "final_head_bias_flat": [-0.006162865553051233, -9.81296252575703e-05, ...],
 "frozen_bitwise_identical": true, "frozen_grad_is_none": true,
 "n_opt_state_params": 2
}
```

第 1 步 loss 从 6.82 回升到 11.32 是动量过冲；第 2 步骤降回 1.10。

## 约束

- 全流程确定性，初始权重/输入/目标都由 `arange` 公式给出，禁止使用随机数。
- 数值对拍 `atol=1e-5, rtol=1e-5`；计数/布尔/flags 必须精确一致。
- `flags` 顺序必须与 `named_parameters()` 一致。
