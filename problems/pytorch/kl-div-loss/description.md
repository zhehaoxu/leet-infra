# kl_div 损失（log-prob 输入与 batchmean 口径）

## 题目描述

知识蒸馏、RLHF 的 KL 惩罚（PPO/GRPO 里对参考模型的偏离项）、VAE 的正则项都调用 `F.kl_div`，而它的接口有两处反直觉：**第一个参数是 $\log q$（学生/策略的对数概率）而不是 $q$**，以及**默认 `reduction="mean"` 在数学上不是 KL**。KL 散度的定义是

$$\mathrm{KL}(P\,\|\,Q) = \sum_{c} p_c \log\frac{p_c}{q_c}, \qquad \text{逐元素项 } \ell_c = p_c(\log p_c - \log q_c)$$

对 batch 内的每个样本各自求和、再对样本取平均，才是数学意义上的 KL——这正是 `"batchmean"` 的口径（除以 $B$）；`"mean"` 却除以 $B\times C$，把 KL 缩小了 $C$ 倍，蒸馏代码里 KL 权重调不准多半就是它。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `log_pred` 是 $\log q$（例如 `log_softmax` 的输出），`target` 是 $p$；`log_target=True` 时 `target` 已经是 $\log p$，逐元素为 $e^{\text{target}}(\text{target}-\text{log\_pred})$。
- `reduction` 四态：`"batchmean"`（默认，和 ÷ 输出第 0 维长度 $B$）/ `"mean"`（和 ÷ 元素个数）/ `"sum"` / `"none"`（广播后的逐元素张量）。
- 逐元素口径：$p>0$ 时 $p(\log p-\log q)$；$p>0$ 且 $q=0$ 时为 $+\infty$；$p=0$ 处按数学约定 $0\log 0 := 0$，贡献恰为 0（**不复制** `F.kl_div` 在 $p=0,q=0$ 处返回 NaN 的实现细节）。
- `target` 允许与 `log_pred` 广播（例如一个共享的目标分布作用到 batch 内每一步）；`reduction` 非法时抛 ValueError。

## 示例

logits $=[[2,0,-1],[0,1,3]]$ 经 `log_softmax(dim=1)` 得 $\log q$，$p=[[0.5,0.3,0.2],[0.1,0.6,0.3]]$（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
elem(none) = [[-0.26165056228637695, 0.2897619903087616, 0.31208163499832153],
              [ 0.0867260992527008,  0.9954123497009277, -0.3102380335330963]]
batchmean  = 0.556046724319458
sum        = 1.112093448638916
mean       = sum / 6
```

首项 $0.5(\log 0.5 - (-0.1694)) = -0.2617$ 为负——逐元素项**可正可负**，只有求和后的 KL 才保证非负。`batchmean = sum/2 = 0.5560`，而 `mean = sum/6`，两者恰差类数 3 倍。

## 约束

- 输入 float32，内部 float64 计算，返回 float32；稠密（$p$ 全 $>0$）时与 `F.kl_div` 逐点对拍 atol=1e-6、rtol=1e-5
- 判题含 one-hot 硬标签退化（$p=1$ 处等于 $-\log q$，即交叉熵）、$q=0$ 的 $+\infty$、$p=0$ 的 0 约定、广播 target 与 4 个非法 `reduction`
- 与 `kl-divergence`（两个分布的标量 KL）的分工：那题考数学定义，本题考 PyTorch 损失函数接口口径。
