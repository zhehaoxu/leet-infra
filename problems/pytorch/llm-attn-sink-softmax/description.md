# SoftMax1（LLN）：给注意力分母加一个虚拟 sink

## 题目描述

StreamingLLM（Xiao et al., ICLR 2024, arXiv 2309.17453）第 3.3 节的观察：LLM 中大量注意力会落在序列最前面几个 token 上，形成 **attention sink**。他们提出的 **SoftMax1**（又称 LLN，less-normalized）把普通的归一化分母加一：

\[
\text{SoftMax1}(x)_i=\frac{e^{x_i}}{1+\sum_{j=1}^{N}e^{x_j}}
\]

多出来的那个 1 就是 \(\exp(0)\)，相当于在 keys 里额外拼一个 logit 恒为 0、value 全零的虚拟 token。于是每一行的概率不再强制加和为 1——和小于 1，缺口恰好被这个 sink 吃掉。本题要求实现带可配置 sink logit 与掩码的这一版归一，并同时把两部分质量都返回出来。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `mask` 中 True = 保留、False = 屏蔽；屏蔽位置的权重必须严格为 0，并且既不参与取最大值、也不进分母。
- 数值稳定：每行取 \(r=\max(\max_{\text{保留位}} x,\ \texttt{sink\_logit})\)，所有 `exp` 的指数都 \(\le 0\)；`sink` 必须参与取最大，否则整行远低于 sink 时会算出 \(\exp(+\text{大数})\)。
- 整行被屏蔽属于合法输入：`weights` 全 0、`sink_weight` 为 1，不得返回 NaN。
- 返回 `(weights, sink_weight)`：`weights` 与 `scores` 同形状，`sink_weight` 形状为 `scores.shape[:-1]`，两者都是 float32；内部用 float64 计算。
- 逐行恒有 `weights.sum(-1) + sink_weight == 1`，因此 `weights.sum(-1) < 1`，缺口就是 sink 占走的概率质量；**不允许**把 `weights` 再除以自身行和来"补回" 1。
- `sink_logit` 取 \(-80\) 这类大负数时应退化为普通的 masked softmax（`sink_weight ≈ 0`）。

## 示例

```
scores = [[ 0.000000,  0.693147],
          [ 0.000000, -3.000000],
          [-1.000000, -2.000000]]
mask   = [[True, True], [True, False], [False, False]]
sink_logit = 0.0

weights     = [[0.250000, 0.500000],
               [0.500000, 0.000000],
               [0.000000, 0.000000]]
sink_weight = [0.250000, 0.500000, 1.000000]
```

第 0 行 \(r=\ln 2\)，指数和为 \(0.5+1=1.5\)，sink 项 \(e^{-\ln 2}=0.5\)，分母为 2，故权重 \([0.25,0.5]\)、sink 权重 0.25。第 1 行只保留第一位，\(e^{0}=1\) 加上 sink 的 1 得到 0.5/0.5，第二位严格为 0。第 2 行整行屏蔽，sink 独占全部质量。

## 约束

- `scores` 必须是 float32/float64 张量，`ndim >= 1`，末维 \(\ge 1\)，且全部有限；出现 NaN 或 \(\pm\infty\) 抛 ValueError（要排除位置请用 `mask`）。
- `mask` 必须是 bool 张量且能广播到 `scores`；非张量、float/整型、无法广播都抛 ValueError。
- `sink_logit` 必须是有限实数；bool、字符串、None、NaN、\(\pm\infty\) 都抛 ValueError（整数可以）。
- 时间复杂度与空间复杂度都是 \(O(\text{元素数})\)。
