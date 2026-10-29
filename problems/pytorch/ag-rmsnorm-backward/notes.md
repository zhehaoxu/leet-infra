# 详解：RMSNorm 反向为什么只剩一项减法

## 出处

RMSNorm 是 LLaMA/Mistral/Gemma 一系的标准归一化（本题库中 `rmsnorm`、`llm-rmsnorm` 是它的正向版本），面试里问反向的场景有两个：写 fused residual+RMSNorm kernel，以及解释「RMSNorm 相比 LayerNorm 省了哪一次归约、梯度形式为什么会变」。第二个问题最能区分是否真推过公式。

## 从缩放不变性推 dX

前向 \(y_i=\gamma_i x_i/r\)，\(r=\sqrt{\frac1H\sum_k x_k^2+eps}\)。注意 RMSNorm 只惩罚**沿 \(x\) 方向**的扰动：对 \(r\) 求导得 \(\partial r/\partial x_j=x_j/(Hr)\)，于是

$$\frac{\partial L}{\partial x_j}=\frac{g_j}{r}-\sum_i g_i\frac{x_i}{r^2}\frac{x_j}{Hr}=\frac1r\Big(g_j-\hat x_j\cdot\text{mean}(g\circ\hat x)\Big)$$

对照 LayerNorm 的 \(\frac1\sigma\big(g-\text{mean}(g)-\hat x\,\text{mean}(g\circ\hat x)\big)\)：少掉的 \(\text{mean}(g)\) 正是 \(\partial\mu/\partial x_j\) 那条路径。几何上看，LayerNorm 把梯度投影到「与全 1 向量和 \(x̂\) 都正交」的子空间，RMSNorm 只投影掉 \(x̂\)（即 \(x\) 本身）这一个方向。

## 核心与易错点

- **照抄 LayerNorm 公式**：多减 \(\text{mean}(g)\) 后，若 \(dY\) 沿末轴为常数、\(\gamma\) 全 1，会错误地给出 0；正确结果应为 \((g-\hat x\,\text{mean}(g\circ\hat x))/r\)，一般非 0。测试里专门用这组入参把两个公式区分开。
- **\(x̂\) 的定义**：这里 \(x̂=x/rms\) 不减均值，直接拿 LayerNorm 的去均值残差代入会引入一阶误差。
- **零输入**：\(x\) 全 0 且 \(eps=0\) 时 \(r=0\)，必须靠 `eps`（默认 1e-6）保护，不能出现 `nan`。
- **dGamma 折叠**：与 LayerNorm 完全一致，沿所有前导轴求和到 `(H,)`。

## 与邻近题的区别

`ag-layernorm-backward` 多一项 \(\text{mean}(g)\) 且 \(x̂\) 去心；`ag-batchnorm-backward` 沿 batch 轴归约并且统计量本身可导；`ag-weightnorm-backward` 处理的是权重重参数化、没有归一化轴。本题的独有考点是「少的这一项会带来可观测的非零梯度差」。

## 对拍口径与复杂度

oracle 用纯 autograd 搭前向 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma`，再取 `dX/dGamma`，float64 atol=1e-9。示例取 \(x=\pm1\)、\(eps=0\) 使 \(rms=1\)，四个分量可手算。复杂度 \(O(BTH)\)，一次末轴平方和归约 + 一次末轴点积归约 + 一次前导轴折叠。
