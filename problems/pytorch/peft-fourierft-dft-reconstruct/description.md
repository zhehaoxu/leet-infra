# PEFT：FourierFT 离散傅里叶基系数重建

## 题目描述

**FourierFT**（arXiv:2405.18201）把权重更新 \(\Delta W\) 看作一个 \(d_\text{out}\times d_\text{in}\) 的实图像，用少量可学习的**离散傅里叶余弦基系数**来稀疏表示它：

\[\Delta W[i,j] = \sum_{(u,v)\in\mathcal F} c_{u,v}\,\cos\!\left(2\pi\left(\frac{u\,i}{d_\text{out}} + \frac{v\,j}{d_\text{in}}\right)\right),\]

其中 \(\mathcal F\) 是学习到的频率位置集合，\(c_{u,v}\) 是对应系数。本题把频率位置和系数作为输入，要求重建 \(\Delta W\) 并合并到基座上：

\[W' = W_0 + \alpha\,\Delta W.\]

频率 \(u\) 的取值范围是 \(0\le u<d_\text{out}\)，\(v\) 的范围是 \(0\le v<d_\text{in}\)。

## 任务

给定 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)、频率位置 `freqs`（形状 \((m,2)\) 的 int64 张量，每行 \((u,v)\)）、对应系数 `coeffs`（长度 \(m\) 的 float32 张量）以及缩放 \(\alpha\)，按上式返回 \(W'\)。

- 可假设 `freqs` 中所有频率都在合法范围内。
- \(\alpha=0\) 时退化为 \(W_0\)。
- 空频率列表（\(m=0\)）时 \(\Delta W=0\)，同样退化为 \(W_0\)。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，float32。

## 提示

- 用行/列坐标网格 `arange(d_out)[:, None]` 与 `arange(d_in)[None, :]` 广播构造相位矩阵。
- 对每个频率 \((u,v)\) 计算 `cos(2*pi*(u*i/d_out + v*j/d_in))`，乘上系数后累加。
- 这是 FourierFT 的简化实余弦基版本；判题只关心给定稀疏频率系数能否正确重建 \(\Delta W\)。
