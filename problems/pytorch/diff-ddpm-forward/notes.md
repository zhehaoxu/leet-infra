# 详解：DDPM 前向加噪 q(x_t|x_0)

## 出处
Ho et al., *Denoising Diffusion Probabilistic Models*（arXiv:2006.11239）式 (5)-(6)：由 \(q(x_1|x_0)*\dots*q(x_t|x_{t-1})\) 的高斯闭合性得到可直接采样的 \(q(x_t|x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)\)，故 \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\)。

## 核心与易错点
1. **两个 sqrt 别写反**：信号系数 \(\sqrt{\bar\alpha_t}\)、噪声系数 \(\sqrt{1-\bar\alpha_t}\)，平方和为 1（方差保持）。写反会让 \(\bar\alpha_t\) 大时反而噪声占主导。
2. **eps 显式传入**：训练时 \(\epsilon\sim\mathcal N(0,I)\) 随机，但判题要确定性，故把噪声作为入参、本题不内部采样。
3. **一步到位**：无需迭代 \(t\) 次单步加噪，直接用 \(\bar\alpha_t\) 闭式，正是 DDPM 高效训练的关键。
4. 支持任意形状，逐元素即可。

## 对拍口径
参考解 `sqrt(ab)*x0 + sqrt(1-ab)*eps`。判题 `_oracle` 用 Python `math.sqrt` 取标量系数再逐元素乘加，并在 \(\bar\alpha=1\) 边界断言 \(x_t=x_0\)（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。

## 复杂度
逐元素 \(O(N)\)。
