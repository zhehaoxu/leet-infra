# 详解：DDPM 反向后验一步 p(x_{t-1}|x_t,x_0)

## 出处
Ho et al., DDPM（arXiv:2006.11239）式 (7)–(11)。把 \(q(x_{t-1}|x_t,x_0)\) 写成关于 \(x_t,\hat x_0\) 的线性组合，均值系数、方差 \(\tilde\beta_t\) 均为闭式；采样时用模型预测 \(\hat x_0=x_\theta\)-换算结果替换真实 \(x_0\)。

## 核心与易错点
1. **三个标量别混**：\(\beta_t\)（单步剩余方差）、\(\bar\alpha_t\)（累积信号）、\(\alpha_t=1-\beta_t\)（单步信号）。分母恒为 \(1-\bar\alpha_t\)。
2. **系数对应项**：\(x_t\) 前是 \(\sqrt{\alpha_t}(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)\)；\(\hat x_0\) 前是 \(\sqrt{\bar\alpha_{t-1}}\beta_t/(1-\bar\alpha_t)\)。两者弄反会得到错误轨迹。
3. **方差用 \(\tilde\beta_t\)**：\(\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\)（不是 \(\beta_t\) 本身，也不是 \(\bar\beta_t\)）。最后一步 \(t=1\) 常约定 \(\bar\alpha_0=1\) 使方差为 0。
4. \(z\) 显式传入保确定性；\(z=0\) 只给均值。

## 对拍口径
参考解向量化：算 \(c_t,c_0,\text{var}\) 后 `c_t*x_t + c_0*x0p + sqrt(var)*z`。判题 `_oracle` 逐元素双循环用 `math.sqrt` 重算同式，并断言 \(z=0\) 时输出等于「去掉噪声项的均值」（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。

## 复杂度
逐元素 \(O(N)\)。
