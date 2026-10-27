# 详解：ε / x₀ / v 三种预测目标互转

## 出处

Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*（arXiv:2206.00364）提出用 \(v=\alpha\epsilon-\sigma x_0\) 作训练目标（蒸馏时学生一次走半步、教师走整步，v 空间里这一步是半旋转，插值才有意义）；SD 2.x / SDXL / Imagen 都提供 v-prediction 开关。

## 核心与易错点

1. **v 的符号不能写反**：\(v=\alpha\epsilon-\sigma x_0\)，不是 \(\sigma x_0-\alpha\epsilon\)。用旋转矩阵记：\([x_t;v]=\begin{bmatrix}\alpha&\sigma\\-\sigma&\alpha\end{bmatrix}[x_0;\epsilon]\)，行列式为 1、范数保持。
2. **反解 \(v\to(x_0,\epsilon)\) 就是把旋转取逆**：\(\epsilon=\alpha v+\sigma x_t\)、\(x_0=\alpha x_t-\sigma v\)，两个式子的 \(\sigma\) 项符号相反，抄错一处就会在 \(\bar\alpha_t\to1\) 时明显偏差。
3. **奇点在两端**：\(\alpha=0\) 时不能由 \(\epsilon\) 反解 \(x_0\)（除以 0），\(\sigma=0\) 时不能由 \(x_0\) 反解 \(\epsilon\)。题目要求这两种情况抛 `ValueError`，而不是返回 inf。
4. **已知量必须原样回填**：`mode="v"` 时输出里的 `eps`/`x0` 是算出来的、但 `v` 必须逐位等于输入，这是判题里最容易被「四舍五入」破坏的一条。

## 对拍口径

参考解向量化三支公式、内部 float64、出口转 float32。判题 `_oracle` 逐元素用 `math.sqrt` 重算并把公式展开成不共享中间量（如 \(x_0=x_t/\alpha-(\sigma/\alpha)\epsilon\)），另加三组结构化断言：前向关系 \(x_t=\alpha x_0+\sigma\epsilon\)、\(v=\alpha\epsilon-\sigma x_0\) 同时成立；\(\epsilon\to x_0\to\epsilon\) 与 \(v\to(x_0,\epsilon)\) 互逆；两端退化（\(\bar\alpha_t=0,1\)）与非法输入。阈值 `allclose(atol=1e-5, rtol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)，无归约、无随机性。
