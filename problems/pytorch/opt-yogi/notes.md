# 详解：Yogi 符号控制的二阶矩

## 出处
Zaheer, Reddi, Sachan, Rawhi, Kumar, *Adaptive Learning Rate Strategies for Non-convex and Stochastic Optimization*（arXiv:1807.01874）。参考实现取 kozistr/pytorch-optimizer 的 `Yogi`（`weight_decouple=True`、`adam_debias=False`、`maximize=False`）。

## 为什么改二阶矩更新
Adam 的 \(v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2\) 在出现一个很大的 \(g^2\) 时会让 \(v\) 一步跳高（把该坐标的学习率长期压小）。Yogi 观察到：若把更新写成“沿 \(\operatorname{sign}(v-g^2)\) 方向、以 \(g^2\) 为幅度、按 \((1-\beta_2)\) 步长**增减** \(v\)”：

\[ v_t = v_{t-1} - (1-\beta_2)\,\operatorname{sign}(v_{t-1}-g_t^2)\, g_t^2 \]

则 \(v\) 每次最多变化 \((1-\beta_2)g_t^2\)、且方向由 \(v\) 与 \(g^2\) 谁大决定——比 Adam 的软平均更“粘滞”，抗突发梯度。可以证明它保持了 \(v\) 对 \(g^2\) 的上界性质（\(v\) 始终是历史 \(g^2\) 的上界估计）。

## 与参考解对齐的三个易错点
1. **状态初值非零**：\(m_0=v_0=\texttt{initial\_accumulator}\)（默认 \(10^{-6}\)），不是 0。写成 0 会与前几步的 \(v\) 偏离。
2. **符号用的是更新前的 \(v\)**：\(\operatorname{sign}(v_{t-1}-g^2)\) 必须在 \(v\) 自身前计算——即先算 `v - g2` 再 `sign_`，再 `addcmul_` 到 `v`。
3. **分母算子顺序**：\(\sqrt{v}/\sqrt{bc_2}+\epsilon\)，与 PyTorch 一致（`v.sqrt().div_(bc2sq).add_(eps)`，其中 `bc2sq=√(1−β2^t)`）。\(bc_1\) 只作用在步长 \(lr/bc_1\) 上。

## 对拍口径
判题用独立向量化重算（`_oracle_yogi`）：用与解不同的写法（`b1*m+(1-b1)*g`、`v-(1-b2)*sign(v-g2)*g2`、`sqrt(v)/sqrt(bc2)+eps` 等代数式）在多形状、多步随机梯度下 `allclose(atol=1e-5, rtol=1e-5)` 比对。`grad is None` 参数连权重衰减也跳过——专项用例覆盖。

## 复杂度
与 Adam 相同：每参数每步 \(O(\text{numel})\)，两份状态 \(m,v\)。
