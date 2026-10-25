# 详解：ADOPT 去偏差 + 前一步二阶矩 + 梯度裁剪

## 出处
Taniguchi, Harada, Minegishi, Oshima, Guo, Nagahara, Imaizumi, Kashiwabara, Suzuki, *ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate*（arXiv:2411.02853）。参考实现取 kozistr/pytorch-optimizer 的 `ADOPT`（`clip_lambda=step**0.25`、`maximize=False`、`cautious/stable_adamw` 关闭）。

> 权重衰减口径：库默认 `weight_decouple=False`（把 \(\lambda p\) 加进梯度）。本题统一采用**解耦**形式 \(p\leftarrow p(1-lr\cdot\text{wd})\)，与合集内 AdamW/Yogi 等保持一致，也避免就地改写 `p.grad`。默认 `weight_decay=0` 时两者无差别。

## ADOPT 相对 Adam 的三处结构改动
1. **去掉偏差修正**：不再出现 \(1-\beta^t\)，动量 \(m\) 直接作为更新方向、步长就是 \(lr\)。作者证明正是偏差修正破坏了任意 \(\beta_2\) 下的收敛性。
2. **用上一步的二阶矩归一**：第 \(t\) 步的分母来自 \(v_{t-1}\)（**在把当前 \(g^2\) 并入之前**）。所以更新顺序是：先用旧 \(v\) 归一 \(g\)、更新 \(m\) 与 \(p\)，**最后**才 `v ← β2·v + (1−β2)·g²`。
3. **首步特殊**：\(t=1\) 时只初始化 \(v\leftarrow g^2\)、**不做参数更新**（旧 \(v=0\) 会除零，故跳过）。这是最容易写错、也是判题专门考察的点。
4. **裁剪**：归一化梯度被夹在 \(\pm t^{1/4}\) 内（`clip_lambda=step**0.25`）；\(t^{1/4}\) 随步数缓慢放宽，早期抑制异常大的归一化步、后期几乎不起作用。分母还逐元素 `clamp(min=eps)`。

## 对拍口径
判题用独立向量化重算（`_oracle_adopt`）：用不同写法（`g/de_nom`、`clamp`、`b1*m+(1-b1)*gt` 等）在多形状多步随机梯度下 `allclose(atol=1e-5, rtol=1e-5)` 比对。`check_first_step_only_inits_v` 专项验证首步除权重衰减外参数不动；`grad is None` 参数整体跳过。

## 复杂度
与 Adam 相同：每参数每步 \(O(\text{numel})\)，两份状态 \(m,v\)。
