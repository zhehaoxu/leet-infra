# 详解：APG 的自适应投影引导

## 出处

*Adaptive Projected Guidance*（arXiv:2410.02416）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/adaptive_projected_guidance.py`：`normalized_guidance`（L211–253）给出「动量 → 截断 → 投影 → \(\eta\) 混合 → 外推」的完整次序，`MomentumBuffer`（L163–170）定义缓冲递推 `running = update + momentum · running`。该文件另有 `adaptive_projected_guidance_mix` 与 `AdaptiveProjectedGuidance` 的 `use_original_formulation` 分支，本题只取 diffusers 原生口径（以 \(\epsilon_u\) 为外推基准）。

## 核心与易错点

1. **平行方向用条件预测，不是差分自身**：\(\hat e=\epsilon_c/\lVert\epsilon_c\rVert\)。改成对 \(\delta\) 归一化就变成完全不同的算子。
2. **次序不能换**：动量先算（缓冲存的是截断前的 \(\delta\)），再截断，最后才投影。先截断后存缓冲会让下一步的动量状态被污染，判题专门检查这一点。
3. **外推基准是无条件预测**：`out = ε_u + s·update`。写成 `ε_c + s·update` 是最常见的错误，会让 \(\eta=1\) 的恒等断言直接失败。
4. **\(\eta\) 只缩放平行分量**：正交分量的系数恒为 1，所以 update 对 \(\eta\) 是线性的，\(\eta=0.5\) 恰为两端均值；\(\eta=1\) 时 `δ⊥+δ∥=δ`，整体退化成裸 CFG。
5. **截断是逐样本标量**：因子 `min(1, τ/‖δ‖)` 在除 batch 外的全部维度上归约。差分全零时范数为 0，靠 `clamp(max=1)` 让因子取 1——直接写 `τ/norm` 会得到 inf 再乘 0，产出 NaN。
6. **归一化的 eps 是 1e-12**：`F.normalize` 的默认值，条件预测为零向量时方向取零，整段差分全部算正交分量。

## 对拍口径

参考解在 float64 下向量化（`torch.linalg.vector_norm` + 广播内积），最后转 float32。判题 `_oracle` 先把 `dims` 重排到张量尾部、逐切片展平，再用 python 标量循环重算范数、单位向量与投影（不复用任何张量归约），可覆盖自定义 `dims`。结构断言四组：\(\eta=1\) 等于裸 CFG、\(\eta=0\) 的正交性与 \(\eta\) 线性、截断后更新范数不超过 \(\tau\) 且未超阈值样本逐位不变、动量递推与「截断不污染缓冲」。阈值 `allclose(atol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)，外加一次逐样本范数归约（也是 \(O(N)\)），另存一份同形状缓冲。
