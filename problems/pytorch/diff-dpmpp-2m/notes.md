# 详解：DPM-Solver++(2M) 二阶更新

## 出处

Lu et al., *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*（arXiv:2211.01095）给出 data-prediction 形式与 multistep 二阶；diffusers 的 `DPMSolverMultistepScheduler`（`algorithm_type="dpmsolver++"`）就是这条式子，`solver_order=1` 即 Euler。

## 核心与易错点

1. **λ 是半个 log-SNR**：\(\lambda=\log(\alpha/\sigma)\)。用 `log(α/σ)` 与用 `log(ᾱ/(1-ᾱ))/2` 数值上一致，但别和 `log(ᾱ/(1-ᾱ))` 混用——差一倍会让指数项整体偏掉。
2. **系数符号**：实现常写成 \(\alpha_t(1-e^{-h})\)，与 \(- \alpha_t(e^{-h}-1)\) 等价；写成 \(1-e^{h}\) 就发散。因为 \(h>0\)，\(1-e^{-h}\in(0,1)\)，系数恒正。
3. **一阶就是 DDIM**：\(\alpha_tD_0+\sigma_t\,(x-\alpha_{s_0}D_0)/\sigma_{s_0}\)，两条式子代数等价。这是判断系数是否写对的最强探针。
4. **二阶只是加 0.5·D1**：\(D_1=(D_0-D_0^{\text{prev}})/r_0\)，所以「二阶 - 一阶 = 系数·0.5·D1」可以脱离两条完整公式单独验算；\(D_1=0\) 时必须退化。
5. **r₀ 的方向**：\(r_0=\lambda_{s_0}-\lambda_{s_1}>0\) 要求 \(s_1\) 更噪。写反会让二阶修正符号整体翻转，采样直接崩。

## 对拍口径

参考解向量化（float64）出结果。判题 `_oracle` 用 sigmoid 化的 λ 写法（`0.5*log(ab/(1-ab))`）逐元素重算，与参考解的 `log(α/σ)` 不同源；另加四组结构性断言：与 DDIM 闭式逐位一致、二阶差分等于 \(coef\cdot 0.5\cdot D_1\)、同预测退化、\(h=0\) 恒等与 \(\bar\alpha_t\to1\) 收敛（用 \(\bar\alpha_t=0.9999999999\) 取极限）。阈值 `allclose(atol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)；多步采样时整个轨迹是 \(O(\text{steps}\cdot N)\)。
