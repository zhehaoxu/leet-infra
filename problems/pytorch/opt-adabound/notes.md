# 详解：AdaBound 动态学习率夹逼

## 出处
AdaBound 论文：*AdaBound: Adaptive Bound Learning Rate Methods*，arXiv:1902.11010（微软，ICLR 2019）。算法骨架对齐 kozistr/pytorch_optimizer 的 `AdaBound`（`ams_bound=False, foreach=False` 的 per-param 路径，`weight_decouple=True`）。

## 一句话机制
Adam 的有效逐坐标学习率是 \(\eta_t/(\sqrt{\hat v}+\epsilon)\)。AdaBound 把它夹在区间 \([\alpha_t,\beta_t]\) 内，区间端点随 \(t\) 从两侧收敛到同一个 `final_lr`：早期区间极宽（等于 Adam），后期区间塌缩成一点（等于固定学习率 SGD）。

## 实现要点（易错点）
1. **夹的是「分母/步长」而非步长本身**：库先 `update = de_nom / step_size`（即 \(\text{denom}/\text{step}\)，是有效学习率的**倒数**），再 `clamp_(lower, upper)`，再 `*= m`，`p -= update`。等价于把逐坐标学习率限制在界内。别把 clamp 作用到 \(m\) 或最终位移上。
2. **动态界**：`lower = flr*(1 − 1/(γt+1))`（下界从 0 升向 flr），`upper = flr*(1 + 1/(γt))`（上界从 ∞ 降到 flr）。`flr = final_lr * lr / base_lr`；`base_lr` 是构造时的初始 lr（本题 lr 恒定，故 flr=final_lr）。
3. **数值稳定常数 1e-15**：库在开方前对 \(v\) 加 `exp_avg_sq_eps=1e-15`（即 \(\sqrt{v+1e^{-15}}\)），随后再加 `eps=1e-8`。两处不要漏。
4. **步长折进了分母**：`step_size = lr·sqrt(1−β2^t)/(1−β1^t)`，把两个偏差修正与 lr 合并；因为除进 `denom` 再夹逼，所以分母用的是**未修正**的 \(\sqrt{v+1e^{-15}}\)。
5. 解耦权重衰减**在矩更新之前**、且对每个有梯度的参数施加；`grad is None` 整体跳过。

## 对拍口径
外部参照用**与参考解相同的张量原语序列**（`lerp_ / mul_+addcmul_ / add(1e-15).sqrt_().add_(eps) / div_(step) / clamp_ / mul_(m) / sub_`）的独立向量化重算——保持同一条实现路径以消除融合算子舍入被 `/step_size` 放大带来的偏差（实测 60 seed ×25 步逐元素 maxdiff=0）。这是本合集唯一刻意对齐原语的题，因为 AdaBound 的夹逼会把 1e-8 级舍入放大到 1e-6 级，独立代数式重算会逼近判题容差边界。

## 复杂度
每参数每步 \(O(\text{numel})\)，额外两份状态 \((m, v)\)，与 Adam 同量级。
