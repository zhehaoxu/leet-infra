# 详解：剪枝微调的 STE 与软阈值重参数化反向

## 出处
STE 在离散随机单元上的用法由 Bengio, Léonard & Courville（*Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*, 2013, arXiv 1308.3432）系统讨论；软阈值重参数化是 WSN（Kusupati et al., *Soft Threshold Weight Reparameterization for Learnable Sparsity*, ICML 2021, arXiv 2002.10509）：\(w\cdot\mathrm{clamp}(|w|/\tau,0,1)\)，阈值 \(\tau\) 变成可学习参数，训练中自动收敛到目标稀疏度。

## 核心与易错点
1. **STE 的梯度是「乘掩码」而非「乘 1」**：直通估计的名字容易误导，被剪位置必须是 0 梯度（否则被剪权重会被重新拉大、剪枝失效）；只有保留位置才直通。
2. **`hard_ste` 的等号**：\(|w|=\tau\) 属保留侧，这是训练/推理一致性的关键边界；写成 `>` 会让阈值点抖动。
3. **WSN 的梯度多一项**：\(y=w\cdot m(w)\) 是两个都依赖 \(w\) 的因子相乘，链式法则给出 \(m+w\,\mathrm{d}m/\mathrm{d}w\)。对 \(|w|<\tau\) 有 \(\mathrm{d}m/\mathrm{d}|w|=1/\tau\)、\(\mathrm{d}|w|/\mathrm{d}w=\mathrm{sign}(w)\)，两项合起来恰好是 \(2|w|/\tau\)；漏掉第二项就退化成「STE 但掩码是软的」，梯度会偏小一半。
4. **\(\tau\) 的正负**：\(\tau\le0\) 时 \(m\equiv1\) 没有稀疏意义，题面限定 \(\tau>0\)。
5. **只给上游梯度、不开 autograd**：这正是真实训练框架里手写 backward 的形态（自定义 autograd.Function 的 `backward`），先把标量式推清楚再落代码。

## 对拍口径
参考解按 `mode` 分支算掩码与梯度并返回 `(w*m, grad, m)`。判题 `_oracle` 用纯 Python 逐元素按定义重算三条输出并逐位对拍。额外断言：`hard_ste` 被剪位置梯度严格为 0、保留位置梯度等于上游值、等号边界保留；`wsn` 的梯度用 `torch.autograd`（float64）对 **同一公式** 求导交叉验证（这是唯一允许出现 autograd 的地方，属于判题侧独立校验）；`wsn` 掩码在 \(|w|\ge\tau\) 饱和为 1；两种模式下前向都满足 \(y=w\odot m\)。

## 复杂度
\(O(N)\) 逐元素；显存 \(O(N)\)。
