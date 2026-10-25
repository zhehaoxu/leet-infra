# 详解：KTO 前景理论损失

## 出处
Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*（arXiv:2402.01306）。价值函数相对参考点 \(z_0\) 定义：desirable 取 \(\sigma(\beta(z-z_0))\)、undesirable 取 \(\sigma(\beta(z_0-z))\)，损失为 \(\lambda_y(1-v)\)。参考点 \(z_0\approx\mathrm{KL}(\pi_\theta\|\pi_\text{ref})\) 论文里在批内估计并 detach；本题把它作为显式常量入参给定，聚焦损失本体。

## 核心与易错点
1. **两方向相反**：好样本奖励 \(z>z_0\)（价值随 \(z-z_0\) 升），坏样本惩罚 \(z>z_0\)（价值随 \(z_0-z\) 升）。方向写反会让坏样本被鼓励。
2. **损失是 1−σ，不是 −log σ**：KTO 直接最小化「未达成的前景价值」\(1-\sigma(\cdot)\)（前景理论的价值线性加权），与 DPO 的 logsigmoid 不同。
3. **λ 权重**：\(\lambda_D,\lambda_U\) 缓解好/坏样本数量不平衡；相等时退化等权。
4. \(z=z_\text{ref}\) 时 \(\sigma(0)=0.5\) → 逐样本损失恰 \(\lambda/2\)，是判方向的探针。

## 对拍口径
参考解向量化 `torch.where(label, λ_D*(1-sigmoid(β(z-z_ref))), λ_U*(1-sigmoid(β(z_ref-z)))).mean()`。判题 `_oracle` 用恒等式 \(1-\sigma(x)=\sigma(-x)\) 把两支改写成 `sigmoid` 的负参数、逐元素 Python 循环重算（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「全样本 z==z_ref → 损失=λ/2」「label 方向」断言。

## 复杂度
逐元素 \(O(N)\)。
