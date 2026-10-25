# 详解：DoRA 幅值-方向分解低秩适配

## 出处
Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*（arXiv:2402.09353）。把权重按 Weight Normalization 的思路拆成「幅值 \(m\)」与「方向 \(\hat V\)」两块：\(W_\text{eff}=m\odot \hat V\)，其中方向 \(\hat V=V/\lVert V\rVert\)、而 \(V=W_0+sBA\) 沿用 LoRA 的低秩合成。仅 \(m\) 与 \(A,B\) 可训练，\(W_0\) 冻结。

## 核心与易错点
1. **范数沿 dim=1（输出行）**：DoRA 对权重的每一行（对应一个输出神经元）单独归一化，\(n\) 形状 \((d_\text{out},1)\)。沿错维度会破坏退化性质。
2. **先合成再归一**：\(V\) 已经把 LoRA 增量加进去了，方向 \(\hat V\) 是对 \(V\) 而非 \(W_0\) 取范数。
3. **幅值广播**：\(m:(d_\text{out},)\) 要 `unsqueeze(1)` 成 \((d_\text{out},1)\) 才能沿输入维正确乘到每一行。
4. **退化验证**：取 \(m=\lVert V\rVert_\text{row}\) 时 \(W_\text{eff}=V\)，DoRA==合并 LoRA 前向；\(\alpha=0\) 且 \(m=\lVert W_0\rVert\) 时回到基座。这两个不变量是判对方向/广播是否写反的探针。

## 对拍口径
参考解批量：\(V=W_0+s(BA)\)、\(n=\text{norm}(V,\text{dim}=1)\)、\(W_\text{eff}=(V/n)\cdot m[:,None]\)、\(y=xW_\text{eff}^\top\)。判题 `_oracle` 显式逐行循环 \(n_i=\sqrt{\sum_j V_{ij}^2}\)、\(W_{ij}=m_i V_{ij}/n_i\)（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设两个退化不变量与 3D 批量。

## 复杂度
相比 LoRA 仅多一个 \((d_\text{out},)\) 幅值向量；每步多一次逐行范数与一次归一化除法。
