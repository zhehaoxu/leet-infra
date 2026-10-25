# 详解：IA3 逐元素激活重标定

## 出处
Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*（arXiv:2205.05638，(IA)³）。论文在 self-attention 的 K、V 与 FFN 中间激活上各乘一个可学习向量，冻结其它一切。本抽象到单层线性输出后的整体重标定，抓住「乘性缩放激活」这一核心算子。

## 核心与易错点
1. **作用在输出维**：\(\lambda\in\mathbb R^{d_\text{out}}\)，\(y=\lambda\odot(xW^\top)\)，即在最后维广播；不是作用在输入维。
2. **广播要跨 batch 与序列**：对 \((b,l,d_\text{out})\) 也按最后维乘同一个 \(\lambda\)。用 `h * lam`（自动广播）即可；若手写循环别把 `lam` 对齐到 `d_in`。
3. **恒等初始化**：\(\lambda=\mathbf 1\) 不改动预训练；\(\lambda=0\) 直接置零该输出维（抑制）。
4. **与折权重的等价**：\(\lambda\odot(xW^\top) = x(\lambda_\text{row}\odot W)^\top\)（把 \(\lambda\) 乘到 \(W\) 的每一行）。故 IA³ 也可 fold 进权重部署。

## 对拍口径
参考解先算 \(h=xW^\top\) 再乘 \(\lambda\)；判题 `_oracle` 走「把 \(\lambda\) 乘进权重行得 \(W'=\lambda[:,\text{None}]\odot W\)，再一次 \(xW'^\top\)」——结构不同、数学等价，`allclose(atol=1e-5,rtol=1e-5)`。另设 \(\lambda=\mathbf1\) 恒等、\(\lambda=0\) 全零、3D 批量断言。

## 复杂度
每层新增可训练参数仅 \(d_\text{out}\)（比 LoRA 的 \(r(d_\text{in}+d_\text{out})\) 更少）；前向一个额外逐元素乘 \(O(n\,d_\text{out})\)。
