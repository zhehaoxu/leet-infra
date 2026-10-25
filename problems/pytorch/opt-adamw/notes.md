# 详解：AdamW 解耦权重衰减

## 出处
Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019（arXiv:1711.05101）。参考实现即 PyTorch 原生 `torch.optim.AdamW`。

## 为什么要"解耦"
在原始 Adam 里做 \(L_2\) 正则，是把 \(\lambda p\) 加到梯度 \(g\) 上再走自适应更新，于是这一项也被 \(\sqrt{\hat v}\) 除过：本来想每步固定收缩参数，实际收缩幅度却被梯度二阶矩放大/缩小，正则强度和梯度噪声尺度耦合。AdamW 把衰减从"进梯度"改成"直接作用在参数上"：

\[ p \leftarrow p - lr\cdot\text{wd}\cdot p \;=\; p\,(1 - lr\cdot\text{wd}) \]

这一步发生在矩估计与自适应更新**之前**，与 \(m,v\) 完全无关，故称"解耦"。这也是为什么 \(\beta_2\) 很大、梯度很稀时 Adam 的 \(L_2\) 会失效，而 AdamW 不受影响。

## 与 torch.optim.AdamW 的逐步对齐（对拍口径）
评测用 `torch.optim.AdamW(..., amsgrad=False, foreach=False, fused=False)` 作外部参照。要对齐的三个易错点：

1. **顺序**：先 `p.mul_(1 - lr*wd)`，再更新 \(m,v\)，最后 `p.addcdiv_`。
2. **偏差修正写法**：PyTorch 把更新写成
   \[ p \leftarrow p - \frac{lr}{1-\beta_1^{t}}\cdot\frac{m}{\sqrt{v}/\sqrt{1-\beta_2^{t}}+\epsilon} \]
   注意分母是 \(\sqrt{v}/\sqrt{1-\beta_2^t}\)（把 \(\hat v\) 的除法放进分母里），\(eps\) 加在这一步之后——**不是**先各自算 \(\hat m,\hat v\) 再 \(eps\) 直接加 \(\sqrt{\hat v}\)。两种写法数学等价但浮点落点不同，为逐元素一致须照抄 PyTorch 的算子顺序：`m/(v.sqrt()/sqrt(bc2) + eps)`。
3. **步计数**：每个参数独立从 1 开始（首步 \(bc_1=1-\beta_1\)），与是否稀疏无关。

`grad is None` 的参数在 PyTorch 里直接 `continue`，权重衰减也不施加——评测有专项用例。

## 复杂度
每参数每步 \(O(\text{numel})\)，额外状态两份（\(m,v\)），与 Adam 相同。
