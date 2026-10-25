# 详解：NovoGrad 逐层梯度归一

## 出处
Ginsburg, Brattoli, Bühlmann, et al., *Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training Deep Networks*（arXiv:1905.11286）。参考实现取 kozistr/pytorch-optimizer 的 `NovoGrad`（`weight_decouple=True`、`grad_averaging` 可选、`adam_debias=False`）。

## 与 Adam 的三点结构差异
NovoGrad 把 Adam 的“逐元素二阶矩 + 偏差修正”换成了“逐层标量归一 + 纯动量”，具体差异：

1. **二阶矩是一个标量**：对某个参数（一整层）先算 \(\lVert g\rVert^2=\sum_i g_i^2\)，对它做 EMA 得到 \(s\)（每层一个数，而非每元素一个数）。分母是 \(\sqrt{s}+\epsilon\)，对整层统一缩放。省显存、且天然做“层归一”。
2. **首步特殊**：库实现里，首次见到某参数时直接令 \(s=\lVert g\rVert^2\)（不是 \(0+(1-\beta_2)\lVert g\rVert^2\)），动量 \(m=\hat g\)（不是 \(\beta_1\cdot0+(1-\beta_1)\hat g\)）。之后才走正常 EMA/动量递推。
3. **没有偏差修正**：更新式 \(p\leftarrow p-lr\cdot m\)，步长就是 `lr`，不出现 \(1-\beta^t\)。\(\hat g\) 累加进动量时的系数默认是 \(1\)；只有 `grad_averaging=True` 才乘 \((1-\beta_1)\)（这是为对齐 Adam 语义提供的开关，默认关）。

## 权重衰减的顺序
论文主张 NovoGrad 与解耦权重衰减配合最好：先算归一化梯度 \(\hat g\)，再对参数做 \(p\leftarrow p(1-lr\cdot\text{wd})\)，最后用 \(\hat g\) 更新动量与参数。本题采用**解耦**形式（与论文推荐一致），衰减作用在 \(p\) 上、不污染梯度。

## 对拍口径
判题用独立的向量化重算做参照（`test_cases.py` 里的 `_oracle_novograd`）：逐元素用与参考解**不同的实现路径**（标量 `pow(2).sum()`、`b2*s+(1-b2)*g2` 代数式等），在多形状（含 2D 权重 + 1D 偏置）、多步随机梯度下 `allclose(atol=1e-5, rtol=1e-5)` 比对。`grad is None` 的参数连同权重衰减都必须跳过——专项用例覆盖。

## 复杂度
每参数每步 \(O(\text{numel})\)（多一次整层求和），额外状态仅一份动量 \(m\) 加每层一个标量 \(s\)，比 Adam 省一半显存。
