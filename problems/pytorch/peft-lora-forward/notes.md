# 详解：LoRA 低秩适配前向

## 出处
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*（arXiv:2106.09685）。核心假设：微调时权重更新 \(\Delta W\) 具有低「内在秩」，故用 \(\Delta W = B A\)（\(B:d_\text{out}\times r\)、\(A:r\times d_\text{in}\)）参数化，训练时冻结 \(W_0\)、只更新 \(A,B\)。缩放因子论文取 \(\alpha/r\)（原文对 \(r\) 固定时常用 \(\alpha=2r\) 即系数恒定，但工程上普遍按 \(\alpha/r\) 显式缩放）。初始化 \(A\sim\mathcal N(0,\sigma^2)\)、\(B=0\)，故训练起点 \(\Delta W=0\)、不破坏预训练。

## 为什么这样写（易错点）
1. **缩放 \(s=\alpha/r\) 只乘旁路**：\(y_0=xW_0^\top\) 不含 \(s\)，只有低秩项 \(y_\Delta=s\,(xA^\top)B^\top\) 乘 \(s\)。忘了 \(s\) 或把它也乘到基座上都会错。
2. **矩阵顺序**：\(\Delta W = B A\)（先 \(A\) 降维、再 \(B\) 升维），形状 \((d_\text{out},d_\text{in})\)。写成 \(A B\) 形状就对不上。
3. **行向量惯例**：本合集沿用 `nn.Linear` 的 \(y=xW^\top\)。故旁路是 \((x A^\top) B^\top\)，不是 \(x A B\)。
4. **合并等价性**：\(y = x(W_0+\frac{\alpha}{r}BA)^\top\) 与分步相加逐元素相等——这是「LoRA 可折叠、部署零延迟开销」的根据，也是本题 oracle 的结构差异点（oracle 走合并矩阵、参考解走分步两矩阵乘）。

## 对拍口径
参考解走「基座 + 分步旁路」两步矩阵乘；判题 `_oracle` 走「先合并 \(W'=W_0+sBA\) 再一次 \(xW'^\top\)」——结构不同但数学等价，`allclose(atol=1e-5,rtol=1e-5)` 比对。检查覆盖：示例值、随机权重对拍、`alpha=0` 退化为纯基座、3D 批量、与合并矩阵等价。

## 复杂度
前向额外 \(O(n\cdot r\cdot(d_\text{in}+d_\text{out}))\)（远小于稠密 \(\Delta W\) 的 \(O(n\,d_\text{in}d_\text{out})\)）；可训练参数 \(r(d_\text{in}+d_\text{out})\)。
