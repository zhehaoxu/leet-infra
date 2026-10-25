# 详解：LoRA 权重合并（merge_and_unload）

## 出处
Hu et al., arXiv:2106.09685；API 名 `merge_and_unload` 来自 huggingface/peft。因 \(\Delta W=BA\) 与 \(W_0\) 同形，可无损折叠：\(W'=W_0+\frac{\alpha}{r}BA\)，此后推理不再需要 \(A,B\)。

## 核心与易错点
1. **缩放只作用在 \(\Delta W\)**：\(W'=W_0+s\,(BA)\)，\(s=\alpha/r\)。别把 \(s\) 也乘到 \(W_0\)。
2. **顺序 \(BA\)**（\(d_\text{out}\times r\) 乘 \(r\times d_\text{in}\)），结果 \((d_\text{out},d_\text{in})\) 与 \(W_0\) 对齐；\(AB\) 会得 \((r,r)\) 直接广播错。
3. **折叠可逆性有限**：合并后无法精确拆回 \(A,B\)（数值上 \(W'-W_0\) 就是那个低秩 \(\Delta W\)），本题只考正向合并。

## 与「LoRA 前向」题的分工
前向题（order 199）考「不合并、分步旁路相加」；本题考「合并成稠密 \(W'\)」。二者对同一 \(x\) 输出逐元素相等——本题用 `check_apply_equivalence` 现场验证 \(x@W'^\top =\) 分步前向，把两题的数学一致性钉死。

## 对拍口径
参考解显式算 \(W_0+s(BA)\)；判题 `_oracle` 用「逐元素 \(W_0[i,j]+\frac{\alpha}{r}\sum_k B[i,k]A[k,j]\)」（三重循环结构、与矩阵乘不同）比对，`allclose(atol=1e-5,rtol=1e-5)`。另设 `alpha=0` 恒等、形状一致、\(x@W'^\top\) 与分步前向等价等断言。

## 复杂度
合并 \(O(d_\text{in}d_\text{out}r)\)、产出一整份 \(d_\text{out}\times d_\text{in}\) 权重（这正是「部署换显存、训练省显存」的取舍）。
