# 详解：LoRA 可训练参数量与显存账

## 出处
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*（arXiv:2106.09685）。核心卖点是把每个权重矩阵的可训练量从 \(d_\text{in}\cdot d_\text{out}\) 降到 \(r(d_\text{in}+d_\text{out})\)，\(r\) 常取 4/8/16，远小于隐藏维。

## 核心与易错点
1. **满秩 vs 低秩**：\(W_0\) 的参数量是「面积」\(d_\text{in}\cdot d_\text{out}\)；LoRA 是两个「边长」乘 \(r\)：\(A:r\cdot d_\text{in}\)、\(B:d_\text{out}\cdot r\)，合计 \(r(d_\text{in}+d_\text{out})\)。
2. **省的是可训练参数**：基座权重仍需**前向**（占显存），LoRA 减少的是要更新/存优化器状态的参数，不是总权重显存。本题只清点「可训练」两侧。
3. **压缩比随维度增长**：单层比值 \(=r(d_\text{in}+d_\text{out})/(d_\text{in}d_\text{out})\)，隐藏维越大越趋近 0，故大模型上 LoRA 相对更省。
4. **多层累加**：不同层维度不同，需逐层求和后再相除得全局 ratio，不能对每层 ratio 取平均。

## 对拍口径
参考解直接两遍累加 `full += din*dout`、`lora += r*(din+dout)`。判题 `_ref` 用列表推导式分别求和（结构不同），`full`/`lora` 用整数 `==` 比较，`ratio` 用 `abs(...-...) < 1e-9`。另设单恒等层（\(r=\min\) 时不省）、\(r=1\) 下界、以及全局 ratio 是求和之比而非逐层均值。

## 复杂度
\(O(L)\) 遍历 \(L\) 层，纯整数运算，无张量。
