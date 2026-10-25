# 详解：Adapter 瓶颈前馈与残差

## 出处
Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*（arXiv:1902.00751）。在每个 Transformer 子层（attention 后、FFN 后）各插一个 Adapter：降维 \(d\to r\) → 非线性 → 升维 \(r\to d\) → 残差。原文非线性用 ReLU，并可选在瓶颈后加 Dropout；本题去掉 Dropout 以保判题确定性。

## 核心与易错点
1. **降维在前、升维在后**：\(W_\text{down}:(r,d)\) 先把 \(d\) 压到 \(r\)，ReLU，\(W_\text{up}:(d,r)\) 再回到 \(d\)。形状顺序反了会维度不匹配。
2. **ReLU 只在瓶颈**：非线性夹在两次投影之间，不是作用在最终输出或残差上。
3. **残差加原输入**：\(y = u + x\)，要求 \(u\) 与 \(x\) 同 \(d\) 维——这正是 \(W_\text{up}\) 输出维必须等于 \(d\) 的原因。
4. **恒等起点**：若 \(W_\text{up}=0\)，则 \(y=x\)（旁路不改变主干），利于加载后不掉点。

## 对拍口径
参考解走「行向量 × 转置」批量化（\(xW_\text{down}^\top\)）；判题 `_oracle` 逐样本用列向量 \(W_\text{down}x_i\to\text{ReLU}\to W_\text{up}(\cdot)+x_i\)（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设 `W_up=0` 纯残差、ReLU 非负性（断言下投影负分量被置零）、3D 批量。

## 复杂度
每个 Adapter 参数 \(2rd\ll d^2\)；前向两次小矩阵乘 + 一次逐元素 ReLU + 一次加法。
