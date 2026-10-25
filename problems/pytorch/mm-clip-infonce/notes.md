# 详解：CLIP 对称 InfoNCE 对比损失

## 出处
Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*（CLIP, arXiv:2103.00020）：对称 InfoNCE / NT-Xent，图像→文本与文本→图像两方向各做以对角为正样本的交叉熵后取平均，温度用可学习 `logit_scale`。

## 核心与易错点
1. **对称 = 两个方向都算 CE**：\(\tfrac12[\mathrm{CE}(s\,IT^\top,y)+\mathrm{CE}(s\,TI^\top,y)]\)。只算一个方向会低估且不对称。
2. **labels 恒为 arange(N)**：对角线是正对，第 \(i\) 行正确类别就是第 \(i\) 列。
3. **温度乘在 logits 上**（`logit_scale` 常≈exp(learned)≈1/τ），越大 softmax 越尖。别把 scale 加在损失外面。
4. 交叉熵内部含 logsumexp，数值稳定由 `F.cross_entropy` 保证。

## 对拍口径
参考解 `0.5*(F.cross_entropy(scale*I@T.T, arange)+ 同 t2i)`。判题 `_oracle` 用纯 Python 逐行 max-shift + `logsumexp` 手算 softmax NLL、两方向分别求均值再平均（结构不同、float64 累加，数值等价），`abs(got-exp)<1e-5`。另设「完全一致特征+大温度→损失趋 0」「对称标量」「identity 相似度」断言。

## 复杂度
矩阵乘 \(O(N^2 D)\) + softmax \(O(N^2)\)。
