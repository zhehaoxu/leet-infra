# 详解：SigLIP Sigmoid 对比损失

## 出处
Zhai et al., *Sigmoid Loss for Language-Image Pre-Training*（arXiv:2303.15343）式 (2)：\(\mathcal{L}=-\frac1B\sum_i\sum_j\log\sigma(y_{ij}(t\,s_{ij}+b))\)，对角 \(y=+1\)、其余 \(-1\)，\(t,b\) 为可学习 scale/bias。逐对独立、无 softmax 全局归一。

## 核心与易错点
1. **是 sigmoid 不是 softmax**：每个 pair 独立二分类，不存在「按行归一」，所以不需要跨设备 gather 整批相似度。
2. **分母是 \(B\) 不是 \(B^2\)**：对全部 \(B^2\) 个 pair 求和后除以 batch 大小 \(B\)（论文伪码如此）；写成平均 over \(B^2\) 会差一个 \(B\) 因子。
3. **label 用 ±1**：\(y_{ij}=2\mathbb{1}[i=j]-1\)，把负对的 \(z\) 也翻成「越大越好」。等价写法是 \(y\odot z\) 再过 `logsigmoid`。
4. **仿射先于 sigmoid**：\(z=t\,s+b\)，别把 bias 漏掉、也别在 loss 外乘。

## 对拍口径
参考解 `-(F.logsigmoid((2*eye-1)*(scale*sim+bias)).sum())/B`。判题 `_oracle` 用 Python 双循环逐 pair `log σ(m)=-log1p(exp(-m))` 稳定式累加、除以 B（结构不同、数值等价），`abs(got-exp)<1e-5`。另设「对角极大、非对角极小 → 损失趋 0」「scale=0、bias 定值时的闭式」「方阵形状/标量返回」断言。

## 复杂度
\(O(B^2)\) 逐对。
