# PEFT：FourierFT 离散傅里叶基系数重建——算法详解

## 1. 公式与出处

FourierFT 出自 arXiv:2405.18201，属于「变换域稀疏适配」家族。它把权重更新 \(\Delta W\) 表示成少量 2D 频率分量的线性组合：

\[\Delta W[i,j] = \sum_{(u,v)\in\mathcal F} c_{u,v}\,\cos\!\left(2\pi\left(\frac{u\,i}{M} + \frac{v\,j}{N}\right)\right).\]

其中 \(M=d_\text{out},N=d_\text{in}\)，\(\mathcal F\) 是可学习频率集合，\(c_{u,v}\) 是可训练系数。合并方式：

\[W' = W_0 + \alpha\,\Delta W.\]

## 2. 实现细节

- 构造行坐标 `i = arange(M)[:, None]` 与列坐标 `j = arange(N)[None, :]`，广播到 \((M,N)\)。
- 对每个 \((u,v)\) 计算相位矩阵 \(\phi_{u,v}[i,j]=2\pi(u i/M + v j/N)\)。
- 累加 `coeff * cos(phase)` 到 \(\Delta W\)。
- 乘以 \(\alpha\) 后加回 \(W_0\)。

## 3. 退化与判题考点

- \(\alpha=0\)：\(\Delta W\) 被抹掉，\(W'=W_0\)。
- 空频率列表：\(\Delta W=0\)，\(W'=W_0\)。
- 仅 \((0,0)\) 频率：\(\cos\equiv 1\)，\(\Delta W\) 是均匀常数矩阵。
- 输出完全由输入 \(W_0,\text{freqs},\text{coeffs},\alpha\) 决定，无随机性。

## 4. 与现有题的分工

- `peft-loha-hadamard`：时域 Hadamard 积低秩更新。
- 本题：频域稀疏系数重建，强调从有限频率分量还原完整权重更新。
