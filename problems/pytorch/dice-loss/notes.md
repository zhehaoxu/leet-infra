# 详解：Soft Dice 损失

## 出处
Dice 系数源于 Sørensen–Dice 相似度；Soft/可微 Dice 损失见 Milletari et al., *V-Net*（3D 医学分割，arXiv:1604.03265）与 MONAI `DiceLoss`、nnU-Net。常见两种分母：线性 \(\sum p+\sum y\)（本题）与平方 \(\sum p^2+\sum y^2\)（Dice Loss squared 变体）。题面已明确取线性形式并加平滑 \(s\)。

## 核心与易错点
1. **沿通道维聚合、不是逐元素**：Dice 是集合重叠度，先把每个样本的 \(C\) 维求和成交叠/存在量，再算比值；逐元素会退化成常数。
2. **分母是 \(|p|+|y|\) 而非并集**：\(\frac{2\sum py}{\sum p+\sum y}\) 对连续概率即 soft Dice，别写成 IoU 的并集分母。
3. **平滑项 \(s\) 同加分子分母**：全零样本不 NaN、loss→0。
4. **逐样本归约**：先每行一个 loss，再按 reduction 对 batch 归约，勿把整个 batch 混一起算一个 Dice（除非题目要求 dice 模式）。

## 对拍口径
参考解 `inter=(p*y).sum(-1)`、`denom=p.sum(-1)+y.sum(-1)`、`loss=1-(2inter+s)/(denom+s)`，按 reduction 归约。判题 `_oracle` 纯 Python 逐样本累加分子分母（结构不同、数值等价）。另设「完美预测 loss=0」「全零样本 loss=0（平滑兜底）」「正负完全相反 prob=[[1,0]] vs y=[[0,1]] 给 loss≈1」「reduction none/sum/mean 一致」断言。

## 复杂度
\(O(N C)\)。
