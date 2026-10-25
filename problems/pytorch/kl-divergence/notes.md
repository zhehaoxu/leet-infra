# 详解：KL 散度（前向与反向）

## 出处
Kullback & Leibler, *On Information and Sufficiency*（Ann. Math. Statist. 1951）。定义 \(\mathrm{KL}(P\|Q)=\sum P\log(P/Q)\)，非对称、非负，满足 Gibbs 不等式。前向 KL（\(\mathrm{KL}(P\|Q)\)）是 MLE/变分下界的核心；反向 KL（\(\mathrm{KL}(Q\|P)\)）因 mode-seeking 特性被用于蒸馏与 VAE 后验近似。

## 核心与易错点
1. **非对称是本题考点**：必须同时返回两个方向，且它们一般不等。判题会断言 `forward_kl != reverse_kl`（在同一对 \(P,Q\) 下）。
2. **\(0\log 0=0\) 约定**：\(P_c=0\) 的项直接跳过，不能因 `log(0)` 产生 NaN/Inf。参考解用掩膜 `P>0` 只累加有效项。
3. **逐行求和再对 batch 取均值**：不是对整矩阵求和，而是每行一个 KL、再 `mean` over \(N\)。
4. **返回标量元组**：judge 支持 tuple 返回值，两个元素都是 0 维张量，逐元素 allclose 比较。

## 对拍口径
参考解 `_kl(P,Q)` 掩膜累加后 `.sum(-1).mean()`，`solve` 返回 `(_kl(P,Q), _kl(Q,P))`。判题 `_oracle` 纯 Python 三重循环逐元素 `a*log(a/b)`（仅 \(a>0\)）逐行累加、除以行数，结构不同、数值等价。另设「\(P==Q\) 双向都≈0」「非对称（fwd≠rev）」「example 数值」断言。

## 复杂度
\(O(N C)\)。
