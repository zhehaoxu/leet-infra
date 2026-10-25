# 详解：乘积量化编码

## 出处
Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*（IEEE TPAMI 2011，inria-00514462）。PQ 对 \(d\) 维子空间各建一套 \(2^b\) 大小码本、独立 VQ，编码即逐段取最近码字；Faiss `IndexPQ::compute_codes` 即此操作。

## 核心与易错点
1. **切段必须按码本的子空间划分**：第 \(j\) 段 = `x[:, j*d:(j+1)*d]` 只与 `codebook[j]`（形状 \((k_{sub}, d)\)）比较。用错码本（如全用 `codebook[0]`）是典型 bug。
2. **笛卡尔积而非单一码本**：整体近似误差是各段误差之**和**，这正是 PQ 用 \(m\cdot\log_2 k_{sub}\) 位达到 \(k=m\cdot k_{sub}^{m}\) 等效码本规模的原因。
3. **子码是「下标」不是「向量」**：输出为 \((n,m)\) 的整数索引。
4. 平方欧氏 + 并列取小下标，与 IVF 分配同源，只是逐段独立。

## 对拍口径
参考解逐段广播 `((sub[:,None,:]-cb[j][None,:,:])**2).sum(-1).argmin(1)` 填进 `(n,m)`。判题 `_oracle` 四重纯 Python 循环（样本×段×码字×维）显式累加平方差、严格 `<` 才更新（结构不同、数值等价）。另设「shape=(n,m)」「子码∈[0,k_sub)」「由已知码字拼回向量再编码应还原该码字（精确重建）」断言。

## 复杂度
\(O(n\cdot m\cdot k_{sub}\cdot d)\)。
