# 详解：稠密矩阵编码为 CSR

## 出处
CSR（Compressed Sparse Row，行压缩存储）是 scipy/PyTorch `to_sparse_csr`/cuSPARSE 的通用稀疏矩阵格式：三数组 `indptr/indices/values`。剪枝后模型的实际存储与稀疏 GEMM 都建立在它之上。

## 核心与易错点
1. **行主序**：`values`/`indices` 必须逐行、行内逐列展开——`torch.nonzero` 天然按此序返回。
2. **`indptr` 长 R+1、是累计 nnz**：`indptr[i]` 是第 \(i\) 行之前的非零总数，`indptr[i+1]-indptr[i]` 即第 \(i\) 行 nnz；首 0 尾 = 总 nnz。构造时最易漏掉末尾那格。
3. **空行**：某行全零时 `indptr` 相邻两值相等，不能跳过。
4. **dtype**：`indptr`/`indices` 是 int64、`values` 跟原矩阵 float32；判题按 dtype 断言。
5. 「非零」以 `!= 0.0` 严格判定，-0.0 视作 0。

## 对拍口径
参考解 `nonzero→counts→cumsum` 向量化拼三数组。判题 `_oracle` 纯 Python 双循环逐元素扫描手工累加（结构不同、结果等价），逐一对拍 `indptr/indices/values`（含 `tolist()` 相等 + values `allclose`）。另设「长度关系 len(indptr)=R+1、len(indices)=len(values)=nnz」「全零矩阵退化」「用三数组能还原原矩阵」断言。

## 复杂度
\(O(RC)\) 扫描、存储 \(O(\text{nnz}+R)\)。
