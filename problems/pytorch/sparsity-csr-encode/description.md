# 稀疏化：稠密矩阵编码为 CSR

## 题目描述

剪枝后网络里绝大多数权重为 0，若仍按稠密布局存储会浪费显存与带宽。**CSR（Compressed Sparse Row）** 是稀疏矩阵的标准压缩存储：只保存非零元素，用三个数组描述——

- `values`：所有非零元素，按**行主序**（逐行、行内逐列）排列；
- `indices`：每个非零元素对应的**列下标**；
- `indptr`：长度 \(R+1\) 的行指针，`indptr[i]:indptr[i+1]` 是第 \(i\) 行非零元素在 `values`/`indices` 中的切片，`indptr[0]=0`、`indptr[R]=nnz`。

## 规则

输入 `dense`（float32，2-D \((R,C)\)，「非零」定义为 `!= 0.0`）。返回三元组 `(indptr, indices, values)`：`indptr`、`indices` 为 int64 张量，`values` 为 float32。非零元素严格按行主序展开。禁止直接返回 `dense.to_sparse_csr()`——需手工构造三数组。

## 示例

`dense=[[1,0,0,2],[0,0,3,0],[4,5,0,0]]`：

```
行0 非零：列0=1, 列3=2
行1 非零：列2=3
行2 非零：列0=4, 列1=5
indptr  = [0, 2, 3, 5]        # 每行结束处的累计 nnz
indices = [0, 3, 2, 0, 1]
values  = [1, 2, 3, 4, 5]
```

## 约束

- CPU float32；`indptr` 单调不减、首 0 尾 = nnz；`indices` 每行内列升序（由行主序扫描自然得到）。
- 全零矩阵：`indptr=[0,0,...,0]`、`indices`/`values` 为空。
- 三数组长度关系：`len(indptr)=R+1`、`len(indices)=len(values)=nnz`。
