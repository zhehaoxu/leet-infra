import torch


def solve(dense):
    """把稠密 2-D 矩阵编码为 CSR：返回 (indptr, indices, values)（行主序）。
    indptr/indices 为 int64、values 为 float32；indptr 长 R+1、indices/values 长 nnz。
    非零=dense!=0.0。禁止直接返回 dense.to_sparse_csr()。"""
    # TODO
    raise NotImplementedError
