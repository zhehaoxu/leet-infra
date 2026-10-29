import torch


def solve(dense, block):
    """把稠密矩阵转成 ELL 与 BSR，并给出 CSR/ELL/BSR 三种格式的存储字节数。
    dense float32 (R,C)；block=(br,bc)，要求 R%br==0、C%bc==0。非零 = != 0.0。
    返回 (ell_values, ell_indices, bsr_values, bsr_indices, bsr_indptr, nbytes)：
      ELL：ell_values float32 (R,M)、ell_indices int64 (R,M)，M = 各行非零数的最大值；
           行内按列升序填，尾部值补 0、索引补 -1；
      BSR：块粒度 (br,bc)，只存「块内至少一个非零」的块，按块行主序；
           bsr_values float32 (NB, br*bc)（块内行主序展平）、bsr_indices int64 (NB,) 块列号、
           bsr_indptr int64 (R//br+1,) 为块行指针；
      nbytes int64 (3,) = [CSR, ELL, BSR] 的字节数（值 float32 记 4B、索引记 8B）。
    禁止 torch.sparse / to_sparse_csr / to_sparse_bsr。"""
    # TODO
    raise NotImplementedError
