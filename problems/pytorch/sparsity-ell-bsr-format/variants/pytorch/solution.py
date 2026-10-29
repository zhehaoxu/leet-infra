"""参考答案：ELL（定长补齐）与 BSR（块粒度）转换 + 三格式字节账。"""

import torch


def solve(dense, block):
    r, c = dense.shape
    br, bc = block
    nz = dense != 0

    counts = nz.sum(dim=1)
    m = int(counts.max())
    order = torch.argsort((~nz).to(torch.int8), dim=1, stable=True)[:, :m]
    mask = torch.gather(nz, 1, order)
    ell_values = torch.gather(dense, 1, order) * mask.to(dense.dtype)
    ell_indices = torch.where(mask, order, torch.full_like(order, -1)).to(torch.long)

    nr, nc = r // br, c // bc
    blocks = dense.reshape(nr, br, nc, bc).permute(0, 2, 1, 3).reshape(nr, nc, br * bc)
    block_nz = blocks.reshape(nr, nc, br * bc).ne(0).any(dim=2)
    flat_nz = block_nz.reshape(-1)
    bsr_values = blocks.reshape(nr * nc, br * bc)[flat_nz]
    bsr_indices = torch.nonzero(block_nz)[:, 1].to(torch.long)
    rows = block_nz.sum(dim=1).to(torch.long)
    bsr_indptr = torch.cat([torch.zeros(1, dtype=torch.long), rows.cumsum(0)])

    nnz = int(nz.sum())
    nb = int(flat_nz.sum())
    csr_bytes = nnz * 4 + nnz * 8 + (r + 1) * 8
    ell_bytes = r * m * 4 + r * m * 8
    bsr_bytes = nb * br * bc * 4 + nb * 8 + (nr + 1) * 8
    nbytes = torch.tensor([csr_bytes, ell_bytes, bsr_bytes], dtype=torch.long)
    return ell_values, ell_indices, bsr_values, bsr_indices, bsr_indptr, nbytes
