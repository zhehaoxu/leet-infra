"""参考解：Lexico 的 KV 稀疏字典编码 = 逐行 OMP + CSR 打包。

上游 lexico/omp.py::omp_v0 用 Gram 矩阵 + Cholesky 增量更新做同一件事（数学上仍是「每轮
对整个已选原子集合重解最小二乘」，见其 F/a_F 变量），omp() 再按 lengths 摊成 CSR 三元组；
cache_utils.py::_compress 把 KV 按 head_dim 合并成行后调用它（max_sparsity=14）。
这里按最直白的口径逐行显式重解，清晰优先于取巧。
"""

import torch


def solve(vectors, dictionary, n_atoms, tol=None):
    if not torch.is_tensor(vectors) or vectors.dim() != 2:
        raise ValueError("vectors 必须是二维 (N, D)")
    if not torch.is_tensor(dictionary) or dictionary.dim() != 2:
        raise ValueError("dictionary 必须是二维 (D, M)")
    for name, t in (("vectors", vectors), ("dictionary", dictionary)):
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是 float32/float64")
    if vectors.dtype != dictionary.dtype:
        raise ValueError("vectors 与 dictionary 的 dtype 必须一致")
    if dictionary.shape[0] != vectors.shape[1]:
        raise ValueError("dictionary 的行数必须等于 vectors 的列数 D")
    m = dictionary.shape[1]
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, int) or n_atoms < 1:
        raise ValueError("n_atoms 必须是正整数")
    if n_atoms > m:
        raise ValueError("n_atoms 不能超过原子个数 M")
    if tol is not None:
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0.0 <= float(tol) < 1.0:
            raise ValueError("tol 必须是 None 或落在 [0, 1) 的浮点数")

    n, d = vectors.shape
    out_dtype = vectors.dtype
    sig = vectors.to(torch.float64)
    dic = dictionary.to(torch.float64)
    tol2 = None if tol is None else float(tol) ** 2

    lengths = []
    picks_all, weights_all = [], []
    recon = torch.zeros((n, d), dtype=torch.float64)
    residual = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        y = sig[i]
        initial = float(y @ y)
        r, picked = y.clone(), []
        w = torch.zeros(0, dtype=torch.float64)
        for _k in range(n_atoms):
            if tol2 is not None and float(r @ r) <= tol2 * initial:    # 先判停，再选原子
                break
            rest = [j for j in range(m) if j not in picked]
            corr = (dic[:, rest].T @ r).abs()
            picks = picked + [rest[int((corr == corr.max()).nonzero()[0])]]  # 同分取小下标
            basis = dic[:, picks]
            w = torch.linalg.lstsq(basis, y.unsqueeze(1)).solution.squeeze(1)
            r = y - basis @ w
            picked = picks
        lengths.append(len(picked))
        picks_all.extend(picked)
        weights_all.extend(w.tolist())
        recon[i] = y - r
        residual[i] = float(r.norm())

    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    indptr = torch.zeros(n + 1, dtype=torch.int64)
    indptr[1:] = lengths_t.cumsum(0)
    return {
        "lengths": lengths_t,
        "csr_indptr": indptr,
        "csr_indices": torch.tensor(picks_all, dtype=torch.int64),
        "csr_values": torch.tensor(weights_all, dtype=out_dtype),
        "residual_norm": residual.to(out_dtype),
        "recon": recon.to(out_dtype),
    }
