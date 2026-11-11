"""862 reference solution: torchaudio MVDR beamforming ledger.

Pure Python (no torch / numpy / scipy).  Complex values cross the boundary as
``[re, im]`` pairs; internally they are Python ``complex`` objects.  The
linear solves that the official code performs with ``torch.linalg.solve`` are
reproduced with complex LU (partial pivoting) + back substitution.

Modes
-----
* ``psd``       cross-channel PSD via outer products + mask normalisation
* ``trace``     batched complex matrix trace
* ``tik_reg``   Tikhonov loading (real diagonal only)
* ``souden``    Souden MVDR weights (solve + complex-trace normalisation)
* ``rtf``       RTF MVDR weights (solve + .real denominator + optional scale)
* ``rtf_power`` RTF estimation by the power method (n_iter schedule)
* ``beamform``  conjugate einsum applying the weights

``rtf_evd`` is intentionally excluded from the judged domain: ``linalg.eigh``
eigenvector phase/sign is LAPACK-implementation-defined, so a RTF is always
passed explicitly instead of being estimated by an eigendecomposition.
"""


# ------------------------------------------------------------------ validation
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_pair(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v)


def pack(x, name="value"):
    """[re, im] nested lists -> nested python complex (floats stay floats)."""
    if _is_pair(x):
        return complex(x[0], x[1])
    if isinstance(x, (list, tuple)):
        return [pack(v, name) for v in x]
    if _is_num(x):
        return float(x)
    raise ValueError(name + " must be rectangular")


def unpack(z):
    if isinstance(z, complex):
        return [z.real, z.imag]
    if isinstance(z, (list, tuple)):
        return [unpack(v) for v in z]
    if isinstance(z, bool):
        return z
    if _is_num(z):
        return z
    return z


def _shape(x):
    if isinstance(x, (list, tuple)):
        return [len(x)] + (_shape(x[0]) if x else [])
    return []


def cshape(x):
    """Logical shape of a complex value encoded as nested [re, im] pairs."""
    if _is_pair(x):
        return []
    if isinstance(x, (list, tuple)):
        return [len(x)] + (cshape(x[0]) if x else [])
    return []


def _need_cx(x, name, ndim):
    s = cshape(x)
    if len(s) < ndim:
        raise ValueError(name + " must be at least %dD" % ndim)
    # every leaf must be a [re, im] pair
    def check(v):
        if isinstance(v, (list, tuple)):
            if _is_pair(v):
                return
            for q in v:
                check(q)
        else:
            raise ValueError(name + " must be rectangular")
    check(x)
    return s


def _need_square(x, name):
    s = cshape(x)
    if s[-1] != s[-2]:
        raise ValueError(name + " last two dimensions should be the same")
    return s


def _need_same(a, b, na, nb):
    if cshape(a) != cshape(b):
        raise ValueError("The dimensions of %s and %s should be the same" % (na, nb))


def _rc(x):
    """Return (branch, int-or-list): int -> int branch; complex list -> tensor."""
    if isinstance(x, int) and not isinstance(x, bool):
        return "int", x
    if isinstance(x, (list, tuple)):
        return "tensor", pack(x, "reference_channel")
    raise TypeError('Expected "int" or "Tensor" for reference_channel')


# ------------------------------------------------------------------ complex linear algebra
def _solve(A, B):
    """Solve A X = B for a batch of small complex square systems.

    A: (F, C, C) complex, B: (F, C, K) complex -> X: (F, C, K) complex.
    Complex LU with partial pivoting + back substitution, mirroring
    ``torch.linalg.solve`` to ~1e-12 for well-conditioned systems.
    """
    F = len(A)
    C = len(A[0])
    K = len(B[0][0])
    out = []
    for f in range(F):
        M = [[A[f][i][j] for j in range(C)] + [B[f][i][j] for j in range(K)]
             for i in range(C)]
        for col in range(C):
            piv = max(range(col, C), key=lambda r: abs(M[r][col]))
            if M[piv][col] == 0:
                raise ValueError("singular matrix in linalg.solve")
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            p = M[col][col]
            for r in range(col + 1, C):
                factor = M[r][col] / p
                if factor == 0:
                    continue
                M[r][col] = 0j
                for c in range(col + 1, C + K):
                    M[r][c] -= factor * M[col][c]
        X = [[0j] * K for _ in range(C)]
        for i in range(C - 1, -1, -1):
            for k in range(K):
                s = M[i][C + k]
                for j in range(i + 1, C):
                    s -= M[i][j] * X[j][k]
                X[i][k] = s / M[i][i]
        out.append(X)
    return out


def _matmul(A, B):
    """(F,C,C) @ (F,C,K) -> (F,C,K)."""
    F, C = len(A), len(A[0])
    K = len(B[0][0])
    out = []
    for f in range(F):
        rows = []
        for i in range(C):
            row = []
            for k in range(K):
                acc = 0j
                for j in range(C):
                    acc += A[f][i][j] * B[f][j][k]
                row.append(acc)
            rows.append(row)
        out.append(rows)
    return out


def _trace(mats):
    """(F,C,C) -> (F,) complex trace."""
    return [sum(mat[i][i] for i in range(len(mat))) for mat in mats]


# ------------------------------------------------------------------ mode helpers
def _psd(specgram, mask, normalize, eps):
    _need_cx(specgram, "specgram", 3)
    sg = pack(specgram, "specgram")                 # (C, F, T)
    C, F, T = len(sg), len(sg[0]), len(sg[0][0])
    # transpose(-3,-2): (C, F, T) -> (F, C, T)
    t = [[[sg[c][f][tt] for tt in range(T)] for c in range(C)] for f in range(F)]
    mk = None
    if mask is not None:
        ms = _shape(mask)
        if ms[:-1] != [F] or ms[-1] != T:
            raise ValueError("The dimensions of mask except the channel dimension "
                             "should be the same as specgram")
        # mask is a REAL (freq, time) list, never [re, im] pairs
        mk = [[float(v) for v in row] for row in mask]
        if normalize:
            mk = [[v / (sum(row) + eps) for v in row] for row in mk]
    outer = [[[[t[f][c][tt] * t[f][e][tt].conjugate() for e in range(C)]
               for c in range(C)] for tt in range(T)] for f in range(F)]
    psd = [[[sum(outer[f][tt][c][e] * (mk[f][tt] if mk is not None else 1.0)
                 for tt in range(T)) for e in range(C)] for c in range(C)]
           for f in range(F)]
    return {
        "psd": unpack(psd), "shape": [F, C, C],
        "mask_norm": unpack(mk) if mk is not None else None,
        "sum_dim": -3,
    }


def _trace_mode(mat, dim1, dim2):
    s = _need_cx(mat, "input", 2)
    m = pack(mat, "input")
    # dim1/dim2 refer to the last two axes for our 3-D (F,C,C) domain;
    # both orders give the same diagonal sum for a square matrix.
    if s[-1] != s[-2]:
        raise ValueError("The size of ``dim1`` and ``dim2`` must be the same.")
    diag = [[m[f][i][i] for i in range(s[-1])] for f in range(s[0])]
    tr = [sum(row) for row in diag]
    return {"trace": unpack(tr), "diag": unpack(diag), "shape": [s[0]]}


def _tik_reg_mode(mat, reg, eps):
    _need_cx(mat, "mat", 2)
    s = _need_square(mat, "mat")
    m = pack(mat, "mat")
    C = s[-1]
    # _compute_mat_trace(mat).real is the real part of the SUM of the diagonal
    tr_real = [sum(m[f][i][i].real for i in range(C)) for f in range(s[0])]
    epsilon = [[[tr_real[f] * reg + eps]] for f in range(s[0])]
    out = [[[m[f][i][j] + (tr_real[f] * reg + eps if i == j else 0.0)
             for j in range(C)] for i in range(C)] for f in range(s[0])]
    eye = [[[1.0 + 0j if i == j else 0j for j in range(C)] for i in range(C)]]
    return {
        "out": unpack(out), "shape": [s[0], C, C],
        "trace_real": tr_real, "epsilon": epsilon,
        "eye": unpack(eye)[0], "channels": C, "reg": reg, "eps": eps,
    }


def _assert_psd(psd_s, psd_n):
    ss, sn = cshape(psd_s), cshape(psd_n)
    if len(ss) < 3 or len(sn) < 3:
        raise ValueError("Expected at least 3D Tensor (..., freq, channel, channel) "
                         "for psd_s and psd_n")
    if ss != sn:
        raise ValueError("The dimensions of psd_s and psd_n should be the same")
    if ss[-1] != ss[-2]:
        raise ValueError("The last two dimensions of psd_s should be the same")


def _souden(psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps):
    _assert_psd(psd_s, psd_n)
    S, N = pack(psd_s, "psd_s"), pack(psd_n, "psd_n")
    F, C = len(S), len(S[0])
    branch, rc = _rc(reference_channel)
    Nreg = _tik_reg_core(N, diag_eps, 1e-8) if diagonal_loading else N
    numerator = _solve(Nreg, S)
    trn = _trace(numerator)                        # COMPLEX trace, no .real
    ws = [[[numerator[f][i][j] / (trn[f] + eps) for j in range(C)]
           for i in range(C)] for f in range(F)]
    if branch == "int":
        weights = [[ws[f][i][rc] for i in range(C)] for f in range(F)]
    else:
        weights = [[sum(ws[f][i][j] * rc[j] for j in range(C)) for i in range(C)]
                   for f in range(F)]
    return {
        "weights": unpack(weights), "shape": [F, C], "branch": branch,
        "psd_n_reg": unpack(Nreg), "numerator": unpack(numerator),
        "trace_numerator": unpack(trn), "ws": unpack(ws),
    }


def _tik_reg_core(mats, reg, eps):
    """Apply _tik_reg to (F,C,C) complex matrices, returning complex matrices."""
    C = len(mats[0])
    tr_real = [sum(m[i][i].real for i in range(C)) for m in mats]
    return [[[mats[f][i][j] + (tr_real[f] * reg + eps if i == j else 0.0)
              for j in range(C)] for i in range(C)] for f in range(len(mats))]


def _rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps):
    _need_cx(rtf, "rtf", 2)
    sn = cshape(psd_n)
    if len(sn) < 3:
        raise ValueError("Expected at least 3D Tensor (..., freq, channel, channel) for psd_n")
    if cshape(rtf) != sn[:-1]:
        raise ValueError("The dimensions of rtf and the dimensions withou the last "
                         "dimension of psd_n should be the same")
    if sn[-1] != sn[-2]:
        raise ValueError("The last two dimensions of psd_n should be the same")
    v, N = pack(rtf, "rtf"), pack(psd_n, "psd_n")
    F, C = len(v), len(v[0])
    branch = "none"
    rc = None
    if reference_channel is not None:
        branch, rc = _rc(reference_channel)
    Nreg = _tik_reg_core(N, diag_eps, 1e-8) if diagonal_loading else N
    B = [[[v[f][i]] for i in range(C)] for f in range(F)]     # (F,C,1)
    numerator = [[X[i][0] for i in range(C)] for X in _solve(Nreg, B)]
    denominator = [sum(v[f][d].conjugate() * numerator[f][d] for d in range(C))
                   for f in range(F)]
    den_real = [z.real for z in denominator]
    w_unnorm = [[numerator[f][i] / (den_real[f] + eps) for i in range(C)]
                for f in range(F)]
    scale = None
    w = [row[:] for row in w_unnorm]
    if rc is not None:
        if branch == "int":
            scale = [v[f][rc].conjugate() for f in range(F)]
        else:
            scale = [sum(v[f][j].conjugate() * rc[j] for j in range(C))
                     for f in range(F)]
        w = [[w[f][i] * scale[f] for i in range(C)] for f in range(F)]
    return {
        "weights": unpack(w), "shape": [F, C], "branch": branch,
        "psd_n_reg": unpack(Nreg), "numerator": unpack(numerator),
        "denominator": unpack(denominator), "denominator_real": den_real,
        "weights_unnorm": unpack(w_unnorm),
        "scale": unpack(scale) if scale is not None else None,
    }


def _rtf_power(psd_s, psd_n, reference_channel, n_iter, diagonal_loading, diag_eps):
    _assert_psd(psd_s, psd_n)
    if n_iter <= 0:
        raise ValueError("The number of iteration must be greater than 0.")
    S, N = pack(psd_s, "psd_s"), pack(psd_n, "psd_n")
    F, C = len(S), len(S[0])
    branch, rc = _rc(reference_channel)
    Nreg = _tik_reg_core(N, diag_eps, 1e-8) if diagonal_loading else N
    phi = _solve(Nreg, S)
    if branch == "int":
        r0 = [[phi[f][i][rc] for i in range(C)] for f in range(F)]
    else:
        r0 = [[sum(phi[f][i][j] * rc[j] for j in range(C)) for i in range(C)]
              for f in range(F)]
    cur = [[[r0[f][i]] for i in range(C)] for f in range(F)]     # (F,C,1)
    n_loop = 0
    if n_iter >= 2:
        for _ in range(n_iter - 2):
            cur = _matmul(phi, cur)
            n_loop += 1
        cur = _matmul(S, cur)
        tail = "psd_s"
    else:
        cur = _matmul(Nreg, cur)
        tail = "psd_n"
    rtf = [[cur[f][i][0] for i in range(C)] for f in range(F)]
    return {
        "rtf": unpack(rtf), "shape": [F, C], "branch": branch,
        "psd_n_reg": unpack(Nreg), "phi": unpack(phi), "rtf_init": unpack(r0),
        "n_loop": n_loop, "tail": tail, "n_iter": n_iter,
    }


def _beamform(weights, specgram):
    _need_cx(weights, "beamform_weights", 2)
    _need_cx(specgram, "specgram", 3)
    w, sg = pack(weights, "beamform_weights"), pack(specgram, "specgram")
    F, C = len(w), len(w[0])
    if len(sg) != C:
        raise ValueError("The dimensions except the last two dimensions of "
                         "beamform_weights should be the same as specgram")
    T = len(sg[0][0])
    if len(sg[0]) != F:
        raise ValueError("beamform_weights freq dimension must match specgram")
    out = [[sum(w[f][c].conjugate() * sg[c][f][t] for c in range(C))
            for t in range(T)] for f in range(F)]
    return {"out": unpack(out), "shape": [F, T],
            "weights_conj": unpack([[z.conjugate() for z in row] for row in w])}


# ------------------------------------------------------------------ dispatch
MODES = ("psd", "trace", "tik_reg", "souden", "rtf", "rtf_power", "beamform")


def solve(mode, **kw):
    if mode == "psd":
        return _psd(kw["specgram"], kw.get("mask"), kw.get("normalize", True),
                    kw.get("eps", 1e-10))
    if mode == "trace":
        return _trace_mode(kw["mat"], kw.get("dim1", -1), kw.get("dim2", -2))
    if mode == "tik_reg":
        return _tik_reg_mode(kw["mat"], kw.get("reg", 1e-7), kw.get("eps", 1e-8))
    if mode == "souden":
        return _souden(kw["psd_s"], kw["psd_n"], kw["reference_channel"],
                       kw.get("diagonal_loading", True), kw.get("diag_eps", 1e-7),
                       kw.get("eps", 1e-8))
    if mode == "rtf":
        return _rtf(kw["rtf"], kw["psd_n"], kw.get("reference_channel"),
                    kw.get("diagonal_loading", True), kw.get("diag_eps", 1e-7),
                    kw.get("eps", 1e-8))
    if mode == "rtf_power":
        return _rtf_power(kw["psd_s"], kw["psd_n"], kw["reference_channel"],
                          kw.get("n_iter", 3), kw.get("diagonal_loading", True),
                          kw.get("diag_eps", 1e-7))
    if mode == "beamform":
        return _beamform(kw["weights"], kw["specgram"])
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
