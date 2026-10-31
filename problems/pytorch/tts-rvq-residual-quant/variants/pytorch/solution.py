def solve(latents, codebooks):
    def _check_num(x, msg):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(msg)
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            raise ValueError(msg)

    if not isinstance(latents, (list, tuple)) or len(latents) == 0:
        raise ValueError("latents must be a non-empty [N, D] list")
    D = None
    for v in latents:
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError("latent vectors must be non-empty")
        if D is None:
            D = len(v)
        elif len(v) != D:
            raise ValueError("latent dims must be equal")
        for x in v:
            _check_num(x, "latents entries must be finite numbers")
    if not isinstance(codebooks, (list, tuple)) or len(codebooks) == 0:
        raise ValueError("codebooks must be a non-empty [K, C, D] list")
    K = len(codebooks)
    for k, cb in enumerate(codebooks):
        if not isinstance(cb, (list, tuple)) or len(cb) == 0:
            raise ValueError(f"codebook {k} must be non-empty")
        for entry in cb:
            if not isinstance(entry, (list, tuple)) or len(entry) != D:
                raise ValueError("codebook entry dims must match latents")
            for x in entry:
                _check_num(x, "codebook entries must be finite numbers")

    N = len(latents)
    codes = [[] for _ in range(K)]
    residual_sq = [[] for _ in range(K)]
    reconstructed = []
    for n in range(N):
        residual = [float(x) for x in latents[n]]
        recon = [0.0] * D
        for k in range(K):
            cb = codebooks[k]
            best_i = 0
            best_d = None
            for i, entry in enumerate(cb):
                d = 0.0
                for j in range(D):
                    diff = residual[j] - entry[j]
                    d += diff * diff
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            entry = cb[best_i]
            codes[k].append(best_i)
            sq = 0.0
            for j in range(D):
                residual[j] -= entry[j]
                recon[j] += entry[j]
                sq += residual[j] * residual[j]
            residual_sq[k].append(sq)
        reconstructed.append(recon)
    return {"codes": codes, "reconstructed": reconstructed,
            "residual_sq": residual_sq}
