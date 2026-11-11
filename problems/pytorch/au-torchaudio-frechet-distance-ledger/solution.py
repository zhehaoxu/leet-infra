"""863 reference solution: torchaudio frechet_distance, pure Python.

The official implementation is

    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
    return a + b - 2 * c

This module reproduces that ledger without torch/numpy. The spectrum of
A = sigma_x @ sigma_y is obtained from its characteristic polynomial
(Faddeev-LeVerrier / Newton identities) and solved in closed form for N <= 3
(quadratic formula and Cardano), with Durand-Kerner as the general fallback.
Complex eigenvalues take the principal square root BEFORE summing real parts.
"""
import cmath
import math

MODES = ("frechet", "mu_term", "trace_term", "sqrt_trace_term",
         "terms", "charpoly", "eigvals")

_KEYS = ("mu_x", "sigma_x", "mu_y", "sigma_y")
_IMAG_EPS = 1e-9


# ---------------------------------------------------------------- input packing
def _is_num(x):
    return isinstance(x, (int, float))


def _pack(x, tag):
    """Validate a rectangular numeric nested list and return float copies."""
    if not isinstance(x, (list, tuple)):
        raise TypeError(tag + " must be a list")
    if len(x) == 0:
        return []
    if isinstance(x[0], (list, tuple)):
        width = len(x[0])
        rows = []
        for row in x:
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise ValueError(tag + " must be rectangular")
            rows.append(_pack(row, tag))
        return rows
    for v in x:
        if not _is_num(v):
            raise ValueError(tag + " must contain numbers")
    return [float(v) for v in x]


def _shape(x):
    out = []
    while isinstance(x, (list, tuple)):
        out.append(len(x))
        if not x:
            break
        x = x[0]
    return tuple(out)


def _torchsize(sh):
    return "torch.Size([" + ", ".join(str(v) for v in sh) + "])"


# ---------------------------------------------------------------- linear algebra
def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def _trace(A):
    return sum(A[i][i] for i in range(len(A)))


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _charpoly(A):
    """Coefficients of det(lambda*I - A), highest power first (monic).

    Newton identities on the power traces tr(A^k) give the elementary
    symmetric polynomials e_k, then coeff[j] = (-1)^(n-j) * e_(n-j).
    """
    n = len(A)
    power = _identity(n)
    ptraces = []
    for _ in range(n):
        power = _matmul(power, A)
        ptraces.append(_trace(power))
    e = [1.0]
    for k in range(1, n + 1):
        s = 0.0
        for i in range(1, k + 1):
            s += ((-1) ** (i - 1)) * e[k - i] * ptraces[i - 1]
        e.append(s / k)
    return [((-1) ** k) * e[k] for k in range(n + 1)]


def _cbrt(z):
    """Principal complex cube root."""
    if z == 0:
        return 0j
    return complex(z) ** (1.0 / 3.0)


def _roots_closed(coeffs):
    """Closed-form roots for degree 1, 2 and 3 (monic polynomial)."""
    c = [complex(v) for v in coeffs]
    n = len(c) - 1
    if n == 1:
        return [-c[1] / c[0]]
    if n == 2:
        a, b, d = c[0], c[1], c[2]
        disc = cmath.sqrt(b * b - 4.0 * a * d)
        return [(-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)]
    if n == 3:
        a2, a1, a0 = c[1] / c[0], c[2] / c[0], c[3] / c[0]
        p = a1 - a2 * a2 / 3.0
        q = 2.0 * a2 ** 3 / 27.0 - a2 * a1 / 3.0 + a0
        disc = cmath.sqrt((q / 2.0) ** 2 + (p / 3.0) ** 3)
        u = _cbrt(-q / 2.0 + disc)
        if abs(u) < 1e-15:
            v = _cbrt(-q / 2.0 - disc)
        else:
            v = -p / (3.0 * u)
        w = complex(-0.5, math.sqrt(3.0) / 2.0)
        w2 = w * w
        shift = a2 / 3.0
        return [u + v - shift, w * u + w2 * v - shift, w2 * u + w * v - shift]
    return None


def _roots_dk(coeffs):
    """Durand-Kerner iteration for degree >= 4."""
    c = [complex(v) for v in coeffs]
    n = len(c) - 1
    z = [(0.4 + 0.9j) * cmath.exp(2j * math.pi * k / n) for k in range(n)]
    for _ in range(600):
        old = z[:]
        delta = 0.0
        for i in range(n):
            val = c[0]
            for cc in c[1:]:
                val = val * old[i] + cc
            den = complex(1.0)
            for j in range(n):
                if j != i:
                    den *= old[i] - old[j]
            if abs(den) < 1e-300:
                continue
            nz = old[i] - val / den
            z[i] = nz
            delta = max(delta, abs(nz - old[i]))
        if delta < 1e-14:
            break
    return z


def _clean(z):
    """LAPACK reports real eigenvalues with a zero imaginary part."""
    out = []
    for r in z:
        if abs(r.imag) < _IMAG_EPS * max(1.0, abs(r.real)):
            out.append(complex(r.real, 0.0))
        else:
            out.append(r)
    return out


def _eigvals(A):
    coeffs = _charpoly(A)
    roots = _roots_closed(coeffs)
    if roots is None:
        roots = _roots_dk(coeffs)
    roots = _clean(roots)
    pairs = [[r.real, r.imag] for r in roots]
    return sorted([(p[0], p[1]) for p in pairs])


def _sqrt_real_sum(evs):
    return sum(cmath.sqrt(complex(re, im)).real for re, im in evs)


# ---------------------------------------------------------------- validation
def _validate(kw):
    for k in _KEYS:
        if k not in kw:
            raise TypeError("missing required argument: " + k)
    mx = _pack(kw["mu_x"], "mu_x")
    sx = _pack(kw["sigma_x"], "sigma_x")
    my = _pack(kw["mu_y"], "mu_y")
    sy = _pack(kw["sigma_y"], "sigma_y")
    shx, shy, shm, shn = _shape(sx), _shape(sy), _shape(mx), _shape(my)
    if len(shm) != 1:
        raise ValueError("Input mu_x must be one-dimensional; got dimension %d." % len(shm))
    if len(shx) != 2:
        raise ValueError("Input sigma_x must be two-dimensional; got dimension %d." % len(shx))
    # Deliberately a chained comparison: this mirrors the official guard and
    # short-circuits exactly the same way for square sigma_x.
    if shx[0] != shx[1] != shm[0]:
        raise ValueError("Each of sigma_x's dimensions must match mu_x's size.")
    if shm != shn:
        raise ValueError("Inputs mu_x and mu_y must have the same shape; got %s and %s."
                         % (_torchsize(shm), _torchsize(shn)))
    if shx != shy:
        raise ValueError("Inputs sigma_x and sigma_y must have the same shape; got %s and %s."
                         % (_torchsize(shx), _torchsize(shy)))
    if shx[0] != shx[1] or shy[0] != shy[1]:
        raise ValueError("A must be batches of square matrices")
    return mx, sx, my, sy


# ---------------------------------------------------------------- modes
def _ledger(kw):
    mx, sx, my, sy = _validate(kw)
    a = sum((x - y) ** 2 for x, y in zip(mx, my))
    b = _trace(sx) + _trace(sy)
    A = _matmul(sx, sy)
    coeffs = _charpoly(A)
    evs = _eigvals(A)
    c = _sqrt_real_sum(evs)
    return {"a": a, "b": b, "A": A, "coeffs": coeffs, "evs": evs,
            "c": c, "F": a + b - 2.0 * c}


def solve(mode, **kw):
    if mode not in MODES:
        raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                         + ", ".join(MODES) + ")")
    L = _ledger(kw)
    if mode == "frechet":
        return L["F"]
    if mode == "mu_term":
        return L["a"]
    if mode == "trace_term":
        return L["b"]
    if mode == "sqrt_trace_term":
        return L["c"]
    if mode == "terms":
        return [L["a"], L["b"], L["c"]]
    if mode == "charpoly":
        return L["coeffs"]
    if mode == "eigvals":
        return [[p[0], p[1]] for p in L["evs"]]
    raise AssertionError("unreachable")
