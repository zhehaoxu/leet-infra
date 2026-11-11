#!/usr/bin/env python3
"""863 judge cases: torchaudio Gaussian Frechet distance / FID core ledger.

Constants below are exact snapshots produced by fixtures863.py from the
verbatim official implementation. Helpers use internal assert, never a
returned bool. No torch, no import of the solution module.
"""

# ---- official numerical fixtures ----
### shared inputs
I2 = [[1.0, 0.0], [0.0, 1.0]]
I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
I4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
MX2 = [1.0, 2.0]
SX2 = [[2.0, 0.3], [0.3, 1.0]]
MY2 = [0.0, 0.0]
MX3 = [1.0, -2.0, 0.5]
SX3 = [[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.0]]
MY3 = [0.0, 1.0, -0.5]
SY3 = [[1.0, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 0.8]]
SXC = [[1.0, 2.0], [-2.0, 1.0]]
SXC2 = [[1.0, 0.3], [-4.0, 1.0]]
SXN = [[-4.0, 0.0], [0.0, -1.0]]
SXS = [[1.0, 1.0], [1.0, 1.0]]
SXD2 = [[4.0, 0.0], [0.0, 9.0]]
SXD3 = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
SXD4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
SXOFF = [[2.0, 9.0, -7.0], [9.0, 1.5, 5.0], [-7.0, 5.0, 1.0]]
SXNS = [[2.0, 1.5], [0.2, 1.0]]
Z2 = [0.0, 0.0]
Z3 = [0.0, 0.0, 0.0]
Z4 = [0.0, 0.0, 0.0, 0.0]
### main N=3 SPD ledger
F_MAIN = 11.204837625048754
MU_MAIN = 11.0
TRACE_MAIN = 7.5
C_MAIN = 3.647581187475623
TERMS_MAIN = [11.0, 7.5, 3.647581187475623]
A_MAIN = [[2.06, 0.75, -0.05000000000000001], [0.6, 1.88, 0.31000000000000005], [-0.06, 0.32, 0.8200000000000001]]
EV_MAIN = [[0.6832924572395724, 0.0], [1.4145993231878078, 0.0], [2.66210821957262, 0.0]]
CP_MAIN = [1.0, -4.76, 6.5514, -2.573154]
### N=1
F_N1 = 5.0
MU_N1 = 4.0
TRACE_N1 = 5.0
C_N1 = 2.0
TERMS_N1 = [4.0, 5.0, 2.0]
A_N1 = [[4.0]]
EV_N1 = [[4.0, 0.0]]
CP_N1 = [1.0, -4.0]
### N=2 SPD
F_SPD2 = 5.19831071716753
MU_SPD2 = 5.0
TRACE_SPD2 = 5.0
C_SPD2 = 2.400844641416235
TERMS_SPD2 = [5.0, 5.0, 2.400844641416235]
A_SPD2 = [[2.0, 0.3], [0.3, 1.0]]
EV_SPD2 = [[0.91690481051547, 0.0], [2.08309518948453, 0.0]]
CP_SPD2 = [1.0, -3.0, 1.9100000000000001]
### diagonal N=2 / N=3 / N=4
F_DIAG2 = 10.0
MU_DIAG2 = 5.0
TRACE_DIAG2 = 15.0
C_DIAG2 = 5.0
TERMS_DIAG2 = [5.0, 15.0, 5.0]
A_DIAG2 = [[4.0, 0.0], [0.0, 9.0]]
EV_DIAG2 = [[4.0, 0.0], [9.0, 0.0]]
CP_DIAG2 = [1.0, -13.0, 36.0]
F_DIAG3 = 2.17157287525381
MU_DIAG3 = 1.0
TRACE_DIAG3 = 10.0
C_DIAG3 = 4.414213562373095
TERMS_DIAG3 = [1.0, 10.0, 4.414213562373095]
A_DIAG3 = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
EV_DIAG3 = [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
CP_DIAG3 = [1.0, -7.0, 14.0, -8.0]
F_N4 = 1.7074712601160549
MU_N4 = 0.0
TRACE_N4 = 14.0
C_N4 = 6.146264369941973
TERMS_N4 = [0.0, 14.0, 6.146264369941973]
A_N4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
EV_N4 = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
CP_N4 = [1.0, -10.0, 35.0, -50.0, 24.0]
### complex spectrum branches
F_COMPLEX = -1.088078598056276
MU_COMPLEX = 0.0
TRACE_COMPLEX = 4.0
C_COMPLEX = 2.544039299028138
TERMS_COMPLEX = [0.0, 4.0, 2.544039299028138]
A_COMPLEX = [[1.0, 2.0], [-2.0, 1.0]]
EV_COMPLEX = [[1.0, -2.0000000000000004], [1.0, 2.0000000000000004]]
CP_COMPLEX = [1.0, -2.0, 5.000000000000002]
F_COMPLEX2 = -0.4571198749139631
MU_COMPLEX2 = 0.0
TRACE_COMPLEX2 = 4.0
C_COMPLEX2 = 2.2285599374569816
TERMS_COMPLEX2 = [0.0, 4.0, 2.2285599374569816]
A_COMPLEX2 = [[1.0, 0.3], [-4.0, 1.0]]
EV_COMPLEX2 = [[1.0, -1.0954451150103321], [1.0, 1.0954451150103321]]
CP_COMPLEX2 = [1.0, -2.0, 2.1999999999999997]
### negative real spectrum
F_NEG = -3.0
MU_NEG = 0.0
TRACE_NEG = -3.0
C_NEG = 0.0
TERMS_NEG = [0.0, -3.0, 0.0]
A_NEG = [[-4.0, 0.0], [0.0, -1.0]]
EV_NEG = [[-4.0, 0.0], [-1.0, 0.0]]
CP_NEG = [1.0, 5.0, 4.0]
### singular covariance
F_SING = 1.1715728752538097
MU_SING = 0.0
TRACE_SING = 4.0
C_SING = 1.4142135623730951
TERMS_SING = [0.0, 4.0, 1.4142135623730951]
A_SING = [[1.0, 1.0], [1.0, 1.0]]
EV_SING = [[0.0, 0.0], [2.0, 0.0]]
CP_SING = [1.0, -2.0, 0.0]
### identity covariances
F_I2 = 0.0
MU_I2 = 0.0
TRACE_I2 = 4.0
C_I2 = 2.0
TERMS_I2 = [0.0, 4.0, 2.0]
A_I2 = [[1.0, 0.0], [0.0, 1.0]]
EV_I2 = [[1.0, 0.0], [1.0, 0.0]]
CP_I2 = [1.0, -2.0, 1.0]
F_I3 = 0.0
MU_I3 = 0.0
TRACE_I3 = 6.0
C_I3 = 3.0
TERMS_I3 = [0.0, 6.0, 3.0]
A_I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
EV_I3 = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
CP_I3 = [1.0, -3.0, 3.0, -1.0]
### non-symmetric but real-spectrum covariance
F_NONSYM = 0.7638914868508095
MU_NONSYM = 0.5
TRACE_NONSYM = 5.0
C_NONSYM = 2.3680542565745952
TERMS_NONSYM = [0.5, 5.0, 2.3680542565745952]
A_NONSYM = [[2.0, 1.5], [0.2, 1.0]]
EV_NONSYM = [[0.7583801512904337, 0.0], [2.241619848709566, 0.0]]
CP_NONSYM = [1.0, -3.0, 1.7]
### off-diagonal only (trace unchanged, F changed)
F_OFFDIAG = 6.385965864231988
MU_OFFDIAG = 11.0
TRACE_OFFDIAG = 7.5
C_OFFDIAG = 6.057017067884006
TERMS_OFFDIAG = [11.0, 7.5, 6.057017067884006]
A_OFFDIAG = [[3.8000000000000003, 10.5, -4.7], [9.3, 4.1000000000000005, 4.15], [-6.0, 4.699999999999999, 1.3]]
EV_OFFDIAG = [[-10.104092324520895, 0.0], [5.454825173275933, 0.0], [13.849267151244977, 0.0]]
CP_OFFDIAG = [1.0, -9.200000000000014, -119.50499999999988, 763.3169999999999]
### mu length decoupled from sigma dimension (chain-comparison quirk)
F_QUIRK = 0.0
MU_QUIRK = 0.0
TRACE_QUIRK = 4.0
C_QUIRK = 2.0
TERMS_QUIRK = [0.0, 4.0, 2.0]
A_QUIRK = [[1.0, 0.0], [0.0, 1.0]]
EV_QUIRK = [[1.0, 0.0], [1.0, 0.0]]
CP_QUIRK = [1.0, -2.0, 1.0]
### identical inputs
F_IDENT = 0.0
MU_IDENT = 0.0
TRACE_IDENT = 9.0
C_IDENT = 4.5
TERMS_IDENT = [0.0, 9.0, 4.5]
A_IDENT = [[4.1, 1.03, -0.24000000000000002], [1.03, 2.38, 0.47000000000000003], [-0.24000000000000002, 0.47000000000000003, 1.05]]
EV_IDENT = [[0.7972456989691227, 0.0], [2.1508301426933754, 0.0], [4.581924158337504, 0.0]]
CP_IDENT = [1.0, -7.530000000000002, 15.222600000000007, -7.856809000000004]
F_IDENT_NEG = -20.0
F_IDENT_COMPLEX = 0.0
### argument swap (AB vs BA share a spectrum)
F_SWAP = 11.204837625048754
### bool / int leaves are numeric (torch.tensor casts them)
F_BOOL = 0.0
F_INT = 5.17157287525381
### scaled covariance
F_SCALED = 11.614512875146268
### float64 dtype of the official environment
EV_DTYPE = 'torch.complex128'

import math

# ------------------------------------------------------------------ helpers
def _near(got, want, tag, rtol=1e-8, atol=1e-9):
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), tag + ": expected a list"
        assert len(got) == len(want), "%s: length %d != %d" % (tag, len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            _near(a, b, "%s[%d]" % (tag, i), rtol, atol)
        return
    if isinstance(want, bool) or isinstance(got, bool):
        assert got is want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, str) or want is None:
        assert got == want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, float) and math.isnan(want):
        assert isinstance(got, float) and math.isnan(got), "%s: %r != nan" % (tag, got)
        return
    assert abs(float(got) - float(want)) <= atol + rtol * abs(float(want)), \
        "%s: %r != %r" % (tag, got, want)


def _raises(fn, tag, needle):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert needle in str(e), "%s: wrong message %r (want %r)" % (tag, str(e), needle)
    except BaseException as e:
        raise AssertionError("%s: wrong exception %s %s" % (tag, type(e).__name__, e))
    else:
        raise AssertionError("%s: no exception" % tag)


def KW(mx, sx, my, sy):
    return dict(mu_x=mx, sigma_x=sx, mu_y=my, sigma_y=sy)


def S(solve, mode, mx, sx, my, sy):
    return solve(mode, **KW(mx, sx, my, sy))


def F(solve, mx, sx, my, sy):
    return S(solve, "frechet", mx, sx, my, sy)


SX23 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
MODES = ("frechet", "mu_term", "trace_term", "sqrt_trace_term",
         "terms", "charpoly", "eigvals")


# ------------------------------------------------------------------ A. frechet ledger
def check_frechet_main(solve):
    _near(F(solve, MX3, SX3, MY3, SY3), F_MAIN, "F main")


def check_frechet_n1(solve):
    _near(F(solve, [3.0], [[4.0]], [1.0], [[1.0]]), F_N1, "F N=1")


def check_frechet_spd2(solve):
    _near(F(solve, MX2, SX2, MY2, I2), F_SPD2, "F SPD2")


def check_frechet_diag2(solve):
    _near(F(solve, MX2, SXD2, MY2, I2), F_DIAG2, "F diag2")


def check_frechet_diag3(solve):
    _near(F(solve, [1.0, 0.0, 0.0], SXD3, Z3, I3), F_DIAG3, "F diag3")


def check_frechet_n4(solve):
    _near(F(solve, Z4, SXD4, Z4, I4), F_N4, "F N=4")


def check_frechet_complex(solve):
    _near(F(solve, Z2, SXC, Z2, I2), F_COMPLEX, "F conjugate pair")


def check_frechet_complex2(solve):
    _near(F(solve, Z2, SXC2, Z2, I2), F_COMPLEX2, "F conjugate pair 2")


def check_frechet_negative(solve):
    _near(F(solve, Z2, SXN, Z2, I2), F_NEG, "F negative spectrum")


def check_frechet_singular(solve):
    _near(F(solve, Z2, SXS, Z2, I2), F_SING, "F singular covariance")


def check_frechet_identity2(solve):
    _near(F(solve, Z2, I2, Z2, I2), F_I2, "F identity N=2")


def check_frechet_identity3(solve):
    _near(F(solve, Z3, I3, Z3, I3), F_I3, "F identity N=3")


def check_frechet_nonsym(solve):
    _near(F(solve, [0.5, -0.5], SXNS, Z2, I2), F_NONSYM, "F non-symmetric sigma")


def check_frechet_offdiag(solve):
    _near(F(solve, MX3, SXOFF, MY3, SY3), F_OFFDIAG, "F off-diagonal heavy")


def check_frechet_quirk_mu_length(solve):
    _near(F(solve, [1.0, 2.0, 3.0], I2, [1.0, 2.0, 3.0], I2), F_QUIRK,
          "F mu length decoupled from sigma dimension")


def check_frechet_swap(solve):
    _near(F(solve, MY3, SY3, MX3, SX3), F_SWAP, "F swap symmetry")


def check_frechet_identical(solve):
    _near(F(solve, MX3, SX3, MX3, SX3), F_IDENT, "F identical inputs")


def check_frechet_identical_negative(solve):
    _near(F(solve, Z2, SXN, Z2, SXN), F_IDENT_NEG,
          "F identical negative-definite is NOT zero")


def check_frechet_identical_complex(solve):
    _near(F(solve, Z2, SXC, Z2, SXC), F_IDENT_COMPLEX,
          "F identical complex-spectrum")


def check_frechet_bool_leaf(solve):
    _near(F(solve, Z2, [[True, False], [False, True]], Z2, I2), F_BOOL,
          "F bool leaves cast to floats")


def check_frechet_int_leaf(solve):
    _near(F(solve, [1, 2], [[2, 0], [0, 1]], [0, 0], I2), F_INT,
          "F int leaves")


def check_frechet_scaled(solve):
    SC = [[v * 3.0 for v in row] for row in SX3]
    SD = [[v * 3.0 for v in row] for row in SY3]
    _near(F(solve, MX3, SC, MY3, SD), F_SCALED, "F scaled covariances")


# ------------------------------------------------------------------ B. terms and components
def check_terms_main(solve):
    _near(S(solve, "terms", MX3, SX3, MY3, SY3), TERMS_MAIN, "terms main")


def check_terms_n1(solve):
    _near(S(solve, "terms", [3.0], [[4.0]], [1.0], [[1.0]]), TERMS_N1, "terms N=1")


def check_terms_complex(solve):
    _near(S(solve, "terms", Z2, SXC, Z2, I2), TERMS_COMPLEX, "terms complex")


def check_terms_negative(solve):
    _near(S(solve, "terms", Z2, SXN, Z2, I2), TERMS_NEG, "terms negative")


def check_terms_singular(solve):
    _near(S(solve, "terms", Z2, SXS, Z2, I2), TERMS_SING, "terms singular")


def check_mu_term_main(solve):
    _near(S(solve, "mu_term", MX3, SX3, MY3, SY3), MU_MAIN, "a main")


def check_mu_term_zero(solve):
    _near(S(solve, "mu_term", MX3, SX3, MX3, SY3), 0.0, "a zero when mu equal")


def check_mu_term_squared(solve):
    _near(S(solve, "mu_term", [1.0, -2.0], I2, Z2, I2), 5.0, "a is a sum of squares")


def check_trace_term_main(solve):
    _near(S(solve, "trace_term", MX3, SX3, MY3, SY3), TRACE_MAIN, "b main")


def check_trace_offdiag_unchanged(solve):
    _near(S(solve, "trace_term", MX3, SXOFF, MY3, SY3), TRACE_MAIN,
          "b ignores off-diagonal")


def check_trace_negative(solve):
    _near(S(solve, "trace_term", Z2, SXN, Z2, I2), TRACE_NEG, "b negative spectrum")


def check_sqrt_term_main(solve):
    _near(S(solve, "sqrt_trace_term", MX3, SX3, MY3, SY3), C_MAIN, "c main")


def check_sqrt_term_negative_zero(solve):
    _near(S(solve, "sqrt_trace_term", Z2, SXN, Z2, I2), C_NEG,
          "c negative roots have zero real part")


def check_sqrt_term_complex_closed_form(solve):
    want = 2.0 * math.sqrt((abs(complex(1.0, 2.0)) + 1.0) / 2.0)
    _near(S(solve, "sqrt_trace_term", Z2, SXC, Z2, I2), want,
          "c principal sqrt closed form")


def check_sqrt_term_complex2(solve):
    _near(S(solve, "sqrt_trace_term", Z2, SXC2, Z2, I2), C_COMPLEX2, "c complex2")


def check_sqrt_term_singular(solve):
    _near(S(solve, "sqrt_trace_term", Z2, SXS, Z2, I2), C_SING,
          "c singular has sqrt(0)=0")


def check_assembly_all_modes(solve):
    kw = KW(MX3, SX3, MY3, SY3)
    a = solve("mu_term", **kw)
    b = solve("trace_term", **kw)
    c = solve("sqrt_trace_term", **kw)
    _near(a + b - 2.0 * c, solve("frechet", **kw), "a+b-2c equals F")


# ------------------------------------------------------------------ C. characteristic polynomial
def check_charpoly_main(solve):
    _near(S(solve, "charpoly", MX3, SX3, MY3, SY3), CP_MAIN, "charpoly main")


def check_charpoly_n1(solve):
    _near(S(solve, "charpoly", [3.0], [[4.0]], [1.0], [[1.0]]), CP_N1, "charpoly N=1")


def check_charpoly_spd2(solve):
    _near(S(solve, "charpoly", MX2, SX2, MY2, I2), CP_SPD2, "charpoly SPD2")


def check_charpoly_diag3(solve):
    _near(S(solve, "charpoly", Z3, SXD3, Z3, I3), CP_DIAG3, "charpoly diag3")


def check_charpoly_n4(solve):
    _near(S(solve, "charpoly", Z4, SXD4, Z4, I4), CP_N4, "charpoly N=4")


def check_charpoly_complex(solve):
    _near(S(solve, "charpoly", Z2, SXC, Z2, I2), CP_COMPLEX, "charpoly complex")


def check_charpoly_negative(solve):
    _near(S(solve, "charpoly", Z2, SXN, Z2, I2), CP_NEG, "charpoly negative")


def check_charpoly_singular(solve):
    _near(S(solve, "charpoly", Z2, SXS, Z2, I2), CP_SING, "charpoly singular")


def check_charpoly_identity3(solve):
    _near(S(solve, "charpoly", Z3, I3, Z3, I3), CP_I3, "charpoly identity N=3")


def check_charpoly_offdiag(solve):
    _near(S(solve, "charpoly", MX3, SXOFF, MY3, SY3), CP_OFFDIAG, "charpoly offdiag")


def check_charpoly_monic(solve):
    for tag, (mx, sx, my, sy) in {
        "n1": ([3.0], [[4.0]], [1.0], [[1.0]]),
        "main": (MX3, SX3, MY3, SY3),
        "n4": (Z4, SXD4, Z4, I4),
    }.items():
        cp = S(solve, "charpoly", mx, sx, my, sy)
        assert len(cp) == len(sx) + 1, tag + ": charpoly length"
        _near(cp[0], 1.0, tag + ": monic")


def check_charpoly_vs_eigvals_vieta(solve):
    for tag, (mx, sx, my, sy) in {
        "main": (MX3, SX3, MY3, SY3),
        "complex": (Z2, SXC, Z2, I2),
        "neg": (Z2, SXN, Z2, I2),
        "n4": (Z4, SXD4, Z4, I4),
    }.items():
        cp = S(solve, "charpoly", mx, sx, my, sy)
        ev = S(solve, "eigvals", mx, sx, my, sy)
        _near(sum(p[0] for p in ev), -cp[1], tag + ": Vieta sum")
        prod = complex(1.0)
        for re, im in ev:
            prod *= complex(re, im)
        _near(prod.real, ((-1) ** len(ev)) * cp[-1], tag + ": Vieta product")


# ------------------------------------------------------------------ D. eigenvalues
def check_eigvals_main(solve):
    _near(S(solve, "eigvals", MX3, SX3, MY3, SY3), EV_MAIN, "eigvals main")


def check_eigvals_n1(solve):
    _near(S(solve, "eigvals", [3.0], [[4.0]], [1.0], [[1.0]]), EV_N1, "eigvals N=1")


def check_eigvals_spd2(solve):
    _near(S(solve, "eigvals", MX2, SX2, MY2, I2), EV_SPD2, "eigvals SPD2")


def check_eigvals_diag2(solve):
    _near(S(solve, "eigvals", MX2, SXD2, MY2, I2), EV_DIAG2, "eigvals diag2")


def check_eigvals_diag3(solve):
    _near(S(solve, "eigvals", Z3, SXD3, Z3, I3), EV_DIAG3, "eigvals diag3")


def check_eigvals_n4(solve):
    _near(S(solve, "eigvals", Z4, SXD4, Z4, I4), EV_N4, "eigvals N=4")


def check_eigvals_identity2(solve):
    _near(S(solve, "eigvals", Z2, I2, Z2, I2), EV_I2, "eigvals identity N=2")


def check_eigvals_identity3(solve):
    _near(S(solve, "eigvals", Z3, I3, Z3, I3), EV_I3, "eigvals identity N=3")


def check_eigvals_complex(solve):
    _near(S(solve, "eigvals", Z2, SXC, Z2, I2), EV_COMPLEX, "eigvals conjugate pair")


def check_eigvals_complex2(solve):
    _near(S(solve, "eigvals", Z2, SXC2, Z2, I2), EV_COMPLEX2, "eigvals conjugate pair 2")


def check_eigvals_negative(solve):
    _near(S(solve, "eigvals", Z2, SXN, Z2, I2), EV_NEG, "eigvals negative")


def check_eigvals_singular(solve):
    _near(S(solve, "eigvals", Z2, SXS, Z2, I2), EV_SING, "eigvals singular")


def check_eigvals_nonsym(solve):
    _near(S(solve, "eigvals", Z2, SXNS, Z2, I2), EV_NONSYM, "eigvals non-symmetric")


def check_eigvals_sorted(solve):
    ev = S(solve, "eigvals", MX3, SX3, MY3, SY3)
    assert ev == sorted([(p[0], p[1]) for p in ev]) or \
        [list(p) for p in sorted([(p[0], p[1]) for p in ev])] == ev, \
        "eigvals must be normalized by (re, im) order"


def check_eigvals_ab_ba(solve):
    _near(S(solve, "eigvals", MX3, SX3, MY3, SY3),
          S(solve, "eigvals", MY3, SY3, MX3, SX3), "AB and BA share a spectrum")


def check_eigvals_real_roots_have_zero_imag(solve):
    for p in S(solve, "eigvals", MX3, SX3, MY3, SY3):
        _near(p[1], 0.0, "SPD product roots are real")


def check_eigvals_complex_branch_not_clamped(solve):
    ev = S(solve, "eigvals", Z2, SXC, Z2, I2)
    assert any(abs(p[1]) > 1.0 for p in ev), "complex branch must keep imaginary parts"


# ------------------------------------------------------------------ E. validation
def check_mu_x_2d_raises(solve):
    _raises(lambda: F(solve, [[1.0, 2.0]], I2, Z2, I2), "mu_x 2D",
            "Input mu_x must be one-dimensional; got dimension 2.")


def check_mu_x_3d_raises(solve):
    _raises(lambda: F(solve, [[[1.0]]], I2, [0.0], I2), "mu_x 3D",
            "Input mu_x must be one-dimensional; got dimension 3.")


def check_sigma_x_1d_raises(solve):
    _raises(lambda: F(solve, MX2, [1.0, 2.0], MY2, I2), "sigma_x 1D",
            "Input sigma_x must be two-dimensional; got dimension 1.")


def check_sigma_x_3d_raises(solve):
    _raises(lambda: F(solve, MX2, [[[1.0]]], MY2, I2), "sigma_x 3D",
            "Input sigma_x must be two-dimensional; got dimension 3.")


def check_mu_y_2d_shape_mismatch(solve):
    _raises(lambda: F(solve, MX2, I2, [[0.0, 0.0]], I2), "mu_y 2D",
            "Inputs mu_x and mu_y must have the same shape")


def check_sigma_y_1d_shape_mismatch(solve):
    _raises(lambda: F(solve, MX2, I2, MY2, [1.0, 1.0]), "sigma_y 1D",
            "Inputs sigma_x and sigma_y must have the same shape")


def check_chain_guard_fires(solve):
    _raises(lambda: F(solve, Z2, SX23, Z2, SX23), "chain guard",
            "Each of sigma_x's dimensions must match mu_x's size.")


def check_chain_guard_short_circuits(solve):
    _raises(lambda: F(solve, Z3, SX23, Z3, SX23), "chain short circuit",
            "square matrices")


def check_mu_shape_mismatch(solve):
    _raises(lambda: F(solve, MX2, I2, [0.0], I2), "mu shape mismatch",
            "Inputs mu_x and mu_y must have the same shape")


def check_sigma_shape_mismatch(solve):
    _raises(lambda: F(solve, Z3, I3, MY3, I2), "sigma shape mismatch",
            "Inputs sigma_x and sigma_y must have the same shape")


def check_guard_order(solve):
    _raises(lambda: F(solve, Z2, SX23, [0.0], I2), "guard 3 before guard 4",
            "Each of sigma_x's dimensions must match mu_x's size.")


def check_non_rectangular_raises(solve):
    _raises(lambda: F(solve, Z2, [[1.0, 2.0], [3.0]], Z2, I2),
            "ragged sigma_x", "must be rectangular")


def check_non_rectangular_mu_raises(solve):
    _raises(lambda: F(solve, [[1.0], [2.0, 3.0]], I2, Z2, I2),
            "ragged mu_x", "must be rectangular")


def check_non_numeric_leaf_raises(solve):
    _raises(lambda: F(solve, [1.0, "x"], I2, Z2, I2),
            "non numeric mu leaf", "must contain numbers")


def check_non_list_argument_raises(solve):
    _raises(lambda: F(solve, 1.0, I2, Z2, I2),
            "non list mu_x", "must be a list")


def check_missing_argument_raises(solve):
    _raises(lambda: solve("frechet", mu_x=MX2, sigma_x=I2, mu_y=MY2),
            "missing sigma_y", "missing required argument: sigma_y")


def check_unknown_mode_message(solve):
    _raises(lambda: solve("evd", **KW(MX3, SX3, MY3, SY3)),
            "unknown mode", "unknown mode")
    try:
        solve("evd", **KW(MX3, SX3, MY3, SY3))
    except (ValueError, TypeError) as e:
        msg = str(e)
        for m in MODES:
            assert m in msg, "unknown-mode message must list %r: %r" % (m, msg)
        assert "evd" in msg, "unknown-mode message must echo the bad mode"
    else:
        raise AssertionError("unknown mode must raise")


# ------------------------------------------------------------------ F. hardening / mechanism
def check_square_sigma_mu_mismatch_silent(solve):
    _near(F(solve, Z2, I3, Z2, I3), 0.0,
          "square 3x3 sigma with length-2 mu is accepted")


def check_a_independent_of_sigma(solve):
    _near(S(solve, "mu_term", MX3, SX3, MY3, SY3),
          S(solve, "mu_term", MX3, I3, MY3, I3), "a independent of sigma")


def check_b_independent_of_offdiag(solve):
    _near(S(solve, "trace_term", MX3, SXOFF, MY3, SY3),
          S(solve, "trace_term", MX3, SX3, MY3, SY3), "b independent of offdiag")


def check_offdiag_changes_f(solve):
    a = F(solve, MX3, SX3, MY3, SY3)
    b = F(solve, MX3, SXOFF, MY3, SY3)
    assert abs(a - b) > 1.0, "off-diagonal must change F through A"


def check_f_can_be_negative(solve):
    assert F(solve, Z2, SXN, Z2, I2) < 0.0, "non-PSD input can give negative F"
    assert F(solve, Z2, SXC, Z2, I2) < 0.0, "complex spectrum can give negative F"


def check_f_symmetry_swap(solve):
    for mx, sx, my, sy in ((MX3, SX3, MY3, SY3), (Z2, SXC, Z2, I2),
                           (Z2, SXN, Z2, I2)):
        _near(F(solve, mx, sx, my, sy), F(solve, my, sy, mx, sx),
              "F symmetric under argument swap")


def check_sqrt_principal_not_clamp(solve):
    c = S(solve, "sqrt_trace_term", Z2, SXC, Z2, I2)
    clamped = sum(math.sqrt(p[0]) for p in
                  S(solve, "eigvals", Z2, SXC, Z2, I2))
    assert abs(c - clamped) > 1e-3, "must take principal sqrt before .real"


def check_negative_root_sqrt_zero(solve):
    _near(S(solve, "sqrt_trace_term", Z2, SXN, Z2, I2), 0.0,
          "sqrt of negative real roots is purely imaginary")


def check_zero_eigenvalue_sqrt_zero(solve):
    ev = S(solve, "eigvals", Z2, SXS, Z2, I2)
    assert any(abs(p[0]) < 1e-12 and abs(p[1]) < 1e-12 for p in ev), \
        "singular covariance must produce a zero eigenvalue"


def check_n1_closed_form(solve):
    _near(F(solve, [3.0], [[4.0]], [1.0], [[1.0]]), 4.0 + 5.0 - 4.0,
          "N=1 hand formula")


def check_int_bool_leaf_equivalence(solve):
    _near(F(solve, Z2, [[True, False], [False, True]], Z2, I2),
          F(solve, Z2, I2, Z2, I2), "bool leaves equal float identity")


def check_modes_all_return_numeric(solve):
    kw = KW(MX3, SX3, MY3, SY3)
    for m in MODES:
        out = solve(m, **kw)
        assert isinstance(out, (int, float, list, tuple)), \
            "%s returned %r" % (m, type(out))


def check_terms_length_and_order(solve):
    t = S(solve, "terms", MX3, SX3, MY3, SY3)
    assert len(t) == 3, "terms must be [a, b, c]"
    _near(t, [MU_MAIN, TRACE_MAIN, C_MAIN], "terms order")


def check_eigvals_length_matches_dimension(solve):
    for sx, sy, mu in ((I2, I2, Z2), (I3, I3, Z3), (SXD4, I4, Z4)):
        ev = S(solve, "eigvals", mu, sx, mu, sy)
        assert len(ev) == len(sx), "one eigenvalue per matrix dimension"


def check_swap_mu_only_changes_a(solve):
    a1 = S(solve, "mu_term", MX3, SX3, MY3, SY3)
    a2 = S(solve, "mu_term", MY3, SX3, MX3, SY3)
    _near(a1, a2, "a symmetric under mu swap")


def check_no_psd_validation(solve):
    for sx in (SXC, SXN, SXS):
        out = F(solve, Z2, sx, Z2, I2)
        assert isinstance(out, (int, float)), "no positivity check upstream"


# ------------------------------------------------------------------ G. mutation hardening
def check_shape_message_includes_torchsize(solve):
    try:
        solve("frechet", mu_x=[1.0, 2.0], sigma_x=I2, mu_y=[0.0], sigma_y=I2)
    except (ValueError, TypeError) as e:
        msg = str(e)
        assert "torch.Size([2])" in msg and "torch.Size([1])" in msg, msg
    else:
        raise AssertionError("mu shape mismatch must raise")


def check_sigma_shape_message_torchsize(solve):
    try:
        solve("frechet", mu_x=Z3, sigma_x=I3, mu_y=MY3, sigma_y=I2)
    except (ValueError, TypeError) as e:
        msg = str(e)
        assert "torch.Size([3, 3])" in msg and "torch.Size([2, 2])" in msg, msg
    else:
        raise AssertionError("sigma shape mismatch must raise")


def check_tuple_inputs_accepted(solve):
    _near(F(solve, (1.0, 2.0), ((2.0, 0.0), (0.0, 1.0)), (0.0, 0.0), I2),
          F_INT, "tuple inputs are accepted")


def check_empty_mu_shape_mismatch(solve):
    _raises(lambda: F(solve, [], I2, [1.0], I2),
            "empty mu_x vs scalar mu_y", "must have the same shape")


def check_non_numeric_sigma_leaf_raises(solve):
    _raises(lambda: F(solve, Z2, [[1.0, 2.0], [3.0, "x"]], Z2, I2),
            "non numeric sigma leaf", "must contain numbers")


def check_eigvals_real_roots_exactly_zero(solve):
    for mx, sx, my, sy in ((Z3, SXD3, Z3, I3), (Z4, SXD4, Z4, I4)):
        for p in S(solve, "eigvals", mx, sx, my, sy):
            assert p[1] == 0.0, \
                "real eigenvalues must carry an exactly zero imaginary part"
