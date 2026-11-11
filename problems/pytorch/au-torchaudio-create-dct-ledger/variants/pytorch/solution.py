#!/usr/bin/env python3
"""867 reference solution: create_dct ledger (torchaudio functional.py:636).

The problem pins torch's default dtype to float64, so this pure-Python double
reference matches the official matrix to floating-point tolerance.

create_dct(n_mfcc, n_mels, norm):
    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')
    n = [0, 1, ..., n_mels-1]  (float)
    k = [0, 1, ..., n_mfcc-1]  (float, column)
    dct[i][j] = cos(pi / n_mels * (n[j] + 0.5) * k[i])   # shape (n_mfcc, n_mels)
    if norm is None:   dct *= 2.0
    else:              dct[0] *= 1/sqrt(2); dct *= sqrt(2/n_mels)
    return dct.t()   # shape (n_mels, n_mfcc)
"""
import math
import typing

MODES = ("matrix", "norm_error", "census")


def _err(e):
    return {"type": type(e).__name__, "msg": str(e)}


def _create_dct(n_mfcc, n_mels, norm):
    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')
    # Official evaluates math.pi / float(n_mels) once at the top; for n_mels=0
    # this raises ZeroDivisionError immediately (not lazily per-element).
    pi_over_nm = math.pi / float(n_mels)
    n = [float(i) for i in range(n_mels)]          # col index of built matrix
    k = [float(i) for i in range(n_mfcc)]          # row index of built matrix
    # built matrix shape (n_mfcc, n_mels): dct[i][j] = cos(pi/n_mels*(n_j+0.5)*k_i)
    dct = [[math.cos(pi_over_nm * (n[j] + 0.5) * k[i])
            for j in range(n_mels)] for i in range(n_mfcc)]
    if norm is None:
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct[i][j] *= 2.0
    else:
        for j in range(n_mels):
            dct[0][j] *= 1.0 / math.sqrt(2.0)
        s = math.sqrt(2.0 / float(n_mels))
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct[i][j] *= s
    # transpose -> shape (n_mels, n_mfcc)
    return [[dct[j][i] for j in range(n_mfcc)] for i in range(n_mels)]


def solve(mode, **kw):
    if mode == "census":
        return {
            "want": "create_dct",
            "dct_ii": True,
            "norm_none_scale": 2.0,
            "norm_ortho_first": True,
            "norm_ortho_scale": True,
            "transpose": True,
            "norm_validation": True,
        }
    if mode == "norm_error":
        try:
            _create_dct(kw.get("n_mfcc", 4), kw.get("n_mels", 8), kw.get("norm", "bogus"))
            return "NO_ERROR"
        except Exception as e:                                       # noqa: BLE001
            return _err(e)
    if mode == "matrix":
        n_mfcc = kw.get("n_mfcc", 4)
        n_mels = kw.get("n_mels", 8)
        norm = kw.get("norm", None)
        try:
            return _create_dct(n_mfcc, n_mels, norm)
        except Exception as e:                                       # noqa: BLE001
            return _err(e)
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")


if __name__ == "__main__":
    torch = __import__("torch")
    torch.set_default_dtype(torch.float64)
    m = solve("matrix", n_mfcc=4, n_mels=8, norm=None)
    mo = solve("matrix", n_mfcc=4, n_mels=8, norm="ortho")
    print("SMOKE 867 ref")
    print("  none[3][2]=", m[3][2], "ortho[3][2]=", mo[3][2])
