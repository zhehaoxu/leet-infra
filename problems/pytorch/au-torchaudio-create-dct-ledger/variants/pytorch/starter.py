#!/usr/bin/env python3
"""867 starter: implement create_dct as three judged modes.

Modes
-----
"matrix"     : build the DCT-II transform matrix for the given n_mfcc, n_mels
               and norm, and return it as a list of lists of Python floats.
               Return shape must be (n_mels, n_mfcc).
"norm_error" : return "NO_ERROR" if norm is valid (None or "ortho"), otherwise
               return an error dict {"type": ..., "msg": ...} whose message is
               EXACTLY 'norm must be either "ortho" or None'.
"census"     : return the source-level fact dict (see the tests for the keys).

Contract
--------
* DCT-II: dct[i][j] = cos(pi / n_mels * (n + 0.5) * k), built as (n_mfcc, n_mels).
* After building, return the TRANSPOSE, so returned[i][j] is (n_mels, n_mfcc).
* norm is None      -> multiply the whole matrix by 2.0.
* norm == "ortho"   -> scale the BUILT row 0 by 1/sqrt(2), then the whole matrix
                       by sqrt(2 / n_mels). (Built row 0 becomes the returned
                       column 0 after the transpose.)
* Any other norm    -> ValueError with message 'norm must be either "ortho" or None'.
* n_mels=0          -> ZeroDivisionError (math.pi / float(n_mels) evaluated at the
                       top level). n_mfcc=0 returns a (n_mels, 0) matrix.
* unknown mode      -> ValueError whose message lists the modes in MODES order.

You must NOT import torch or numpy. Pure Python + math only.
"""
import math

MODES = ("matrix", "norm_error", "census")


def _err(e):
    return {"type": type(e).__name__, "msg": str(e)}


def _create_dct(n_mfcc, n_mels, norm):
    # TODO: implement the DCT-II matrix as described above.
    raise NotImplementedError("implement _create_dct")


def solve(mode, **kw):
    if mode == "matrix":
        n_mfcc = kw.get("n_mfcc", 4)
        n_mels = kw.get("n_mels", 8)
        norm = kw.get("norm", None)
        try:
            return _create_dct(n_mfcc, n_mels, norm)
        except Exception as e:                                       # noqa: BLE001
            return _err(e)
    if mode == "norm_error":
        try:
            _create_dct(kw.get("n_mfcc", 4), kw.get("n_mels", 8), kw.get("norm", "bogus"))
            return "NO_ERROR"
        except Exception as e:                                       # noqa: BLE001
            return _err(e)
    if mode == "census":
        # TODO: return the census dict required by the tests.
        raise NotImplementedError("implement census")
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
