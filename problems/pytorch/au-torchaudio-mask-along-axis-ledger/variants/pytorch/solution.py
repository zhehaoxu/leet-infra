"""865 reference solution: torchaudio SpecAugment mask ledger, pure Python.

The solve() entry point mirrors official865.py::comp() mode-for-mode:

    param | rand_calls | interval | mask | output | identity | errors
    tensor_value | census

No torch, no RNG and no audio IO: `torch.rand` draws are explicit inputs.
The shared variant computes its interval in float32 (torch.rand(1) has no
dtype argument) while the iid variant uses the spectrogram dtype; float32
arithmetic is emulated bit-for-bit with struct.pack/unpack.
"""

import math
import struct

MODES = (
    "param",
    "rand_calls",
    "interval",
    "mask",
    "output",
    "identity",
    "errors",
    "tensor_value",
    "census",
)

WANT = ["_get_mask_param", "mask_along_axis_iid", "mask_along_axis"]


# --------------------------------------------------------------- validation
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _shape(x):
    if not isinstance(x, (list, tuple)):
        return []
    if len(x) == 0:
        return [0]
    inner = _shape(x[0])
    for row in x[1:]:
        if _shape(row) != inner:
            raise ValueError("spec must be rectangular")
    return [len(x)] + inner


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _nest(flat, shape):
    if len(shape) == 0:
        assert len(flat) == 1
        return float(flat[0])
    if len(shape) == 1:
        assert len(flat) == shape[0]
        return [float(v) for v in flat]
    step = _prod(shape[1:])
    return [_nest(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _pack(x, name):
    """Rectangular numeric nested list -> nested floats."""
    if not isinstance(x, (list, tuple)):
        raise ValueError("%s must be rectangular" % name)
    sh = _shape(x)
    for leaf in _flatten(x):
        if not _is_num(leaf):
            raise ValueError("%s must contain numbers" % name)
    flat = [float(v) for v in _flatten(x)]
    return _nest(flat, sh)


def default_spec(shape=(2, 3, 4)):
    n = _prod(shape)
    return _nest([float(1 + i) for i in range(n)], list(shape))


def default_draws(lead, variant):
    if variant == "shared":
        return [0.7], [0.25]
    n = _prod(lead)
    v = [((0.31 + 0.17 * k) % 1.0) for k in range(n)]
    s = [((0.13 + 0.29 * k) % 1.0) for k in range(n)]
    return _nest(v, list(lead)), _nest(s, list(lead))


def _q(v, dtype):
    """One scalar arithmetic step in the recorded rand dtype."""
    v = float(v)
    if dtype == "float32":
        return struct.unpack("f", struct.pack("f", v))[0]
    return v


def _err(kind, msg):
    return {"type": kind, "msg": msg}


# ------------------------------------------------------------------- config
def _cfg(kw):
    c = {
        "variant": "iid",
        "spec": default_spec(),
        "dtype": "float64",
        "axis": None,
        "mask_param": 3,
        "mask_value": 0.0,
        "mask_value_tensor": False,
        "p": 1.0,
        "value_draws": None,
        "start_draws": None,
    }
    c.update(kw)
    c["spec"] = _pack(c["spec"], "spec")
    sh = _shape(c["spec"])
    if len(sh) < 1:
        raise ValueError("spec must be rectangular")
    if c["axis"] is None:
        c["axis"] = len(sh) - 1
    if c["value_draws"] is None or c["start_draws"] is None:
        dv, ds = default_draws(sh[:-2], c["variant"])
        if c["value_draws"] is None:
            c["value_draws"] = dv
        if c["start_draws"] is None:
            c["start_draws"] = ds
    c["value_draws"] = _pack(c["value_draws"], "value_draws") \
        if isinstance(c["value_draws"], (list, tuple)) else [float(c["value_draws"])]
    c["start_draws"] = _pack(c["start_draws"], "start_draws") \
        if isinstance(c["start_draws"], (list, tuple)) else [float(c["start_draws"])]
    return c


# -------------------------------------------------------------- parameter
def _get_mask_param(mask_param, p, axis_length):
    if p == 1.0:
        return mask_param
    return min(mask_param, math.trunc(axis_length * p))


# ----------------------------------------------------------------- runner
def _run(c):
    """Return the full ledger for one configuration."""
    sh = _shape(c["spec"])
    dim = len(sh)
    axis = c["axis"]
    # The official dispatcher selects mask_along_axis_iid only for the literal
    # string "iid"; anything else falls through to mask_along_axis.
    variant = "iid" if c["variant"] == "iid" else "shared"
    p = c["p"]

    if dim < (2 if variant == "shared" else 3):
        msg = ("Spectrogram must have at least two dimensions "
               "(time and frequency) (%d given)." % dim) if variant == "shared" \
            else ("Spectrogram must have at least three dimensions "
                  "(%d given)." % dim)
        return {"err": _err("ValueError", msg), "calls": [],
                "iv": None, "rows": [], "out": None, "identity": False}
    if axis not in [dim - 2, dim - 1]:
        msg = ("Only Frequency and Time masking are supported "
               "(axis %d and axis %d supported; %d given)."
               % (dim - 2, dim - 1, axis))
        return {"err": _err("ValueError", msg), "calls": [],
                "iv": None, "rows": [], "out": None, "identity": False}
    if not 0.0 <= p <= 1.0:
        msg = "The value of p must be between 0.0 and 1.0 (%s given)." % p
        return {"err": _err("ValueError", msg), "calls": [],
                "iv": None, "rows": [], "out": None, "identity": False}

    alen = sh[axis]
    param = _get_mask_param(c["mask_param"], p, alen)
    if param < 1:
        return {"err": None, "calls": [], "iv": None, "rows": [],
                "out": c["spec"], "identity": True}

    lead = sh[:-2]
    rand_shape = [1] if variant == "shared" else list(lead)
    dtype = "float32" if variant == "shared" else c["dtype"]
    for name in ("value_draws", "start_draws"):
        got = _prod(_shape(c[name])) if isinstance(c[name], list) else 1
        if got != _prod(rand_shape):
            msg = ("draw size %d does not match torch.rand shape %s"
                   % (got, str(tuple(rand_shape))))
            return {"err": _err("ValueError", msg), "calls": [],
                    "iv": None, "rows": [], "out": None, "identity": False}

    vdraw = [_q(v, dtype) for v in _flatten(c["value_draws"])]
    sdraw = [_q(v, dtype) for v in _flatten(c["start_draws"])]
    calls = [
        {"shape": list(rand_shape), "dtype": dtype,
         "values": _nest(vdraw, rand_shape)},
        {"shape": list(rand_shape), "dtype": dtype,
         "values": _nest(sdraw, rand_shape)},
    ]

    def interval(v, s):
        val = _q(_q(v, dtype) * param, dtype)
        mn = _q(_q(s, dtype) * (alen - val), dtype)
        st = math.trunc(mn)
        en = st + math.trunc(val)
        return {"value": val, "min_value": mn, "start": st, "end": en,
                "width": en - st}

    if variant == "shared":
        iv = interval(vdraw[0], sdraw[0])
        if iv["width"] >= param:
            return {"err": _err("ValueError",
                                "Number of columns to be masked should be "
                                "less than mask_param"),
                    "calls": calls, "iv": iv,
                    "rows": _mask_row(alen, iv), "out": None,
                    "identity": False}
        rows = _mask_row(alen, iv)
    else:
        iv = _map2(vdraw, sdraw, list(lead), interval)
        rows = _map1(iv, list(lead), lambda node: _mask_row(alen, node))

    value = c["mask_value"]
    if c["mask_value_tensor"] and isinstance(value, (list, tuple)):
        return {"err": _err("RuntimeError",
                            "non-scalar tensor mask_value cannot be repeated "
                            "to specgram shape"),
                "calls": calls, "iv": iv, "rows": rows, "out": None,
                "identity": False}
    out = _apply(c["spec"], rows, axis, dim, variant, float(value))
    return {"err": None, "calls": calls, "iv": iv, "rows": rows,
            "out": out, "identity": False}


def _map1(node, shape, fn):
    if len(shape) == 0:
        return fn(node)
    return [_map1(x, shape[1:], fn) for x in node]


def _map2(a, b, shape, fn):
    if len(shape) == 0:
        return fn(a[0], b[0])
    step = _prod(shape[1:])
    return [_map2(a[i * step:(i + 1) * step], b[i * step:(i + 1) * step],
                  shape[1:], fn) for i in range(shape[0])]


def _mask_row(alen, iv):
    return [int(iv["start"] <= d < iv["end"]) for d in range(alen)]


def _apply(spec, rows, axis, dim, variant, value):
    if variant == "shared":
        def rec(node, depth):
            if depth == dim - 2:
                if axis == dim - 2:
                    return [[value if rows[f] else v for v in col]
                            for f, col in enumerate(node)]
                return [[value if rows[t] else v for t, v in enumerate(col)]
                        for col in node]
            return [rec(x, depth + 1) for x in node]

        return rec(spec, 0)
    lead = _shape(spec)[:-2]

    def rec(node, rnode, depth):
        if depth == len(lead):
            if axis == dim - 2:
                return [[value if rnode[f] else v for v in col]
                        for f, col in enumerate(node)]
            return [[value if rnode[t] else v for t, v in enumerate(col)]
                    for col in node]
        return [rec(a, b, depth + 1) for a, b in zip(node, rnode)]

    return rec(spec, rows, 0)


# ------------------------------------------------------------------- modes
def _errors():
    cases = {
        "dim_shared": dict(variant="shared", spec=[1.0, 2.0, 3.0],
                           axis=0, mask_param=2),
        "dim_iid": dict(variant="iid", spec=[[1.0, 2.0, 3.0],
                                             [4.0, 5.0, 6.0]],
                        axis=1, mask_param=2),
        "axis_oob": dict(axis=0, mask_param=2),
        "axis_negative": dict(axis=-1, mask_param=2),
        "p_high": dict(p=1.2, mask_param=2),
        "p_low": dict(p=-0.1, mask_param=2),
        "width_guard": dict(variant="shared", mask_param=3,
                            value_draws=[1.5], start_draws=[0.25]),
        "iid_no_guard": dict(variant="iid", mask_param=3,
                             value_draws=[1.5, 0.4],
                             start_draws=[0.25, 0.6]),
    }
    out = {}
    for name, over in cases.items():
        r = _run(_cfg(over))
        out[name] = (r["err"]["type"] + ": " + r["err"]["msg"]
                     if r["err"] else "NO_ERROR")
    return out


def _census():
    return {
        "want": list(WANT),
        "shared_rand_count": 2,
        "iid_rand_count": 2,
        "shared_guard": True,
        "iid_guard": False,
        "p_shortcut": True,
        "int_trunc": True,
        "doc_floor": True,
        "iid_transpose": 2,
        "iid_repeat": True,
        "iid_where": True,
        "shared_dim_guard": True,
        "iid_dim_guard": True,
        "shared_reshape": True,
        "iid_long_split": True,
        "shared_long_split": True,
        "p_default": 1.0,
    }


def solve(mode, **kw):
    if mode == "errors":
        return _errors()
    if mode == "census":
        return _census()
    if mode == "tensor_value":
        c2 = dict(kw)
        c2["mask_value_tensor"] = True
        c2["variant"] = "iid"
        r = _run(_cfg(c2))
        return r["err"] if r["err"] else r["out"]
    c = _cfg(kw)
    r = _run(c)
    if mode == "param":
        sh = _shape(c["spec"])
        alen = sh[c["axis"]] if 0 <= c["axis"] < len(sh) else sh[-1]
        return _get_mask_param(c["mask_param"], c["p"], alen)
    if mode == "rand_calls":
        return r["calls"]
    if mode == "identity":
        return {"same_object": bool(r["identity"]),
                "rand_calls": len(r["calls"]), "err": r["err"]}
    if mode == "output":
        return r["err"] if r["err"] else r["out"]
    if mode == "interval":
        return r["iv"]
    if mode == "mask":
        return r["rows"]
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
