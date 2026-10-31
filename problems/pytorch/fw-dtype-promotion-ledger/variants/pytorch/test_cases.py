import torch

DTYPES = [
    "bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
]
WRAPPED = ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]
SCALARS = ["bool", "int", "float"]

# 题面样例：由参考实现与本机 torch 2.14.1 双向实测得出
EXAMPLES = [
    (("tensor", "bool"), ("scalar", "int"), None, "float32", ("int64", "ok")),
    (("tensor", "bool"), ("scalar", "float"), None, "float32", ("float32", "ok")),
    (("tensor", "uint8"), ("scalar", "int"), None, "float32", ("uint8", "ok")),
    (("tensor", "int8"), ("wrapped", "int64"), None, "float32", ("int8", "ok")),
    (("tensor", "int8"), ("tensor", "uint8"), None, "float32", ("int16", "ok")),
    (("tensor", "float16"), ("tensor", "bfloat16"), None, "float32", ("float32", "ok")),
    (("tensor", "int64"), ("tensor", "float16"), None, "float32", ("float16", "ok")),
    (("tensor", "float64"), ("tensor", "complex64"), None, "float32", ("complex128", "ok")),
    (("wrapped", "float64"), ("tensor", "bool"), None, "float32", ("float64", "ok")),
    (("tensor", "float16"), ("scalar", "float"), None, "float64", ("float16", "ok")),
    (("tensor", "float16"), ("scalar", "float"), "int8", "float32", ("float16", "cast_error")),
    (("tensor", "uint8"), ("tensor", "int8"), "uint8", "float32", ("int16", "ok")),
    (("tensor", "bool"), ("scalar", "int"), "bool", "float32", ("int64", "cast_error")),
    (("tensor", "complex64"), ("tensor", "float64"), "complex64", "float32", ("complex128", "ok")),
]


class _default_dtype:
    """临时切换 torch.get_default_dtype()；本题只切 float32/float64。"""

    def __init__(self, dt):
        self.dt = dt

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, self.dt))
        return self

    def __exit__(self, *exc):
        torch.set_default_dtype(self.prev)
        return False


def _operand(kind, dt):
    """把 (kind, dtype) 落成真实操作数；值一律取 1，不参与类型判定。"""
    if kind == "tensor":
        return torch.ones(3, dtype=getattr(torch, dt))
    if kind == "wrapped":
        return torch.ones((), dtype=getattr(torch, dt))
    return {"bool": True, "int": 1, "float": 1.0}[dt]


def _torch_promote(lhs, rhs, default_float):
    with _default_dtype(default_float):
        got = torch.result_type(_operand(*lhs), _operand(*rhs))
    return str(got).replace("torch.", "")


def _torch_gate(lhs, rhs, out_dt, default_float):
    """torch 亲测写回：add(..., out=) 抛 RuntimeError 即 cast_error。"""
    with _default_dtype(default_float):
        a = _operand(*lhs)
        b = _operand(*rhs)
        o = torch.zeros(3, dtype=getattr(torch, out_dt))
        try:
            torch.add(a, b, out=o)
        except RuntimeError:
            return "cast_error"
    return "ok"


def _kinds():
    return ([("tensor", d) for d in DTYPES]
            + [("wrapped", d) for d in WRAPPED]
            + [("scalar", d) for d in SCALARS])


def check_example(solve):
    for lhs, rhs, out, df, want in EXAMPLES:
        got = tuple(solve(lhs, rhs, out, df))
        assert got == want, f"{lhs} ⊗ {rhs} (out={out}, default_float={df}): got {got}, want {want}"


def check_promotion_matches_torch(solve):
    """全网格：23×23 组操作数 × 两种 default_float，逐组对齐 torch.result_type。"""
    kinds = _kinds()
    n = 0
    for df in ("float32", "float64"):
        for lhs in kinds:
            for rhs in kinds:
                want = _torch_promote(lhs, rhs, df)
                got = tuple(solve(lhs, rhs, None, df))
                n += 1
                assert got[0] == want, f"{lhs} ⊗ {rhs} (default_float={df}): got {got}, want {want}"
                assert got[1] == "ok", f"{lhs} ⊗ {rhs}: out=None 时 status 必须是 ok，got {got}"
    assert n == 1058, f"网格规模变化：{n}"


def check_writeback_gate_matches_torch(solve):
    """写回门槛：276 组操作数 × 12 个 out dtype，逐格对齐 torch.add(..., out=)。"""
    lhs_pool = ([("tensor", d) for d in DTYPES]
                + [("scalar", d) for d in SCALARS]
                + [("wrapped", d) for d in WRAPPED])
    n = 0
    for lhs in lhs_pool:
        for rhs in [("tensor", d) for d in DTYPES]:
            for out_dt in DTYPES:
                want = _torch_gate(lhs, rhs, out_dt, "float32")
                got = tuple(solve(lhs, rhs, out_dt, "float32"))
                n += 1
                assert got[1] == want, f"写回 {lhs} ⊗ {rhs} -> out={out_dt}: got {got}, want status {want}"
    assert n == 3312, f"写回网格规模变化：{n}"


def check_wrapped_number_only_wins_when_strictly_higher(solve):
    """标量/0 维数只有在 category 严格更高时才参与——最容易写反的一组对照。"""
    must_keep = {
        (("tensor", "float16"), ("scalar", "float")): "float16",
        (("tensor", "bfloat16"), ("scalar", "bool")): "bfloat16",
        (("tensor", "uint8"), ("scalar", "int")): "uint8",
        (("tensor", "int8"), ("wrapped", "int64")): "int8",
        (("tensor", "int64"), ("wrapped", "bool")): "int64",
        (("tensor", "float32"), ("wrapped", "int64")): "float32",
        (("tensor", "float16"), ("wrapped", "float64")): "float16",
        (("tensor", "complex64"), ("wrapped", "float64")): "complex64",
    }
    for (lhs, rhs), want in must_keep.items():
        got = tuple(solve(lhs, rhs))
        assert got[0] == want, f"{lhs} ⊗ {rhs}: got {got}, want {want}"
    assert solve(("tensor", "bool"), ("tensor", "bool"), "uint8") == ("bool", "ok")


def check_all_wrapped_numbers(solve):
    """没有有维张量时：0 维张量全量参与 join，裸 Python 数仍走严格更高规则。"""
    assert solve(("wrapped", "uint8"), ("wrapped", "int8"))[0] == "int16"
    assert solve(("wrapped", "bool"), ("wrapped", "uint8"))[0] == "uint8"
    assert solve(("wrapped", "uint8"), ("scalar", "int"))[0] == "uint8"
    assert solve(("wrapped", "int8"), ("scalar", "int"))[0] == "int8"
    assert solve(("scalar", "int"), ("wrapped", "uint8"))[0] == "uint8"
    assert solve(("wrapped", "bool"), ("scalar", "int"))[0] == "int64"
    assert solve(("wrapped", "bool"), ("scalar", "float"))[0] == "float32"
    assert solve(("wrapped", "float32"), ("scalar", "int"))[0] == "float32"
    assert solve(("scalar", "bool"), ("scalar", "int"))[0] == "int64"


def check_complex_lattice(solve):
    """复数格：只有**参与集合里**出现 float64 才抬到 complex128；0 维数与 Python 数抬不动。"""
    assert solve(("tensor", "complex64"), ("tensor", "float32"))[0] == "complex64"
    assert solve(("tensor", "complex64"), ("tensor", "float64"))[0] == "complex128"
    assert solve(("tensor", "complex64"), ("wrapped", "float32"))[0] == "complex64"
    assert solve(("tensor", "complex64"), ("wrapped", "float64"))[0] == "complex64"
    assert solve(("tensor", "complex64"), ("scalar", "float"), None, "float64")[0] == "complex64"
    assert solve(("tensor", "complex64"), ("scalar", "float"), None, "float32")[0] == "complex64"
    assert solve(("tensor", "complex64"), ("scalar", "int"), None, "float64")[0] == "complex64"
    assert solve(("tensor", "complex128"), ("tensor", "float16"))[0] == "complex128"


def check_invalid_inputs_raise(solve):
    assert tuple(solve(("tensor", "float16"), ("tensor", "float32"), None, "float32")) == ("float32", "ok")
    bad = [
        (("tensor", "complex32"), ("tensor", "float32"), None, "float32"),
        (("tensor", "float16"), ("wrapped", "complex64"), None, "float32"),
        (("tensor", "float16"), ("wrapped", "bfloat16"), None, "float32"),
        (("tensor", "float16"), ("scalar", "complex"), None, "float32"),
        (("matrix", "float16"), ("tensor", "float32"), None, "float32"),
        (("tensor",), ("tensor", "float32"), None, "float32"),
        ("tensor", ("tensor", "float32"), None, "float32"),
        (("tensor", "float16"), ["tensor", "float32"], "half", "float32"),
        (("tensor", "float16"), ["tensor", "float32"], None, "bfloat16"),
        (("tensor", "float16"), ["tensor", "float32"], 0, "float32"),
    ]
    for case in bad:
        try:
            solve(*case)
        except ValueError:
            continue
        raise AssertionError(f"非法输入未抛 ValueError: {case}")


check_example.example = True
