"""PyTorch 类型提升：算术结果 dtype 与写回门槛。

规则出处：pytorch/pytorch `docs/source/tensor_attributes.md` 的 type-promotion-doc 一节
（发布为 Tensor attribute 页的 "Type Promotion Mechanics"）。本实现只按文档规则的三段式
（category 高低 -> 参与集合 -> 格内取最小「装得下」的类型）计算，不调用 torch.result_type。
"""

DTYPES = [
    "bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
]
# 四档 category：boolean < integral < floating < complex（uint8 也是 integral）
CATEGORY = {
    "bool": 0,
    "uint8": 1, "int8": 1, "int16": 1, "int32": 1, "int64": 1,
    "float16": 2, "bfloat16": 2, "float32": 2, "float64": 2,
    "complex64": 3, "complex128": 3,
}
# 0 维张量（wrapped number）允许的 dtype：半精度与复数标量不在本题定义域内
WRAPPED_DTYPES = ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]
SCALAR_DTYPES = ["bool", "int", "float"]
# 整数的值域（决定谁装得下谁）
INT_RANGE = {
    "bool": (0, 1),
    "uint8": (0, 255),
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-9223372036854775808, 9223372036854775807),
}
INT_POOL = ["bool", "uint8", "int8", "int16", "int32", "int64"]
# float16 与 bfloat16 同层：互相装不下，所以两者同时出现要升到 float32
FLOAT_LEVEL = {"float16": 0, "bfloat16": 0, "float32": 1, "float64": 2}
FLOAT_BY_LEVEL = {0: ["float16"], 1: ["float32"], 2: ["float64"]}
COMPLEX_LEVEL = {"complex64": 1, "complex128": 2}


def _check_operand(name, op):
    if not isinstance(op, (tuple, list)) or len(op) != 2:
        raise ValueError(f"{name} must be a (kind, dtype) pair, got {op!r}.")
    kind, dt = op
    if not isinstance(kind, str) or not isinstance(dt, str):
        raise ValueError(f"{name} must hold two strings, got {op!r}.")
    if kind == "tensor":
        if dt not in DTYPES:
            raise ValueError(f"{name}: unknown dtype {dt!r} for a dimensioned tensor.")
    elif kind == "wrapped":
        if dt not in WRAPPED_DTYPES:
            raise ValueError(f"{name}: a zero-dim operand must be one of {WRAPPED_DTYPES}, got {dt!r}.")
    elif kind == "scalar":
        if dt not in SCALAR_DTYPES:
            raise ValueError(f"{name}: a Python scalar must be one of {SCALAR_DTYPES}, got {dt!r}.")
    else:
        raise ValueError(f"{name}: unknown kind {kind!r}, expected 'tensor' / 'wrapped' / 'scalar'.")
    return kind, dt


def _scalar_dtype(dt, default_float):
    return {"bool": "bool", "int": "int64", "float": default_float}[dt]


def _int_holds(a, b):
    alo, ahi = INT_RANGE[a]
    blo, bhi = INT_RANGE[b]
    return alo <= blo and ahi >= bhi


def _join(dt_list):
    """在最高 category 内取「装得下需要装的那些操作数」的最小 dtype。"""
    cat = max(CATEGORY[d] for d in dt_list)
    if cat == 0:
        return "bool"
    if cat == 1:
        for cand in INT_POOL:
            if all(_int_holds(cand, d) for d in dt_list if CATEGORY[d] == 1):
                return cand
    if cat == 2:
        floats = {d for d in dt_list if CATEGORY[d] == 2}
        top = max(FLOAT_LEVEL[d] for d in floats)
        if top >= 1:
            return "float32" if top == 1 else "float64"
        if len(floats) == 1:
            return next(iter(floats))
        return "float32"  # float16 与 bfloat16 同层相遇
    level = max(
        [COMPLEX_LEVEL[d] for d in dt_list if CATEGORY[d] == 3]
        + [FLOAT_LEVEL[d] for d in dt_list if CATEGORY[d] == 2 and FLOAT_LEVEL[d] >= 1]
        + [1]
    )
    return "complex64" if level == 1 else "complex128"


def _promote(operands, default_float):
    dimensioned = [dt for kind, dt in operands if kind == "tensor"]
    wrapped = [dt for kind, dt in operands if kind == "wrapped"]
    wrapped += [_scalar_dtype(dt, default_float) for kind, dt in operands if kind == "scalar"]
    if not dimensioned and wrapped:
        # 没有有维张量时，0 维张量不按「wrapped number」降级，而是像普通张量一样全量参与 join；
        # 继续走严格更高规则的只有裸 Python 数。
        n_py = sum(1 for kind, _ in operands if kind == "scalar")
        tensors, wrapped = wrapped[: len(wrapped) - n_py], wrapped[len(wrapped) - n_py:]
        dimensioned = tensors
    cat_dim = max((CATEGORY[d] for d in dimensioned), default=None)
    cat_scal = max((CATEGORY[d] for d in wrapped), default=None)
    if cat_scal is not None and (cat_dim is None or cat_scal > cat_dim):
        # Python 数的 category 严格更高：结果只需装下该 category 的那些数
        return _join([d for d in wrapped if CATEGORY[d] == cat_scal])
    if cat_dim is not None:
        return _join(dimensioned)
    return _join(wrapped)


def _accepts(out_dtype, promoted):
    """算术结果写回 out 的三条禁令；其余一律放行（哪怕会截断）。"""
    co, cp = CATEGORY[out_dtype], CATEGORY[promoted]
    if co == 1 and cp >= 2:
        return False
    if co == 0 and cp != 0:
        return False
    if co < 3 and cp == 3:
        return False
    return True


def solve(lhs, rhs, out=None, default_float="float32"):
    if default_float not in ("float32", "float64"):
        raise ValueError(f"`default_float` must be 'float32' or 'float64', got {default_float!r}.")
    if out is not None and (not isinstance(out, str) or out not in DTYPES):
        raise ValueError(f"`out` must be None or one of {DTYPES}, got {out!r}.")
    a = _check_operand("lhs", lhs)
    b = _check_operand("rhs", rhs)

    promoted = _promote([a, b], default_float)
    if out is None:
        return promoted, "ok"
    return promoted, ("ok" if _accepts(out, promoted) else "cast_error")
