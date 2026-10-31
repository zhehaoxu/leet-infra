"""张量布局账：shape/stride/storage_offset/contiguous 在视图操作序列上的推演。

规则出处：pytorch/pytorch 的 ATen 源码
- aten/src/ATen/TensorUtils.cpp `computeStride_impl`（view/reshape 能否复用同一段存储、
  以及复用后的 stride 如何算出，本文件 `_compute_strides` 是它的直译）
- aten/src/ATen/native/TensorShape.cpp：`reshape`（view 失败则 clone 再 view）、
  `inferUnsqueezeGeometry`（新增维的 stride）、`inferSqueezeGeometry`（删 size-1 维、
  其余 stride 原样保留）、`expand`+`inferExpandGeometry`、`slice`（负数归一化与夹紧）、
  `view_impl`。
- TensorImpl 的连续性判定：numel==0 恒为连续，否则从最内维累乘期望 stride。
"""

OPS = {
    "transpose",
    "permute",
    "unsqueeze",
    "squeeze",
    "narrow",
    "select",
    "slice",
    "expand",
    "reshape",
    "flatten",
    "contiguous",
}


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _contiguous_strides(shape):
    """C 连续布局的 stride；size<=1 的维不参与累乘（与 torch 的期望值推进一致）。"""
    strides = [1] * len(shape)
    expected = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = expected
        if shape[i] != 0:
            expected *= max(shape[i], 1)
    return strides


def _is_contiguous(shape, strides):
    if _numel(shape) == 0:
        return True
    expected = 1
    for i in range(len(shape) - 1, -1, -1):
        if strides[i] != expected and shape[i] != 1:
            return False
        if shape[i] != 0:
            expected *= shape[i]
    return True


def _compute_strides(old_shape, old_strides, new_shape):
    """at::detail::computeStride 的直译；None 表示无法用同一段存储表示该形状。"""
    if not old_shape:
        return [1] * len(new_shape)
    numel = _numel(old_shape)
    if numel == 0:
        if list(old_shape) == list(new_shape):
            return list(old_strides)
        new_strides = [0] * len(new_shape)
        for vd in range(len(new_shape) - 1, -1, -1):
            if vd == len(new_shape) - 1:
                new_strides[vd] = 1
            else:
                new_strides[vd] = max(new_shape[vd + 1], 1) * new_strides[vd + 1]
        return new_strides

    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base_stride = old_strides[-1]
    tensor_numel = 1
    view_numel = 1
    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]
        if tensor_d == 0 or (
            old_shape[tensor_d - 1] != 1
            and old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride
        ):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = old_strides[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return new_strides


def _infer_shape(sizes, numel):
    """reshape 的 -1 推断（at::infer_size）。"""
    if sizes.count(-1) > 1:
        raise ValueError("only one dimension can be inferred")
    if -1 not in sizes:
        shape = list(sizes)
        if _numel(shape) != numel:
            raise ValueError(f"shape {shape} is invalid for input of size {numel}")
        return shape
    rest = 1
    for s in sizes:
        if s != -1:
            rest *= s
    if rest == 0:
        raise ValueError(
            f"cannot reshape tensor of {numel} elements into shape {list(sizes)} "
            "because the unspecified dimension size -1 can be any value and is unknown"
        )
    if numel % rest != 0:
        raise ValueError(f"shape {list(sizes)} is invalid for input of size {numel}")
    shape = [numel // rest if s == -1 else s for s in sizes]
    if _numel(shape) != numel:
        raise ValueError(f"shape {shape} is invalid for input of size {numel}")
    return shape


def _wrap(dim, ndim, what):
    d = dim + ndim if dim < 0 else dim
    if not 0 <= d < ndim:
        raise ValueError(f"{what} dimension {dim} is out of bounds for tensor of dimension {ndim}")
    return d


def _pos(dim, ndim_plus_one, what):
    """unsqueeze/squeeze 允许的维范围上界是 ndim（含）。"""
    d = dim + ndim_plus_one if dim < 0 else dim
    if not 0 <= d < ndim_plus_one:
        raise ValueError(f"{what} dimension {dim} is out of bounds for tensor of dimension {ndim_plus_one - 1}")
    return d


def _ints(args, what):
    out = []
    for a in args:
        if not isinstance(a, int) or isinstance(a, bool):
            raise ValueError(f"{what} arguments must be ints, got {a!r}.")
        out.append(a)
    return out


def solve(shape, strides, offset, ops):
    shape = _ints(shape, "shape")
    strides = _ints(strides, "strides")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError(f"offset must be an int, got {offset!r}.")
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same length.")
    if any(s < 0 for s in shape):
        raise ValueError("shape entries must be non-negative.")
    if any(s < 0 for s in strides):
        raise ValueError("strides entries must be non-negative.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if not isinstance(ops, (list, tuple)):
        raise ValueError("ops must be a list of op lists.")

    sizes, st, off = list(shape), list(strides), offset
    ledger = []
    for op in ops:
        if not isinstance(op, (list, tuple)) or not op:
            raise ValueError(f"each op must be a non-empty list, got {op!r}.")
        kind = op[0]
        if not isinstance(kind, str) or kind not in OPS:
            raise ValueError(f"unknown op {kind!r}, expected one of {sorted(OPS)}.")
        if kind == "squeeze":
            if len(op) != 2:
                raise ValueError("squeeze needs (dim_or_None).")
            squeeze_dim = None if op[1] is None else _ints([op[1]], "op 'squeeze'")[0]
            args = []
        else:
            args = _ints(op[1:], f"op {kind!r}")
        ndim = len(sizes)
        result_kind = "view"

        if kind == "transpose":
            if len(args) != 2:
                raise ValueError("transpose needs (dim0, dim1).")
            a = _wrap(args[0], ndim, "transpose")
            b = _wrap(args[1], ndim, "transpose")
            sizes[a], sizes[b] = sizes[b], sizes[a]
            st[a], st[b] = st[b], st[a]
        elif kind == "permute":
            dims = [_wrap(d, ndim, "permute") for d in args]
            if len(dims) != ndim or sorted(dims) != list(range(ndim)):
                raise ValueError(f"permute needs a permutation of range({ndim}), got {args}.")
            sizes = [sizes[d] for d in dims]
            st = [st[d] for d in dims]
        elif kind == "unsqueeze":
            if len(args) != 1:
                raise ValueError("unsqueeze needs (dim).")
            d = _pos(args[0], ndim + 1, "unsqueeze")
            new_stride = 1 if d >= ndim else sizes[d] * st[d]
            sizes = sizes[:d] + [1] + sizes[d:]
            st = st[:d] + [new_stride] + st[d:]
        elif kind == "squeeze":
            if squeeze_dim is None:
                keep = [i for i in range(ndim) if sizes[i] != 1]
            elif ndim == 0:
                keep = []
            else:
                d = _wrap(squeeze_dim, ndim, "squeeze")
                keep = [i for i in range(ndim) if i != d or sizes[i] != 1]
            sizes = [sizes[i] for i in keep]
            st = [st[i] for i in keep]
        elif kind == "narrow":
            if len(args) != 3:
                raise ValueError("narrow needs (dim, start, length).")
            d = _wrap(args[0], ndim, "narrow")
            start, length = args[1], args[2]
            if length < 0:
                raise ValueError("narrow length must be non-negative.")
            s = start + sizes[d] if start < 0 else start
            if s < 0 or s + length > sizes[d]:
                raise ValueError(
                    f"narrow: start ({start}) + length ({length}) exceeds dimension size ({sizes[d]})."
                )
            off += s * st[d]
            sizes[d] = length
        elif kind == "select":
            if len(args) != 2:
                raise ValueError("select needs (dim, index).")
            d = _wrap(args[0], ndim, "select")
            idx = args[1] + sizes[d] if args[1] < 0 else args[1]
            if not 0 <= idx < sizes[d]:
                raise ValueError(f"select: index {args[1]} out of range for dimension {d} of size {sizes[d]}.")
            off += idx * st[d]
            sizes = sizes[:d] + sizes[d + 1:]
            st = st[:d] + st[d + 1:]
        elif kind == "slice":
            if len(args) != 4:
                raise ValueError("slice needs (dim, start, end, step).")
            d = _wrap(args[0], ndim, "slice")
            start, end, step = args[1], args[2], args[3]
            if step <= 0:
                raise ValueError("slice step must be positive.")
            size = sizes[d]
            if start < 0:
                start += size
            if end < 0:
                end += size
            start = 0 if start < 0 else (size if start >= size else start)
            if end < start:
                end = start
            elif end >= size:
                end = size
            off += start * st[d]
            sizes[d] = (end - start + step - 1) // step
            st[d] *= step
        elif kind == "expand":
            if len(args) < ndim:
                raise ValueError(
                    f"expand: the number of sizes ({len(args)}) must be >= the number of dimensions ({ndim})."
                )
            shift = len(args) - ndim
            new_sizes, new_strides = list(args), [0] * len(args)
            for i, size_i in enumerate(new_sizes):
                if size_i < -1:
                    raise ValueError(f"expand: size must be -1 or non-negative, got {size_i} at position {i}.")
            for i in range(ndim):
                d = shift + i
                target = sizes[i] if new_sizes[d] == -1 else new_sizes[d]
                new_sizes[d] = target
                if target == sizes[i]:
                    new_strides[d] = st[i]
                elif sizes[i] == 1:
                    new_strides[d] = 0
                else:
                    raise ValueError(
                        f"expand: the expanded size of the tensor ({target}) must match the "
                        f"existing size ({sizes[i]}) at non-singleton dimension {d}."
                    )
            for d in range(shift - 1, -1, -1):
                if new_sizes[d] != 1 or d + 1 >= len(new_sizes):
                    new_strides[d] = 0
                else:
                    new_strides[d] = new_sizes[d + 1] * new_strides[d + 1]
            sizes, st = new_sizes, new_strides
        elif kind == "reshape":
            shape_new = _infer_shape(list(args), _numel(sizes))
            computed = _compute_strides(sizes, st, shape_new)
            if computed is None:
                result_kind = "copy"
                sizes, st, off = shape_new, _contiguous_strides(shape_new), 0
            else:
                sizes, st = shape_new, computed
        elif kind == "flatten":
            if len(args) != 2:
                raise ValueError("flatten needs (dim_start, dim_end).")
            if ndim == 0:
                sizes, st = [1], [1]
            else:
                a = _wrap(args[0], ndim, "flatten")
                b = _wrap(args[1], ndim, "flatten")
                if a > b:
                    raise ValueError(
                        f"flatten has invalid args: start_dim {args[0]} cannot come after end_dim {args[1]}."
                    )
                if a != b:
                    shape_new = sizes[:a] + [_numel(sizes[a:b + 1])] + sizes[b + 1:]
                    computed = _compute_strides(sizes, st, shape_new)
                    if computed is None:
                        result_kind = "copy"
                        sizes, st, off = shape_new, _contiguous_strides(shape_new), 0
                    else:
                        sizes, st = shape_new, computed
        elif kind == "contiguous":
            if args:
                raise ValueError("contiguous takes no arguments.")
            if not _is_contiguous(sizes, st):
                result_kind = "copy"
                st, off = _contiguous_strides(sizes), 0

        ledger.append({
            "shape": list(sizes),
            "strides": list(st),
            "offset": off,
            "contiguous": _is_contiguous(sizes, st),
            "kind": result_kind,
        })
    return ledger
