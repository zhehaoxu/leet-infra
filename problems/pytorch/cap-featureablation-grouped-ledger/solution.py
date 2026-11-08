#!/usr/bin/env python3
"""856 reference solution: Captum FeatureAblation grouped-ablation ledger.

Pure Python, no torch, no global RNG.  Five independent modes reproduce the
official ``captum/attr/_core/feature_ablation.py`` mechanisms:

    mask_range  : _format_feature_mask / _get_feature_range_and_mask /
                  _get_feature_counts / _attribute_progress_setup
    ablate      : _construct_ablated_input
    schedule    : _ith_input_ablation_generator
    eval_out    : _process_initial_eval / _process_ablated_out /
                  _generate_result
    attribution : the whole FeatureAblation.attribute pipeline
"""

import math


# ---------------------------------------------------------------------------
# nested-list shape utilities (all values are explicit ints/floats)
# ---------------------------------------------------------------------------
def shape_of(x):
    if not isinstance(x, list):
        return []
    if not x:
        return [0]
    return [len(x)] + shape_of(x[0])


def numel(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def flatten(x):
    if not isinstance(x, list):
        return [x]
    out = []
    for v in x:
        out.extend(flatten(v))
    return out


def unflatten(vals, shape):
    if not shape:
        return vals[0]
    step = numel(shape[1:])
    return [unflatten(vals[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def reshape(x, shape):
    vals = flatten(x)
    if numel(shape) != len(vals):
        raise ValueError("cannot reshape %d elements into %r" % (len(vals), shape))
    return unflatten(vals, shape)


def zeros(shape, value=0):
    return unflatten([value] * numel(shape), shape)


def broadcast_shapes(a, b):
    ra, rb = list(reversed(a)), list(reversed(b))
    out = []
    for i in range(max(len(a), len(b))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError("shapes are not broadcastable: %r and %r" % (a, b))
        out.append(max(da, db))
    return list(reversed(out))


def broadcast_to(x, shape):
    s = shape_of(x)
    if s == shape:
        return x
    target = broadcast_shapes(s, shape)
    if not s:
        return unflatten([x] * numel(target), target)
    off = len(target) - len(s)

    def build(rest, prefix):
        if not rest:
            v = x
            for m, d in enumerate(s):
                v = v[0 if d == 1 else prefix[off + m]]
            return v
        return [build(rest[1:], prefix + [i]) for i in range(rest[0])]

    return build(target, [])


def bop(a, b, op):
    sh = broadcast_shapes(shape_of(a), shape_of(b))
    av, bv = flatten(broadcast_to(a, sh)), flatten(broadcast_to(b, sh))
    return unflatten([op(x, y) for x, y in zip(av, bv)], sh)


def map_vals(x, fn):
    if not isinstance(x, list):
        return fn(x)
    return [map_vals(v, fn) for v in x]


def sum_axis0(x):
    s = shape_of(x)
    if not s:
        raise ValueError("sum_axis0 needs at least one dimension")
    acc = zeros(s[1:])
    for row in x:
        acc = bop(acc, row, lambda p, q: p + q)
    return acc


def repeat_rows(x, k):
    return [row for _ in range(k) for row in x]


def slice_rows(x, n):
    return x[:n]


# ---------------------------------------------------------------------------
# input contracts
# ---------------------------------------------------------------------------
def _ppe(kw):
    p = kw.get("perturbations_per_eval", 1)
    if isinstance(p, bool):
        p = int(p)                      # official isinstance(True, int) quirk
    if not isinstance(p, int) or p < 1:
        raise ValueError("perturbations_per_eval must be an integer and at least 1")
    return p


def _mask_shape_ok(mshape, ishape):
    if len(mshape) != len(ishape):
        raise ValueError("feature mask must have the same rank as the input")
    if mshape[0] not in (1, ishape[0]):
        raise ValueError("feature mask first dim must be 1 or num_examples")
    for md, idim in zip(mshape[1:], ishape[1:]):
        if md not in (1, idim):
            raise ValueError("feature mask shape is not broadcastable to input")


def _validate_mask(mask, ishape):
    mshape = shape_of(mask)
    _mask_shape_ok(mshape, ishape)
    for v in flatten(mask):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError("feature mask values must be non-negative integers")
    return mshape


def _default_masks(inputs):
    out = []
    offset = 0
    for x in inputs:
        s = shape_of(x)
        per = numel(s[1:])
        out.append(unflatten(list(range(offset, offset + per)), [1] + s[1:]))
        offset += per
    return out


def _fmt_masks(inputs, feature_mask):
    if feature_mask is None:
        return _default_masks(inputs)
    if not isinstance(feature_mask, list) or len(feature_mask) != len(inputs):
        raise ValueError("feature mask count must match inputs")
    for m, x in zip(feature_mask, inputs):
        _validate_mask(m, shape_of(x))
    return feature_mask


def _fmt_baselines(inputs, baselines):
    if baselines is None:
        return [0 for _ in inputs]
    if not isinstance(baselines, list) or len(baselines) != len(inputs):
        raise ValueError("baselines count must match inputs")
    for b, x in zip(baselines, inputs):
        if isinstance(b, list):
            bs = shape_of(b)
            if bs and bs[0] not in (1, shape_of(x)[0]):
                raise ValueError("baseline first dim must be 1 or num_examples")
        elif not isinstance(b, (int, float)) or isinstance(b, bool):
            raise ValueError("baseline must be a number or a broadcastable array")
    return baselines


def _mask_eq(mask, j):
    return map_vals(mask, lambda v: 1 if v == j else 0)


def _expand_add(add, k):
    if add is None:
        return None
    out = []
    for a in add:
        out.append(repeat_rows(a, k) if isinstance(a, list) else a)
    return out


def _expand_target(target, k):
    if isinstance(target, list):
        return list(target) * k
    return target


def _mask_count(mask):
    vals = flatten(mask)
    if not vals:
        return 0
    return max(vals) - min(vals) + 1


# ---------------------------------------------------------------------------
# mode 1: mask_range
# ---------------------------------------------------------------------------
def _mask_range(kw):
    inp = kw["input"]
    ishape = shape_of(inp)
    if len(ishape) < 2:
        raise ValueError("input must have a leading example dimension")
    ppe = _ppe(kw)
    mask = kw.get("feature_mask")
    if mask is None:
        offset = int(kw.get("offset", 0))
        if offset < 0:
            raise ValueError("offset must be non-negative")
        m = unflatten(list(range(offset, offset + numel(ishape[1:]))),
                      [1] + ishape[1:])
    else:
        _validate_mask(mask, ishape)
        m = mask
    vals = flatten(m)
    mn, mx = min(vals), max(vals)
    num_features = mx + 1
    count = mx - mn + 1
    ppe_clamped = min(ppe, num_features)
    chunks = []
    pos = mn
    while pos < num_features:
        end = min(pos + ppe_clamped, num_features)
        chunks.append([pos, end])
        pos = end
    return {
        "mask": m,
        "mask_shape": shape_of(m),
        "numel_per_example": numel(ishape[1:]),
        "min_feature": mn,
        "num_features": num_features,
        "feature_count": count,
        "loop_start": mn,
        "num_steps": num_features - mn,
        "ppe_clamped": ppe_clamped,
        "chunks": chunks,
        "num_chunks": len(chunks),
        "progress_total": math.ceil(count / ppe) + 1,
    }


# ---------------------------------------------------------------------------
# mode 2: ablate
# ---------------------------------------------------------------------------
def _ablate(kw):
    expanded = kw["expanded_input"]
    eshape = shape_of(expanded)
    if len(eshape) < 2:
        raise ValueError("expanded_input must have shape (n_features, n_examples, ...)")
    mask = kw["input_mask"]
    mshape = shape_of(mask)
    for v in flatten(mask):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("input_mask values must be integers")
    baseline = kw.get("baseline", 0)
    start = kw["start_feature"]
    end = kw["end_feature"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("start_feature and end_feature must be integers")
    if start >= end:
        raise ValueError("start_feature must be less than end_feature")
    if start < 0:
        raise ValueError("start_feature must be non-negative")
    mask_full_shape = [end - start] + mshape
    try:
        merged = broadcast_shapes(mask_full_shape, eshape)
    except ValueError:
        raise ValueError("input_mask is not broadcastable to expanded_input")
    if merged != eshape:
        raise ValueError("input_mask broadcast changes the expanded_input shape")
    current_mask = [_mask_eq(mask, j) for j in range(start, end)]
    keep = map_vals(current_mask, lambda v: 1 - v)
    term1 = bop(expanded, keep, lambda a, b: a * b)
    term2 = bop(baseline, current_mask, lambda a, b: a * b)
    ablated = bop(term1, term2, lambda a, b: a + b)
    return {
        "mask": current_mask,
        "mask_shape": shape_of(current_mask),
        "ablated": ablated,
        "ablated_shape": shape_of(ablated),
        "num_ablated_features": end - start,
        "ones_per_feature": [sum(flatten(r)) for r in current_mask],
        "baseline_shape": shape_of(baseline),
    }


# ---------------------------------------------------------------------------
# mode 3: schedule
# ---------------------------------------------------------------------------
def _generate(inputs, masks, baselines, ppe, index, target, add, extra_args):
    ishapes = [shape_of(x) for x in inputs]
    N = ishapes[0][0]
    if index < 0 or index >= len(inputs):
        raise ValueError("index out of range for inputs")
    mask_i = masks[index]
    vals = flatten(mask_i)
    mn, mx = min(vals), max(vals)
    num_features = mx + 1
    count = mx - mn + 1
    ppe_eff = min(ppe, num_features)
    feat_shape = ishapes[index][1:]

    baseline = baselines[index]
    if isinstance(baseline, list):
        base_shape = [1] + shape_of(baseline)
        base_vals = reshape(baseline, base_shape)
    else:
        base_shape = []
        base_vals = baseline

    if ppe_eff > 1:
        repeated = [repeat_rows(x, ppe_eff) for x in inputs]
        add_rep = _expand_add(add, ppe_eff)
        target_rep = _expand_target(target, ppe_eff)
    else:
        repeated = [list(x) for x in inputs]
        add_rep = add
        target_rep = target

    records = []
    pos = mn
    while pos < num_features:
        K = min(ppe_eff, num_features - pos)
        if K != ppe_eff:
            cur = [slice_rows(rep, K * N) for rep in repeated]
            cur_add = _expand_add(add, K)
            cur_target = _expand_target(target, K)
        else:
            cur = [list(rep) for rep in repeated]
            cur_add = add_rep
            cur_target = target_rep
        reshaped = unflatten(flatten(cur[index]), [K, N] + feat_shape)
        current_mask = [_mask_eq(mask_i, j) for j in range(pos, pos + K)]
        keep = map_vals(current_mask, lambda v: 1 - v)
        term1 = bop(reshaped, keep, lambda a, b: a * b)
        term2 = bop(base_vals, current_mask, lambda a, b: a * b)
        ablated = bop(term1, term2, lambda a, b: a + b)
        cur[index] = unflatten(flatten(ablated), [K * N] + feat_shape)
        records.append({
            "start": pos,
            "end": pos + K,
            "k": K,
            "batch_size": K * N,
            "current_inputs": cur,
            "current_mask": current_mask,
            "current_target": cur_target,
            "current_additional_args": cur_add,
        })
        pos += K

    selected = {k: (v[index] if isinstance(v, tuple) else v)
                for k, v in extra_args.items()}
    meta = {
        "ppe_effective": ppe_eff,
        "num_examples": N,
        "num_features": num_features,
        "loop_start": mn,
        "feature_count": count,
        "min_feature": mn,
        "baseline_reshape": base_shape,
        "repeated_batch_size": ppe_eff * N,
        "extra_args_selected": selected,
        "num_chunks": len(records),
        "chunks": [[r["start"], r["end"]] for r in records],
        "progress_total": sum(math.ceil(_mask_count(m) / ppe) for m in masks) + 1,
    }
    return records, meta


def _schedule(kw):
    inputs = kw["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list of input arrays")
    ppe = _ppe(kw)
    masks = _fmt_masks(inputs, kw.get("feature_mask"))
    baselines = _fmt_baselines(inputs, kw.get("baselines"))
    add = kw.get("additional_forward_args")
    if add is not None and not isinstance(add, list):
        raise ValueError("additional_forward_args must be a list")
    records, meta = _generate(inputs, masks, baselines, ppe,
                              int(kw.get("index", 0)), kw.get("target"), add,
                              kw.get("extra_args", {}))
    return dict(meta, yields=records)


# ---------------------------------------------------------------------------
# mode 4: eval_out
# ---------------------------------------------------------------------------
def _initial_state(initial, input_shapes, use_weights, initial_dtype):
    if isinstance(initial, bool):
        raise ValueError("initial_eval must be a number or an array")
    if isinstance(initial, int):
        dtype = "int64"
        shp = []
        flat_vals = [initial]
    elif isinstance(initial, float):
        dtype = "float64"
        shp = []
        flat_vals = [initial]
    else:
        dtype = initial_dtype
        shp = shape_of(initial)
        flat_vals = flatten(initial)
    n_outputs = numel(shp) if shp else 1
    if len(flat_vals) != n_outputs:
        raise ValueError("initial_eval shape and values disagree")
    flattened = unflatten(flat_vals, [1, n_outputs])
    total = [zeros([n_outputs] + s[1:]) for s in input_shapes]
    weights = [zeros([n_outputs] + s[1:], 0.0) for s in input_shapes] \
        if use_weights else []
    return dtype, shp, n_outputs, flattened, total, weights


def _accumulate(state, masks, modified, batch_sizes, ppe,
                num_examples, index, use_weights, input_rank):
    dtype, shp, n_outputs, flattened, total, weights = state
    shape_valid = False
    diffs = []
    for c, mod in enumerate(modified):
        mod_shape = shape_of(mod)
        if ppe > 1 and not shape_valid:
            n_perturb = batch_sizes[c] / num_examples
            if not (mod_shape and shp and mod_shape[0] == n_perturb * shp[0]):
                raise ValueError(
                    "When perturbations_per_eval > 1, forward_func's output "
                    "should be a tensor whose 1st dim grow with the input batch size")
            shape_valid = True
        mod_vals = flatten(mod)
        if len(mod_vals) % n_outputs != 0:
            raise ValueError("modified eval cannot reshape to (-1, n_outputs)")
        mod2 = unflatten(mod_vals, [len(mod_vals) // n_outputs, n_outputs])
        diff = bop(flattened, mod2, lambda a, b: a - b)
        diff_shape = shape_of(diff) + [1] * (input_rank - 1)
        diff = unflatten(flatten(diff), diff_shape)
        diffs.append(diff)
        prod = bop(diff, masks[c], lambda a, b: a * b)
        total[index] = bop(total[index], sum_axis0(prod), lambda a, b: a + b)
        if use_weights:
            mf = map_vals(masks[c], lambda v: float(v))
            weights[index] = bop(weights[index], sum_axis0(mf), lambda a, b: a + b)
    return diffs


def _eval_out(kw):
    initial = kw["initial_eval"]
    input_shapes = kw["input_shapes"]
    if not isinstance(input_shapes, list) or not input_shapes:
        raise ValueError("input_shapes must be a non-empty list of shapes")
    ppe = _ppe(kw)
    index = int(kw.get("index", 0))
    if index < 0 or index >= len(input_shapes):
        raise ValueError("index out of range for input_shapes")
    use_weights = bool(kw.get("use_weights", False))
    state = _initial_state(initial, input_shapes, use_weights,
                           kw.get("initial_dtype", "float64"))
    dtype, shp, n_outputs, flattened, total, weights = state
    modified = kw["modified_evals"]
    masks = kw["masks"]
    batch_sizes = kw["batch_sizes"]
    if not (len(modified) == len(masks) == len(batch_sizes)):
        raise ValueError("modified_evals, masks and batch_sizes lengths must agree")
    diffs = _accumulate(state, masks, modified, batch_sizes, ppe,
                        kw["num_examples"], index, use_weights,
                        len(input_shapes[index]))
    if use_weights:
        attrib = [bop(t, w, lambda a, b: a / b if b != 0 else float("nan"))
                  for t, w in zip(total, weights)]
    else:
        attrib = list(total)
    return {
        "attributions": attrib,
        "weights": weights,
        "flattened_initial_eval": flattened,
        "attrib_type": dtype,
        "n_outputs": n_outputs,
        "total_attrib_shapes": [shape_of(t) for t in total],
        "eval_diffs": diffs,
        "initial_shape": shp,
    }


# ---------------------------------------------------------------------------
# mode 5: attribution (end-to-end)
# ---------------------------------------------------------------------------
def _weight_for(weights, t):
    if isinstance(weights, list) and weights and isinstance(weights[0], list):
        return weights[t]
    return weights


def _forward(inputs, add_args, target, weights, kind, n_classes):
    B = shape_of(inputs[0])[0]
    bias = 0.0
    if add_args is not None:
        for a in add_args:
            if isinstance(a, list):
                raise ValueError("additional_forward_args must be scalar numbers "
                                 "in attribution mode")
            bias += float(a)
    flats = []
    for x in inputs:
        flats.append([flatten(row) for row in x])
    if kind in ("aggregate", "python_int", "python_float"):
        total = bias + sum(sum(flat) for ff in flats for flat in ff)
        if kind == "python_int":
            return int(total)
        if kind == "python_float":
            return float(total)
        return total
    rows = []
    for b in range(B):
        if kind == "multi":
            rows.append([bias + sum(float(_weight_for(weights, t)[c][j]) *
                                   flats[t][b][j]
                                   for t in range(len(inputs))
                                   for j in range(len(flats[t][b])))
                         for c in range(n_classes)])
        else:
            val = bias
            for t in range(len(inputs)):
                w = _weight_for(weights, t)
                for j, x in enumerate(flats[t][b]):
                    val += float(w[j]) * (x * x if kind == "quadratic" else x)
            rows.append(val)
    if kind == "multi":
        if isinstance(target, list):
            if len(target) != B:
                raise ValueError("Target list length does not match output!")
            return [row[t] for row, t in zip(rows, target)]
        if isinstance(target, int):
            return [row[target] for row in rows]
        raise ValueError("multi forward needs an integer or list target")
    return rows


def _attribution(kw):
    inputs = kw["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list of input arrays")
    tuple_inputs = bool(kw.get("tuple_inputs", False))
    if not tuple_inputs and len(inputs) != 1:
        raise ValueError("single-input attribution needs exactly one input array")
    ppe = _ppe(kw)
    masks = _fmt_masks(inputs, kw.get("feature_mask"))
    baselines = _fmt_baselines(inputs, kw.get("baselines"))
    add = kw.get("additional_forward_args")
    if add is not None and not isinstance(add, list):
        raise ValueError("additional_forward_args must be a list")
    target = kw.get("target")
    weights = kw["forward_weights"]
    kind = kw.get("forward_kind", "linear")
    if kind not in ("linear", "quadratic", "multi", "aggregate",
                    "python_int", "python_float"):
        raise ValueError("unknown forward_kind: %r" % (kind,))
    n_classes = int(kw.get("n_classes", 2))
    use_weights = bool(kw.get("use_weights", False))
    show_progress = bool(kw.get("show_progress", False))

    init = _forward(inputs, add, target, weights, kind, n_classes)
    input_shapes = [shape_of(x) for x in inputs]
    dtype, shp, n_outputs, flattened, total, acc_weights = _initial_state(
        init, input_shapes, use_weights, "float64")
    forward_shapes = [shape_of(init)]
    chunks = []
    shape_valid = False
    num_examples = input_shapes[0][0]
    for i in range(len(inputs)):
        if numel(input_shapes[i]) == 0:
            continue
        records, meta = _generate(inputs, masks, baselines, ppe, i,
                                  target, add, kw.get("extra_args", {}))
        for rec in records:
            mod = _forward(rec["current_inputs"], rec["current_additional_args"],
                           rec["current_target"], weights, kind, n_classes)
            forward_shapes.append(shape_of(mod))
            chunks.append({"input_index": i, "start": rec["start"],
                           "end": rec["end"], "batch_size": rec["batch_size"]})
            mod_shape = shape_of(mod)
            if ppe > 1 and not shape_valid:
                n_perturb = rec["batch_size"] / num_examples
                if not (mod_shape and shp and mod_shape[0] == n_perturb * shp[0]):
                    raise ValueError(
                        "When perturbations_per_eval > 1, forward_func's output "
                        "should be a tensor whose 1st dim grow with the input batch size")
                shape_valid = True
            mod_vals = flatten(mod)
            if len(mod_vals) % n_outputs != 0:
                raise ValueError("modified eval cannot reshape to (-1, n_outputs)")
            mod2 = unflatten(mod_vals, [len(mod_vals) // n_outputs, n_outputs])
            diff = bop(flattened, mod2, lambda a, b: a - b)
            diff = unflatten(flatten(diff),
                             shape_of(diff) + [1] * (len(input_shapes[i]) - 1))
            prod = bop(diff, rec["current_mask"], lambda a, b: a * b)
            total[i] = bop(total[i], sum_axis0(prod), lambda a, b: a + b)
            if use_weights:
                mf = map_vals(rec["current_mask"], lambda v: float(v))
                acc_weights[i] = bop(acc_weights[i], sum_axis0(mf),
                                     lambda a, b: a + b)
    if use_weights:
        attrib = [bop(t, w, lambda a, b: a / b if b != 0 else float("nan"))
                  for t, w in zip(total, acc_weights)]
    else:
        attrib = list(total)
    return {
        "attributions": attrib if tuple_inputs else attrib[0],
        "weights": acc_weights,
        "n_outputs": n_outputs,
        "attrib_type": dtype,
        "flattened_initial_eval": flattened,
        "initial_shape": shp,
        "num_forwards": len(forward_shapes),
        "forward_shapes": forward_shapes,
        "chunks": chunks,
        "progress_total": sum(math.ceil(_mask_count(m) / ppe) for m in masks) + 1
        if show_progress else None,
    }


# ---------------------------------------------------------------------------
def solve(mode, **kw):
    if mode == "mask_range":
        return _mask_range(kw)
    if mode == "ablate":
        return _ablate(kw)
    if mode == "schedule":
        return _schedule(kw)
    if mode == "eval_out":
        return _eval_out(kw)
    if mode == "attribution":
        return _attribution(kw)
    raise ValueError("unknown mode: %r" % (mode,))
