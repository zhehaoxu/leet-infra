"""851 参考解：Captum NoiseTunnel 加噪平滑账（纯 Python）。

对应官方源码（pytorch/captum, main 分支）：
  captum/attr/_core/noise_tunnel.py::NoiseTunnel.attribute
  captum/attr/_utils/common.py::_validate_noise_tunnel_type
  captum/_utils/common.py::_expand_and_update_baselines

五个 mode：
  noise_layout   —— 加噪布局（repeat_interleave / stdevs 类型与展开）
  batch_schedule —— nt_samples_batch_size 的 min / 整除 partition / remaining
  aggregate      —— smoothgrad / smoothgrad_sq / vargrad 三种聚合口径
  baseline_expand—— baseline 的 repeat_interleave 条件与分布抽样
  noise_tunnel   —— 完整编排（含 delta 的 cat 拼接）

数值口径：全程 Python float（float64）；随机噪声与抽样索引一律显式入参。
所有非法输入统一抛 ValueError。
"""

import math

NOISE_TUNNEL_TYPES = ["smoothgrad", "smoothgrad_sq", "vargrad"]


# ----------------------------------------------------------------- 基础工具
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_shape(node, shape, name):
    """校验 node 是给定形状的嵌套列表。"""
    dims = list(shape)
    cur = node
    for i, want in enumerate(dims):
        if not isinstance(cur, (list, tuple)):
            raise ValueError("%s must be a nested list of shape %s" % (name, tuple(dims)))
        if len(cur) != want:
            raise ValueError("%s must be a nested list of shape %s (got dim %d = %d)"
                             % (name, tuple(dims), i, len(cur)))
        cur = cur[0] if want else None
    return node


def _check_flat(node, n, name):
    if not isinstance(node, (list, tuple)):
        raise ValueError("%s must be a list" % (name,))
    if len(node) != n:
        raise ValueError("%s must have %d elements (got %d)" % (name, n, len(node)))
    for v in node:
        if isinstance(v, (list, tuple)):
            raise ValueError("%s must be a flat list" % (name,))
    return node


def _check_matrix(node, rows, cols, name):
    _check_shape(node, (rows, cols), name)
    for r in node:
        for v in r:
            if not _is_num(v):
                raise ValueError("%s must contain numbers only" % (name,))
    return node


def _as_float(v, name):
    if not _is_num(v):
        raise ValueError("%s must be a number" % (name,))
    return float(v)


def _repeat_interleave(rows, n):
    """torch.repeat_interleave(x, n, dim=0)：每个样本连续重复 n 次。"""
    out = []
    for row in rows:
        for _ in range(n):
            out.append(list(row))
    return out


def _repeat(rows, n):
    """torch.repeat(n, 1)：整体交替重复（错误候选，用于双向断言）。"""
    out = []
    for _ in range(n):
        for row in rows:
            out.append(list(row))
    return out


def _validate_stdevs(stdevs, n_inputs):
    """官方 add_noise_to_inputs 的 stdevs 分支。

    - float        -> stdevs_ = (stdevs,) * len(inputs)   【唯一可用路径】
    - int / bool   -> 官方 assert isinstance(stdevs, float) 失败
    - tuple/list   -> 先 assert 长度匹配；但官方源码里 stdevs_ 只在 float 分支
                      赋值，tuple 分支的 zip(inputs, stdevs_) 会 UnboundLocalError
                      ⇒ 本题统一显式抛 ValueError 并说明这是官方 bug。
    """
    if isinstance(stdevs, bool):
        raise ValueError("stdevs must be a float (bool is not accepted)")
    if isinstance(stdevs, float):
        return [float(stdevs)] * n_inputs
    if isinstance(stdevs, int):
        raise ValueError("stdevs must be type float, got int %r "
                         "(the official code asserts isinstance(stdevs, float))"
                         % (stdevs,))
    if isinstance(stdevs, (list, tuple)):
        if len(stdevs) != n_inputs:
            raise ValueError("the number of stdevs values (%d) must equal the "
                             "number of inputs (%d)" % (len(stdevs), n_inputs))
        raise ValueError("stdevs as a tuple triggers an official bug: `stdevs_` "
                         "is only assigned in the float branch, so zip(inputs, "
                         "stdevs_) raises UnboundLocalError (see pyre-fixme[61])")
    raise ValueError("stdevs must be a float, got %r" % (type(stdevs),))


def _validate_nt_type(nt_type):
    if nt_type not in NOISE_TUNNEL_TYPES:
        raise ValueError("Noise types must be either `smoothgrad`, "
                         "`smoothgrad_sq` or `vargrad`. Given %r" % (nt_type,))
    return nt_type


def _batch_schedule(nt_samples, nt_samples_batch_size):
    """官方 attribute 主体里的分批算式。"""
    if not isinstance(nt_samples, int) or isinstance(nt_samples, bool) or nt_samples <= 0:
        raise ValueError("nt_samples must be a positive integer")
    bs = nt_samples_batch_size
    if bs is not None:
        if not isinstance(bs, int) or isinstance(bs, bool) or bs <= 0:
            raise ValueError("nt_samples_batch_size must be a positive integer or None")
    eff = nt_samples if bs is None else min(nt_samples, bs)
    partition = nt_samples // eff
    remaining = nt_samples - partition * eff
    batch_sizes = [eff] * partition
    if remaining > 0:
        batch_sizes.append(remaining)
    return {
        "effective_batch_size": eff,
        "partition": partition,
        "remaining": remaining,
        "batch_sizes": batch_sizes,
        "total": sum(batch_sizes),
    }


def _accumulate(attributions, bsz, batch_sizes, nt_samples, d):
    """复刻 update_sum_attribution_and_sq 的跨 partition 累加。"""
    if len(attributions) != bsz * nt_samples:
        raise ValueError("attributions must have bsz*nt_samples rows (got %d, "
                         "expected %d)" % (len(attributions), bsz * nt_samples))
    for row in attributions:
        _check_flat(row, d, "attributions row")
    total_sum = [[0.0] * d for _ in range(bsz)]
    total_sum_sq = [[0.0] * d for _ in range(bsz)]
    offset = 0
    part_records = []
    for k in batch_sizes:
        part = attributions[offset:offset + bsz * k]
        offset += bsz * k
        # view(bsz, k, d) -> sum(dim=1)
        p_sum = [[0.0] * d for _ in range(bsz)]
        p_sum_sq = [[0.0] * d for _ in range(bsz)]
        for bi in range(bsz):
            for si in range(k):
                row = part[bi * k + si]
                for di in range(d):
                    v = float(row[di])
                    p_sum[bi][di] += v
                    p_sum_sq[bi][di] += v * v
        for bi in range(bsz):
            for di in range(d):
                total_sum[bi][di] += p_sum[bi][di]
                total_sum_sq[bi][di] += p_sum_sq[bi][di]
        part_records.append({"k": k, "sum": p_sum, "sum_sq": p_sum_sq})
    expected = [[total_sum[bi][di] / nt_samples for di in range(d)]
                for bi in range(bsz)]
    expected_sq = [[total_sum_sq[bi][di] / nt_samples for di in range(d)]
                   for bi in range(bsz)]
    vargrad = [[expected_sq[bi][di] - expected[bi][di] * expected[bi][di]
                for di in range(d)] for bi in range(bsz)]
    return {
        "sum": total_sum,
        "sum_sq": total_sum_sq,
        "expected": expected,
        "expected_sq": expected_sq,
        "smoothgrad": expected,
        "smoothgrad_sq": expected_sq,
        "vargrad": vargrad,
        "part_records": part_records,
    }


# ----------------------------------------------------------- mode 1
def noise_layout(inputs, stdevs, nt_samples, noise):
    """add_noise_to_input / add_noise_to_inputs 的布局账。

    inputs : [b][d]
    stdevs : float（或 tuple/list —— 官方 bug，本题显式报错）
    nt_samples : int
    noise  : [b*nt_samples][d]（显式给出的高斯噪声，避免随机性）
    """
    if not isinstance(nt_samples, int) or isinstance(nt_samples, bool) or nt_samples <= 0:
        raise ValueError("nt_samples must be a positive integer")
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs must be a non-empty nested list [b][d]")
    b = len(inputs)
    d = len(inputs[0])
    _check_matrix(inputs, b, d, "inputs")
    stdevs_ = _validate_stdevs(stdevs, 1)   # 单输入 => len(inputs)==1
    expanded_rows = b * nt_samples
    _check_matrix(noise, expanded_rows, d, "noise")

    # 官方：stdev_expanded = torch.tensor(stdev).repeat(input_expanded_size)
    # 逐元素都是同一个 stdev；dtype 是 float32（本题用 float 表示）
    stdev_expanded = [[float(stdevs_[0])] * d for _ in range(expanded_rows)]
    # 官方：input.repeat_interleave(nt_samples_partition, dim=0) + noise
    repeated = _repeat_interleave(inputs, nt_samples)
    perturbed = [[float(repeated[i][j]) + float(noise[i][j]) for j in range(d)]
                 for i in range(expanded_rows)]
    source_index = [i // nt_samples for i in range(expanded_rows)]
    # 错误候选：torch.repeat 的交替顺序
    repeat_alt = _repeat(inputs, nt_samples)
    repeat_alt_source = [i % b for i in range(expanded_rows)]
    return {
        "expanded_size": [expanded_rows, d],
        "stdevs_": stdevs_,
        "stdev_expanded": stdev_expanded,
        "repeat_interleave": repeated,
        "perturbed": perturbed,
        "source_index": source_index,
        "repeat_alt": repeat_alt,
        "repeat_alt_source": repeat_alt_source,
        "n_inputs": 1,
        "bsz": b,
        "nt_samples": nt_samples,
        "d": d,
    }


# ----------------------------------------------------------- mode 2
def batch_schedule(nt_samples, nt_samples_batch_size=None):
    """nt_samples_batch_size 的 min / 整除 partition / remaining 调度账。"""
    return _batch_schedule(nt_samples, nt_samples_batch_size)


# ----------------------------------------------------------- mode 3
def aggregate(attributions, nt_samples, nt_type="smoothgrad",
              nt_samples_batch_size=None):
    """三种聚合口径。attributions : [bsz*nt_samples][d]（展平，C 序）。"""
    _validate_nt_type(nt_type)
    if not isinstance(nt_samples, int) or isinstance(nt_samples, bool) or nt_samples <= 0:
        raise ValueError("nt_samples must be a positive integer")
    if not isinstance(attributions, (list, tuple)) or not attributions:
        raise ValueError("attributions must be a non-empty nested list")
    d = len(attributions[0])
    if len(attributions) % nt_samples != 0:
        raise ValueError("attributions row count must be divisible by nt_samples")
    bsz = len(attributions) // nt_samples
    sched = _batch_schedule(nt_samples, nt_samples_batch_size)
    acc = _accumulate(attributions, bsz, sched["batch_sizes"], nt_samples, d)
    return {
        "attributions": acc[nt_type],
        "smoothgrad": acc["smoothgrad"],
        "smoothgrad_sq": acc["smoothgrad_sq"],
        "vargrad": acc["vargrad"],
        "sum": acc["sum"],
        "sum_sq": acc["sum_sq"],
        "expected": acc["expected"],
        "expected_sq": acc["expected_sq"],
        "effective_batch_size": sched["effective_batch_size"],
        "partition": sched["partition"],
        "remaining": sched["remaining"],
        "batch_sizes": sched["batch_sizes"],
        "bsz": bsz,
        "d": d,
        "nt_samples": nt_samples,
        "nt_type": nt_type,
    }


# ----------------------------------------------------------- mode 4
def baseline_expand(baselines, inputs, n_samples,
                    draw_baseline_from_distrib=False, baseline_indices=None):
    """_expand_and_update_baselines 的两条分支。

    baselines : [num_ref][d]（或标量 baseline 用 [1][d] 表示）
    inputs    : [bsz][d]
    n_samples : int（本 partition 的样本数）
    baseline_indices : 长度 n_samples*bsz 的显式索引（draw=True 时必给）
    """
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs must be a non-empty nested list [bsz][d]")
    bsz = len(inputs)
    d = len(inputs[0])
    _check_matrix(inputs, bsz, d, "inputs")
    if not isinstance(baselines, (list, tuple)) or not baselines:
        raise ValueError("baselines must be a non-empty nested list [num_ref][d]")
    num_ref = len(baselines)
    _check_matrix(baselines, num_ref, d, "baselines")

    if draw_baseline_from_distrib:
        need = n_samples * bsz
        if baseline_indices is None:
            raise ValueError("baseline_indices must be provided when "
                             "draw_baseline_from_distrib is True")
        _check_flat(baseline_indices, need, "baseline_indices")
        idx = [int(v) for v in baseline_indices]
        for v in idx:
            if v < 0 or v >= num_ref:
                raise ValueError("baseline index %d out of range [0, %d)"
                                 % (v, num_ref))
        expanded = [list(baselines[v]) for v in idx]
        return {
            "mode": "draw",
            "expanded_baselines": expanded,
            "num_drawn": need,
            "indices": idx,
            "num_ref": num_ref,
            "shape": [need, d],
        }

    # 官方默认分支：仅当 baseline.shape[0] == input.shape[0] 且 > 1 才扩展
    if num_ref == bsz and bsz > 1:
        expanded = _repeat_interleave(baselines, n_samples)
        return {
            "mode": "repeat_interleave",
            "expanded_baselines": expanded,
            "num_drawn": 0,
            "indices": None,
            "num_ref": num_ref,
            "shape": [len(expanded), d],
        }
    return {
        "mode": "unchanged",
        "expanded_baselines": [list(r) for r in baselines],
        "num_drawn": 0,
        "indices": None,
        "num_ref": num_ref,
        "shape": [num_ref, d],
    }


# ----------------------------------------------------------- mode 5
def noise_tunnel(inputs, noise, attributions, nt_type="smoothgrad",
                 nt_samples=5, nt_samples_batch_size=None, stdevs=1.0,
                 return_convergence_delta=False, delta_parts=None):
    """完整编排：加噪 -> 分批累加 -> 聚合 -> delta 拼接。"""
    _validate_nt_type(nt_type)
    if not isinstance(nt_samples, int) or isinstance(nt_samples, bool) or nt_samples <= 0:
        raise ValueError("nt_samples must be a positive integer")
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs must be a non-empty nested list [b][d]")
    b = len(inputs)
    d = len(inputs[0])
    _check_matrix(inputs, b, d, "inputs")
    stdevs_ = _validate_stdevs(stdevs, 1)
    sched = _batch_schedule(nt_samples, nt_samples_batch_size)
    expanded_rows = b * nt_samples
    _check_matrix(noise, expanded_rows, d, "noise")
    _check_matrix(attributions, expanded_rows, d, "attributions")

    repeated = _repeat_interleave(inputs, nt_samples)
    # ★ 官方每个 partition 独立调用 add_noise_to_inputs(k)，
    #   即整体 perturbed 是 concat_p(repeat_interleave(inputs, k_p))，
    #   **不是** 一次性 repeat_interleave(inputs, nt_samples)。
    perturbed = []
    source_index = []
    off = 0
    for k in sched["batch_sizes"]:
        rep_k = _repeat_interleave(inputs, k)
        for i in range(b * k):
            perturbed.append([float(rep_k[i][j]) + float(noise[off + i][j])
                              for j in range(d)])
            source_index.append(i // k)
        off += b * k
    acc = _accumulate(attributions, b, sched["batch_sizes"], nt_samples, d)

    delta = None
    if return_convergence_delta:
        if delta_parts is None:
            raise ValueError("delta_parts must be provided when "
                             "return_convergence_delta is True")
        if len(delta_parts) != len(sched["batch_sizes"]):
            raise ValueError("delta_parts must have one entry per batch partition")
        delta = []
        for k, part in zip(sched["batch_sizes"], delta_parts):
            _check_flat(part, b * k, "delta_part")
            delta.extend([float(v) for v in part])
    return {
        "attributions": acc[nt_type],
        "smoothgrad": acc["smoothgrad"],
        "smoothgrad_sq": acc["smoothgrad_sq"],
        "vargrad": acc["vargrad"],
        "sum": acc["sum"],
        "sum_sq": acc["sum_sq"],
        "expected": acc["expected"],
        "expected_sq": acc["expected_sq"],
        "perturbed": perturbed,
        "stdevs_": stdevs_,
        "effective_batch_size": sched["effective_batch_size"],
        "partition": sched["partition"],
        "remaining": sched["remaining"],
        "batch_sizes": sched["batch_sizes"],
        "delta": delta,
        "delta_shape": [len(delta)] if delta is not None else None,
        "bsz": b,
        "d": d,
        "nt_samples": nt_samples,
        "nt_type": nt_type,
    }


# ----------------------------------------------------------------- 统一入口
def solve(mode, **kw):
    if mode == "noise_layout":
        return noise_layout(kw["inputs"], kw["stdevs"], kw["nt_samples"],
                            kw["noise"])
    if mode == "batch_schedule":
        return batch_schedule(kw["nt_samples"], kw.get("nt_samples_batch_size"))
    if mode == "aggregate":
        return aggregate(kw["attributions"], kw["nt_samples"],
                         kw.get("nt_type", "smoothgrad"),
                         kw.get("nt_samples_batch_size"))
    if mode == "baseline_expand":
        return baseline_expand(kw["baselines"], kw["inputs"], kw["n_samples"],
                               kw.get("draw_baseline_from_distrib", False),
                               kw.get("baseline_indices"))
    if mode == "noise_tunnel":
        return noise_tunnel(kw["inputs"], kw["noise"], kw["attributions"],
                            kw.get("nt_type", "smoothgrad"),
                            kw.get("nt_samples", 5),
                            kw.get("nt_samples_batch_size"),
                            kw.get("stdevs", 1.0),
                            kw.get("return_convergence_delta", False),
                            kw.get("delta_parts"))
    raise ValueError("unknown mode: %r" % (mode,))
