import math


def _real(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(what + " must be an int or a float")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(what + " is not representable as a float")
    if not math.isfinite(result):
        raise ValueError(what + " must be finite")
    return result


def _shards(shards, what):
    """检查一个分片列表，返回 (float 化后的分片, 行数, 列数)。"""
    if not isinstance(shards, list) or not shards:
        raise ValueError(what + " must be a non-empty list of shards")
    converted = []
    rows = None
    width = None
    for shard in shards:
        if not isinstance(shard, list) or not shard:
            raise ValueError(what + " shards must be non-empty lists of rows")
        if rows is None:
            rows = len(shard)
        elif len(shard) != rows:
            raise ValueError(what + " shards must all have the same row count")
        out_shard = []
        for row in shard:
            if not isinstance(row, list) or not row:
                raise ValueError(what + " rows must be non-empty lists")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(what + " rows must all have the same width")
            out_shard.append([_real(value, what + " entry") for value in row])
        converted.append(out_shard)
    return converted, rows, width


def _check_arguments(q_shards, k_shards, v_shards, causal, scale, dtype_bytes):
    q, rows, width = _shards(q_shards, "q_shards")
    k, k_rows, k_width = _shards(k_shards, "k_shards")
    v, v_rows, v_width = _shards(v_shards, "v_shards")
    count = len(q)
    if len(k) != count or len(v) != count:
        raise ValueError("q_shards, k_shards and v_shards must have the same rank count")
    for name, got_rows, got_width in (("k_shards", k_rows, k_width), ("v_shards", v_rows, v_width)):
        if got_rows != rows:
            raise ValueError(name + " must have the same shard length as q_shards")
        if got_width != width:
            raise ValueError(name + " must have the same head width as q_shards")
    if not isinstance(causal, bool):
        raise ValueError("causal must be a bool")
    rate = _real(scale, "scale")
    if rate <= 0.0:
        raise ValueError("scale must be strictly positive")
    if isinstance(dtype_bytes, bool) or not isinstance(dtype_bytes, int):
        raise ValueError("dtype_bytes must be an int")
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be at least 1")
    return q, k, v, count, rows, width, rate, dtype_bytes


def _rank_pass(q, k, v, rank, causal, rate, payload, count, length, width):
    """单 rank 的 C 步环上重放：返回 (out, traces, sources, skipped, comm)。"""
    live_max = [-math.inf] * length
    live_sum = [0.0] * length
    live_acc = [[0.0] * width for _ in range(length)]
    traces, sources, skipped, comm = [], [], [], []
    for step in range(count):
        source = (rank - step) % count
        sources.append(source)
        dead = causal and source > rank
        skipped.append(dead)
        if not dead:
            for i in range(length):
                query = q[rank][i]
                query_pos = rank * length + i
                logits = []
                for j in range(length):
                    if causal and source * length + j > query_pos:
                        continue
                    key = k[source][j]
                    dot = 0.0
                    for d in range(width):
                        dot += query[d] * key[d]
                    logits.append((rate * dot, v[source][j]))
                block_max = max(entry[0] for entry in logits)
                new_max = block_max if block_max > live_max[i] else live_max[i]
                corr = math.exp(live_max[i] - new_max)
                live_sum[i] *= corr
                row = live_acc[i]
                for d in range(width):
                    row[d] *= corr
                for logit, value in logits:
                    weight = math.exp(logit - new_max)
                    live_sum[i] += weight
                    for d in range(width):
                        row[d] += weight * value[d]
                live_max[i] = new_max
        traces.append([(live_max[i], live_sum[i]) for i in range(length)])
        comm.append(payload if step < count - 1 else 0)
    out = [[live_acc[i][d] / live_sum[i] for d in range(width)] for i in range(length)]
    return out, traces, sources, skipped, comm


def solve(q_shards, k_shards, v_shards, causal, scale, dtype_bytes):
    q, k, v, count, length, width, rate, dtype_bytes = _check_arguments(
        q_shards, k_shards, v_shards, causal, scale, dtype_bytes
    )
    payload = 2 * length * width * dtype_bytes
    out, traces, sources, skipped, comm = [], [], [], [], []
    for rank in range(count):
        rank_out, rank_traces, rank_sources, rank_skipped, rank_comm = _rank_pass(
            q, k, v, rank, causal, rate, payload, count, length, width
        )
        out.append(rank_out)
        traces.append(rank_traces)
        sources.append(rank_sources)
        skipped.append(rank_skipped)
        comm.append(rank_comm)
    return out, traces, sources, skipped, comm
