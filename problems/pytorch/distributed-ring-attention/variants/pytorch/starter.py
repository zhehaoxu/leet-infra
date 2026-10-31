def solve(q_shards, k_shards, v_shards, causal, scale, dtype_bytes):
    """中文契约：
    q_shards / k_shards / v_shards 是等长的 list，长度 C >= 1 是 rank 数；每项是一个分片，
    即 S >= 1 行、每行 D >= 1 个数的 list。三个参数的 C 必须相同，分片行数 S 与行宽 D
    也必须相同。元素必须是非 bool、能转成有限 float 的 int/float。
    causal 必须是 bool（0/1 不算）。scale 是严格正、能转成有限 float 的 int/float。
    dtype_bytes 是 >= 1 的 int（bool 不算）。非法入参一律抛 ValueError。

    全局序列是各 rank 分片按 rank 下标拼接，N = C * S；rank r 的第 i 行就是全局第 r*S+i 行。
    模拟 C 步环上重放：第 s 步（0 <= s < C）rank r 收到 KV 分片 src = (r - s) % C，
    于是每个 rank 在 C 步里恰好见到每个分片一次，第 0 步见到的就是自己那块。
    causal 为真且 src > r 时，这一块的 key 全部晚于本 rank 的最后一行，
    整块跳过：不做 exp、不改状态，但 KV 照样接收并转发。
    因此第 r 个 rank 的 source 序列是 r, r-1, ..., 0, C-1, ..., r+1。

    logit = scale * (q 行 · k 行)。因果掩码只保留全局 key 位置 <= r*S+i 的 key。
    每行维护运行最大值 m、分母 l 与分子 acc（D 维），按在线 softmax 递推：
      blockmax = 本块保留 key 的 logit 最大值（掩码后至少还剩一个 key）
      new_max  = max(m, blockmax)
      corr     = exp(m - new_max)            # m 为 -inf 时自然得 0.0
      l        = l * corr + Σ exp(logit - new_max)
      acc[d]   = acc[d] * corr + Σ exp(logit - new_max) * v[d]
    处理完这块后 m = new_max。末态 out[r][i][d] = acc[d] / l。

    返回五元组 (out, traces, sources, skipped, comm_bytes)，全 Python 内置类型：
    - out: C 个分片、每片 S 行 D 列的 float。
    - traces: traces[r][s][i] 是 (m, l) 二元组，即第 s 步结束后第 i 行的运行最大值与分母；
      被整块跳过的步原样照抄上一步（第 0 步永远不会被跳过）。
    - sources: sources[r][s] = src（int）。
    - skipped: 布尔，标记该步是否整块被因果掩码吃掉。
    - comm_bytes: comm_bytes[r][s] = 2 * S * D * dtype_bytes（当 s < C-1），否则 0；
      跳过的块照样要转发，通信账不因跳过而减少。

    English contract:
    q_shards, k_shards and v_shards are equally long lists of C >= 1 shards; a
    shard has S >= 1 rows and each row has D >= 1 numbers. The three arguments
    must agree on C, on the shard length S and on the row width D. Entries must
    be ints or floats (not bool) that convert to a finite float. causal must be a
    bool (0/1 does not count). scale is a strictly positive int or float that
    converts to a finite float. dtype_bytes is an int >= 1 (not bool). Any
    illegal argument raises ValueError.

    The global sequence is the concatenation of the shards in rank order,
    N = C * S; row i of rank r is global row r*S+i. Replay C steps around the
    ring: in step s (0 <= s < C) rank r receives KV shard src = (r - s) % C, so
    every rank sees every shard exactly once and step 0 is its own shard. When
    causal is true and src > r, the whole block lies after the rank's last query
    row and is skipped entirely: no exp, no state change, yet the KV block is
    still received and forwarded. Hence rank r's source order is
    r, r-1, ..., 0, C-1, ..., r+1.

    logit = scale * (q row . k row). The causal mask keeps only keys whose global
    position is <= r*S+i. Each row keeps a running max m, denominator l and
    numerator acc (D dims), updated online:
      blockmax = max logit over the surviving keys of this block (never empty)
      new_max  = max(m, blockmax)
      corr     = exp(m - new_max)            # naturally 0.0 when m is -inf
      l        = l * corr + sum exp(logit - new_max)
      acc[d]   = acc[d] * corr + sum exp(logit - new_max) * v[d]
    After the block, m = new_max. The final result is out[r][i][d] = acc[d] / l.

    Return the 5-tuple (out, traces, sources, skipped, comm_bytes) of plain
    Python built-ins:
    - out: C shards of S rows and D floats.
    - traces: traces[r][s][i] is the pair (m, l) after step s for row i; a
      skipped step copies the previous step verbatim (step 0 is never skipped).
    - sources: sources[r][s] = src (int).
    - skipped: bool, whether the whole block was swallowed by the causal mask.
    - comm_bytes: comm_bytes[r][s] = 2 * S * D * dtype_bytes when s < C-1, else
      0; a skipped block is still forwarded, so skipping never lowers the bill.
    """
    raise NotImplementedError
