def solve(sizes, weights, num_samples, temperature):
    """中文契约：
    sizes 是非空 list[int]，每个数据源的条数（1..40 个源，0 <= sizes[i] <= 100000）。
    weights 是 list，长度等于 len(sizes)，每项是非负、能转成有限 float 的 int/float。
    num_samples 是非负 int（0..200000），本 epoch 的样本预算。
    temperature 是正数、能转成有限 float 的 int/float。非法入参一律抛 ValueError（bool 既不算 int 也不算 float）。

    返回五元组 (probs, counts, schedule, doc_indices, flat_indices)，全 Python 内置类型：
    - expo = 1.0 / temperature；e[i] = 0.0（当 sizes[i] == 0）否则 weights[i] ** expo；
      total 是 e 各项之和；probs[i] = e[i] / total（浮点容差 1e-12，结合顺序不计）。
      total 必须有限且 > 0。
    - counts: list[int]，最大余额法。q[i] = probs[i] * num_samples，base[i] = floor(q[i])，
      extra = num_samples - sum(base)，下标按 (q[i] - base[i]) 降序、平局取下标小者在前，
      前 extra 个各 +1；sum(counts) == num_samples。
    - schedule: list[int]，长度 num_samples。按源下标升序逐源放置，该源第 j 次发射
      （0 <= j < counts[i]）的理想槽 floor(j * num_samples / counts[i]) 起向后线性扫描，
      落在第一个空闲槽写 i；counts[i] == 0 的源不占槽。
    - doc_indices: list[int]，某源第 j 次发射落点处写 j % sizes[i]（循环越采样）。
    - flat_indices: list[int]，offsets 是 sizes 的 exclusive 前缀和，
      flat_indices[t] = offsets[schedule[t]] + doc_indices[t]。

    English contract:
    sizes is a non-empty list of ints, records per source (1..40 sources,
    0 <= sizes[i] <= 100000). weights is a list as long as sizes, each entry a
    non-negative int or float that converts to a finite float. num_samples is a
    non-negative int (0..200000), the sample budget for this epoch. temperature is
    a positive int or float that converts to a finite float. Any illegal argument
    raises ValueError (bool counts as neither int nor float).

    Return the 5-tuple (probs, counts, schedule, doc_indices, flat_indices) of
    plain Python built-ins:
    - expo = 1.0 / temperature; e[i] = 0.0 when sizes[i] == 0, else weights[i] ** expo;
      total is the sum of the entries of e; probs[i] = e[i] / total, compared with a
      1e-12 tolerance so the association order does not matter, and total must be
      finite and strictly positive.
    - counts: list of ints by largest remainder. q[i] = probs[i] * num_samples,
      base[i] = floor(q[i]), extra = num_samples - sum(base); order the indices by
      (q[i] - base[i]) descending, ties by the smaller index first, and add 1 to the
      first extra entries; sum(counts) == num_samples.
    - schedule: list of ints of length num_samples. Place sources in ascending index
      order; emission j of a source (0 <= j < counts[i]) has ideal slot
      floor(j * num_samples / counts[i]) and scans forward to the first free slot,
      where i is written. A source with counts[i] == 0 takes no slot.
    - doc_indices: list of ints; where emission j of a source lands, write
      j % sizes[i] (cyclic oversampling).
    - flat_indices: list of ints; offsets is the exclusive prefix sum of sizes and
      flat_indices[t] = offsets[schedule[t]] + doc_indices[t].
    """
    raise NotImplementedError
