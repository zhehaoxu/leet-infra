def solve(docs, n, perms, bands, rows, threshold):
    """中文契约：
    docs 是 list[list[int]]，0..400 篇文档，每篇 0..12000 个非负整数 token id。
    n 是正整数（shingle 宽度）。perms 是非空 list[(a, b)]，长度 num_perm 满足
    1 <= num_perm <= 64，系数满足 1 <= a < MOD、0 <= b < MOD，MOD = 2**61 - 1。
    bands、rows 是正整数且 bands * rows == num_perm。threshold 是 [0, 1] 内的
    int 或 float。非法入参一律抛 ValueError（bool 不当作 int 或 float）。

    返回五元组 (sig, candidates, duplicates, labels, kept)，全 Python 内置类型：
    - sig: list[list[int]]，形状 (len(docs), num_perm)，
      sig[i][p] = min((a_p * sid + b_p) % MOD for sid in S(docs[i]))；
      sid 由窗口内 Horner 折叠得到：sid = 0 起步，逐步 sid = (sid * 31 + t) % MOD；
      S 是这些 sid 的集合（重复只算一次），窗口数 max(0, len(d) - n + 1)；
      S 为空（len(d) < n）时该文档每一格都填 MOD。
    - candidates: list[tuple[int, int]]，第 k 带取**连续**切片
      sig[i][k*rows:(k+1)*rows]，切片相等的文档同桶，桶内两两成对 (p, q) 且 p < q；
      各带候选对取并集去重后按字典序升序。
    - duplicates: candidates 的子集，判据是 len(S_p & S_q) / len(S_p | S_q) >= threshold
      的 Python 浮点除法（并集为空记 0.0），同样去重升序。
    - labels: list[int]，labels[i] 是 i 所在连通分量（边取 duplicates）的最小下标。
    - kept: list[int]，各分量代表的升序列表，等于 sorted(set(labels))。

    English contract:
    docs is a list of 0..400 documents, each a list of 0..12000 non-negative token
    ids. n is a positive integer (shingle width). perms is a non-empty list of
    (a, b) pairs whose length num_perm obeys 1 <= num_perm <= 64, with
    1 <= a < MOD and 0 <= b < MOD for MOD = 2**61 - 1. bands and rows are
    positive integers with bands * rows == num_perm. threshold is an int or float
    inside [0, 1]. Any illegal argument raises ValueError (bool counts as neither
    int nor float).

    Return the 5-tuple (sig, candidates, duplicates, labels, kept) of plain
    Python built-ins:
    - sig: list of lists of ints, shape (len(docs), num_perm), where
      sig[i][p] = min((a_p * sid + b_p) % MOD for sid in S(docs[i])); sid is the
      Horner fold of one window, starting at 0 and doing sid = (sid * 31 + t) % MOD
      per token; S is the set of those sid values (repeats counted once) over the
      max(0, len(d) - n + 1) windows, and an empty S fills every entry with MOD.
    - candidates: list of (p, q) tuples with p < q, built from the contiguous
      slice sig[i][k*rows:(k+1)*rows] of each band, pairing documents whose slice
      is identical, then de-duplicated across bands and sorted lexicographically.
    - duplicates: the subset of candidates with
      len(S_p & S_q) / len(S_p | S_q) >= threshold under Python float division
      (an empty union scores 0.0), de-duplicated and ascending.
    - labels: list of ints, labels[i] is the smallest index of i's connected
      component when duplicates are treated as undirected edges.
    - kept: ascending list of component representatives, equal to
      sorted(set(labels)).
    """
    raise NotImplementedError
