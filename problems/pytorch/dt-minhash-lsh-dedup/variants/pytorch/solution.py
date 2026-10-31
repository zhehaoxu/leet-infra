"""参考解：手写 shingle 折叠、MinHash 签名、连续分带、精确复核、并查集聚簇。

四个口径点：

1. **shingle 集是 sid 的集合**，不是窗口的列表——同一 sid 出现多次只留一个，签名与
   Jaccard 都基于它。窗口数 `max(0, len(d) - n + 1)`，短于 `n` 的文档得到空集。
2. **空集签名的每格填 MOD**（min 的单位元），于是空集文档之间必然同桶、互为候选，
   但它们的并集为空、Jaccard 记 0.0，除非阈值为 0.0 否则不会被判重复。
3. **分带用连续切片**：第 k 带是 `sig[k*rows:(k+1)*rows]`。跨步取样
   （`sig[k::bands]`）在数值上也是 `bands * rows` 个位置，看着同样合理，却是另一种
   分带方案，候选对会不同。
4. **候选对跨带取并集去重**：一对文档在多个带里撞上只算一次；`duplicates` 只能是
   `candidates` 的子集，不做全量两两复核。

聚簇用并查集，最后把「分量的最小下标」写回每一篇，因此 `labels` 与 union-find 内部
的根编号无关，`kept == sorted(set(labels))`。
"""

BASE = 31
MOD = 2 ** 61 - 1


def _check(docs, n, perms, bands, rows, threshold):
    if not isinstance(docs, list):
        raise ValueError("docs must be a list of lists of ints")
    for doc in docs:
        if not isinstance(doc, list):
            raise ValueError("every document must be a list of ints")
        for token in doc:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("token ids must be non-negative ints")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive int")
    if not isinstance(perms, list) or not perms:
        raise ValueError("perms must be a non-empty list")
    for perm in perms:
        if not isinstance(perm, (tuple, list)) or len(perm) != 2:
            raise ValueError("each permutation must be a pair (a, b)")
        a, b = perm[0], perm[1]
        for coefficient in (a, b):
            if isinstance(coefficient, bool) or not isinstance(coefficient, int):
                raise ValueError("permutation coefficients must be ints")
        if not 1 <= a < MOD or not 0 <= b < MOD:
            raise ValueError("coefficients out of range")
    for value in (bands, rows):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("bands and rows must be positive ints")
    if bands * rows != len(perms):
        raise ValueError("bands * rows must equal num_perm")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be an int or float")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")


def _shingle_sets(docs, n):
    sets = []
    for doc in docs:
        current = set()
        for start in range(max(0, len(doc) - n + 1)):
            sid = 0
            for token in doc[start:start + n]:
                sid = (sid * BASE + token) % MOD
            current.add(sid)
        sets.append(current)
    return sets


def _signatures(sets, perms):
    size = len(perms)
    signatures = []
    for current in sets:
        if not current:
            signatures.append([MOD] * size)
            continue
        row = []
        for a, b in perms:
            best = MOD
            for sid in current:
                value = (a * sid + b) % MOD
                if value < best:
                    best = value
            row.append(best)
        signatures.append(row)
    return signatures


def _candidate_pairs(signatures, bands, rows):
    found = set()
    for band in range(bands):
        lo = band * rows
        buckets = {}
        for index, row in enumerate(signatures):
            buckets.setdefault(tuple(row[lo:lo + rows]), []).append(index)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    found.add((members[i], members[j]))
    return sorted(found)


def _solve_clustering(duplicates, count):
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in duplicates:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    smallest = {}
    for index in range(count):
        root = find(index)
        if root not in smallest or index < smallest[root]:
            smallest[root] = index
    labels = [smallest[find(index)] for index in range(count)]
    return labels, sorted(set(labels))


def solve(docs, n, perms, bands, rows, threshold):
    _check(docs, n, perms, bands, rows, threshold)
    sets = _shingle_sets(docs, n)
    signatures = _signatures(sets, perms)
    candidates = _candidate_pairs(signatures, bands, rows)
    duplicates = []
    for left, right in candidates:
        union = len(sets[left] | sets[right])
        score = len(sets[left] & sets[right]) / union if union else 0.0
        if score >= threshold:
            duplicates.append((left, right))
    labels, kept = _solve_clustering(duplicates, len(docs))
    return signatures, candidates, duplicates, labels, kept
