BASE = 31
MOD = 2 ** 61 - 1


def _oracle(docs, n, perms, bands, rows, threshold):
    """结构独立：sid 用幂次展开 sum(t_k * 31**(n-1-k)) 后统一取模（参考解是逐位 Horner
    取模），签名把哈希值排序后取首元（参考解是边扫边比），候选对是 all-pairs 逐带元素
    比较（参考解是建桶），聚簇用邻接表 BFS（参考解是并查集）。四条路径互不共用中间结果。
    """
    powers = [pow(BASE, n - 1 - k, MOD) for k in range(n)]
    sets = []
    for doc in docs:
        current = set()
        for start in range(len(doc) - n + 1):
            total = 0
            for k, weight in enumerate(powers):
                total += doc[start + k] * weight
            current.add(total % MOD)
        sets.append(current)

    signature = []
    for current in sets:
        row = []
        for a, b in perms:
            hashed = sorted((a * x + b) % MOD for x in current)
            row.append(hashed[0] if hashed else MOD)
        signature.append(row)

    count = len(docs)
    candidates = []
    for i in range(count):
        for j in range(i + 1, count):
            for band in range(bands):
                lo = band * rows
                clash = True
                for offset in range(rows):
                    if signature[i][lo + offset] != signature[j][lo + offset]:
                        clash = False
                        break
                if clash:
                    candidates.append((i, j))
                    break

    duplicates = []
    for i, j in candidates:
        inter = len([x for x in sets[i] if x in sets[j]])
        union = len(sets[i]) + len(sets[j]) - inter
        score = inter / union if union else 0.0
        if score >= threshold:
            duplicates.append((i, j))

    adjacency = {}
    for i, j in duplicates:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    labels = [-1] * count
    for seed in range(count):
        if labels[seed] != -1:
            continue
        component = []
        stack = [seed]
        seen = {seed}
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        representative = min(component)
        for node in component:
            labels[node] = representative
    return signature, candidates, duplicates, labels, sorted(set(labels))


def _verify(solve, args, frozen=None):
    docs, n, perms, bands, rows, threshold = args
    docs_snapshot = [list(doc) for doc in docs]
    perms_snapshot = [list(perm) for perm in perms]
    expected = _oracle(*args)
    if frozen is not None:
        assert expected == frozen, ("oracle disagrees with the hand-computed fixture", expected, frozen)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 5, f"expected a 5-tuple, got {type(result)}"
    sig, candidates, duplicates, labels, kept = result
    assert sig == expected[0], ("sig", sig, expected[0])
    assert candidates == expected[1], ("candidates", candidates, expected[1])
    assert duplicates == expected[2], ("duplicates", duplicates, expected[2])
    assert labels == expected[3], ("labels", labels, expected[3])
    assert kept == expected[4], ("kept", kept, expected[4])

    assert isinstance(sig, list) and len(sig) == len(docs)
    for row in sig:
        assert isinstance(row, list) and len(row) == len(perms), ("signature row shape", row)
    for pair_list in (candidates, duplicates):
        assert isinstance(pair_list, list), "pair lists must be lists"
        for pair in pair_list:
            assert isinstance(pair, tuple) and len(pair) == 2, ("pair must be a 2-tuple", pair)
            assert pair[0] < pair[1], ("pair must be ordered", pair)
        assert pair_list == sorted(set(pair_list)), ("pair list must be ascending and unique", pair_list)
    assert set(duplicates) <= set(candidates), ("duplicates must be proposed by banding", duplicates, candidates)

    for index, label in enumerate(labels):
        assert isinstance(label, int) and not isinstance(label, bool)
        assert label <= index, ("label is a component minimum", index, label)
        assert labels[label] == label, ("representative labels itself", label)
    assert kept == sorted(set(labels)), ("kept is the ascending representative list", kept)

    for index, row in enumerate(sig):
        if len(docs[index]) < n:
            assert all(value == MOD for value in row), ("empty shingle set fills MOD", docs[index], row)
        else:
            assert all(0 <= value < MOD for value in row), ("signature entries lie in [0, MOD)", row)

    assert docs == docs_snapshot, "docs must not be mutated"
    assert [list(perm) for perm in perms] == perms_snapshot, "perms must not be mutated"


PERM4 = [(1, 0), (3, 7), (5, 1), (2, 3)]
PERM2 = [(1, 0), (2, 0)]
PERM_WRAP = [(1, 0), (2 ** 31, 0)]
# 乘 2**k 模 2**61-1 就是 61 位循环左移，于是同一个文档在不同置换下的
# argmin 会不是一条线上的，分带方案才能看出差别
PERM_ROT = [(2 ** 31, 0), (2 ** 32, 0), (2 ** 33, 0), (2 ** 34, 0)]


def check_example(solve):
    ex1 = ([[1, 2, 3], [1, 2, 3, 2, 3], [4, 5], [7]], 2, PERM4, 2, 2, 0.5)
    frozen1 = (
        [[33, 106, 166, 69],
         [33, 106, 166, 69],
         [129, 394, 646, 261],
         [MOD, MOD, MOD, MOD]],
        [(0, 1)],
        [(0, 1)],
        [0, 0, 2, 3],
        [0, 2, 3],
    )
    _verify(solve, ex1, frozen1)

    ex2 = ([[1, 2, 3], [1, 4, 5], [2, 3, 6], [1, 2, 3], [], []], 1, PERM2, 2, 1, 0.7)
    frozen2 = (
        [[1, 2], [1, 2], [2, 4], [1, 2], [MOD, MOD], [MOD, MOD]],
        [(0, 1), (0, 3), (1, 3), (4, 5)],
        [(0, 3)],
        [0, 1, 2, 0, 4, 5],
        [0, 1, 2, 4, 5],
    )
    _verify(solve, ex2, frozen2)


def check_matches_oracle(solve):
    cases = [
        ([], 1, PERM2, 2, 1, 0.5),
        ([[5, 6, 7]], 2, PERM2, 2, 1, 0.5),
        ([[1, 2], [1, 2], [1, 2]], 2, PERM4, 2, 2, 1.0),
        ([[], [], []], 3, PERM4, 4, 1, 0.0),
        ([[1, 2, 1, 2]], 2, PERM4, 1, 4, 0.5),
        ([[1, 2, 1, 2], [2, 1, 2, 1]], 2, PERM4, 1, 4, 0.5),
        ([[1, 2, 3, 4], [1, 2, 3], [2, 3, 4]], 3, PERM4, 2, 2, 0.5),
        ([[7] * 30, [7] * 29, [7, 8] * 15], 2, PERM4, 2, 2, 0.5),
        ([[0, 0, 0], [0, 0, 1], [1, 0, 0]], 3, PERM4, 2, 2, 0.0),
        ([[MOD, 1], [0, 1]], 1, PERM2, 1, 2, 1.0),
        ([[MOD + 5, 7], [5, 7], [MOD, MOD]], 2, PERM_WRAP, 1, 2, 1.0),
        ([[1, 2, 3], [1, 2, 3, 4], [1, 2]], 1, [(1, 0), (1, 0)], 1, 2, 0.5),
        ([[3, 1, 4, 1, 5], [3, 1, 4, 1, 6]], 2, PERM4, 4, 1, 0.3),
        ([[3, 1, 4, 1, 5], [3, 1, 4, 1, 6]], 2, PERM4, 2, 2, 0.3),
        ([[3, 1, 4, 1, 5], [3, 1, 4, 1, 6]], 2, PERM4, 1, 4, 0.3),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, PERM4, 2, 2, 0.5),
        ([[2 ** 40, 2 ** 40 + 1, 5], [2 ** 40 + 1, 5, 9]], 2, PERM_WRAP, 1, 2, 0.5),
        ([[1, 2, 3], [1, 2, 3], [4, 5, 6]], 2, [(MOD - 1, 0), (MOD - 1, MOD - 1), (1, MOD - 1), (2 ** 30, 7)], 4, 1, 0.9),
        ([[9, 8, 7, 6, 5, 4], [9, 8, 7, 6, 5, 3], [1, 2, 3]], 4, PERM_WRAP, 1, 2, 0.4),
        ([[1], [1, 1], [1, 1, 1]], 1, PERM4, 1, 4, 0.5),
    ]
    # 一个确定性的伪随机批：LCG 造 40 篇文档，长度与字母表都参差
    state = 12345
    corpus = []
    for _ in range(40):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        size = state % 25
        tokens = []
        for _ in range(size):
            state = (state * 1103515245 + 12345) % (2 ** 31)
            tokens.append(state % 9)
        corpus.append(tokens)
    cases.append((corpus, 3, PERM_WRAP, 1, 2, 0.6))
    cases.append((corpus, 2, PERM4, 2, 2, 0.5))
    cases.append((corpus, 1, PERM_WRAP, 2, 1, 0.2))
    for args in cases:
        _verify(solve, args)


def check_banding_nominates_and_exact_check_disposes(solve):
    # rows = 1：band0 看 min sid，band1 看 (2**31 * x) % MOD 的 min，提名更松
    docs = [[1, 2 ** 30], [5, 2 ** 30], [1, 5]]
    args = (docs, 1, PERM_WRAP, 2, 1, 0.3)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig == [[1, 1], [5, 1], [1, 2147483648]], sig
    assert candidates == [(0, 1), (0, 2)], candidates
    # (1, 2) 的真实 Jaccard 同样是 1/3，但分带从未提名它
    assert duplicates == [(0, 1), (0, 2)], duplicates
    assert labels == [0, 0, 0] and kept == [0], (labels, kept)
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 阈值抬到 0.5，同样的提名一对都不算重复，三篇各自成簇
    args = (docs, 1, PERM_WRAP, 2, 1, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 1), (0, 2)] and duplicates == []
    assert labels == [0, 1, 2] and kept == [0, 1, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # rows = 2：整条签名相同才提名，上面两对都掉出候选
    args = (docs, 1, PERM_WRAP, 1, 2, 0.3)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [] and duplicates == []
    assert labels == [0, 1, 2] and kept == [0, 1, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_true_duplicate_never_patched_in(solve):
    # d0 与 d1 真重复（J = 2/3 >= 0.5），但 min sid 不同、单带单行提不到名
    docs = [[2, 3, 4], [1, 2, 3]]
    args = (docs, 1, [(1, 0)], 1, 1, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig == [[2], [1]], sig
    assert candidates == [] and duplicates == []
    assert labels == [0, 1] and kept == [0, 1]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 加上一个提名得到的对：d2 与 d0 共享 min sid，且 J = 3/4
    docs = [[2, 3, 4], [1, 2, 3], [2, 3, 4, 9]]
    args = (docs, 1, [(1, 0)], 1, 1, 0.4)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 2)], candidates
    assert duplicates == [(0, 2)], duplicates
    assert labels == [0, 1, 0] and kept == [0, 1]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_empty_shingle_set_rules(solve):
    # 三篇都取不出窗口 -> 同桶 -> 互为候选，但并集为空记 0.0
    args = ([[], [4], [7, 8]], 3, PERM2, 2, 1, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig == [[MOD, MOD], [MOD, MOD], [MOD, MOD]], sig
    assert candidates == [(0, 1), (0, 2), (1, 2)], candidates
    assert duplicates == [], duplicates
    assert labels == [0, 1, 2] and kept == [0, 1, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 阈值 0.0 时 0.0 >= 0.0 成立，空集文档全部并成一簇
    args = ([[], [4], [7, 8]], 3, PERM2, 2, 1, 0.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert duplicates == [(0, 1), (0, 2), (1, 2)], duplicates
    assert labels == [0, 0, 0] and kept == [0]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 空文档与有 shingle 的文档不会同桶（MOD 不等于任何合法格值）
    args = ([[], [1, 2, 3]], 3, PERM2, 2, 1, 0.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [] and duplicates == []
    assert labels == [0, 1] and kept == [0, 1]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_sid_is_folded_modulo(solve):
    # (MOD, 1) 与 (0, 1)：n=1 时 sid 分别是 {0, 1} 与 {0, 1}，两个置换的 min 都是 0
    args = ([[MOD, 1], [0, 1]], 1, PERM2, 1, 2, 1.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig == [[0, 0], [0, 0]], sig
    assert candidates == [(0, 1)] and duplicates == [(0, 1)]
    assert labels == [0, 0] and kept == [0]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # n=2 时 (MOD+5, 7) 与 (5, 7) 折出同一个 sid = 162
    args = ([[MOD + 5, 7], [5, 7]], 2, PERM4, 2, 2, 1.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig[0] == [162, 493, 811, 327], sig[0]
    assert sig[1] == sig[0], sig
    assert duplicates == [(0, 1)] and labels == [0, 0]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 不取模就会算成 2*MOD+5 与 5 的两个不同集合
    args = ([[2 * MOD + 5, 7], [5, 7]], 2, PERM4, 2, 2, 1.0)
    _verify(solve, args)


def check_shingle_set_not_multiset(solve):
    # 窗口重复：S 是集合，两篇的 sid 集都是 {33, 63} -> J = 1.0；
    # 把窗口当成多重集的话会变成 2/4 = 0.5，在阈值 1.0 下掉出 duplicates
    docs = [[1, 2, 1], [1, 2, 1, 2, 1]]
    args = (docs, 2, PERM4, 1, 4, 1.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig[0] == sig[1] == [33, 106, 166, 69], sig
    assert candidates == [(0, 1)], candidates
    assert duplicates == [(0, 1)], duplicates
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # d0 的 sid 集 {33, 65, 95} 是 d1 的 {33, 65} 的超集，J = 2/3，与窗口个数无关
    docs = [[1, 2, 3, 2], [1, 2, 3]]
    args = (docs, 2, PERM4, 1, 4, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig[0] == sig[1], sig
    assert duplicates == [(0, 1)], duplicates
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_threshold_boundary_is_inclusive(solve):
    # |{1,2,3} & {1,2,4}| / |{1,2,3,4}| = 2/4 = 0.5，取等号算重复
    docs = [[1, 2, 3], [1, 2, 4]]
    args = (docs, 1, PERM2, 1, 2, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 1)], candidates
    assert duplicates == [(0, 1)], duplicates
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 阈值刚好高一点就掉下去
    args = (docs, 1, PERM2, 1, 2, 0.5000001)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert duplicates == [] and labels == [0, 1]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 0.4999 小于 0.5；先四舍五入再比就会伪造成重复
    docs = [list(range(4999)), list(range(10000))]
    args = (docs, 1, PERM2, 1, 2, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 1)], candidates
    assert duplicates == [] and labels == [0, 1] and kept == [0, 1]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 阈值降到 0.4999 以下才收
    args = (docs, 1, PERM2, 1, 2, 0.4999)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert duplicates == [(0, 1)], duplicates
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # threshold 传整数 1 等价 1.0，只认全等集合
    docs = [[1, 2], [1, 2], [1, 2, 3]]
    args = (docs, 1, PERM2, 2, 1, 1)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert (0, 1) in candidates and (0, 2) in candidates and (1, 2) in candidates, candidates
    assert duplicates == [(0, 1)], duplicates
    assert labels == [0, 0, 2] and kept == [0, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_clusters_are_transitive(solve):
    # 提名三对，但只有 0~1 与 1~2 过复核：0~2 靠传递性并进同一簇
    docs = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 9, 5]]
    args = (docs, 1, PERM2, 1, 2, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 1), (0, 2), (1, 2)], candidates
    assert duplicates == [(0, 1), (1, 2)], duplicates
    assert labels == [0, 0, 0] and kept == [0], (labels, kept)
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 代表取分量的最小下标，而不是并查集里最后落到的根：唯一的重复边落在靠后的下标上
    docs = [[1, 8], [9, 0], [2, 3], [2, 3]]
    args = (docs, 1, PERM2, 1, 2, 1.0)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(2, 3)] and duplicates == [(2, 3)], (candidates, duplicates)
    assert labels == [0, 1, 2, 2] and kept == [0, 1, 2], (labels, kept)
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # 同一口径的长下标版：1 与 4 共享 min sid，其余下标互不同桶；边 (1, 4) 只能把标签定在 1
    docs = [[3], [4], [5], [6], [4, 9], [7]]
    args = (docs, 1, [(1, 0), (1, 0)], 2, 1, 0.3)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(1, 4)], candidates
    assert duplicates == [(1, 4)], duplicates
    assert labels == [0, 1, 2, 3, 1, 5] and kept == [0, 1, 2, 3, 5], (labels, kept)
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_bands_are_contiguous_slices(solve):
    # sid 集 {2**30} 与 {2**30, 2**28}：循环左移后签名是 [1,2,4,8] 与 [1,2,1,2]，
    # 连续分带（0,1 / 2,3）在第一带撞车，跨步分带（0,2 / 1,3）两带都不撞
    docs = [[2 ** 30], [2 ** 30, 2 ** 28], [7]]
    args = (docs, 1, PERM_ROT, 2, 2, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert sig == [[1, 2, 4, 8], [1, 2, 1, 2],
                   [15032385536, 30064771072, 60129542144, 120259084288]], sig
    assert candidates == [(0, 1)], candidates
    assert duplicates == [(0, 1)], duplicates
    assert labels == [0, 0, 2] and kept == [0, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # rows = 1：第 0、1 位各自撞车，跨带并集去重后仍只有一对
    args = (docs, 1, PERM_ROT, 4, 1, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [(0, 1)], candidates
    assert duplicates == [(0, 1)], duplicates
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))

    # rows = 4：整条签名相同才提名，这一对掉出候选
    args = (docs, 1, PERM_ROT, 1, 4, 0.5)
    sig, candidates, duplicates, labels, kept = _oracle(*args)
    assert candidates == [] and duplicates == []
    assert labels == [0, 1, 2] and kept == [0, 1, 2]
    _verify(solve, args, (sig, candidates, duplicates, labels, kept))


def check_scale(solve):
    state = 7
    corpus = []
    for _ in range(300):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        size = state % 40
        doc = []
        for _ in range(size):
            state = (state * 1103515245 + 12345) % (2 ** 31)
            doc.append(state % 500)
        corpus.append(doc)
    perms = [(a, b) for a, b in
             [(1, 0), (3, 11), (2 ** 31, 5), (7, MOD - 1), (2 ** 30, 2 ** 30), (11, 3), (13, 17), (19, 0)]]
    for bands, rows in ((8, 1), (4, 2), (2, 4), (1, 8)):
        args = (corpus, 4, perms, bands, rows, 0.5)
        _verify(solve, args)
    args = (corpus, 2, perms, 8, 1, 0.2)
    _verify(solve, args)
    long_doc = [[i for i in range(300)], [i for i in range(1, 301)]]
    _verify(solve, (long_doc, 5, perms, 4, 2, 0.9))


def check_invalid(solve):
    def expect_raises(args, why):
        try:
            solve(*args)
        except ValueError:
            return
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{why}: wrong exception {type(exc).__name__}: {exc}")
        raise AssertionError(f"{why}: no ValueError raised")

    good_docs = [[1, 2, 3]]
    good_perms = [(1, 0)]
    docs, n, perms, bands, rows, threshold = good_docs, 2, good_perms, 1, 1, 0.5

    # docs
    expect_raises((None, n, perms, bands, rows, threshold), "docs is None")
    expect_raises((( [1, 2 ],), n, perms, bands, rows, threshold), "docs is a tuple")
    expect_raises(([ (1, 2) ], n, perms, bands, rows, threshold), "inner doc is a tuple")
    expect_raises(([["1"]], n, perms, bands, rows, threshold), "token is a str")
    expect_raises(([[1.0]], n, perms, bands, rows, threshold), "token is a float")
    expect_raises(([[None]], n, perms, bands, rows, threshold), "token is None")
    expect_raises(([[True]], n, perms, bands, rows, threshold), "token is a bool")
    expect_raises(([[-1]], n, perms, bands, rows, threshold), "token is negative")
    # n
    expect_raises((docs, 0, perms, bands, rows, threshold), "n is zero")
    expect_raises((docs, -1, perms, bands, rows, threshold), "n is negative")
    expect_raises((docs, 2.0, perms, bands, rows, threshold), "n is a float")
    expect_raises((docs, True, perms, bands, rows, threshold), "n is a bool")
    expect_raises((docs, "2", perms, bands, rows, threshold), "n is a str")
    expect_raises((docs, None, perms, bands, rows, threshold), "n is None")
    # perms
    expect_raises((docs, n, [], bands, rows, threshold), "perms empty")
    expect_raises((docs, n, (), bands, rows, threshold), "perms is a tuple")
    expect_raises((docs, n, None, bands, rows, threshold), "perms is None")
    expect_raises((docs, n, [(1,)], bands, rows, threshold), "perm has one coefficient")
    expect_raises((docs, n, [(1, 0, 2)], 1, 3, threshold), "perm has three coefficients")
    expect_raises((docs, n, [[1, 0], (1,)], 1, 2, threshold), "second perm too short")
    expect_raises((docs, n, [(1.0, 0)], 1, 1, threshold), "a is a float")
    expect_raises((docs, n, [(1, True)], 1, 1, threshold), "b is a bool")
    expect_raises((docs, n, [(0, 0)], 1, 1, threshold), "a is zero")
    expect_raises((docs, n, [(-1, 0)], 1, 1, threshold), "a is negative")
    expect_raises((docs, n, [(MOD, 0)], 1, 1, threshold), "a equals MOD")
    expect_raises((docs, n, [(1, MOD)], 1, 1, threshold), "b equals MOD")
    expect_raises((docs, n, [(1, -2)], 1, 1, threshold), "b is negative")
    # bands / rows
    expect_raises((docs, n, perms, 0, 1, threshold), "bands is zero")
    expect_raises((docs, n, perms, 1, 0, threshold), "rows is zero")
    expect_raises((docs, n, perms, 1.0, 1, threshold), "bands is a float")
    expect_raises((docs, n, perms, True, 1, threshold), "bands is a bool")
    expect_raises((docs, n, perms, 2, 2, threshold), "bands * rows != num_perm")
    expect_raises((docs, n, perms, 3, 1, threshold), "3 bands over 1 perm")
    # threshold
    expect_raises((docs, n, perms, 1, 1, -0.1), "threshold negative")
    expect_raises((docs, n, perms, 1, 1, 1.1), "threshold above one")
    expect_raises((docs, n, perms, 1, 1, "0.5"), "threshold is a str")
    expect_raises((docs, n, perms, 1, 1, None), "threshold is None")
    expect_raises((docs, n, perms, 1, 1, True), "threshold is a bool")

    # 合法边界必须放行
    _verify(solve, ([], 1, [(1, 0)], 1, 1, 0.0))
    _verify(solve, ([[1, 2]], 2, [[1, 0]], 1, 1, 1))
    _verify(solve, ([[1, 2]], 5, [(MOD - 1, MOD - 1)], 1, 1, 0.5))
    _verify(solve, ([[0]], 1, [(1, 0)], 1, 1, 0.0))
