import torch

EVAL_DOCS = [
    [1, 2, 3, 6, 4],       # e0: 3-gram = {(1,2,3),(2,3,6),(3,6,4)}
    [1, 5, 3, 6, 4],       # e1: {(1,5,3),(5,3,6),(3,6,4)}
    [1, 2],                # e2: 长度 2 < n=3 -> 空集
]
TRAIN_DOCS = [
    [1, 2, 3, 6, 4],          # t0: 与 e0 完全相同
    [1, 2, 3, 6, 4, 7, 8, 9],  # t1: 前缀就是 t0，另有 3 个自己的 3-gram
    [1, 5, 3, 6, 4, 2, 9],     # t2: 完整覆盖 e1 的三个 3-gram
]
N = 3
THRESHOLD = 0.6

EXAMPLE_CONTAM = [
    [1.0, 1.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0],
    [0.0, 0.0, 0.0],
]
EXAMPLE_JACCARD = [
    [1.0, 0.5, 1.0 / 7.0],
    [0.2, 0.125, 0.6],
    [0.0, 0.0, 0.0],
]
EXAMPLE_FLAGGED = [0, 1]


def _key(seq):
    return "|".join(str(int(x)) for x in seq)


def _oracle(eval_docs, train_docs, n):
    """独立重算：不用任何集合代数。shingle 编码成字符串键，存进普通 list（保持出现次序、
    手工判重），交集大小用显式双重循环数，并集用容斥 |A|+|B|-|A∩B|。"""

    def shingle_list(doc):
        m = len(doc)
        out = []
        seen = {}
        i = 0
        while i <= m - n:
            parts = []
            j = 0
            while j < n:
                parts.append(str(int(doc[i + j])))
                j += 1
            k = "|".join(parts)
            if k not in seen:
                seen[k] = True
                out.append(k)
            i += 1
        return out, seen

    A = [shingle_list(d) for d in eval_docs]
    B = [shingle_list(d) for d in train_docs]

    contam = []
    jac = []
    for (alist, _aset) in A:
        row_c = []
        row_j = []
        for (blist, bset) in B:
            inter = 0
            for x in alist:
                if x in bset:
                    inter += 1
            if len(alist) == 0:
                row_c.append(0.0)
            else:
                row_c.append(inter / len(alist))
            union = len(alist) + len(blist) - inter
            row_j.append(inter / union if union > 0 else 0.0)
        contam.append(row_c)
        jac.append(row_j)
    return contam, jac


def _close(a, b):
    return torch.allclose(
        torch.tensor(float(a), dtype=torch.float64),
        torch.tensor(float(b), dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def _mat_equal(a, b):
    b = torch.tensor(b, dtype=torch.float64)
    assert list(a.shape) == list(b.shape), (a.shape, b.shape)
    return torch.allclose(
        a.double(),
        b,
        atol=1e-5,
        rtol=1e-5,
    )


def check_example(solve):
    contam, jac, flagged = solve(EVAL_DOCS, TRAIN_DOCS, N, THRESHOLD)
    assert _mat_equal(contam, EXAMPLE_CONTAM), contam
    assert _mat_equal(jac, EXAMPLE_JACCARD), jac
    assert torch.equal(flagged, torch.tensor(EXAMPLE_FLAGGED, dtype=torch.int64)), flagged


check_example.example = True


def check_matches_oracle(solve):
    cases = [
        (EVAL_DOCS, TRAIN_DOCS, 3, 0.6),
        (EVAL_DOCS, TRAIN_DOCS, 1, 0.5),
        (EVAL_DOCS, TRAIN_DOCS, 4, 0.5),
        ([[1, 1, 1]], [[1, 1]], 2, 0.9),
        ([[5], [6, 7]], [[5, 5, 5], [7, 6], []], 1, 0.0),
        ([[], []], [[1, 2, 3]], 2, 0.0),
        ([[1, 2], [1, 2]], [[1, 2], [1, 2]], 2, 1.0),
    ]
    for ev, tr, n, th in cases:
        contam, jac, flagged = solve(ev, tr, n, th)
        ec, ej = _oracle(ev, tr, n)
        assert _mat_equal(contam, ec), (ev, tr, n, contam, ec)
        assert _mat_equal(jac, ej), (ev, tr, n, jac, ej)
        assert list(flagged.shape) == [len(flagged)]
        assert flagged.dtype == torch.int64


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(300)
    for _ in range(6):
        ne = int(torch.randint(1, 5, (1,), generator=g))
        nt = int(torch.randint(1, 5, (1,), generator=g))
        n = int(torch.randint(1, 5, (1,), generator=g))
        ev = [
            [int(x) for x in torch.randint(1, 6, (int(torch.randint(0, 9, (1,), generator=g)),), generator=g)]
            for _ in range(ne)
        ]
        tr = [
            [int(x) for x in torch.randint(1, 6, (int(torch.randint(0, 12, (1,), generator=g)),), generator=g)]
            for _ in range(nt)
        ]
        for th in (0.0, 0.25, 0.75, 1.0):
            contam, jac, flagged = solve(ev, tr, n, th)
            ec, ej = _oracle(ev, tr, n)
            assert _mat_equal(contam, ec), (ev, tr, n, th, contam, ec)
            assert _mat_equal(jac, ej), (ev, tr, n, th, jac, ej)
            # flagged 必须与"非空行的最大值 >= th"完全一致（空行永远不参与）
            assert flagged.tolist() == _expected_flagged(ec, th, ev, n), (ev, tr, n, th, flagged)


def _expected_flagged(contam_rows, threshold, ev, n):
    out = []
    for i, row in enumerate(contam_rows):
        if len(ev[i]) < n:
            continue
        if row and max(row) >= threshold:
            out.append(i)
    return out


def check_doc_shorter_than_n(solve):
    # 退化 1：长度 < n 的文档 shingle 集为空 -> 整行 0，且阈值放到 0 也不被标记
    contam, jac, flagged = solve([[1, 2], [1, 2, 3]], [[1, 2, 3]], 3, 0.0)
    assert _mat_equal(contam, [[0.0], [1.0]]), contam
    assert _mat_equal(jac, [[0.0], [1.0]]), jac
    assert torch.equal(flagged, torch.tensor([1], dtype=torch.int64)), flagged


def check_diagonal_is_one_when_sets_coincide(solve):
    # 退化 2：评测集就是训练集 -> 对角线上的包含率与 Jaccard 都是 1，阈值取 1.0 仍全员命中
    docs = [[1, 2, 3, 4], [7, 8, 9]]
    contam, jac, flagged = solve(docs, docs, 3, 1.0)
    assert _mat_equal(contam, [[1.0, 0.0], [0.0, 1.0]]), contam
    assert _mat_equal(jac, [[1.0, 0.0], [0.0, 1.0]]), jac
    assert torch.equal(flagged, torch.tensor([0, 1], dtype=torch.int64)), flagged
    assert float(contam[0, 0]) == 1.0 and float(contam[1, 1]) == 1.0


def check_disjoint_corpus(solve):
    # 退化 3：词表完全不相交 -> 两个矩阵全 0；th > 0 无人被标记，th = 0 全员被标记（边界口径）
    ev = [[1, 2, 3]]
    tr = [[8, 9, 10]]
    contam, jac, flagged = solve(ev, tr, 2, 0.01)
    assert float(contam.max()) == 0.0 and float(jac.max()) == 0.0
    assert flagged.numel() == 0, flagged
    _, _, flagged0 = solve(ev, tr, 2, 0.0)
    assert torch.equal(flagged0, torch.tensor([0], dtype=torch.int64)), flagged0


def check_repeated_shingles_collapse(solve):
    # 退化 4：集合口径，重复 shingle 只算一次 —— [1,1,1] 的 2-gram 只有 (1,1)
    contam, jac, flagged = solve([[1, 1, 1]], [[1, 1]], 2, 0.5)
    assert _mat_equal(contam, [[1.0]]), contam
    assert _mat_equal(jac, [[1.0]]), jac
    assert torch.equal(flagged, torch.tensor([0], dtype=torch.int64)), flagged
    # 频率版会给出 2/2 = 1 与 2/1 -> 越界，集合版永远 <= 1
    assert float(contam.max()) <= 1.0


def check_empty_corpora(solve):
    # 退化 5：任一侧为空 -> 形状正确、没有文档被标记（即使阈值是 0）
    contam, jac, flagged = solve([[1, 2, 3]], [], 2, 0.0)
    assert list(contam.shape) == [1, 0] and list(jac.shape) == [1, 0]
    assert flagged.numel() == 0, flagged
    contam, jac, flagged = solve([], [[1, 2, 3]], 2, 0.0)
    assert list(contam.shape) == [0, 1] and flagged.numel() == 0
    z = torch.zeros(0, 0, dtype=torch.float32)
    c0, j0, f0 = solve([], [], 2, 0.0)
    assert c0.shape == j0.shape == z.shape, (c0.shape, j0.shape)
    assert f0.numel() == 0


def check_threshold_above_one(solve):
    # 退化 6：阈值 > 1 -> 包含率上界是 1，永远无法命中
    _, _, flagged = solve(EVAL_DOCS, TRAIN_DOCS, N, 1.0 + 1e-9)
    assert flagged.numel() == 0, flagged
    _, _, f2 = solve(EVAL_DOCS, TRAIN_DOCS, N, 1.5)
    assert f2.numel() == 0, f2


def check_containment_beats_jaccard_on_long_docs(solve):
    # 同一个 3-gram 全量命中，训练文档越长 Jaccard 越被稀释，包含率不动
    ev = [[1, 2, 3, 6, 4]]
    pad = list(range(20, 40))
    tr = [[7, 8, 9, 1, 2, 3, 6, 4, 11, 12] + pad]
    contam, jac, _ = solve(ev, tr, 3, 0.6)
    assert _close(float(contam[0, 0]), 1.0), contam
    assert float(jac[0, 0]) < 0.2, jac
    # 恒等式：Jaccard <= 包含率（并集 >= |A|）
    assert float(jac.max()) <= float(contam.max()) + 1e-6


def check_duplicate_train_columns(solve):
    # 训练集里复制一篇文档 -> 多出一列相同的值，flagged 不变（但"命中次数"会翻倍）
    base_c, base_j, base_f = solve(EVAL_DOCS, TRAIN_DOCS, N, THRESHOLD)
    dup_c, dup_j, dup_f = solve(EVAL_DOCS, TRAIN_DOCS + [list(TRAIN_DOCS[0])], N, THRESHOLD)
    assert _mat_equal(dup_c[:, :3], base_c.tolist()), dup_c
    assert _mat_equal(dup_c[:, 3], base_c[:, 0].tolist()), dup_c
    assert torch.equal(dup_f, base_f), (dup_f, base_f)


def check_n1_unigram_sets(solve):
    # n = 1：shingle 集就是出现过的 token 集合，与词频无关
    contam, jac, flagged = solve([[1, 2, 2]], [[2, 3]], 1, 0.5)
    assert _mat_equal(contam, [[0.5]]), contam        # A = {1, 2}, B = {2, 3}, 交集 {2}
    assert _mat_equal(jac, [[1.0 / 3.0]]), jac        # 并集 {1,2,3} -> 1/3
    assert torch.equal(flagged, torch.tensor([0], dtype=torch.int64)), flagged  # 0.5 >= 0.5 命中


def check_boundary_threshold_inclusive(solve):
    # 阈值边界必须算命中：max 包含率恰为 0.5，th = 0.5 命中、th = 0.5 + eps 不命中
    ev = [[1, 2, 3]]            # 2-gram 集合 {(1,2),(2,3)}
    tr = [[1, 2, 9]]            # {(1,2),(2,9)} -> 交集 1 个 -> 包含率 1/2
    contam, _, f_hit = solve(ev, tr, 2, 0.5)
    assert _close(float(contam[0, 0]), 0.5), contam
    assert torch.equal(f_hit, torch.tensor([0], dtype=torch.int64)), f_hit
    _, _, f_miss = solve(ev, tr, 2, 0.5 + 1e-6)
    assert f_miss.numel() == 0, f_miss
