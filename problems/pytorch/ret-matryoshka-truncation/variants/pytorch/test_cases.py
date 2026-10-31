import math
import random


def _oracle_rank(q64, docs64, d, top_k):
    def norm(v):
        s = math.sqrt(sum(x * x for x in v))
        s = max(s, 1e-12)  # 零向量归一化为全 0（与 F.normalize eps 语义一致）
        return [x / s for x in v]

    q = norm(q64[:d])
    sims = []
    for row in docs64:
        v = norm(row[:d])
        sims.append(sum(a * b for a, b in zip(v, q)))
    order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
    return {"sims": sims, "top": order[:top_k]}


def _oracle(query_emb, doc_embs, dims, top_k):
    D = len(query_emb)
    q64 = [float(x) for x in query_emb]
    docs64 = [[float(x) for x in row] for row in doc_embs]
    full = _oracle_rank(q64, docs64, D, top_k)
    truncated = {d: _oracle_rank(q64, docs64, d, top_k) for d in dims}
    full_set = set(full["top"])
    for d, led in truncated.items():
        led["overlap_full"] = len(set(led["top"]) & full_set)
    return {"full": full, "truncated": truncated, "n_docs": len(docs64)}


def _compare(solve, query_emb, doc_embs, dims, top_k):
    got = solve(query_emb, doc_embs, dims, top_k)
    exp = _oracle(query_emb, doc_embs, dims, top_k)
    assert got["n_docs"] == exp["n_docs"]
    assert got["full"]["top"] == exp["full"]["top"], (got["full"], exp["full"])
    assert got["truncated"].keys() == exp["truncated"].keys()
    for d in exp["truncated"]:
        g, e = got["truncated"][d], exp["truncated"][d]
        assert g["top"] == e["top"], (d, g["top"], e["top"])
        assert g["overlap_full"] == e["overlap_full"], (d, g, e)
        for gs, es in zip(g["sims"], e["sims"]):
            assert abs(gs - es) < 1e-6, (d, gs, es)
    return got


def check_anchor_dims(solve):
    # 手算锚点：2 维嵌入按 幅角 与 第一维符号 分层，截断后排名退化可手算
    import torch
    q = torch.tensor([1.0, 1.0])            # 45°
    docs = torch.tensor(
        [
            [2.0, 0.5],    # 14°,  cos(31°)≈0.857, dim1 符号 +
            [1.0, 2.0],    # 63°,  cos(18°)≈0.951, dim1 符号 +
            [-1.0, 1.0],   # 135°, cos(90°)=0,    dim1 符号 -
            [0.5, 0.1],    # 11°,  cos(34°)≈0.829, dim1 符号 +
        ]
    )
    led = _compare(solve, q, docs, [1, 2], 3)
    # d=2（full）：|angle-45°| 升序 → doc1(18°) > doc0(31°) > doc3(34°) > doc2(90°)
    assert led["full"]["top"] == [1, 0, 3]
    assert abs(led["full"]["sims"][1] - 3 / math.sqrt(10)) < 1e-9  # (1+2)/(√2·√5)
    # d=1：只剩第一维符号（幅值被归一化抹掉），sims = [1, 1, -1, 1]，并列取小 index
    assert led["truncated"][1]["top"] == [0, 1, 3]
    assert abs(led["truncated"][1]["sims"][2] - (-1.0)) < 1e-9
    # 截断 top3 恰好等于 full top3
    assert led["truncated"][1]["overlap_full"] == 3


def check_full_equals_truncated_at_d(solve):
    import torch
    rng = torch.Generator().manual_seed(657)
    D, N = 8, 6
    q = torch.randn(D, generator=rng)
    docs = torch.randn(N, D, generator=rng)
    led = _compare(solve, q, docs, [D], 3)
    assert led["truncated"][D]["top"] == led["full"]["top"]
    assert led["truncated"][D]["overlap_full"] == 3


def check_rank_preservation_spectrum(solve):
    # 嵌入设计成「低维已近似高维」：前 2 维为主成分
    import torch
    base = torch.tensor([[1.0, 0.1], [0.9, -0.2], [0.5, 0.8], [0.2, 0.9]])
    noise = torch.tensor([[0.01, -0.01],
                          [-0.01, 0.02],
                          [0.02, 0.01],
                          [-0.02, 0.0]])
    docs = torch.cat([base, noise], dim=1)
    q = torch.tensor([1.0, 0.2, 0.005, 0.005])
    led = _compare(solve, q, docs, [2, 4], 2)
    assert led["truncated"][2]["top"] == led["full"]["top"]  # 排名完全保持
    assert led["truncated"][2]["overlap_full"] == 2


def check_tie_breaks_by_index(solve):
    import torch
    q = torch.tensor([1.0, 0.0])
    docs = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]])
    # 一维截断下 docs 的第一维全 0（零向量 → sim 全 0 并列）→ 取小 index
    led = _compare(solve, q, docs, [1], 3)
    sims1 = led["truncated"][1]["sims"]
    assert all(abs(s) < 1e-9 for s in sims1)
    assert led["truncated"][1]["top"] == [0, 1, 2]
    assert led["truncated"][1]["overlap_full"] == 3  # top3 集合相同


def check_negative_similarity_ordering(solve):
    import torch
    q = torch.tensor([1.0, 0.001])
    docs = torch.tensor(
        [
            [-2.0, 5.0],   # 全维下与 query 近正交
            [0.5, 0.001],  # 同方向
            [-0.1, -1.0],  # 全维反向
        ]
    )
    led = _compare(solve, q, docs, [1], 3)
    # d=1：一维截断只剩符号 → sims = [-1, 1, -1]，负值并列取小 index
    assert led["truncated"][1]["top"] == [1, 0, 2]
    # 全维：doc1（同向）第一；doc0 近正交（-0.37）比 doc2 反向（-0.10）更小 → 殿后
    assert led["full"]["top"] == [1, 2, 0]
    assert led["truncated"][1]["overlap_full"] == 3


def check_random_against_oracle(solve):
    import torch
    rng = random.Random(657)
    for _ in range(30):
        D = rng.randint(3, 12)
        N = rng.randint(3, 10)
        dims = sorted(rng.sample(range(1, D + 1), rng.randint(1, min(4, D))))
        top_k = rng.randint(1, N)
        q = torch.tensor([rng.uniform(-2, 2) for _ in range(D)])
        docs = torch.tensor([[rng.uniform(-2, 2) for _ in range(D)] for _ in range(N)])
        _compare(solve, q, docs, dims, top_k)


def check_invalid_inputs(solve):
    import torch
    q = torch.tensor([1.0, 2.0, 3.0])
    docs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for kwargs in (
        {"dims": [0]},
        {"dims": [4]},
        {"dims": [1.5]},
        {"top_k": 0},
        {"top_k": True},
        {"top_k": 3},
    ):
        call = {"dims": [1], "top_k": 2}
        call.update(kwargs)
        try:
            solve(q, docs, **call)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    try:
        solve(torch.tensor([1.0, 2.0]), docs, [1], 2)  # query/doc 维度不匹配
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for dim mismatch")
