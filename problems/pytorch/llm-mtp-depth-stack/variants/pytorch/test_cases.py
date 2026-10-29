import math

import torch


def _rms(vals, eps):
    mean_square = sum(v * v for v in vals) / len(vals)
    denom = math.sqrt(mean_square + eps)
    return [v / denom for v in vals]


def _oracle(hidden, embeddings, W, eps):
    """独立 oracle：逐 (batch, 位置) 用 python 标量做两段 RMSNorm、拼接与投影。"""
    batch, seq, dim = hidden.shape
    depth = W.shape[0]
    h = hidden.tolist()
    emb = embeddings.tolist()
    weights = W.tolist()

    states = []
    for k in range(depth):
        wk = weights[k]
        rows = []
        for b in range(batch):
            sample = []
            for s in range(seq):
                fused = _rms(h[b][s], eps) + _rms(emb[b][k + 1 + s], eps)
                sample.append([sum(fused[f] * wk[f][e] for f in range(2 * dim)) for e in range(dim)])
            rows.append(sample)
        h = rows
        states.append(torch.tensor(rows, dtype=torch.float64))
    return states


def check_example(solve):
    hidden = torch.tensor([[[1.0, 2.0], [0.5, -1.0]]])
    embeddings = torch.tensor([[[0.5, 0.5], [1.0, -1.0], [0.25, 0.75], [-1.0, 0.5]]])
    W = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
        ]
    )

    states = solve(hidden, embeddings, W, eps=1e-6)
    assert isinstance(states, tuple) and len(states) == 2, "应返回长度为 D 的 tuple"
    assert all(s.dtype == torch.float32 and s.shape == (1, 2, 2) for s in states)

    first = [[[round(float(v), 6) for v in row] for row in batch] for batch in states[0].tolist()]
    assert first == [[[1.132455, 0.764911], [0.856061, -0.594091]]], first
    second = [[[round(float(v), 6) for v in row] for row in batch] for batch in states[1].tolist()]
    assert second == [[[2.356817, 1.395533], [-0.806298, 0.529388]]], second


check_example.example = True


def check_rms_scale_invariance(solve):
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64)
    embeddings = torch.randn(2, 5, 4, dtype=torch.float64)
    W = torch.randn(2, 8, 4, dtype=torch.float64) * 0.5

    base = solve(hidden, embeddings, W, eps=1e-12)[0]
    # RMSNorm 对正数放缩不敏感：隐状态整体乘 7 不改变 h^1
    scaled_hidden = solve(hidden * 7.0, embeddings, W, eps=1e-12)[0]
    assert torch.allclose(base, scaled_hidden, atol=1e-5), "隐状态应先在特征维上做 RMSNorm"

    # 第 1 层用到的嵌入（列 1..S）整体放缩同样不影响
    embeddings2 = embeddings.clone()
    embeddings2[:, 1 : 1 + hidden.shape[1]] *= 9.0
    scaled_emb = solve(hidden, embeddings2, W, eps=1e-12)[0]
    assert torch.allclose(base, scaled_emb, atol=1e-5), "嵌入也应单独做 RMSNorm"


def check_depth_chaining(solve):
    torch.manual_seed(1)
    hidden = torch.randn(2, 2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, 3, dtype=torch.float64)
    W = torch.randn(2, 6, 3, dtype=torch.float64)

    W2 = W.clone()
    W2[1] = torch.randn(6, 3, dtype=torch.float64)
    base = solve(hidden, embeddings, W)
    changed_second = solve(hidden, embeddings, W2)
    assert torch.allclose(base[0], changed_second[0], atol=1e-7), "第 2 层的权重不应影响 h^1"
    assert not torch.allclose(base[1], changed_second[1], atol=1e-4), "改第 2 层应改变 h^2"

    W1 = W.clone()
    W1[0] = torch.randn(6, 3, dtype=torch.float64)
    changed_first = solve(hidden, embeddings, W1)
    assert not torch.allclose(base[0], changed_first[0], atol=1e-4), "改第 1 层应改变 h^1"
    assert not torch.allclose(base[1], changed_first[1], atol=1e-4), "h^2 依赖 h^1，应一并改变"


def check_embedding_shift(solve):
    torch.manual_seed(2)
    batch, seq, depth, dim = 2, 3, 2, 3
    hidden = torch.randn(batch, seq, dim, dtype=torch.float64)
    embeddings = torch.randn(batch, seq + depth, dim, dtype=torch.float64)
    W = torch.randn(depth, 2 * dim, dim, dtype=torch.float64)

    base = solve(hidden, embeddings, W)

    # 位置 0 的嵌入对应 h^0 已经消化的 token，任何深度都不该再用它
    shifted = embeddings.clone()
    shifted[:, 0] += 5.0
    changed = solve(hidden, shifted, W)
    for k in range(depth):
        assert torch.allclose(base[k], changed[k], atol=1e-7), f"第 {k + 1} 层不应读取位置 0 的嵌入"

    # 位置 S+1 只落在最后一层（k=D 时读 [D, D+S)）的窗口内
    shifted = embeddings.clone()
    shifted[:, seq + 1] += 5.0
    changed = solve(hidden, shifted, W)
    assert torch.allclose(base[0], changed[0], atol=1e-7), "第 1 层不应读取位置 S+1 的嵌入"
    assert not torch.allclose(base[1], changed[1], atol=1e-4), "第 2 层应读取位置 S+1 的嵌入"

    # 位置 1 落在第 1 层的窗口 (1..S) 内
    shifted = embeddings.clone()
    shifted[:, 1] += 5.0
    changed = solve(hidden, shifted, W)
    assert not torch.allclose(base[0], changed[0], atol=1e-4), "第 1 层应读取位置 1 的嵌入"


def check_against_oracle(solve):
    torch.manual_seed(3)
    for batch, seq, depth, dim in ((2, 3, 2, 3), (1, 2, 3, 2), (3, 1, 1, 4)):
        hidden = torch.randn(batch, seq, dim, dtype=torch.float64)
        embeddings = torch.randn(batch, seq + depth, dim, dtype=torch.float64)
        W = torch.randn(depth, 2 * dim, dim, dtype=torch.float64)

        got = solve(hidden, embeddings, W, eps=1e-6)
        want = _oracle(hidden, embeddings, W, 1e-6)
        assert len(got) == depth
        for k in range(depth):
            assert torch.allclose(got[k], want[k].to(torch.float32), atol=1e-5), (
                f"B={batch}, S={seq}, D={depth}, d={dim} 的第 {k + 1} 层与 oracle 不一致"
            )

    # 不同 eps 也要对上
    hidden = torch.randn(2, 2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, 3, dtype=torch.float64)
    W = torch.randn(2, 6, 3, dtype=torch.float64)
    for eps in (1e-8, 1e-2):
        got = solve(hidden, embeddings, W, eps=eps)
        want = _oracle(hidden, embeddings, W, eps)
        for k in range(2):
            assert torch.allclose(got[k], want[k].to(torch.float32), atol=1e-5), f"eps={eps} 时第 {k + 1} 层不一致"


def check_invalid(solve):
    hidden = torch.zeros(1, 2, 3)
    embeddings = torch.zeros(1, 4, 3)
    W = torch.zeros(2, 6, 3)

    for bad in (
        lambda: solve(hidden[0], embeddings, W),
        lambda: solve(hidden, embeddings[0], W),
        lambda: solve(hidden, embeddings, W[0]),
        lambda: solve([[0.0, 1.0]], embeddings, W),
        lambda: solve(hidden, torch.zeros(2, 4, 3), W),
        lambda: solve(hidden, torch.zeros(1, 4, 4), W),
        lambda: solve(hidden, torch.zeros(1, 3, 3), W),
        lambda: solve(hidden, embeddings, torch.zeros(2, 5, 3)),
        lambda: solve(hidden, embeddings, torch.zeros(2, 6, 2)),
        lambda: solve(hidden, embeddings, torch.zeros(0, 6, 3)),
        lambda: solve(hidden, embeddings, W, eps=0.0),
        lambda: solve(hidden, embeddings, W, eps=-1e-6),
        lambda: solve(hidden, embeddings, W, eps=float("nan")),
        lambda: solve(hidden, embeddings, W, eps=True),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
