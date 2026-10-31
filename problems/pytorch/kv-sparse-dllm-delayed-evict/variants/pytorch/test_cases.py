import numpy as np
import torch

KEYS = {"avg_q", "scores", "heads_avg", "pooled", "keep_idx"}
TOL = {torch.float32: 3e-5, torch.float64: 1e-9}


def _oracle(k, q_block, bef, block_len, keep_ratio, kernel):
    """numpy 独立重算：剔块拼接 → 块内 query 均值 → 逐 head 点积 → 头平均 → 单侧 max_pool → 稳定 topk。"""
    kn, qn = k.numpy().astype(np.float64), q_block.numpy().astype(np.float64)
    batch, kv_heads, _, _ = kn.shape
    cand = np.concatenate([kn[:, :, :bef], kn[:, :, bef + block_len:]], axis=2)
    rep = np.repeat(cand, qn.shape[1] // kv_heads, axis=1)                 # GQA 复制
    avg_q = qn.mean(axis=2)
    scores = np.einsum("bhd,bhnd->bhn", avg_q, rep)
    heads_avg = scores.mean(axis=1)
    width, half, pooled = heads_avg.shape[-1], kernel // 2, np.empty_like(heads_avg)
    for n in range(width):                                                 # 两端只剩单侧窗口
        pooled[:, n] = heads_avg[:, max(0, n - half):min(width, n + half + 1)].max(axis=1)
    keep_num = int(width * keep_ratio)
    keep = np.zeros((batch, keep_num), dtype=np.int64)
    for b in range(batch):
        keep[b] = sorted(range(width), key=lambda i: (-pooled[b, i], i))[:keep_num]
    return {"avg_q": avg_q, "scores": scores, "heads_avg": heads_avg,
            "pooled": pooled, "keep": keep}


def _verify(solve, k, q_block, bef, block_len, keep_ratio=0.5, kernel=3):
    bk, bq = k.clone(), q_block.clone()
    got, tol = solve(k, q_block, bef, block_len, keep_ratio, kernel), TOL[k.dtype]
    batch, _, kv_len, head_dim = k.shape
    heads, width, keep_num = q_block.shape[1], kv_len - block_len, int((kv_len - block_len) * keep_ratio)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(k, bk) and torch.equal(q_block, bq), "输入被就地修改了"
    assert got["avg_q"].shape == (batch, heads, head_dim), got["avg_q"].shape
    assert got["scores"].shape == (batch, heads, width), got["scores"].shape
    assert got["heads_avg"].shape == (batch, width) and got["pooled"].shape == (batch, width)
    assert got["keep_idx"].shape == (batch, keep_num), got["keep_idx"].shape
    assert got["keep_idx"].dtype == torch.int64, "keep_idx 必须 int64"
    for key in ("avg_q", "scores", "heads_avg", "pooled"):
        assert got[key].dtype == k.dtype and torch.isfinite(got[key]).all(), f"{key} dtype/数值不对"
    want = _oracle(k, q_block, bef, block_len, keep_ratio, kernel)
    for key in ("avg_q", "scores", "heads_avg", "pooled"):
        _close2(got[key], torch.tensor(want[key], dtype=k.dtype), tol)
    assert got["keep_idx"].tolist() == want["keep"].tolist(), (
        f"keep_idx：{got['keep_idx'].tolist()} vs {want['keep'].tolist()}")
    return got


def _close2(a, b, tol):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def check_example(solve):
    """题面示例：块内两个 9 被整段剔出，两个头平均后 max_pool 把峰值摊给邻居。"""
    k = torch.tensor([[[[2.0, 0.0], [0.0, 1.0], [9.0, 9.0],
                        [9.0, 9.0], [-1.0, 0.0], [0.0, -1.0]]]])              # (1, 1, 6, 2)
    q = torch.tensor([[[[1.0, 0.0], [1.0, 2.0]], [[0.0, 1.0], [0.0, 0.0]]]])  # (1, 2, 2, 2)
    got = _verify(solve, k, q, 2, 2, 0.5, 3)
    assert got["avg_q"].tolist() == [[[1.0, 1.0], [0.0, 0.5]]], got["avg_q"].tolist()
    assert got["heads_avg"].tolist() == [[1.0, 0.75, -0.5, -0.75]], got["heads_avg"].tolist()
    assert got["pooled"].tolist() == [[1.0, 1.0, 0.75, -0.5]], got["pooled"].tolist()
    assert got["keep_idx"].tolist() == [[0, 1]], got["keep_idx"].tolist()


check_example.example = True


def check_block_window_is_excluded(solve):
    """当前块整段不进候选（延迟一步）；也没有 sink / 最近窗口，首尾 token 一样可以被丢。"""
    k = torch.tensor([[[[1.0], [2.0], [99.0], [98.0], [7.0], [0.5]]]])
    q = torch.ones(1, 1, 2, 1)
    got = _verify(solve, k, q, 2, 2, 1.0, 1)
    assert got["heads_avg"].tolist() == [[1.0, 2.0, 7.0, 0.5]], "块内的 99/98 必须消失，块外按原顺序保留"
    assert got["scores"].shape[-1] == 4, "候选长度 = 缓存长度 - block_len"
    sparse = _verify(solve, k, q, 2, 2, 0.34, 1)                   # 4 个候选只留 1 个
    pos = [i if i < 2 else i + 2 for i in sparse["keep_idx"].tolist()[0]]
    assert pos == [4], f"最强的候选是原位置 4（7.0），got {pos}"
    assert 0 not in pos and 5 not in pos, "第一个和最后一个 token 没有任何豁免"


def check_avg_q_is_block_mean(solve):
    """avg_q 是块内 query 的算术均值（双向视图），不是最后一行：块内顺序倒过来结果一字不变。"""
    torch.manual_seed(0)
    k = torch.randn(1, 2, 9, 3, dtype=torch.float64)
    q = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    a = _verify(solve, k, q, 4, 3)
    assert torch.allclose(a["avg_q"], q.mean(dim=2), rtol=0, atol=1e-12), "avg_q 不是块内均值"
    b = solve(k, q.flip(2), 4, 3)
    for key in ("avg_q", "scores", "heads_avg", "pooled"):
        _close2(a[key], b[key], 1e-12)
    assert a["keep_idx"].tolist() == b["keep_idx"].tolist()


def check_gqa_uses_repeat_interleave(solve):
    """GQA：kv head 按 repeat_interleave 复制（不是 tile），4 个 query head 各用自己那一组的 kv head。"""
    torch.manual_seed(1)
    k = torch.randn(1, 2, 7, 2, dtype=torch.float64)
    k[0, 0] *= 5.0
    q = torch.randn(1, 4, 2, 2, dtype=torch.float64)
    got = _verify(solve, k, q, 1, 2, 1.0, 1)
    avg, cand = q.mean(dim=2)[0], torch.cat([k[:, :, :1], k[:, :, 3:]], dim=2)[0]
    for head in range(4):
        _close2(got["scores"][0, head], avg[head] @ cand[head // 2].T, 1e-12)


def check_heads_averaged_before_pooling(solve):
    """顺序是先跨 head 平均再 max_pool，不是每个 head 各自 pool 再平均（两头峰值位置不同）。"""
    k = torch.tensor([[[[9.0, 9.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]])    # 位置 0 是当前块
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])                         # head0 看 e0、head1 看 e1
    got = _verify(solve, k, q, 0, 1, 1.0, 3)                    # 候选 = 位置 1、2、3
    assert got["scores"].tolist() == [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], got["scores"].tolist()
    assert got["heads_avg"].tolist() == [[0.5, 0.0, 0.5]], "必须先跨 head 平均"
    assert got["pooled"].tolist() == [[0.5, 0.5, 0.5]], "逐 head 池化再平均会得到 [0.75, 0.5, 0.75]"


def check_boundary_pooling_is_one_sided(solve):
    """两端是单侧窗口（-inf 填充而不是补 0）：整段为负时两端仍是负数；kernel=1 退化成原分数。"""
    torch.manual_seed(2)
    k = -torch.rand(1, 1, 5, 2, dtype=torch.float64)
    q = torch.ones(1, 1, 1, 2, dtype=torch.float64)
    got = _verify(solve, k, q, 0, 1, 1.0, 3)
    a, p = got["heads_avg"][0], got["pooled"][0]
    assert p[0] == max(a[0], a[1]) and p[-1] == max(a[-2], a[-1]), "两端只能和存在的邻居取 max"
    assert p[0] < 0 and p[-1] < 0, "补 0 会把负分数抬到 0"
    assert p[1] == max(a[0], a[1], a[2]) and p[-2] == max(a[-3], a[-2], a[-1])
    same = solve(k, q, 0, 1, 1.0, 1)
    _close2(same["pooled"], same["heads_avg"], 1e-12)


def check_keep_budget_and_ties(solve):
    """keep_num = int(候选长度 * keep_ratio) 向下取整、可以是 0；全同分时按下标从小到大取。"""
    k = torch.ones(1, 1, 6, 1, dtype=torch.float64)             # 5 个候选、全同分
    q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    assert _verify(solve, k, q, 0, 1, 0.5, 1)["keep_idx"].tolist() == [[0, 1]]
    assert _verify(solve, k, q, 0, 1, 0.59, 1)["keep_idx"].tolist() == [[0, 1]], "int(5*0.59)=2"
    assert _verify(solve, k, q, 0, 1, 0.2, 1)["keep_idx"].tolist() == [[0]], "int(1.0)=1"
    empty = _verify(solve, k, q, 0, 1, 0.19, 1)                 # int(0.95)=0：一个都不留
    assert empty["keep_idx"].shape == (1, 0), "keep_num=0 时返回空索引，而不是报错"


def check_dtype_sweep(solve):
    """float32 输入的四个浮点键都是 float32、keep_idx 是 int64，数值与 float64 路径在 1e-5 内一致。"""
    torch.manual_seed(3)
    k = torch.randn(2, 2, 10, 3, dtype=torch.float64)
    q = torch.randn(2, 4, 4, 3, dtype=torch.float64)
    lo, hi = _verify(solve, k.float(), q.float(), 3, 4, 0.7), solve(k, q, 3, 4, 0.7)
    for key in ("avg_q", "scores", "heads_avg", "pooled"):
        assert lo[key].dtype == torch.float32, f"{key} 应为 float32"
        _close2(lo[key].double(), hi[key], 3e-5)


def check_value_errors(solve):
    """块窗口越界/无候选、block_len 与 q_block 不符、GQA 不整除、参数非法，一律 ValueError。"""
    k, q = torch.randn(1, 2, 6, 2), torch.randn(1, 4, 2, 2)
    bad = [(k, q, 5, 2, 0.5, 3), (k, torch.randn(1, 4, 6, 2), 0, 6, 0.5, 3),
           (k, q, 0, 3, 0.5, 3), (k, torch.randn(1, 3, 2, 2), 0, 2, 0.5, 3),
           (k, q, -1, 2, 0.5, 3), (k, q, 0, 2, 0.0, 3), (k, q, 0, 2, 1.5, 3),
           (k, q, 0, 2, 0.5, 2), (k.double(), q, 0, 2, 0.5, 3), (k[0], q[0], 0, 2, 0.5, 3)]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")


def check_random_against_oracle(solve):
    """随机形状/比例/kernel 与 numpy oracle 全键对拍（含 batch=2、Hq==Hkv、Hq=4:Hkv=2、kernel=5）。"""
    cases = [(1, 1, 7, 2, 2, 0.5, 3), (2, 2, 9, 4, 3, 0.7, 3), (1, 2, 12, 2, 4, 0.3, 5),
             (2, 1, 6, 8, 2, 1.0, 1), (1, 2, 5, 4, 1, 0.5, 3)]
    for seed, (batch, kv_heads, kv_len, heads, block_len, ratio, kernel) in enumerate(cases):
        torch.manual_seed(100 + seed)
        k = torch.randn(batch, kv_heads, kv_len, 3, dtype=torch.float64)
        q = torch.randn(batch, heads, block_len, 3, dtype=torch.float64)
        _verify(solve, k, q, kv_len - block_len - 3, block_len, ratio, kernel)
