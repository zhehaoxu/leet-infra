import numpy as np
import torch

KEYS = {"weights", "removed", "merged_k", "merged_v"}


def _rand_case(seed, ml, mr, heads=2, dim=3, rows=1, prefix_len=2, suffix_len=1):
    torch.manual_seed(seed)
    merged_len = ml + mr - prefix_len - suffix_len
    k_l = torch.randn(ml, heads, dim, dtype=torch.float64)
    v_l = torch.randn(ml, heads, dim, dtype=torch.float64)
    k_r = torch.randn(mr, heads, dim, dtype=torch.float64)
    v_r = torch.randn(mr, heads, dim, dtype=torch.float64)
    metric = torch.rand(heads, rows, merged_len, dtype=torch.float64)
    bias = torch.rand(merged_len + 5, dtype=torch.float64) * 0.4
    mask = torch.ones(merged_len, dtype=torch.bool)
    mask[:prefix_len] = False
    mask[merged_len - suffix_len:] = False
    return k_l, v_l, k_r, v_r, metric, bias, mask


def _oracle(k_l, v_l, k_r, v_r, metric, bias, mask, reduce, prefix_len, suffix_len):
    """numpy 独立重算：affix 取均值 → 拼接 → 反序偏置 → 保护掩码 → 取 r 个最小。"""
    kl, vl = k_l.detach().numpy().astype(np.float64), v_l.detach().numpy().astype(np.float64)
    kr, vr = k_r.detach().numpy().astype(np.float64), v_r.detach().numpy().astype(np.float64)
    ml, mr = kl.shape[0], kr.shape[0]

    def merge(a, b):
        return np.concatenate([(a[:prefix_len] + b[:prefix_len]) / 2.0,
                               a[prefix_len:ml - suffix_len], b[prefix_len:mr - suffix_len],
                               (a[ml - suffix_len:] + b[mr - suffix_len:]) / 2.0], axis=0)

    merged_k, merged_v = merge(kl, kr), merge(vl, vr)
    size = merged_k.shape[0]
    weight = metric.detach().numpy().astype(np.float64)[:, -1, :].max(axis=0)
    weight = weight - bias.detach().numpy().astype(np.float64)[np.arange(size - 1, -1, -1)]
    weight = np.where(mask.detach().numpy().astype(bool), weight, np.inf)
    drop = set(sorted(range(size), key=lambda j: (weight[j], j))[:reduce])
    keep = np.array([j for j in range(size) if j not in drop], dtype=np.int64)
    return {
        "weights": torch.tensor(weight, dtype=k_l.dtype),
        "removed": torch.tensor(sorted(drop), dtype=torch.int64),
        "merged_k": torch.tensor(merged_k[keep], dtype=k_l.dtype),
        "merged_v": torch.tensor(merged_v[keep], dtype=k_l.dtype),
    }


def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, reduce, prefix_len, suffix_len):
    """结构 + numpy oracle 全键对拍。"""
    saved = [t.clone() for t in (k_l, v_l, k_r, v_r, metric, bias, mask)]
    got = solve(k_l, v_l, k_r, v_r, metric, bias, mask, reduce, prefix_len, suffix_len)
    for old, new, name in zip(saved, (k_l, v_l, k_r, v_r, metric, bias, mask),
                              ("k_left", "v_left", "k_right", "v_right", "metric", "bias", "mask")):
        assert torch.equal(old, new), f"{name} 被就地修改了"
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - prefix_len - suffix_len
    assert got["weights"].shape == (size,), f"weights 形状 {tuple(got['weights'].shape)} != ({size},)"
    assert got["removed"].shape == (reduce,) and got["removed"].dtype == torch.int64
    assert got["merged_k"].shape == (size - reduce,) + tuple(k_l.shape[1:]), got["merged_k"].shape
    assert got["merged_v"].shape == got["merged_k"].shape
    for key in ("weights", "merged_k", "merged_v"):
        assert got[key].dtype == k_l.dtype, f"{key} dtype {got[key].dtype} != {k_l.dtype}"
    removed = got["removed"].tolist()
    assert removed == sorted(set(removed)), f"removed 必须升序无重复：{removed}"
    assert all(0 <= j < size for j in removed), f"removed 越界：{removed}"
    assert all(bool(mask[j]) for j in removed), f"保护区不能被删：{removed}"
    protected = (~mask).nonzero().flatten().tolist()
    assert all(float(got["weights"][j]) == float("inf") for j in protected), "保护区权重必须是 +inf"
    assert all(np.isfinite(float(got["weights"][j])) for j in range(size) if j not in protected)
    want = _oracle(k_l, v_l, k_r, v_r, metric, bias, mask, reduce, prefix_len, suffix_len)
    _close(got["weights"][got["weights"].isfinite()], want["weights"][want["weights"].isfinite()])
    assert (got["weights"].isinf() == want["weights"].isinf()).all(), "保护区位置必须恰为 +inf"
    assert removed == want["removed"].tolist(), f"removed：{removed} vs {want['removed'].tolist()}"
    _close(got["merged_k"], want["merged_k"])
    _close(got["merged_v"], want["merged_v"])
    return got


def check_example(solve):
    """题面示例：M_L=6、M_R=5、prefix=2、suffix=1 → M=8，r=3，删掉下标 2/3/4。"""
    k_l = torch.tensor([10.0, 12.0, 1.0, 2.0, 3.0, 40.0]).reshape(6, 1, 1)
    k_r = torch.tensor([20.0, 30.0, 7.0, 8.0, 50.0]).reshape(5, 1, 1)
    metric = torch.tensor([[[0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]])
    bias = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    mask = torch.tensor([False, False, True, True, True, True, True, False])
    got = _verify(solve, k_l, k_l, k_r, k_r, metric, bias, mask, 3, 2, 1)
    _close(got["merged_k"][:, 0, 0], torch.tensor([15.0, 21.0, 7.0, 8.0, 45.0]))
    assert got["removed"].tolist() == [2, 3, 4], got["removed"].tolist()
    finite = torch.tensor([-0.15, 0.0, 0.15, 0.3, 0.45], dtype=torch.float32)
    _close(got["weights"][mask], finite)


check_example.example = True


def check_merge_averages_the_shared_affix(solve):
    """prefix/suffix 逐元素取两个孩子的均值，context 直接拼接；顺序是 [prefix, L, R, suffix]。"""
    k_l = torch.arange(6.0).reshape(6, 1, 1)
    k_r = torch.arange(5.0).reshape(5, 1, 1) + 100.0
    bias = torch.zeros(12)
    metric = torch.ones(1, 1, 8)
    mask = torch.ones(8, dtype=torch.bool)
    got = _verify(solve, k_l, k_l, k_r, k_r, metric, bias, mask, 2, 2, 1)
    want = torch.tensor([50.0, 51.0, 2.0, 3.0, 4.0, 102.0, 103.0, 54.5])
    assert got["merged_k"].shape == (6, 1, 1)
    assert 0.0 not in got["merged_k"].flatten().tolist(), "左 chunk 的 prefix 值不该原样留下"
    full = _verify(solve, k_l, k_l, k_r, k_r, metric, bias, mask, 0, 2, 1)
    _close(full["merged_k"][:, 0, 0], want)


def check_bias_is_indexed_from_the_end(solve):
    """rel(j) = (M-1) - j：bias[0] 打在最后一个位置，bias[M-1] 打在第一个位置。"""
    k_l, v_l, k_r, v_r, metric, bias, mask = _rand_case(11, 7, 6, prefix_len=1, suffix_len=1)
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - 2
    raw = metric[:, -1, :].max(dim=0).values
    got = _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, 4, 1, 1)
    for j in mask.nonzero().flatten().tolist():
        _close(got["weights"][j], raw[j] - bias[size - 1 - j])
    forward = (raw - bias[:size])[mask]
    assert not torch.allclose(got["weights"][mask], forward), "偏置必须从末尾数，不能顺着位置加"


def check_metric_uses_last_row_and_head_max(solve):
    """只读 metric 的最后一行、按 head 取 max；行号 -1 不是 0，也不是按 head 求均值。"""
    k_l, v_l, k_r, v_r, _, bias, mask = _rand_case(13, 6, 6, heads=3, rows=4, prefix_len=1, suffix_len=1)
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - 2
    metric = torch.arange(3 * 4 * size, dtype=torch.float64).reshape(3, 4, size) * 1e-3
    got = _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, 3, 1, 1)
    want = metric[:, -1, :].max(dim=0).values - bias[:size][torch.arange(size - 1, -1, -1)]
    _close(got["weights"][mask], want[mask])
    assert not torch.allclose(got["weights"][mask], (metric[:, 0, :].max(dim=0).values
                                                     - bias[:size][torch.arange(size - 1, -1, -1)])[mask])


def check_protected_positions_are_inf_and_kept(solve):
    """reduction_mask 为 False 的 affix 位置是 +inf，永远进不了最小的那一批。"""
    k_l, v_l, k_r, v_r, metric, bias, mask = _rand_case(17, 9, 8, prefix_len=3, suffix_len=2)
    got = _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, 7, 3, 2)
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - 5
    protected = (~mask).nonzero().flatten().tolist()
    assert protected == [0, 1, 2, size - 2, size - 1], protected
    assert all(float(got["weights"][j]) == float("inf") for j in protected)
    assert not set(protected) & set(got["removed"].tolist()), "保护区被删了"


def check_removed_are_the_smallest(solve):
    """removed 恰好是最小的 r 个（保护区除外）。"""
    k_l, v_l, k_r, v_r, metric, bias, mask = _rand_case(19, 10, 9)
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - 3
    for reduce in (1, 4, 7, size - 5):
        got = _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, reduce, 2, 1)
        free = [j for j in range(size) if bool(mask[j])]
        order = sorted(free, key=lambda j: (float(got["weights"][j]), j))
        assert got["removed"].tolist() == sorted(order[:reduce]), (
            f"r={reduce} 应删掉最小的 {reduce} 个：{got['removed'].tolist()}")


def check_ties_remove_the_smaller_index(solve):
    """同分时先删更小的下标（torch.topk 不定义同分顺序，这里钉死）。"""
    k_l = torch.zeros(6, 1, 1)
    k_r = torch.zeros(6, 1, 1)
    metric = torch.zeros(1, 1, 8)
    bias = torch.zeros(8)
    mask = torch.ones(8, dtype=torch.bool)
    got = _verify(solve, k_l, k_l, k_r, k_r, metric, bias, mask, 3, 2, 2)
    assert got["weights"].tolist() == [0.0] * 8, got["weights"].tolist()
    assert got["removed"].tolist() == [0, 1, 2], got["removed"].tolist()


def check_zero_reduction_keeps_everything(solve):
    """r = 0 时一个都不删，但权重照常计算（上游会短路成全 1，那是占位值）。"""
    k_l, v_l, k_r, v_r, metric, bias, mask = _rand_case(23, 8, 7)
    got = _verify(solve, k_l, v_l, k_r, v_r, metric, bias, mask, 0, 2, 1)
    assert got["removed"].numel() == 0 and got["removed"].dtype == torch.int64
    size = int(k_l.shape[0]) + int(k_r.shape[0]) - 3
    assert got["merged_k"].shape[0] == size, "r=0 时长度应保持合并后的 M"
    assert not torch.all(got["weights"][mask] == 1.0), "权重不能退化成占位全 1"


def check_random_against_oracle(solve):
    """随机形状 / affix / r 与 numpy oracle 全键对拍。"""
    cases = [(9, 8, 2, 3, 2, 1, 5), (12, 6, 3, 4, 3, 2, 8), (5, 5, 1, 2, 1, 0, 3),
             (20, 14, 2, 5, 4, 4, 16), (7, 9, 2, 3, 0, 2, 6)]
    for seed, (ml, mr, heads, dim, prefix_len, suffix_len, reduce) in enumerate(cases):
        args = _rand_case(200 + seed, ml, mr, heads, dim, 2, prefix_len, suffix_len)
        _verify(solve, *args, reduce, prefix_len, suffix_len)


def check_dtype_sweep(solve):
    """float32 输入 → float32 输出，数值与 float64 结果在 1e-5 内一致。"""
    k_l, v_l, k_r, v_r, metric, bias, mask = _rand_case(29, 11, 9, heads=2, dim=4)
    lo = solve(*(t.float() for t in (k_l, v_l, k_r, v_r, metric, bias)), mask, 6, 2, 1)
    hi = solve(k_l, v_l, k_r, v_r, metric, bias, mask, 6, 2, 1)
    for key in ("weights", "merged_k", "merged_v"):
        assert lo[key].dtype == torch.float32, f"{key} 应为 float32"
        finite = hi[key].isfinite()
        torch.testing.assert_close(lo[key][finite], hi[key].float()[finite], rtol=1e-5, atol=1e-5)
    assert lo["removed"].dtype == torch.int64
    assert lo["removed"].tolist() == hi["removed"].tolist()


def check_value_errors(solve):
    """形状 / dtype / affix / r 非法一律 ValueError。"""
    k_l = torch.randn(6, 2, 3)
    v_l = torch.randn(6, 2, 3)
    k_r = torch.randn(5, 2, 3)
    v_r = torch.randn(5, 2, 3)
    metric = torch.rand(2, 1, 8)
    bias = torch.rand(9)
    mask = torch.tensor([False, False, True, True, True, True, True, False])
    bad = [(k_l[0], v_l, k_r, v_r, metric, bias, mask, 2, 2, 1),        # k_left 非三维
           (k_l, v_l, k_r.double(), v_r, metric, bias, mask, 2, 2, 1),   # 左右 dtype 不同
           (k_l, v_l, k_r, v_r.double(), metric, bias, mask, 2, 2, 1),   # k/v dtype 不同
           (k_l, v_l, torch.randn(5, 3, 3), v_r, metric, bias, mask, 2, 2, 1),
           (k_l, v_l, torch.randn(5, 2, 4), v_r, metric, bias, mask, 2, 2, 1),
           (k_l, torch.randn(7, 2, 3), k_r, v_r, metric, bias, mask, 2, 2, 1),
           (k_l, v_l, k_r, v_r, torch.rand(3, 1, 8), bias, mask, 2, 2, 1),
           (k_l, v_l, k_r, v_r, torch.rand(2, 1, 7), bias, mask, 2, 2, 1),
           (k_l, v_l, k_r, v_r, metric, torch.rand(7), mask, 2, 2, 1),   # bias 太短
           (k_l, v_l, k_r, v_r, metric, bias, mask.long(), 2, 2, 1),     # 掩码非 bool
           (k_l, v_l, k_r, v_r, metric, bias, mask[:7], 2, 2, 1),        # 掩码长度不对
           (k_l, v_l, k_r, v_r, metric, bias, mask, 8, 2, 1),            # r >= M
           (k_l, v_l, k_r, v_r, metric, bias, mask, 6, 2, 1),            # r > 可剪位置数
           (k_l, v_l, k_r, v_r, metric, bias, mask, 2.0, 2, 1),          # r 非整数
           (k_l, v_l, k_r, v_r, metric, bias, mask, -1, 2, 1),           # r < 0
           (k_l, v_l, k_r, v_r, metric, bias, mask, 2, 0, 0),            # affix 全零
           (k_l, v_l, k_r, v_r, metric, bias, mask, 2, 4, 2)]            # affix 超过 chunk 长度
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")
