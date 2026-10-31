import numpy as np
import torch

KEYS = {"keys", "values", "counts", "scores", "merged_slots"}


def _f64(x):
    return torch.tensor(x, dtype=torch.float64)


def _bool(x):
    return torch.tensor(x, dtype=torch.bool)


def _oracle(k, v, s, c, dense, nk, nv, ns, metric):
    """numpy 逐 head 重算：非 dense 最低分 → dense 最近邻 → 计数加权合并 → 新 token 顶替。"""
    kn, vn, sn, cn, nk, nv, ns = (t.detach().numpy().astype(np.float64)
                                 for t in (k, v, s, c, nk, nv, ns))
    mask, (t_slots, n_heads) = dense.numpy(), kn.shape[:2]
    ok, ov, oc, os_ = kn.copy(), vn.copy(), cn.copy(), sn.copy()
    slots = np.zeros((n_heads, 2), dtype=np.int64)
    dyn, dn = [t for t in range(t_slots) if not mask[t]], [t for t in range(t_slots) if mask[t]]
    for h in range(n_heads):
        if dn:
            i = min(dyn, key=lambda t: (sn[t, h], t))
            dist = lambda t: (((kn[t, h] - kn[i, h]) ** 2).sum() if metric == "l2"
                              else -(kn[t, h] * kn[i, h]).sum())
            j = min(dn, key=lambda t: (dist(t), t))     # 严格比较 → 同分取更小的槽位号
            w_i, w_j = cn[i, h], cn[j, h]
            ok[j, h] = (w_i * kn[i, h] + w_j * kn[j, h]) / (w_i + w_j)
            ov[j, h] = (w_i * vn[i, h] + w_j * vn[j, h]) / (w_i + w_j)
            oc[j, h], os_[j, h] = w_i + w_j, sn[i, h] + sn[j, h]
        else:
            i, j = int(np.argmin(sn[:, h])), -1
        slots[h] = (i, j)
        ok[i, h], ov[i, h], oc[i, h], os_[i, h] = nk[h], nv[h], 1.0, ns[h]
    return {"keys": torch.tensor(ok, dtype=k.dtype), "values": torch.tensor(ov, dtype=k.dtype),
            "counts": torch.tensor(oc, dtype=k.dtype), "scores": torch.tensor(os_, dtype=k.dtype),
            "merged_slots": torch.tensor(slots)}


def _verify(solve, k, v, s, c, dense, nk, nv, ns, metric="l2"):
    snap = [x.clone() for x in (k, v, s, c, nk, nv, ns)]
    got = solve(k, v, s, c, dense, nk, nv, ns, metric)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert all(torch.equal(a, b) for a, b in zip((k, v, s, c, nk, nv, ns), snap)), "输入被就地修改了"
    n_heads = k.shape[1]
    want = _oracle(k, v, s, c, dense, nk, nv, ns, metric)
    assert got["merged_slots"].dtype == torch.int64 and got["merged_slots"].shape == (n_heads, 2)
    assert (got["counts"] > 0).all() and (got["merged_slots"] < k.shape[0]).all(), "计数须为正、槽位须在界内"
    tol = 3e-5 if k.dtype == torch.float32 else 2e-9
    for key in ("keys", "values", "counts", "scores"):
        assert got[key].dtype == k.dtype, f"{key} dtype {got[key].dtype} != {k.dtype}"
        torch.testing.assert_close(got[key], want[key], rtol=tol, atol=tol)
    assert got["merged_slots"].tolist() == want["merged_slots"].tolist(), (
        f"merged_slots：{got['merged_slots'].tolist()} vs {want['merged_slots'].tolist()}")
    return got


def check_example(solve):
    """题面示例：T=5、H=1、D=2，i=2 合进 j=0，计数 3+1=4，分数 1+(-2)=-1。"""
    k = _f64([[[0.0, 0.0]], [[4.0, 4.0]], [[1.0, 0.0]], [[2.0, 2.0]], [[0.0, 1.0]]])
    got = _verify(solve, k, 10 * k, _f64([[1.0], [2.0], [-2.0], [5.0], [3.0]]),
                  _f64([[1.0], [1.0], [3.0], [2.0], [1.0]]), _bool([1, 1, 0, 0, 0]),
                  _f64([[3.0, 3.0]]), _f64([[30.0, 30.0]]), _f64([1.5]))
    assert got["merged_slots"].tolist() == [[2, 0]], got["merged_slots"].tolist()
    assert got["keys"][0].tolist() == [[0.75, 0.0]] and got["values"][0].tolist() == [[7.5, 0.0]]
    assert (got["counts"][0].item(), got["scores"][0].item()) == (4.0, -1.0)
    assert got["keys"][2].tolist() == [[3.0, 3.0]] and got["counts"][2].item() == 1.0
    assert got["scores"][2].item() == 1.5


check_example.example = True


def check_metric_and_ties(solve):
    """l2 / dot 落在不同 dense 槽位；tail 可被选中；i 同分与 j 等距都取更小的槽位号。"""
    k = _f64([[[1.0, 0.0]], [[2.0, 0.0]], [[1.0, 1.0]]])
    args = (k, 10 * k, _f64([[5.0], [5.0], [-1.0]]), _f64([[1.0]] * 3), _bool([1, 1, 0]),
            _f64([[9.0, 9.0]]), _f64([[9.0, 9.0]]), _f64([0.0]))
    l2, dot = _verify(solve, *args, "l2"), _verify(solve, *args, "dot_product")
    assert l2["merged_slots"].tolist() == [[2, 0]] and dot["merged_slots"].tolist() == [[2, 1]]
    assert l2["keys"][0].tolist() == [[1.0, 0.5]] and dot["keys"][1].tolist() == [[1.5, 0.5]]
    even = _verify(solve, _f64([[[0.0, 0.0]], [[2.0, 0.0]], [[1.0, 0.0]]]),
                   _f64([[[0.0, 0.0]]] * 3), _f64([[9.0], [9.0], [1.0]]), _f64([[1.0]] * 3),
                   _bool([1, 1, 0]), _f64([[0.0, 0.0]]), _f64([[1.0, 1.0]]), _f64([0.0]))
    assert even["merged_slots"].tolist() == [[2, 0]], "等距时 j 应取更小的 dense 槽位号"


def check_counts_are_inherited(solve):
    """计数相加、新 token 计数为 1；分数相加而非取 max；全同分时 i 取更小槽位号。"""
    k = _f64([[[1.0], [5.0]], [[10.0], [6.0]], [[20.0], [7.0]], [[30.0], [8.0]]])
    got = _verify(solve, k, 10 * k, _f64([[0.0, 0.0], [3.0, -1.0], [-1.0, 3.0], [2.0, 2.0]]),
                  _f64([[2.0, 2.0], [5.0, 5.0], [3.0, 3.0], [1.0, 1.0]]), _bool([1, 0, 0, 0]),
                  _f64([[99.0], [99.0]]), _f64([[0.0], [0.0]]), _f64([0.5, 0.5]))
    assert got["merged_slots"].tolist() == [[2, 0], [1, 0]], got["merged_slots"].tolist()
    assert got["counts"][0].tolist() == [5.0, 7.0] and got["scores"][0].tolist() == [-1.0, -1.0]
    assert abs(got["keys"][0, 0, 0].item() - 12.4) < 1e-12      # (3*20 + 2*1) / 5
    assert abs(got["keys"][0, 1, 0].item() - 40 / 7) < 1e-12    # (5*6 + 2*5) / 7
    assert got["counts"][2, 0].item() == 1.0 and got["scores"][2, 0].item() == 0.5
    ties = _verify(solve, _f64([[[1.0]], [[2.0]], [[3.0]], [[4.0]]]), _f64([[[0.0]]] * 4),
                   _f64([[2.0]] * 4), _f64([[1.0]] * 4), _bool([1, 0, 0, 0]),
                   _f64([[0.0]]), _f64([[1.0]]), _f64([0.0]))
    assert ties["merged_slots"].tolist() == [[1, 0]], "i 同分取更小的非 dense 槽位"


def check_degenerate_dense(solve):
    """dense 为空：不合并、全局最低分被顶替、j=-1；dense 全 True：抛 ValueError。"""
    k = _f64([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    got = _verify(solve, k, k, _f64([[1.0], [5.0], [0.5], [2.0]]), _f64([[1.0], [2.0], [3.0], [4.0]]),
                  torch.zeros(4, dtype=torch.bool), _f64([[9.0]]), _f64([[9.0]]), _f64([0.0]))
    assert got["merged_slots"].tolist() == [[2, -1]], "没有 dense 时 j=-1"
    assert (got["keys"][0, 0, 0].item(), got["counts"][2, 0].item()) == (1.0, 1.0), "不该有合并写入"
    args = (_f64([[[1.0]], [[2.0]]]), _f64([[[1.0]], [[2.0]]]), _f64([[1.0], [2.0]]),
            _f64([[1.0], [1.0]]), torch.ones(2, dtype=torch.bool),
            _f64([[3.0]]), _f64([[3.0]]), _f64([1.0]))
    try:
        solve(*args)
    except ValueError:
        return
    raise AssertionError("dense 全 True 时应抛 ValueError（没有非 dense 槽位可合并）")


def check_dtype_and_random_oracle(solve):
    """float32 出 float32；随机形状、两端 dense 规模、两种 metric 全键对拍。"""
    torch.manual_seed(300)
    got = _verify(solve, torch.randn(9, 2, 3), torch.randn(9, 2, 3), torch.randn(9, 2),
                  torch.ones(9, 2), _bool([1, 0, 0, 0, 1, 0, 0, 0, 0]), torch.randn(2, 3),
                  torch.randn(2, 3), torch.randn(2), "dot_product")
    assert got["keys"].dtype == torch.float32, "float32 输入的输出必须仍是 float32"
    for seed, (t, h, d) in enumerate([(3, 1, 1), (17, 2, 8)]):
        torch.manual_seed(310 + seed)
        k, v = torch.randn(t, h, d, dtype=torch.float64) * 3, torch.randn(t, h, d, dtype=torch.float64)
        s = torch.randn(t, h, dtype=torch.float64)
        c = torch.randint(1, 6, (t, h)).to(torch.float64)
        for n_dense in (0, 1, t - 1):
            dense = _bool([1] * n_dense + [0] * (t - n_dense))
            for metric in ("l2", "dot_product"):
                _verify(solve, k, v, s, c, dense, torch.randn(h, d, dtype=torch.float64),
                        torch.randn(h, d, dtype=torch.float64), torch.randn(h, dtype=torch.float64),
                        metric)


def check_value_errors(solve):
    """未知 metric、dense 越界、counts 非正、dtype 或形状不符都要抛 ValueError。"""
    k, v = torch.randn(6, 2, 3), torch.randn(6, 2, 3)
    s, c, d = torch.randn(6, 2), torch.ones(6, 2), _bool([1, 0, 0, 0, 0, 0])
    nk, nv, ns = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2)
    bad = [
        (k, v, s, c, d, nk, nv, ns, "cosine"),
        (k, v, s, c, d.float(), nk, nv, ns, "l2"),
        (k, v, s, torch.zeros(6, 2), d, nk, nv, ns, "l2"),
        (k, v, s.double(), c, d, nk, nv, ns, "l2"), (k, v, s, c, d, nk[:, :2], nv, ns, "l2"),
    ]
    for idx, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {idx} 个非法输入应抛 ValueError（metric / dense / counts / dtype / 形状）")
