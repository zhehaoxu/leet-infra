import numpy as np
import torch

KEYS = ("lengths", "csr_indptr", "csr_indices", "csr_values", "residual_norm", "recon")
TOL = {torch.float32: 1e-5, torch.float64: 1e-9}

# 题面示例的字典：D=3、M=5，原子是列
_DIC = [[1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0]]
_VEC = [[2.0, 1.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]


def _oracle(vectors, dictionary, n_atoms, tol):
    """独立 numpy 重算：每轮先判停、再按 |相关系数| 贪心选原子，并对整个已选集合重解最小二乘。"""
    y_all = np.asarray(vectors, dtype=np.float64)
    dic = np.asarray(dictionary, dtype=np.float64)
    n_rows, dim = y_all.shape
    m = dic.shape[1]
    tol2 = None if tol is None else float(tol) ** 2
    lengths, indices, values, res_norm, recon = [], [], [], [], []
    for i in range(n_rows):
        y = y_all[i]
        initial = float(y @ y)
        r, picked = y.copy(), []
        w = np.zeros(0)
        for _k in range(n_atoms):
            if tol2 is not None and float(r @ r) <= tol2 * initial:
                break
            rest = [j for j in range(m) if j not in picked]
            corr = np.abs(dic[:, rest].T @ r)
            cand = np.flatnonzero(corr == corr.max())
            picked = picked + [rest[cand[0]]]
            basis = dic[:, picked]
            w, *_ = np.linalg.lstsq(basis, y, rcond=None)
            r = y - basis @ w
        lengths.append(len(picked))
        indices.extend(picked)
        values.extend(float(v) for v in w)
        res_norm.append(float(np.linalg.norm(r)))
        recon.append(y - r)
    indptr = [0]
    for ln in lengths:
        indptr.append(indptr[-1] + ln)
    return {"lengths": np.array(lengths, np.int64), "csr_indptr": np.array(indptr, np.int64),
            "csr_indices": np.array(indices, np.int64), "csr_values": np.array(values, np.float64),
            "residual_norm": np.array(res_norm, np.float64),
            "recon": np.array(recon, np.float64).reshape(n_rows, dim)}


def _inputs(dtype=torch.float64):
    return torch.tensor(_VEC, dtype=dtype), torch.tensor(_DIC, dtype=dtype)


def _verify(solve, vectors, dictionary, n_atoms, tol=None):
    before = (vectors.clone(), dictionary.clone())
    got = solve(vectors, dictionary, n_atoms, tol)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"键不对：{sorted(got)}"
    assert torch.equal(vectors, before[0]) and torch.equal(dictionary, before[1]), "输入被就地修改了"
    n_rows, dim = vectors.shape
    want = _oracle(vectors.numpy(), dictionary.numpy(), n_atoms, tol)
    for key in ("lengths", "csr_indptr", "csr_indices"):
        assert got[key].dtype == torch.int64, f"{key} 应为 int64"
        assert torch.equal(got[key], torch.from_numpy(want[key])), f"{key} 与 oracle 不一致"
    total = int(want["lengths"].sum())
    assert got["csr_values"].shape == (total,) and got["csr_indices"].shape == (total,)
    assert got["csr_indptr"].shape == (n_rows + 1,)
    assert got["recon"].shape == (n_rows, dim) and got["residual_norm"].shape == (n_rows,)
    for key in ("csr_values", "residual_norm", "recon"):
        assert got[key].dtype == vectors.dtype, f"{key} 的 dtype 应跟随输入"
        torch.testing.assert_close(got[key], torch.tensor(want[key], dtype=vectors.dtype),
                                   rtol=TOL[vectors.dtype], atol=TOL[vectors.dtype],
                                   msg=f"{key} 与 oracle 不一致")
    return got


def check_example(solve):
    """题面示例：三行、n_atoms=2，第 2 行的两次相关系数并列取小下标。"""
    vectors, dic = _inputs()
    got = _verify(solve, vectors, dic, 2)
    assert got["lengths"].tolist() == [2, 2, 2]
    assert got["csr_indptr"].tolist() == [0, 2, 4, 6]
    assert got["csr_indices"].tolist() == [3, 1, 2, 0, 1, 0], "行内是被选中的先后顺序"
    torch.testing.assert_close(got["csr_values"], torch.tensor(
        [2.0, -0.5, 3.0, 0.0, 0.5, 1.0], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(got["residual_norm"], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(got["recon"][0], torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(got["recon"][2] + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                               vectors[2], rtol=0, atol=1e-12, msg="第 2 行还差原子 2 没拿到")


def check_tol_stops_before_the_next_pick(solve):
    """判停在选原子之前：lengths 是已选个数，residual_norm 也报在停止点上。"""
    vectors, dic = _inputs()
    for tol, lens in ((None, [2, 2, 2]), (0.9, [1, 1, 1]), (0.5, [1, 1, 2])):
        got = _verify(solve, vectors, dic, 2, tol)
        assert got["lengths"].tolist() == lens, (tol, got["lengths"].tolist())
        assert got["csr_indptr"].tolist() == [0] + np.cumsum(lens).tolist()
    one = _verify(solve, vectors, dic, 2, 0.9)
    assert one["csr_indices"].tolist() == [3, 2, 1], "每行都只拿到第一个原子"
    assert abs(float(one["residual_norm"][2]) ** 2 - 2.0) < 1e-12, "残差报在停止点上"
    zero = _verify(solve, torch.zeros(1, 3, dtype=torch.float64), dic, 2, 0.0)
    assert zero["lengths"].tolist() == [0] and zero["csr_indices"].numel() == 0
    assert torch.equal(zero["recon"], torch.zeros(1, 3, dtype=torch.float64)), "lengths==0 的行 recon 全零"


def check_refits_every_weight_each_round(solve):
    """第二轮的加入会改写第一轮原子的权重：原子的权重从 1.5 变成 2.0。"""
    vectors, dic = _inputs()
    solo = _verify(solve, vectors[:1], dic, 1)
    both = _verify(solve, vectors[:1], dic, 2)
    assert solo["csr_indices"].tolist() == [3] and both["csr_indices"].tolist() == [3, 1]
    torch.testing.assert_close(solo["csr_values"], torch.tensor([1.5], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(both["csr_values"], torch.tensor([2.0, -0.5], dtype=torch.float64), rtol=0, atol=1e-12)
    assert float(solo["residual_norm"][0]) > float(both["residual_norm"][0])


def check_selection_is_abs_correlation_with_ties(solve):
    """打分取绝对值（负相关照样中选），并列一律取最小下标。"""
    dic = torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    got = _verify(solve, torch.tensor([[4.0, 0.0]], dtype=torch.float64), dic, 1)
    assert got["csr_indices"].tolist() == [1], "|-8| > |4|，原子 1 中选"
    torch.testing.assert_close(got["csr_values"], torch.tensor([-2.0], dtype=torch.float64), rtol=0, atol=1e-12)
    dup = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    got2 = _verify(solve, torch.tensor([[5.0, 0.0]], dtype=torch.float64), dup, 2)
    assert got2["csr_indices"].tolist() == [0, 1], "三个原子同分，按 0、1 依次入选"
    torch.testing.assert_close(got2["csr_values"], torch.tensor([2.5, 2.5], dtype=torch.float64), rtol=0, atol=1e-9)


def check_no_repeat_and_budget_is_a_cap(solve):
    """同一行不重复选原子；n_atoms 是上限，早停时 lengths 更小，跑满时等于 n_atoms。"""
    rng = np.random.default_rng(7)
    dic = torch.tensor(rng.normal(size=(6, 5)), dtype=torch.float64)
    vectors = torch.tensor(rng.normal(size=(4, 6)), dtype=torch.float64) * 3
    for n_atoms in (1, 3, 5):
        got = _verify(solve, vectors, dic, n_atoms)
        assert got["lengths"].tolist() == [n_atoms] * 4 and int(got["csr_indptr"][-1]) == 4 * n_atoms
        for i in range(4):
            row = got["csr_indices"][got["csr_indptr"][i]:got["csr_indptr"][i + 1]].tolist()
            assert len(set(row)) == len(row) and all(0 <= j < 5 for j in row)
    narrow = torch.eye(6, dtype=torch.float64)[:, :3]
    exact = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    got = _verify(solve, exact, narrow, 3)
    assert got["lengths"].tolist() == [3] and got["csr_indices"].tolist() == [0, 1, 2]
    torch.testing.assert_close(got["csr_values"], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), rtol=0, atol=1e-12)


def check_csr_slices_rebuild_recon(solve):
    """indptr 切出来的每段自己乘回原子列，正好拼出该行的 recon。"""
    vectors, dic = _inputs()
    got = _verify(solve, vectors, dic, 3)
    for i in range(3):
        lo, hi = int(got["csr_indptr"][i]), int(got["csr_indptr"][i + 1])
        mine = (dic[:, got["csr_indices"][lo:hi]] * got["csr_values"][lo:hi]).sum(dim=1)
        torch.testing.assert_close(mine, got["recon"][i], rtol=1e-12, atol=1e-12)
        assert float((vectors[i] - got["recon"][i]).norm()) == float(got["residual_norm"][i])


def check_dtype_and_no_mutation(solve):
    """float32 入 float32 出，选中的原子与 float64 完全一致；内部一律 float64。"""
    rng = np.random.default_rng(11)
    dic64 = torch.tensor(rng.normal(size=(8, 5)), dtype=torch.float64) / np.sqrt(8)
    vec64 = torch.tensor(rng.normal(size=(4, 8)), dtype=torch.float64) * 2
    hi = _verify(solve, vec64, dic64, 3)
    lo = _verify(solve, vec64.float(), dic64.float(), 3)
    assert lo["recon"].dtype == torch.float32 and lo["residual_norm"].dtype == torch.float32
    assert torch.equal(lo["csr_indices"], hi["csr_indices"]) and torch.equal(lo["lengths"], hi["lengths"])
    torch.testing.assert_close(lo["csr_values"].double(), hi["csr_values"], rtol=1e-4, atol=1e-4)


def check_random_against_oracle(solve):
    """多形状 × 多 n_atoms × tol 有无的网格对拍；行整体放大不改变贪心顺序。"""
    rng = np.random.default_rng(20261030)
    for n_rows, dim, m, n_atoms, tol in [(3, 5, 8, 1, None), (4, 6, 4, 2, 0.9), (2, 7, 9, 4, None),
                                         (5, 4, 6, 3, 0.3), (1, 3, 5, 2, 0.0), (6, 5, 5, 5, None)]:
        dic = torch.tensor(rng.normal(size=(dim, m)) / np.sqrt(dim), dtype=torch.float64)
        vec = torch.tensor(rng.normal(size=(n_rows, dim)) * 2.0, dtype=torch.float64)
        _verify(solve, vec, dic, min(n_atoms, m), tol)
    vectors, dic = _inputs()
    base = _verify(solve, vectors, dic, 2)
    twin = _verify(solve, vectors * 10.0, dic, 2)
    assert torch.equal(base["csr_indices"], twin["csr_indices"]), "行整体放大不改变贪心顺序"
    for key in ("csr_values", "residual_norm"):
        torch.testing.assert_close(twin[key], base[key] * 10.0, rtol=1e-9, atol=1e-9)


def check_value_errors(solve):
    """维度、dtype 一致、形状、n_atoms 与 tol 的非法口径全部抛 ValueError。"""
    vectors, dic = _inputs()
    _verify(solve, vectors, dic, 2)
    bad = [(vectors[0], dic, 2, None), (vectors, dic[0], 2, None),
           (vectors, dic[:, :1], 2, None), (vectors.to(torch.float32), dic, 2, None),
           (vectors.to(torch.bfloat16), dic.to(torch.bfloat16), 2, None),
           (vectors, dic, 0, None), (vectors, dic, -1, None),
           (vectors, dic, 2.0, None), (vectors, dic, True, None), (vectors, dic, 6, None),
           (vectors, dic, 2, 1.0), (vectors, dic, 2, -0.1), (vectors, dic, 2, True),
           (vectors, dic, 2, "0.5")]
    for i, (v, d, na, t) in enumerate(bad):
        try:
            solve(v, d, na, t)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法输入没抛 ValueError：n_atoms={na}, tol={t!r}")
    # n_atoms == M 合法；用高斯随机字典，避免精确拟合行的并列判序
    rng = np.random.default_rng(3)
    rdic = torch.tensor(rng.normal(size=(5, 5)), dtype=torch.float64)
    rvec = torch.tensor(rng.normal(size=(3, 5)), dtype=torch.float64) * 2
    assert _verify(solve, rvec, rdic, 5)["lengths"].tolist() == [5] * 3, "n_atoms == M 合法"


check_example.example = True
