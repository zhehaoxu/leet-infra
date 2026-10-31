"""独立 numpy oracle（逐 token、逐码本向量化算欧氏距离）+ 定向 checks。"""

import numpy as np
import torch

KEYS = {"scale", "residual_norm", "indices", "quantized", "error"}

def _perm(x, d, inv=False):
    """inv=False：切段后转置（上游 view/transpose/reshape）；inv=True：split+stack 复原。"""
    sh = x.shape
    if inv:
        return np.stack([x[..., m * d:(m + 1) * d] for m in range(sh[-1] // d)], axis=-1).reshape(sh)
    return x.reshape(*sh[:-1], -1, d).swapaxes(-1, -2).reshape(sh)

def _oracle(x, codebooks, steps, reorder):
    xn = x.detach().numpy().astype(np.float64)
    sh = xn.shape
    tables = [cb.detach().numpy().astype(np.float64) for cb in codebooks]
    n_cb, _, dhat = tables[0].shape
    rows = xn.reshape(-1, sh[-1])
    if reorder:
        rows = _perm(rows, dhat)
    mu = rows.mean(axis=1, keepdims=True)
    scale = np.sqrt(((rows - mu) ** 2).sum(axis=1) / (rows.shape[1] - 1))    # correction=1
    scale[scale == 0.0] = 1.0
    cur, norms, idx = rows / scale[:, None], [], []
    for table in tables:
        m = table.shape[1]
        trail, picks = [np.linalg.norm(cur, axis=1)], []
        for _ in range(steps):
            seg = cur.reshape(-1, n_cb, dhat)
            step, add = np.zeros((seg.shape[0], n_cb), np.int64), np.zeros_like(cur)
            for k in range(n_cb):
                dist = ((seg[:, k, None, :] - table[k][None]) ** 2).sum(-1)   # 直接差方，不用展开式
                tie = dist == dist.min(axis=1, keepdims=True)
                step[:, k] = np.argmax(tie, axis=1) + k * m                   # 首个 True = 最小下标
                add[:, k * dhat:(k + 1) * dhat] = table[k][step[:, k] - k * m]
            cur = cur - add
            picks.append(step)
            trail.append(np.linalg.norm(cur, axis=1))
        norms.append(np.stack(trail, 0))
        idx.append(np.stack(picks, 0))
    recon = rows / scale[:, None] - cur
    if reorder:
        recon = _perm(recon, dhat, True)
    return {"scale": scale.reshape(sh[:-1] + (1,)),
            "residual_norm": np.stack(norms, 0).reshape((len(tables), steps + 1) + sh[:-1]),
            "indices": np.stack(idx, 0).reshape((len(tables), steps) + sh[:-1] + (n_cb,)),
            "quantized": (recon * scale[:, None]).reshape(sh),
            "error": (np.linalg.norm(cur, axis=1) * scale).reshape(sh[:-1])}

def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)

def _verify(solve, x, codebooks, steps, reorder):
    before = (x.clone(), [cb.clone() for cb in codebooks])
    got, want = solve(x, codebooks, steps, reorder), _oracle(x, codebooks, steps, reorder)
    b, ell, dim = (int(v) for v in x.shape)
    n_cb, nq = int(codebooks[0].shape[0]), len(codebooks)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(x, before[0]) and all(torch.equal(a, b) for a, b in zip(codebooks, before[1])), \
        "输入被就地修改了"
    shapes = {"scale": (b, ell, 1), "residual_norm": (nq, steps + 1, b, ell),
              "indices": (nq, steps, b, ell, n_cb), "quantized": (b, ell, dim), "error": (b, ell)}
    for key, shape in shapes.items():
        assert tuple(got[key].shape) == shape, f"{key} 形状 {tuple(got[key].shape)} != {shape}"
        if key != "indices":
            assert got[key].dtype == x.dtype, f"{key} dtype {got[key].dtype} != {x.dtype}"
            _close(got[key], torch.tensor(want[key], dtype=x.dtype))
    assert got["indices"].dtype == torch.int64, f"indices 必须是 int64，got {got['indices'].dtype}"
    assert int(got["indices"].max()) < n_cb * max(int(cb.shape[1]) for cb in codebooks), "indices 越界"
    assert got["indices"].tolist() == want["indices"].tolist(), f"indices：{got['indices'].tolist()} vs {want['indices'].tolist()}"
    return got

def _cb(rows):
    return torch.tensor(rows, dtype=torch.float64)
_EXAMPLE_CB = _cb([[[0.5, 1.0], [-0.05, -0.1]], [[1.5, 3.0], [-0.1, -0.2]]])

def check_example(solve):
    """题面示例：B=L=1、D=4、dhat=2、N=2、两轮残差、reorder_channel=True。"""
    got = _verify(solve, _cb([[[1.0, 3.0, 2.0, 6.0]]]), [_EXAMPLE_CB], 2, True)
    _close(got["scale"], _cb([[[2.160246899469287]]]))
    _close(got["residual_norm"],
           _cb([[[[3.2732683535398857]], [[0.26226555239285204]], [[0.038319636922768574]]]]))
    assert got["indices"].tolist() == [[[[[0, 2]]], [[[1, 3]]]]], got["indices"].tolist()
    want_q = [[[0.9721111047611791, 3.0243456592570017, 1.9442222095223582, 6.048691318514003]]]
    _close(got["quantized"], _cb(want_q))
    _close(got["error"], _cb([[0.08277987685119961]]))

check_example.example = True

def check_offsets_and_residual_steps(solve):
    """下标是「条目号 + k*M」；轨迹首值 ||z||，末段范数 × scale 就是 error；std=0 → scale=1。"""
    got = _verify(solve, _cb([[[1.0, 3.0, 2.0, 6.0]]]), [_EXAMPLE_CB], 2, True)
    traj = got["residual_norm"][0, :, 0, 0]
    assert torch.all(traj[1:] <= traj[:-1] + 1e-12) and traj[0] > traj[-1] * 50, \
        f"这份嵌套码本下两轮应当明显收敛：{traj.tolist()}"
    _close(got["error"], (traj[-1] * got["scale"][0, 0, 0]).reshape(1, 1))
    assert got["indices"][0, 1].tolist() == [[[1, 3]]], "第二轮应改选补残差的条目（第二码本偏移 +2）"
    x = _cb([[[1.0, 2.0, 1.0, 2.0]]])
    got2 = _verify(solve, x, [_cb([[[1.0, 2.0], [9.0, 9.0]], [[9.0, 9.0], [1.0, 2.0]]])], 1, False)
    assert got2["indices"][0, 0, 0, 0].tolist() == [0, 3], "第二码本选条目 1 时应是 M+1"
    const = _verify(solve, _cb([[[5.0, 5.0, 5.0, 5.0]]]), [_EXAMPLE_CB], 1, False)
    assert const["scale"].item() == 1.0, "std=0 必须换成 1，且不能出现 NaN"

def check_reorder_pairs_cross_channels(solve):
    """dhat=2、N=2 时重排把 [c0,c1,c2,c3] 变成 [c0,c2,c1,c3]：段变成 (0,2) 与 (1,3)。"""
    x = _cb([[[0.0, 0.0, 1.0, -1.0]]])
    cb = _cb([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
    plain, over = _verify(solve, x, [cb], 1, False), _verify(solve, x, [cb], 1, True)
    assert plain["indices"][0, 0, 0, 0].tolist() == [0, 2], plain["indices"].tolist()
    assert over["indices"][0, 0, 0, 0].tolist() == [1, 2], (
        f"重排后第一段是 (c0,c2)=[0,1.2247]，应选到条目 1：{over['indices'].tolist()}")
    _close(plain["scale"], over["scale"])
    assert not torch.allclose(plain["quantized"], over["quantized"]), "重排必须改变重建"

def check_ties_prefer_smaller_entry(solve):
    """并列取更小的条目下标：与 ± 条目等距，以及逐位相同的重复条目。"""
    cb = _cb([[[1.0, 0.0], [-1.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]])
    got = _verify(solve, _cb([[[0.0, 0.0, 1.0, -1.0]]]), [cb], 1, False)      # 首段与 ±1 等距
    assert got["indices"][0, 0, 0, 0].tolist() == [0, 2], f"等距必须取小下标：{got['indices'].tolist()}"
    zero = _verify(solve, _cb([[[0.0, 0.0, 0.0, 0.0]]]), [cb], 2, False)
    assert zero["indices"][0, 0, 0, 0].tolist() == [0, 2], "全零 token 的首轮也取小下标"

def check_multiple_quantizers(solve):
    """第二个 quantizer 接手第一个的残差：轨迹首尾衔接、误差再降一个量级。"""
    x = _cb([[[1.0, 3.0, 2.0, 6.0]]])
    fine = _cb([[[-0.04, -0.07], [0.0, 0.0]], [[-0.11, -0.22], [0.0, 0.0]]])
    one = _verify(solve, x, [_EXAMPLE_CB], 1, True)
    two = _verify(solve, x, [_EXAMPLE_CB, fine], 1, True)
    _close(two["residual_norm"][1, 0], two["residual_norm"][0, -1])
    assert two["error"].item() < one["error"].item() * 0.1, "第二个 quantizer 应把残差再压一个量级"

def check_value_errors(solve):
    x, cb = _cb([[[1.0, 2.0, 3.0, 4.0]]]), _EXAMPLE_CB
    bad = [("x 不是三维", (x[0], [cb])), ("x 是整型", (x.to(torch.int64), [cb])),
           ("D=1 时 std 为 NaN", (_cb([[[1.0]]]), [_cb([[[0.5]]])])),
           ("D != N*dhat", (x, [_cb([[[1.0, 2.0], [3.0, 4.0]]])])),
           ("codebooks 为空", (x, [])), ("codebook 是二维", (x, [_cb([[1.0, 2.0]])])),
           ("codebook dtype 不同", (x, [cb.float()])),
           ("各 codebook 的 dhat 不同", (x, [cb, _cb([[[1.0], [2.0]]] * 4)])),
           ("num_residual_steps=0", (x, [cb], 0)), ("num_residual_steps 是 bool", (x, [cb], True)),
           ("reorder_channel 非 bool", (x, [cb], 1, 1))]
    for tag, args in bad:
        try:
            solve(*args)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{tag} 抛了 {type(exc).__name__} 而不是 ValueError") from exc
        raise AssertionError(f"{tag} 没有抛 ValueError")

def check_random_against_oracle(solve):
    """随机形状 × 多轮 × 多量化器 × 两种 reorder（含 float32 一档）与 oracle 全键对拍。"""
    gen = torch.Generator().manual_seed(20261101)

    def roll(low, high):
        return int(torch.randint(low, high, (1,), generator=gen))

    for case in range(8):
        b, ell, n_cb, dhat = roll(1, 3), roll(1, 4), roll(1, 4), roll(2, 5)
        x = torch.randn(b, ell, n_cb * dhat, generator=gen, dtype=torch.float64)
        x = torch.round(x * 4.0) / 4.0 if case % 3 == 0 else x
        if case == 5:
            x = x.float()                                   # 出口 dtype 必须是输入 dtype
        cbs = [torch.randn(n_cb, roll(1, 4), dhat, generator=gen, dtype=x.dtype)
               for _ in range(roll(1, 3))]
        _verify(solve, x, cbs, roll(1, 4), bool(case % 2))
