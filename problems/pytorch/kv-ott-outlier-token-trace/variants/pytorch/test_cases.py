import numpy as np
import torch


KEYS = {"quant_len", "scores", "new_indices", "sink_id", "sink_scores", "keys_out", "values_out"}


def _mk(amps, heads=(1.0, 3.0), d=2, dtype=torch.float64):
    """keys[0,h,t,:] = amps[t] * heads[h]，values = keys / 2。"""
    rows = [[amps[t] * heads[h] for t in range(len(amps))] for h in range(len(heads))]
    k = torch.tensor(rows, dtype=dtype).unsqueeze(-1).unsqueeze(0).expand(1, -1, -1, d)
    k = k.contiguous()
    return k, k / 2


def _backfill(arr4, idx):
    """上游 L139-140：对选中那批 token 求均值，再逐 head 广播写回各自的槽位。

    numpy 会把花式索引结果挪到最前面（torch 保位置），所以这里显式按 head 循环、
    用 np.take 锁定 token 轴，避免和参考解共用同一套广播语义。
    """
    c = arr4[0]
    sel = np.asarray(idx, dtype=np.int64)
    for h in range(c.shape[0]):
        c[h, sel, :] = np.take(c[h], idx, axis=0).mean(axis=0)


def _oracle(k, v, off, sid, ssc, gs, rl, sn, smax):
    kn = k.numpy().astype(np.float64)
    f = kn.shape[2]
    quant_len = max(0, (f - rl) // gs * gs)
    chunk = kn[:, :, :quant_len, :].copy()
    chunk_v = v.numpy().astype(np.float64)[:, :, :quant_len, :].copy()
    scores = np.abs(chunk[0]).mean(axis=(0, 2))                      # (S,)
    old_id = np.asarray(sid, dtype=np.int64).tolist()
    old_sc = np.asarray(ssc, dtype=np.float64).tolist()

    new_local, new_scores = [], []
    if quant_len:
        if not old_id:
            order = np.argsort(scores, kind="stable")[:sn]
            new_local, new_scores = order.tolist(), scores[order].tolist()
        else:
            threshold = old_sc[sn - 1]
            mask = np.flatnonzero(scores < threshold)
            if mask.size and len(old_id) < smax:
                cand = scores[mask]
                order = np.argsort(cand, kind="stable")[:sn]
                new_local, new_scores = mask[order].tolist(), cand[order].tolist()

    kept_id, kept_scores = old_id, old_sc
    if new_local:
        kept_id = old_id + [int(i) + off for i in new_local]
        kept_scores = old_sc + list(new_scores)
        perm = np.argsort(np.asarray(kept_scores, dtype=np.float64), kind="stable")
        kept_id = [kept_id[i] for i in perm]
        kept_scores = [kept_scores[i] for i in perm]
        idx = sorted(int(i) for i in new_local)
        _backfill(chunk, idx)
        _backfill(chunk_v, idx)
    return {"quant_len": quant_len, "scores": scores, "new_indices": new_local,
            "sink_id": kept_id, "sink_scores": kept_scores,
            "keys_out": chunk, "values_out": chunk_v}


def _verify(solve, k, v, off, sid, ssc, **kw):
    got = solve(k, v, off, sid, ssc, **kw)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    exp = _oracle(k, v, off, sid, ssc, kw["group_size"], kw["residual_length"],
                  kw["sink_num"], kw["sink_max_size"])
    assert torch.is_tensor(got["quant_len"]) and got["quant_len"].dtype == torch.int64
    assert got["quant_len"].item() == exp["quant_len"], (got["quant_len"].item(), exp["quant_len"])
    n = exp["quant_len"]
    for key, shape in (("scores", (n,)), ("keys_out", (1, k.shape[1], n, k.shape[3])),
                       ("values_out", (1, k.shape[1], n, k.shape[3]))):
        t = got[key]
        assert tuple(t.shape) == shape, f"{key} 形状 {tuple(t.shape)} != {shape}"
        assert t.dtype == k.dtype, f"{key} dtype 变成了 {t.dtype}"
    assert torch.allclose(got["scores"].to(torch.float64),
                          torch.tensor(exp["scores"], dtype=torch.float64)), \
        f"scores {got['scores'].tolist()} != {exp['scores'].tolist()}"
    assert got["new_indices"].dtype == torch.int64
    assert got["new_indices"].tolist() == exp["new_indices"], \
        f"new_indices {got['new_indices'].tolist()} != {exp['new_indices']}"
    assert got["sink_id"].tolist() == exp["sink_id"], f"sink_id {got['sink_id'].tolist()} != {exp['sink_id']}"
    assert torch.allclose(got["sink_scores"].to(torch.float64),
                          torch.tensor(exp["sink_scores"], dtype=torch.float64)), \
        f"sink_scores {got['sink_scores'].tolist()} != {exp['sink_scores']}"
    assert torch.allclose(got["keys_out"].to(torch.float64),
                          torch.tensor(exp["keys_out"], dtype=torch.float64)), \
        f"keys_out {got['keys_out'].tolist()}"
    assert torch.allclose(got["values_out"].to(torch.float64),
                          torch.tensor(exp["values_out"], dtype=torch.float64)), \
        f"values_out {got['values_out'].tolist()}"
    return got, exp


def _ids_scores(*pairs):
    return (torch.tensor([p[0] for p in pairs], dtype=torch.int64),
            torch.tensor([p[1] for p in pairs], dtype=torch.float64))


def check_example(solve):
    """题面两步：首步取最小两个，次步门槛卡在第 sink_num 小、整表按分数重排。"""
    kw = dict(group_size=2, residual_length=1, sink_num=2, sink_max_size=5)
    k, v = _mk([2.0, 0.5, 1.5, 1.0, 3.5])
    z = torch.zeros(0, dtype=torch.int64)
    r1, _ = _verify(solve, k, v, 0, z, torch.zeros(0, dtype=torch.float64), **kw)
    assert r1["scores"].tolist() == [4.0, 1.0, 3.0, 2.0]
    assert r1["new_indices"].tolist() == [1, 3]
    assert r1["sink_id"].tolist() == [1, 3] and r1["sink_scores"].tolist() == [1.0, 2.0]
    assert r1["keys_out"][0].tolist() == [[[2.0, 2.0], [0.75, 0.75], [1.5, 1.5], [0.75, 0.75]],
                                          [[6.0, 6.0], [2.25, 2.25], [4.5, 4.5], [2.25, 2.25]]]
    k2, v2 = _mk([2.5, 0.25, 4.0, 0.75, 3.0, 1.25, 3.5, 1.75])
    r2, _ = _verify(solve, k2, v2, 4, r1["sink_id"], r1["sink_scores"], **kw)
    assert r2["new_indices"].tolist() == [1, 3]
    assert r2["sink_id"].tolist() == [5, 1, 7, 3]
    assert r2["sink_scores"].tolist() == [0.5, 1.0, 1.5, 2.0]
    assert r2["keys_out"][0, 0, :, 0].tolist() == [2.5, 0.5, 4.0, 0.5, 3.0, 1.25]


check_example.example = True


def check_quant_len_floors_and_keeps_residual(solve):
    """向下取整到整组、至少留 residual_length；F 不够时整步空转、状态原样回吐。"""
    kw = dict(group_size=4, residual_length=3, sink_num=2, sink_max_size=9)
    for F, want in [(3, 0), (6, 0), (7, 4), (10, 4), (11, 8)]:
        k, v = _mk([float(i) for i in range(1, F + 1)], heads=(1.0,), d=1)
        got, _ = _verify(solve, k, v, 0, torch.zeros(0, dtype=torch.int64),
                         torch.zeros(0, dtype=torch.float64), **kw)
        assert got["quant_len"].item() == want, (F, got["quant_len"].item(), want)
    k, v = _mk([1.0, 2.0, 3.0], heads=(1.0,), d=1)
    sid, ssc = _ids_scores((7, 0.5), (9, 1.5))
    got, _ = _verify(solve, k, v, 5, sid, ssc, **kw)
    assert got["quant_len"].item() == 0
    assert got["sink_id"].tolist() == [7, 9] and got["sink_scores"].tolist() == [0.5, 1.5]
    assert got["keys_out"].numel() == 0


def check_threshold_is_configured_sink_num(solve):
    """门槛是 sink_scores[sink_num-1]：追到 4 条后仍然用第 2 小，比第 3/4 小但大于第 2 的不入选。"""
    kw = dict(group_size=2, residual_length=0, sink_num=2, sink_max_size=9)
    sid, ssc = _ids_scores((0, 0.5), (1, 1.0), (2, 4.0), (3, 5.0))
    k, v = _mk([3.0, 2.0], heads=(1.0,), d=1)                    # scores = [3.0, 2.0]
    got, _ = _verify(solve, k, v, 8, sid, ssc, **kw)             # 门槛 1.0，两个都不小
    assert got["new_indices"].tolist() == []
    assert got["sink_id"].tolist() == [0, 1, 2, 3]
    k, v = _mk([3.0, 0.25, 2.0, 0.2], heads=(1.0,), d=1)         # 门槛 1.0 -> 0.25 与 0.2 进
    got, _ = _verify(solve, k, v, 8, sid, ssc, **kw)
    assert got["new_indices"].tolist() == [3, 1]                 # 按分数升序
    assert got["sink_id"].tolist() == [11, 9, 0, 1, 2, 3]        # 新分数 0.2 / 0.25 插到最前


def check_strict_less_than_threshold(solve):
    """等于门槛的分不入选，严格小于才进。"""
    kw = dict(group_size=2, residual_length=0, sink_num=2, sink_max_size=9)
    sid, ssc = _ids_scores((0, 1.0), (1, 2.0))
    k, v = _mk([2.0, 0.5], heads=(1.0,), d=1)                    # 门槛 2.0：2.0 不进、0.5 进
    got, _ = _verify(solve, k, v, 4, sid, ssc, **kw)
    assert got["new_indices"].tolist() == [1]
    assert got["sink_id"].tolist() == [5, 0, 1] and got["sink_scores"].tolist() == [0.5, 1.0, 2.0]


def check_max_size_blocks_but_can_overshoot(solve):
    """上限判的是进入之前的条数：已满则整步不动；未满可一次加满 sink_num，允许顶超上限。"""
    base = dict(group_size=4, residual_length=0, sink_num=2)
    k, v = _mk([0.1, 0.2, 3.0, 4.0], heads=(1.0,), d=1)             # scores = [0.1, 0.2, 3, 4]
    sid, ssc = _ids_scores((100, 0.5), (101, 1.5))                  # 门槛 = ssc[sink_num-1] = 1.5
    got, _ = _verify(solve, k, v, 50, sid, ssc, sink_max_size=2, **base)
    assert got["new_indices"].tolist() == []                        # 0.1 / 0.2 够小但条目已满额
    assert got["sink_id"].tolist() == [100, 101]
    assert got["sink_scores"].tolist() == [0.5, 1.5]
    got, _ = _verify(solve, k, v, 50, sid, ssc, sink_max_size=3, **base)
    assert got["new_indices"].tolist() == [0, 1]                    # 进入前 2 条 < 3，两条全进来
    assert got["sink_id"].tolist() == [50, 51, 100, 101]            # 顶超上限，只重排不裁剪
    assert got["sink_scores"].tolist() == [0.1, 0.2, 0.5, 1.5]
    col = got["keys_out"][0, 0, :, 0].tolist()                   # (0.1 + 0.2) / 2 广播写回两槽
    assert abs(col[0] - 0.15) < 1e-12 and abs(col[1] - 0.15) < 1e-12
    assert col[2] == 3.0 and col[3] == 4.0


def check_backfill_is_token_mean_in_index_order(solve):
    """回填是「本次新增那批 token」的逐 head 均值：分数序与下标序取数同结果，两个 head 各算各的。"""
    kw = dict(group_size=4, residual_length=0, sink_num=2, sink_max_size=9)
    heads = (1.0, 5.0)
    k, v = _mk([4.0, 1.0, 3.0, 0.2], heads=heads, d=1)           # scores=[4,1,3,0.2]*3
    got, exp = _verify(solve, k, v, 0, torch.zeros(0, dtype=torch.int64),
                       torch.zeros(0, dtype=torch.float64), **kw)
    assert got["new_indices"].tolist() == [3, 1]                 # 分数升序
    col = got["keys_out"][0, 0, :, 0].tolist()                   # head0: 0.2 与 1.0 的均值 = 0.6
    assert abs(col[1] - 0.6) < 1e-12 and abs(col[3] - 0.6) < 1e-12
    assert col[0] == 4.0 and col[2] == 3.0
    col1 = got["keys_out"][0, 1, :, 0].tolist()                  # head1: (1.0 + 5.0) / 2 = 3.0
    assert abs(col1[1] - 3.0) < 1e-12 and abs(col1[3] - 3.0) < 1e-12


def check_scores_ignore_values(solve):
    """metric(k, v) 只吃 K：换掉 V，scores / 选择 / keys_out 全不变，values_out 跟着新 V 回填。"""
    kw = dict(group_size=3, residual_length=0, sink_num=2, sink_max_size=9)
    k, v1 = _mk([3.0, 1.0, 2.0, 0.5, 4.0, 1.5], heads=(2.0, 4.0), d=2)
    v2 = torch.ones_like(v1) * 7.0
    z = torch.zeros(0, dtype=torch.int64)
    a, _ = _verify(solve, k, v1, 0, z, torch.zeros(0, dtype=torch.float64), **kw)
    b = solve(k, v2, 0, z, torch.zeros(0, dtype=torch.float64), **kw)
    assert torch.equal(a["scores"], b["scores"]) and torch.equal(a["sink_id"], b["sink_id"])
    assert torch.equal(a["keys_out"], b["keys_out"])
    assert not torch.allclose(a["values_out"], b["values_out"])
    assert torch.allclose(b["values_out"][0, 0, 0, :], torch.tensor([7.0, 7.0], dtype=torch.float64))


def check_no_mutation_and_dtypes(solve):
    """不就地改输入；float32 进 float32 出，sink 张量原样不被改写。"""
    kw = dict(group_size=2, residual_length=0, sink_num=2, sink_max_size=9)
    k, v = _mk([1.0, 4.0, 2.0, 8.0], heads=(1.0, 3.0), d=2, dtype=torch.float32)
    sid, ssc = _ids_scores((0, 0.5), (1, 3.0))
    kc, vc, ic, sc = k.clone(), v.clone(), sid.clone(), ssc.clone()
    got = solve(k, v, 6, sid, ssc.to(torch.float32), **kw)
    assert torch.equal(k, kc) and torch.equal(v, vc)
    assert torch.equal(sid, ic) and torch.equal(ssc, sc)
    assert got["keys_out"].dtype == torch.float32 and got["sink_scores"].dtype == torch.float32
    assert got["sink_id"].dtype == torch.int64 and got["new_indices"].dtype == torch.int64


def check_resort_is_stable_and_absolute(solve):
    """同分时旧条目排在新条目前面；新条目的下标要加 quantized_len 偏移。"""
    kw = dict(group_size=2, residual_length=0, sink_num=2, sink_max_size=5)
    sid, ssc = _ids_scores((2, 1.0), (3, 2.0))                   # 门槛 ssc[1] = 2.0
    k, v = _mk([1.0, 9.0], heads=(1.0,), d=1)                    # 新分数 1.0 与旧条目同分
    got, _ = _verify(solve, k, v, 20, sid, ssc, **kw)
    assert got["new_indices"].tolist() == [0]
    assert got["sink_id"].tolist() == [2, 20, 3]                 # 旧的 2 排在同分新条目前
    assert got["sink_scores"].tolist() == [1.0, 1.0, 2.0]


def check_against_oracle(solve):
    """随机对拍：不同 H / D / dtype / 偏移 / 追踪长度，全部与纯 numpy oracle 一致。"""
    rng = np.random.default_rng(20261031)
    for trial in range(24):
        h = int(rng.integers(1, 4))
        d = int(rng.integers(1, 4))
        f = int(rng.integers(h * d + 1, 26))
        dtype = torch.float64 if trial % 2 else torch.float32
        k = torch.from_numpy(rng.normal(size=(1, h, f, d)).astype(
            np.float32 if dtype is torch.float32 else np.float64)).clone()
        v = torch.from_numpy(rng.normal(size=(1, h, f, d)).astype(
            np.float32 if dtype is torch.float32 else np.float64)).clone()
        gs = int(rng.integers(1, 6))
        rl = int(rng.integers(0, 7))
        sn = int(rng.integers(1, 5))
        smax = int(rng.integers(sn + 1, 12))
        quant_len = max(0, (f - rl) // gs * gs)
        if quant_len < sn:
            continue
        if trial % 3:
            scores = np.abs(k[0].numpy().astype(np.float64)).mean(axis=(0, 2))
            order = np.argsort(scores, kind="stable")[:sn]
            sid = torch.tensor([int(i) + 11 for i in order], dtype=torch.int64)
            # 门槛取历史分的一半，避免与本次分数在 float32 舍入下恰好相等，让严格小于的判定有歧义
            ssc = torch.tensor(scores[order] * 0.5, dtype=dtype)
        else:
            sid = torch.zeros(0, dtype=torch.int64)
            ssc = torch.zeros(0, dtype=dtype)
        _verify(solve, k, v, 11, sid, ssc, group_size=gs, residual_length=rl,
                sink_num=sn, sink_max_size=smax)


def check_value_errors(solve):
    """维度、dtype、batch、sink 长度与参数范围的不合法口径全部抛 ValueError。"""
    kw = dict(group_size=2, residual_length=1, sink_num=2, sink_max_size=5)
    k, v = _mk([1.0, 2.0, 3.0, 4.0])
    z = torch.zeros(0, dtype=torch.int64)
    zz = torch.zeros(0, dtype=torch.float64)
    sid, ssc = _ids_scores((0, 0.5), (1, 1.5))
    bad = [(k[0], v[0], 0, z, zz, kw),                             # keys 掉到三维
           (k, v[:, :, :, :1], 0, z, zz, kw),                      # k/v 形状不一致
           (k.float(), v, 0, z, zz, kw),                            # k/v dtype 不一致
           (k, v, 0, z.float(), zz, kw),                            # sink_id 不是 int64
           (k, v, 0, z, zz.float(), kw),                            # sink_scores 与 keys 不同 dtype
           (k, v, 0, z.tolist(), zz, kw),                           # sink_id 不是张量
           (k, v, 0, z, zz, dict(kw, group_size=0)),
           (k, v, 0, z, zz, dict(kw, sink_num=0)),
           (k, v, 0, z, zz, dict(kw, sink_max_size=0)),
           (k, v, 0, z, zz, dict(kw, sink_max_size=True)),
           (k, v, -1, z, zz, kw), (k, v, 0, z, zz, dict(kw, residual_length=-1)),
           (k, v, 0, sid[:1], ssc[:1], kw),                         # 追踪条数不足 sink_num
           (k, v, 0, sid, ssc[:1], kw),                             # 两个长度不一致
           (k, v, 0, sid.to(torch.int32), ssc, kw),
           (k, v, 0.0, z, zz, kw)]
    for i, (kk, vv, off, s1, s2, opts) in enumerate(bad):
        try:
            solve(kk, vv, off, s1, s2, **opts)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法输入没抛 ValueError")
    r = solve(k, v, 0, z, zz, **kw)                             # F=4 -> quant_len=2 >= sink_num=2
    assert r["quant_len"].item() == 2
    try:
        solve(k, v, 0, z, zz, group_size=1, residual_length=0, sink_num=5, sink_max_size=5)
    except ValueError:
        pass
    else:
        raise AssertionError("sink_num 超过可量化 token 数应抛 ValueError")
