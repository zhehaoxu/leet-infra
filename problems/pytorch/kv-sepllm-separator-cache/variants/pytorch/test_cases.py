import numpy as np
import torch


def _ids(values):
    return torch.tensor(values, dtype=torch.int64)


def _raises(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"应抛 ValueError：{label}")


def _oracle(ids, sep_ids, kvs, init, cap, local, boundary):
    """numpy 独立重算：打 keep 掩码（分隔符 FIFO 淘汰最旧的）后 np.take 聚集。"""
    seq = [int(v) for v in ids.tolist()]
    sep = {int(v) for v in (sep_ids.tolist() if torch.is_tensor(sep_ids) else sep_ids)}
    L, right, keep = len(seq), max(boundary, init), np.zeros(len(seq), dtype=bool)
    keep[:init] = keep[L - local:] = True
    queue = (list(range(init, right)) + [p for p in range(right, L - local) if seq[p] in sep])[-cap:]
    keep[np.asarray(queue, dtype=np.int64)] = True
    idx = np.flatnonzero(keep)
    gather = lambda ts: [torch.from_numpy(np.take(t.detach().numpy().astype(np.float64), idx, 1)).to(t.dtype) for t in ts]
    return {"kept": torch.from_numpy(idx), "boundary": init + len(queue), "kv": gather(kvs),
            "token_ids": torch.tensor([seq[p] for p in idx.tolist()], dtype=torch.int64)}


def _kv(L, dtype=torch.float64, heads=1, dim=1, layers=1):
    """K 的数值等于槽位号、V 取其负，便于从输出反查保留下来的位置。"""
    base = torch.arange(L, dtype=dtype).reshape(1, L, 1)
    return [base.repeat(heads, 1, dim)] * layers, [(-base).repeat(heads, 1, dim)] * layers


def _verify(solve, ids, sep, keys, values, init, cap, local, boundary=0):
    snap = (ids.clone(), [t.clone() for t in keys + values])
    got = solve(ids, sep, keys, values, init, cap, local, boundary)
    assert isinstance(got, dict) and set(got) == {"kept", "token_ids", "keys", "values", "boundary"}, f"键不对：{sorted(got)}"
    assert torch.equal(ids, snap[0]) and all(torch.equal(a, b) for a, b in zip(keys + values, snap[1])), "输入被就地修改"
    L, keep = int(ids.shape[0]), got["kept"].tolist()
    n = len(keep)
    assert got["kept"].dtype == torch.int64 and keep == sorted(set(keep)) and n <= init + cap + local, (
        f"kept 需 int64、升序去重且不超过 init+s+local：{keep}")
    assert keep[:init] == list(range(init)), f"前 init 个槽位必须原样是 sink：{keep}"
    assert keep[-local:] == list(range(L - local, L)), f"末尾必须是本地窗：{keep}"
    assert got["token_ids"].dtype == torch.int64 and tuple(got["token_ids"].shape) == (n,)
    shapes = {(tuple(t.shape), t.dtype) for t in got["keys"] + got["values"]}
    assert shapes == {((keys[0].shape[0], n, keys[0].shape[2]), keys[0].dtype)}, "聚集后的形状/dtype 不对"
    want = _oracle(ids, sep, keys + values, init, cap, local, boundary)
    assert keep == want["kept"].tolist(), f"kept：{keep} vs 期望 {want['kept'].tolist()}"
    assert got["token_ids"].tolist() == want["token_ids"].tolist(), f"token_ids 不符，期望 {want['token_ids'].tolist()}"
    tol = 1e-5 if keys[0].dtype == torch.float32 else 2e-9
    for a, b in zip(got["keys"] + got["values"], want["kv"]):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
    assert got["boundary"] == want["boundary"] and type(got["boundary"]) is int, f"boundary：{got['boundary']} vs 期望 {want['boundary']}"
    return got


def check_example(solve):
    """题面示例：L=9、init=2、s=2、local=3，过去窗口 3 个分隔符只留最新的 2 个。"""
    got = _verify(solve, _ids([9, 9, 1, 5, 1, 2, 3, 7, 1]), {1, 2}, *_kv(9), 2, 2, 3, 0)
    assert (got["kept"].tolist(), got["token_ids"].tolist()) == ([0, 1, 4, 5, 6, 7, 8], [9, 9, 1, 2, 3, 7, 1])
    assert got["boundary"] == 4 and got["keys"][0][0, :, 0].tolist() == [0, 1, 4, 5, 6, 7, 8], got["keys"][0]


check_example.example = True


def check_newest_separators_win(solve):
    """超容量只留最新的 s 个；boundary=0 与 boundary=init 都是首次压缩，结果必须相同。"""
    first = _verify(solve, _ids([0, 1, 5, 1, 1, 9, 1, 3]), {1}, *_kv(8), 1, 2, 2, 0)
    warm = _verify(solve, _ids([0, 1, 5, 1, 1, 9, 1, 3]), {1}, *_kv(8), 1, 2, 2, 1)
    assert (first["kept"].tolist(), first["boundary"], warm["kept"].tolist()) == ([0, 3, 4, 6, 7], 3, first["kept"].tolist()), (
        f"应淘汰位置 1、边界锁在 init+s、boundary=init 与 0 同结果，got {first['kept'].tolist()} / {first['boundary']}")
    full = _verify(solve, _ids([8, 8, 1, 1, 1, 4, 1, 7, 7, 7]), {1}, *_kv(10), 2, 3, 3, 5)
    assert (full["kept"].tolist(), full["boundary"]) == ([0, 1, 3, 4, 6, 7, 8, 9], 5), "应淘汰位置 2、边界保持 5"


def check_region_grows_under_cap(solve):
    """未到容量：分隔符区按出现顺序增长；窗口没有分隔符时边界与累积区原样保留。"""
    first = _verify(solve, _ids([7, 7, 1, 5, 1, 3, 9, 9]), {1}, *_kv(8), 2, 4, 2, 0)
    assert (first["kept"].tolist(), first["boundary"]) == ([0, 1, 2, 4, 6, 7], 4), "两个分隔符 → 边界 init+2"
    second = _verify(solve, torch.cat([first["token_ids"], _ids([1, 5, 9])]), {1}, *_kv(9), 2, 4, 2, first["boundary"])
    assert (second["token_ids"].tolist(), second["boundary"]) == ([7, 7, 1, 1, 1, 5, 9], 5), "累积 2 + 新 1 → init+3"
    quiet = _verify(solve, _ids([8, 8, 1, 1, 7, 7, 7, 3, 3]), {1}, *_kv(9), 2, 3, 2, 4)
    assert (quiet["kept"].tolist(), quiet["boundary"]) == ([0, 1, 2, 3, 7, 8], 4), "窗口无分隔符 → 边界不动"


def check_sink_and_local_do_not_consume_budget(solve):
    """sink 前缀与本地窗里的分隔符都不占 sep_cache_size 名额，也不推进 boundary。"""
    got = _verify(solve, _ids([1, 1, 1, 5, 9, 7, 4, 4]), {1}, *_kv(8), 3, 2, 2, 0)
    assert (got["kept"].tolist(), got["boundary"]) == ([0, 1, 2, 6, 7], 3), "窗口无分隔符 → 边界停在 init"
    got2 = _verify(solve, _ids([1, 1, 1, 1, 1, 9, 4, 4]), {1}, *_kv(8), 3, 2, 2, 0)
    assert (got2["kept"].tolist(), got2["boundary"]) == ([0, 1, 2, 3, 4, 6, 7], 5), "窗口 2 个 → 边界 init+2"
    tail = _verify(solve, _ids([9, 1, 1, 5, 1, 1, 1, 1]), {1}, *_kv(8), 1, 1, 4, 0)
    assert (tail["kept"].tolist(), tail["boundary"]) == ([0, 2, 4, 5, 6, 7], 2), "本地窗 4 个分隔符不计名额"


def check_boundary_indexes_the_compacted_cache(solve):
    """boundary 是压紧后槽位空间的下标：token_ids[init:boundary] 正好是保留下来的分隔符。"""
    got = _verify(solve, _ids([9, 1, 1, 5, 1, 2, 1, 7, 1]), {1, 2}, *_kv(9), 2, 3, 2, 0)
    assert got["boundary"] == 5 and got["kept"].tolist()[2:5] == [4, 5, 6], "分隔符区是压紧后的 [init, boundary)"
    assert got["token_ids"][2:5].tolist() == [1, 2, 1], "压紧后下标 2..4 必须正好是留下的 3 个分隔符"


def check_boundary_precondition(solve):
    """上游 assert seq_len - local_size > sep_exrange：过去窗口为空必须抛 ValueError。"""
    for local, boundary in ((4, 0), (3, 3), (3, 99), (3, -1), (3, 1)):
        _raises(lambda: solve(_ids([9, 1, 1, 1, 1, 5]), {1}, *_kv(6), 2, 2, local, boundary), f"L=6, local={local}, b={boundary}")


def check_chained_stream(solve):
    """连续 4 次压缩（每次追加 4 个 token）：边界逐次推进、缓存不超上界、逐步对拍。"""
    stream = [7, 7, 1, 4, 2, 9, 1, 8, 5, 2, 3, 1, 6, 6, 2, 8, 1, 4, 9, 9, 1, 3, 7, 2, 5, 5]
    init, cap, local, boundary, ids = 2, 3, 4, 0, _ids(stream[:10])
    for step in range(4):
        got = _verify(solve, ids, {1, 2}, *_kv(ids.numel()), init, cap, local, boundary)
        boundary, ids = got["boundary"], torch.cat([got["token_ids"], _ids(stream[10 + 4 * step:14 + 4 * step])])
        assert ids.numel() - local > boundary, "下一步必须满足断言前提"
    assert boundary == init + cap, f"流够长时右边界最终锁在 init+s，got {boundary}"


def check_random_against_oracle(solve):
    """随机 id / 边界 / 层数 / dtype，全键与 numpy oracle 对拍。"""
    rng = np.random.default_rng(2025)
    for seed, (init, cap, local, L) in enumerate([(2, 3, 4, 12), (1, 5, 2, 20), (4, 2, 6, 30), (3, 4, 3, 16)]):
        ids = _ids(rng.integers(0, 6, size=L).tolist())
        dtype, boundary = ((torch.float64, 0) if seed % 2 == 0 else (torch.float32, min(init + cap, L - local - 1)))
        sep = torch.tensor([1, 2], dtype=torch.int64) if seed % 3 == 0 else [1, 2]
        _verify(solve, ids, sep, *_kv(L, dtype=dtype, heads=2, dim=3, layers=2), init, cap, local, boundary)


def check_no_mutation_and_dtype(solve):
    """float32 进 float32 出；非连续视图与连续副本一致；输入不被就地修改。"""
    big = torch.arange(48, dtype=torch.float32).reshape(1, 24, 2)
    k, v = big[:, ::2, :], -(big[:, ::2, :])               # 12 个槽位，seq 维步长 2
    ids = _ids([1, 9, 1, 4, 1, 5, 9, 9, 1, 7, 3, 3])
    got = _verify(solve, ids, {1}, [k], [v], 2, 2, 3, 0)
    twin = _verify(solve, ids, [1], [k.contiguous()], [v.contiguous()], 2, 2, 3, 0)
    assert (got["kept"].tolist(), got["keys"][0].dtype, got["token_ids"].tolist()) == ([0, 1, 4, 8, 9, 10, 11], torch.float32, twin["token_ids"].tolist()), "非连续视图与连续副本必须同结果"


def check_value_errors(solve):
    """非法形状 / dtype / 尺寸 / id 集合一律抛 ValueError，不许断言崩或静默截断。"""
    ids, (keys, values) = _ids([9, 1, 5, 1, 1, 9]), _kv(6)
    bad = [
        (ids.reshape(1, 6), {1}, keys, values, 1, 1, 2, 0),                 # token_ids 非一维
        (torch.tensor([], dtype=torch.int64), {1}, *_kv(0), 1, 1, 2, 0), (ids.float(), {1}, keys, values, 1, 1, 2, 0),
        (ids, {1}, [torch.zeros(1, 5, 1)], values, 1, 1, 2, 0),             # seq 维与 id 不符
        (ids, {1}, [_kv(6)[0][0], torch.zeros(2, 6, 1)], values, 1, 1, 2, 0),  # 层间 head 数不符
        (ids, {1}, [k.float() for k in keys], values, 1, 1, 2, 0),          # K/V dtype 不同
        (ids, {1}, [t.to(torch.int32) for t in keys], [t.to(torch.int32) for t in values], 1, 1, 2, 0),
        (ids, {1}, keys, values, 0, 1, 2, 0), (ids, {1}, keys, values, 1, 0, 2, 0),  # 尺寸 < 1
        (ids, {1}, keys, values, 1, 1, 0, 0), (ids, {1}, keys, values, 1.0, 1, 2, 0),  # 非 int 参数
        (ids, {1}, keys, values, 1, True, 2, 0), (ids, {1.5}, keys, values, 1, 1, 2, 0),  # bool / 非 int id
    ]
    for i, args in enumerate(bad):
        _raises(lambda: solve(*args), f"第 {i} 个非法输入")
