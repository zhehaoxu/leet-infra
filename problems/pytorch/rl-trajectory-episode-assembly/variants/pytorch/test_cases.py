import torch


def _li(xs):
    return torch.tensor(xs, dtype=torch.long)


def _fa(xs):
    return torch.tensor(xs, dtype=torch.float32)


def _oracle(episode_id, terminated, truncated, actions, rewards, values, final_value):
    """独立重算（与 solution 的向量化 cumsum/argsort/bincount/gather 路径结构完全不同）：
    显式逐段扫描 → python sorted() 排段 → 段内逐步 for 循环三分规则 → 纯 list 拼接成张量。"""
    ep = [int(x) for x in episode_id.tolist()]
    trunc = [int(x) for x in truncated.tolist()]
    a = [int(x) for x in actions.tolist()]
    r = [float(x) for x in rewards.tolist()]
    v = [float(x) for x in values.tolist()]
    fv = [float(x) for x in final_value.tolist()]
    T = len(ep)
    if T == 0:
        e64 = torch.zeros(0, dtype=torch.long)
        e32 = torch.zeros(0, dtype=torch.float32)
        return (e64.clone(), e32.clone(), e32.clone(), e64.clone(),
                e32.clone(), e64.clone(), e64.clone())

    # 1) 显式扫描出极大连续段：(id, 出现段序号, [步下标])
    segments = []
    for t in range(T):
        if t == 0 or ep[t] != ep[t - 1]:
            segments.append((ep[t], len(segments), []))
        segments[-1][2].append(t)

    # 2) 段按 id 升序（python sorted，非 argsort），段内朴素 for 循环
    oa, orr, ov, od, onv, olens, ostarts = [], [], [], [], [], [], []
    pos = 0
    for _seg_id, seg_no, idxs in sorted(segments, key=lambda s: s[0]):
        L = len(idxs)
        olens.append(L)
        ostarts.append(pos)
        pos += L
        for k in range(L):
            t = idxs[k]
            done = 1 if k == L - 1 else 0
            if k + 1 < L:
                nv = v[idxs[k + 1]]        # 同段后继
            elif trunc[t] == 1:
                nv = fv[seg_no]            # 截断 bootstrap
            else:
                nv = 0.0                   # 真终止不 bootstrap
            oa.append(a[t])
            orr.append(r[t])
            ov.append(v[t])
            od.append(done)
            onv.append(nv)

    return (_li(oa), _fa(orr), _fa(ov), _li(od), _fa(onv), _li(olens), _li(ostarts))


# ---- 样例用例（与题面示例逐位一致，示例输出即本参考解运行所得） --------------------
def check_example(solve):
    out = solve(_li([7, 7, 7, 3, 3, 8, 8, 8]),
                _li([0, 0, 1, 0, 0, 0, 0, 0]),
                _li([0, 0, 0, 0, 1, 0, 0, 1]),
                _li([10, 11, 12, 20, 21, 30, 31, 32]),
                _fa([1.0, 0.0, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0]),
                _fa([0.5, 0.625, 0.75, 0.25, 0.375, 0.125, 0.25, 0.3125]),
                _fa([0.9375, 0.875, 0.625]))
    assert len(out) == 7
    got_a, got_r, got_v, got_d, got_nv, got_len, got_start = out
    assert torch.equal(got_a, _li([20, 21, 10, 11, 12, 30, 31, 32])), got_a
    assert torch.allclose(got_r, _fa([0.5, 2.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0]), atol=1e-5, rtol=1e-5), got_r
    assert torch.allclose(got_v, _fa([0.25, 0.375, 0.5, 0.625, 0.75, 0.125, 0.25, 0.3125]), atol=1e-5, rtol=1e-5), got_v
    assert torch.equal(got_d, _li([0, 1, 0, 0, 1, 0, 0, 1])), got_d
    assert torch.allclose(got_nv, _fa([0.375, 0.875, 0.625, 0.75, 0.0, 0.25, 0.3125, 0.625]), atol=1e-5, rtol=1e-5), got_nv
    assert torch.equal(got_len, _li([2, 3, 3])), got_len
    assert torch.equal(got_start, _li([0, 2, 5])), got_start
    assert got_a.dtype == torch.long and got_r.dtype == torch.float32 and got_nv.dtype == torch.float32


check_example.example = True


# ---- 退化/规则探针 ---------------------------------------------------------------
def check_cross_episode_leak_probe(solve):
    """截断末步若天真 roll(-1) 会拿到邻段 value（0.25/0.5），与正确的 0.875、0.0 均可区分。
    flag 口径按题面保证：每段只在**末步**带一个 flag——id=6 段末步 t=1 截断（bootstrap
    final_value[0]=0.875），id=1 段末步 t=3 真终止（0）。"""
    out = solve(_li([6, 6, 1, 1]),
                _li([0, 0, 0, 1]),
                _li([0, 1, 0, 0]),
                _li([60, 61, 10, 11]),
                _fa([0.5, 0.5, 1.0, 0.0]),
                _fa([0.5, 0.75, 0.25, 0.5]),
                _fa([0.875, 0.9375]))
    a, r, v, d, nv, ln, st = out
    assert torch.equal(a, _li([10, 11, 60, 61])), a
    assert torch.allclose(r, _fa([1.0, 0.0, 0.5, 0.5]), atol=1e-5, rtol=1e-5), r
    assert torch.allclose(v, _fa([0.25, 0.5, 0.5, 0.75]), atol=1e-5, rtol=1e-5), v
    assert torch.equal(d, _li([0, 1, 0, 1])), d
    assert torch.allclose(nv, _fa([0.5, 0.0, 0.75, 0.875]), atol=1e-5, rtol=1e-5), nv
    assert torch.equal(ln, _li([2, 2])), ln
    assert torch.equal(st, _li([0, 2])), st


def check_single_step_segments(solve):
    """两个单步 episode（首末同一步），一真终止一截断；单步截断也必须 bootstrap。"""
    out = solve(_li([5, 2, 2, 3]),
                _li([1, 0, 0, 0]),
                _li([0, 0, 1, 1]),
                _li([50, 20, 21, 30]),
                _fa([1.0, 0.5, 0.5, 0.0]),
                _fa([0.25, 0.5, 0.75, 1.0]),
                _fa([0.9375, 0.875, 0.625]))
    a, r, v, d, nv, ln, st = out
    assert torch.equal(a, _li([20, 21, 30, 50])), a
    assert torch.allclose(r, _fa([0.5, 0.5, 0.0, 1.0]), atol=1e-5, rtol=1e-5), r
    assert torch.allclose(v, _fa([0.5, 0.75, 1.0, 0.25]), atol=1e-5, rtol=1e-5), v
    assert torch.equal(d, _li([0, 1, 1, 1])), d
    assert torch.allclose(nv, _fa([0.75, 0.875, 0.625, 0.0]), atol=1e-5, rtol=1e-5), nv
    assert torch.equal(ln, _li([2, 1, 1])), ln
    assert torch.equal(st, _li([0, 2, 3])), st


def check_all_truncated(solve):
    """全截断：done 位的 next_value 全为非零 bootstrap（含 0.0625 这种小值，不许当成 0）。"""
    out = solve(_li([4, 4, 4, 1, 1]),
                _li([0, 0, 0, 0, 0]),
                _li([0, 0, 1, 0, 1]),
                _li([40, 41, 42, 10, 11]),
                _fa([1.0, 1.0, 1.0, 1.0, 1.0]),
                _fa([0.5, 0.25, 0.125, 0.75, 1.0]),
                _fa([0.0625, 0.875]))
    a, r, v, d, nv, ln, st = out
    assert torch.equal(a, _li([10, 11, 40, 41, 42])), a
    assert torch.allclose(v, _fa([0.75, 1.0, 0.5, 0.25, 0.125]), atol=1e-5, rtol=1e-5), v
    assert torch.equal(d, _li([0, 1, 0, 0, 1])), d
    assert torch.allclose(nv, _fa([1.0, 0.875, 0.25, 0.125, 0.0625]), atol=1e-5, rtol=1e-5), nv
    assert torch.equal(ln, _li([2, 3])), ln
    assert torch.equal(st, _li([0, 2])), st
    assert bool((nv[d == 1] != 0).all()), nv


def check_all_terminated(solve):
    """全真终止：所有 done 位 next_value 恰为 0，final_value 一概不被使用。"""
    out = solve(_li([2, 2, 9]),
                _li([0, 1, 1]),
                _li([0, 0, 0]),
                _li([20, 21, 90]),
                _fa([0.5, 0.5, 1.0]),
                _fa([0.25, 0.5, 1.0]),
                _fa([0.5, 0.75]))
    a, r, v, d, nv, ln, st = out
    assert torch.equal(a, _li([20, 21, 90])), a
    assert torch.allclose(v, _fa([0.25, 0.5, 1.0]), atol=1e-5, rtol=1e-5), v
    assert torch.equal(d, _li([0, 1, 1])), d
    assert torch.allclose(nv, _fa([0.5, 0.0, 0.0]), atol=1e-5, rtol=1e-5), nv
    assert torch.equal(ln, _li([2, 1])), ln
    assert torch.equal(st, _li([0, 2])), st
    assert bool((nv[d == 1] == 0).all()), nv


def check_empty_input(solve):
    """T=0/K=0：七个 0 元素张量、dtype 正确、不抛异常。"""
    out = solve(torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32))
    assert len(out) == 7
    for i, o in enumerate(out):
        assert o.numel() == 0, (i, o.shape)
        assert o.dtype == (torch.long if i in (0, 3, 5, 6) else torch.float32), (i, o.dtype)


def check_sorted_ids_identity(solve):
    """id 已升序 → 训练序=出现序（gather 应退化为恒等），与乱序共用同一套规则。"""
    out = solve(_li([1, 1, 2, 3]),
                _li([0, 1, 0, 0]),
                _li([0, 0, 1, 1]),
                _li([10, 11, 20, 30]),
                _fa([1.0, 0.0, 1.0, 0.0]),
                _fa([0.125, 0.25, 0.5, 0.75]),
                _fa([0.9375, 0.875, 0.625]))
    a, r, v, d, nv, ln, st = out
    assert torch.equal(a, _li([10, 11, 20, 30])), a
    assert torch.allclose(v, _fa([0.125, 0.25, 0.5, 0.75]), atol=1e-5, rtol=1e-5), v
    assert torch.equal(d, _li([0, 1, 1, 1])), d
    assert torch.allclose(nv, _fa([0.25, 0.0, 0.875, 0.625]), atol=1e-5, rtol=1e-5), nv
    assert torch.equal(ln, _li([2, 1, 1])), ln
    assert torch.equal(st, _li([0, 2, 3])), st


# ---- 与独立 oracle 对拍（显式循环 vs 向量化） -------------------------------------
# (episode_id, terminated, truncated, final_value)；actions/rewards/values 由下标公式生成，全程无随机。
_BATTERY = [
    ([7, 7, 7, 3, 3, 8, 8, 8], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1],
     [0.9375, 0.875, 0.625]),
    ([6, 6, 1, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0.875, 0.9375]),
    ([5, 2, 2, 3], [1, 0, 0, 0], [0, 0, 1, 1], [0.9375, 0.875, 0.625]),
    ([4, 4, 4, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0.0625, 0.875]),
    ([2, 2, 9], [0, 1, 1], [0, 0, 0], [0.5, 0.75]),
    ([1, 1, 2, 3], [0, 1, 0, 0], [0, 0, 1, 1], [0.9375, 0.875, 0.625]),
    ([3], [1], [0], [0.25]),
    ([3], [0], [1], [0.125]),
    ([0], [0], [1], [0.5]),
    ([11, 11, 11, 11, 2, 2, 7, 7, 7, 7, 7, 1, 3],
     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0.5, 0.25, 0.875, 0.625, 0.75]),
    ([9, 9, 9, 9, 9, 8, 1, 1, 1, 2, 2, 2, 2, 2, 3],
     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0.375, 0.625, 0.125, 0.5, 0.875]),
]


def _battery_args(ids, term, trunc, fv):
    T = len(ids)
    actions = _li([10 * (i % 5) + i for i in range(T)])
    rewards = _fa([((i * 7) % 5) / 4.0 for i in range(T)])
    values = _fa([((i * 11) % 7) / 8.0 for i in range(T)])
    return _li(ids), _li(term), _li(trunc), actions, rewards, values, _fa(fv)


def check_matches_oracle_battery(solve):
    for ids, term, trunc, fv in _BATTERY:
        args = _battery_args(ids, term, trunc, fv)
        got = solve(*args)
        exp = _oracle(*args)
        assert torch.equal(got[0], exp[0]), (ids, got[0], exp[0])
        assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), (ids, got[1], exp[1])
        assert torch.allclose(got[2], exp[2], atol=1e-5, rtol=1e-5), (ids, got[2], exp[2])
        assert torch.equal(got[3], exp[3]), (ids, got[3], exp[3])
        assert torch.allclose(got[4], exp[4], atol=1e-5, rtol=1e-5), (ids, got[4], exp[4])
        assert torch.equal(got[5], exp[5]), (ids, got[5], exp[5])
        assert torch.equal(got[6], exp[6]), (ids, got[6], exp[6])


def check_invariants_length_offsets(solve):
    """结构性不变量探针（不依赖 oracle）：
    ⓪ 输入自检（题面保证）：每段恰在末步带且只带一个 flag，其余步两 flag 均为 0；
    ① sum(lengths)==T；② starts[0]=0 且 starts 是 lengths 的 exclusive 前缀和；
    ③ sum(dones)==K==len(lengths)；④ 训练序第 e 段的 values 恰等于「第 e 小的 id 的出现段」
       原序切片（重排分段正确、段内保序；注意 seg_arr 用**出现序**下标去取出现序 values，
       而 seg_tr 用**训练序**下标取输出 values，两者不能混用同一条列表）；
    ⑤ 真终止不 bootstrap：done 位中 nv==0 的数量恰等于 terminated flag 总数；
    ⑥ 截断必 bootstrap：done 位中 nv!=0 的数量恰等于 truncated flag 总数，且这些 nv 全落在
       final_value 值集内；⑦ 非 done 位的 nv 必等于训练序下一位的 values（同段后继）。"""
    for ids, term, trunc, fv in _BATTERY:
        args = _battery_args(ids, term, trunc, fv)
        a, r, v, d, nv, ln, st = solve(*args)
        T = len(ids)
        # ⓪ 题面保证：flag 只出现在各段末步且二者异或，其余步全 0。
        term_t = [int(x) for x in args[1].tolist()]
        trunc_t = [int(x) for x in args[2].tolist()]
        seg_end = [t == T - 1 or ids[t + 1] != ids[t] for t in range(T)]
        for t in range(T):
            xor = term_t[t] + trunc_t[t]
            assert xor == (1 if seg_end[t] else 0), (ids, t, term_t, trunc_t)
        assert len(fv) == sum(int(x) for x in seg_end), (ids, fv)
        lens = [int(x) for x in ln.tolist()]
        starts = [int(x) for x in st.tolist()]
        nv_list = [float(x) for x in nv.tolist()]
        v_arr = [float(x) for x in args[5].tolist()]      # 出现序的 V(s_t)
        v_train = [float(x) for x in v.tolist()]          # 训练序的 V(s_t)
        K = len(set(ids))
        assert sum(lens) == T, (ids, ln)                       # ① 步数账
        assert len(lens) == K and len(starts) == K, (ids, ln, st)
        assert starts[0] == 0, (ids, st)                        # ② offsets 账
        for e in range(1, K):
            assert starts[e] == starts[e - 1] + lens[e - 1], (ids, st, lens)
        assert int(d.sum().item()) == K, (ids, d)               # ③ 边界数
        # ④ 分段重排正确：训练序第 e 段 = 第 e 小 id 段的原序 values
        arrival_by_id = {}
        for t, eid in enumerate(ids):
            arrival_by_id.setdefault(eid, []).append(t)
        sorted_ids = sorted(arrival_by_id)
        for e in range(K):
            seg_arr = [v_arr[t] for t in arrival_by_id[sorted_ids[e]]]
            seg_tr = v_train[starts[e]:starts[e] + lens[e]]
            assert seg_arr == seg_tr, (ids, e, seg_arr, seg_tr)
        # ⑤⑥ done 位的 bootstrap 账（电池内 final_value 全非零，故 nv==0 ⟺ 真终止）
        zero_boot = sum(1 for i in range(T) if int(d[i].item()) == 1 and nv_list[i] == 0.0)
        nonzero_boot = sum(1 for i in range(T) if int(d[i].item()) == 1 and nv_list[i] != 0.0)
        assert zero_boot == sum(term), (ids, nv, d, term)
        assert nonzero_boot == sum(trunc), (ids, nv, d, trunc)
        fset = set(_fa(fv).tolist())
        for i in range(T):
            if int(d[i].item()) == 1 and nv_list[i] != 0.0:
                assert nv_list[i] in fset, (ids, i, nv_list[i], fset)
        # ⑦ 非 done 位：nv == 下一训练位的 values
        for i in range(T - 1):
            if int(d[i].item()) == 0:
                assert abs(nv_list[i] - v_train[i + 1]) < 1e-6, (ids, i, nv_list, v_train)
