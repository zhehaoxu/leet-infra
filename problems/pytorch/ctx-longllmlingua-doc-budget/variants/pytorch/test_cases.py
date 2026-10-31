import numpy as np

KEYS = {"rank_order", "selected", "dynamic_ratios", "tau", "window_plan",
        "thresholds", "keep_masks"}


def _oracle(doc_token_ids, doc_ppl, token_loss, target_token, context_budget="+100",
            force_context_ids=None, reorder_context="original",
            dynamic_context_compression_ratio=0.0, condition_in_question="none",
            iterative_size=200):
    docs = [list(d) for d in doc_token_ids]
    ppl = np.asarray(doc_ppl, dtype=np.float64)
    loss = np.asarray(token_loss, dtype=np.float64)
    n_docs = len(docs)

    direct = -1.0 if condition_in_question == "none" else 1.0
    rank_order = [int(i) for i in np.argsort(ppl * direct, kind="stable")]

    tt = 100.0 if target_token < 0 else float(target_token)
    op, num = context_budget[0], float(context_budget[1:])
    tt = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
          "*": lambda a, b: a * b, "/": lambda a, b: a / b}[op](tt, num)

    used = list(force_context_ids) if force_context_ids is not None else []
    budget = tt  # 窗口阶段用求值后的原值，选文档的扣减只影响停止条件
    for idx in rank_order:
        budget -= len(docs[idx])
        if idx not in used:
            used.append(idx)
        if budget < 0:
            break
    original_used = used

    if reorder_context == "original":
        used = sorted(used)
    else:
        evens = [x for i, x in enumerate(used) if i % 2 == 0]
        odds = [x for i, x in enumerate(used) if i % 2 == 1]
        used = evens + odds[::-1]

    lam = abs(float(dynamic_context_compression_ratio))
    n_used = len(original_used)
    if lam > 0:
        step = lam / (n_used - 1) if n_used > 1 else 0.0
        vals = [i * step for i in range(-(n_used - 1), n_used, 2)][::-1]
        rmap = dict(zip(original_used, vals))
        dynamic_ratios = [rmap[i] for i in used]
    else:
        dynamic_ratios = [0.0] * len(used)

    context_length = [len(docs[i]) + 2 for i in used]
    tau = tt / (sum(context_length) + 1)
    doc_ratios = [max(min(1.0, tau + dr), 0.0) for dr in dynamic_ratios]

    # 独立实现：按「剩余容量」模拟，而不是上游的 last 游标
    windows, cur, capacity = [], [], iterative_size - 1
    for seg_len, seg_ratio in zip(context_length, doc_ratios):
        if seg_len >= capacity:
            cur.append([capacity, seg_ratio])
            windows.append(cur)
            rem = seg_len - capacity
            k = rem // iterative_size
            windows.extend([[[iterative_size, seg_ratio]] for _ in range(k)])
            rem -= k * iterative_size
            capacity = iterative_size - rem
            cur = [[rem, seg_ratio]] if rem else []
        else:
            capacity -= seg_len
            cur.append([seg_len, seg_ratio])
    if cur:
        windows.append(cur)

    filtered = np.asarray([v for v in loss if v != 10000.0], dtype=np.float64)
    total = len(loss)
    thresholds, keep_masks = [], []
    flat = 0  # 上游 end 每 pair 推进 iterative_size：第 g 个 pair 覆盖块 [g*it, g*it+delta_end)
    for window in windows:
        for delta_end, ratio in window:
            if ratio == 1.0:
                thr = float("-inf")
            else:
                k = max(0, min(len(filtered) - 1, int(len(filtered) * ratio) - 1))
                thr = float(np.sort(filtered)[::-1][k])
            mask = np.ones(total, dtype=bool)
            lo = flat * iterative_size
            hi = min(lo + delta_end, total)
            for j in range(lo, hi):
                if j > 0:
                    mask[j] = loss[j] > thr
            thresholds.append(thr)
            keep_masks.append([bool(x) for x in mask])
            flat += 1
    return {
        "rank_order": rank_order,
        "selected": used,
        "dynamic_ratios": dynamic_ratios,
        "tau": tau,
        "window_plan": windows,
        "thresholds": thresholds,
        "keep_masks": keep_masks,
    }


def _verify(solve, *args, **kwargs):
    docs_snapshot = [list(d) for d in args[0]]
    got = solve(*args, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    want = _oracle(*args, **kwargs)
    assert got["rank_order"] == want["rank_order"], (got["rank_order"], want["rank_order"])
    assert got["selected"] == want["selected"], (got["selected"], want["selected"])
    assert len(got["dynamic_ratios"]) == len(want["dynamic_ratios"])
    for a, b in zip(got["dynamic_ratios"], want["dynamic_ratios"]):
        assert abs(float(a) - float(b)) < 1e-9, (a, b)
    assert abs(float(got["tau"]) - float(want["tau"])) < 1e-9
    assert got["window_plan"] == want["window_plan"], (
        got["window_plan"], want["window_plan"])
    assert len(got["thresholds"]) == len(want["thresholds"]) == len(got["keep_masks"])
    for a, b in zip(got["thresholds"], want["thresholds"]):
        if a == float("-inf") or b == float("-inf"):
            assert a == b, (a, b)
        else:
            assert abs(float(a) - float(b)) < 1e-9, (a, b)
    for a, b in zip(got["keep_masks"], want["keep_masks"]):
        assert a == b
    assert [list(d) for d in args[0]] == docs_snapshot, "不应修改输入文档"
    return got


def check_example(solve):
    """题面示例一：3 文档预算 + two_stage 关 + 动态压缩率 + 小窗计划。"""
    got = _verify(
        solve,
        [[10, 11, 12], [20, 21], [30, 31, 32, 33]],
        [2.0, 8.0, 5.0],
        [3.0, 1.0, 4.0, 1.5, 3.5, 0.5, 2.5, 2.0, 2.8],
        10,
        context_budget="+5", reorder_context="two_stage",
        dynamic_context_compression_ratio=0.6, iterative_size=8,
    )
    assert got["rank_order"] == [1, 2, 0]
    assert got["selected"] == [1, 0, 2], got["selected"]
    assert np.allclose(got["dynamic_ratios"], [0.6, -0.6, 0.0]), got["dynamic_ratios"]
    assert abs(got["tau"] - 0.9375) < 1e-12
    assert got["window_plan"] == [[[4, 1.0], [3, 0.3375]], [[2, 0.3375], [6, 0.9375]]]
    assert got["thresholds"][0] == float("-inf")
    assert abs(got["thresholds"][1] - 3.0) < 1e-9
    assert abs(got["thresholds"][2] - 3.0) < 1e-9
    assert abs(got["thresholds"][3] - 1.0) < 1e-9
    assert got["keep_masks"][0] == [True] * 9
    assert got["keep_masks"][1] == [True] * 8 + [False]
    assert all(all(m) for m in got["keep_masks"][2:])


check_example.example = True


def check_matches_oracle(solve):
    """多组合对拍：条件方向、force、two_stage、λ、预算四则、哨兵、λ=0、单文档。"""
    base = dict(
        context_budget="+0", iterative_size=10,
    )
    cases = [
        # question 感知方向：升序（条件 PPL 越低越相关）
        dict(docs=[[1, 2, 3], [4, 5], [6, 7, 8, 9]], ppl=[4.0, 1.0, 2.5],
             loss=[float(i + 1) for i in range(9)], tt=8,
             extra=dict(condition_in_question="question", **base)),
        # force_context_ids：强制文档占预算但不必排最前
        dict(docs=[[1] * 4, [2] * 3, [3] * 5], ppl=[1.0, 2.0, 3.0],
             loss=[float(i + 1) for i in range(12)], tt=6,
             extra=dict(force_context_ids=[0], **base)),
        # 预算乘法与除法
        dict(docs=[[1] * 5, [2] * 4], ppl=[2.0, 1.0],
             loss=[float(i + 1) for i in range(9)], tt=5,
             extra=dict(context_budget="*2", iterative_size=6)),
        dict(docs=[[1] * 5, [2] * 4], ppl=[2.0, 1.0],
             loss=[float(i + 1) for i in range(9)], tt=50,
             extra=dict(context_budget="/5", iterative_size=6)),
        # two_stage：5 篇全选中，奇数位反转才与不反转不同
        dict(docs=[[1] * 2, [2] * 3, [3] * 2, [4] * 3, [5] * 2],
             ppl=[1.0, 2.0, 3.0, 4.0, 5.0],
             loss=[float(i + 1) for i in range(12)], tt=20,
             extra=dict(reorder_context="two_stage", iterative_size=40)),
        # target_token<0 先归 100
        dict(docs=[[1] * 3, [2] * 3], ppl=[1.0, 2.0],
             loss=[float(i + 1) for i in range(6)], tt=-7,
             extra=dict(iterative_size=5)),
        # 哨兵 10000 不参与阈值统计
        dict(docs=[[1] * 5, [2] * 4], ppl=[1.0, 2.0],
             loss=[3.0, 10000.0, 4.0, 1.0, 2.0, 0.5, 2.5, 1.5, 3.5], tt=6,
             extra=dict(iterative_size=7)),
        # λ>0、N==1：只选中文档一个，动态比率为 0
        dict(docs=[[1] * 3, [2] * 8], ppl=[5.0, 1.0],
             loss=[float(i + 1) for i in range(11)], tt=4,
             extra=dict(dynamic_context_compression_ratio=0.8, iterative_size=6)),
        # 单文档 + 大 λ + ratio 撞 1.0（预算充足）
        dict(docs=[[1] * 10], ppl=[1.0],
             loss=[float(i + 1) for i in range(10)], tt=100,
             extra=dict(dynamic_context_compression_ratio=0.5, iterative_size=4)),
    ]
    for case in cases:
        _verify(solve, case["docs"], case["ppl"], case["loss"], case["tt"], **case["extra"])


def check_budget_break_and_force(solve):
    """预算耗尽即停：排序靠后的文档拿不到预算；force 文档即使概率低也保留。"""
    # 文档 token 长度 [6, 2, 4]，预算 5+0=5：rank 降序 [2 (3.0), 1 (2.0), 0 (1.0)]
    got = _verify(solve, [[1] * 6, [2] * 2, [3] * 4], [1.0, 2.0, 3.0],
                  [float(i + 1) for i in range(12)], 5,
                  context_budget="+0", iterative_size=50)
    assert got["rank_order"] == [2, 1, 0]
    assert got["selected"] == [1, 2], got["selected"]  # doc2 扣 4 剩 1；doc1 扣 2 剩 -1 停；doc0 未入选

    # force [0]：used 从 [0] 起步，排序遍历照常扣预算
    got = _verify(solve, [[1] * 6, [2] * 2, [3] * 4], [1.0, 2.0, 3.0],
                  [float(i + 1) for i in range(12)], 5,
                  context_budget="+0", force_context_ids=[0], iterative_size=50)
    assert got["selected"] == [0, 1, 2], got["selected"]


def check_window_plan_k_extension(solve):
    """超长文档跨多个整窗：余量整除出 k 个 (iterative_size, r) 整窗，再留余数窗。"""
    # 文档 len 22（+2=24），iterative_size=5：1+24=25>=5 → (4,r)，last=1+24-5=20
    # 20>5 → k=4 个整窗，last=0 → 余数窗为空；再补一条短文档进新窗
    got = _verify(solve, [[1] * 22, [2] * 3], [1.0, 2.0],
                  [float(i + 1) for i in range(25)], 100,
                  context_budget="+0", iterative_size=5)
    r = got["window_plan"]
    assert len(r) == 6, r
    assert r[0] == [[4, 1.0]]
    assert all(w == [[5, 1.0]] for w in r[1:])


def check_threshold_direction_and_sentinel(solve):
    """loss > threshold 保留高 NLL；哨兵 10000 只退出统计、不影响 mask；等值丢弃。"""
    loss = [2.0, 10000.0, 6.0, 4.0, 4.0, 1.0]
    got = _verify(solve, [[1] * 6], [1.0], loss, 100,
                  context_budget="+0", iterative_size=100)
    # 预算 100：tau = 100/(6+2+1) = 11.1 → clamp 1.0 → ratio 1.0 → -inf 全留
    assert got["thresholds"] == [float("-inf")]
    assert got["keep_masks"][0] == [True] * 6

    # 预算压到 tau<1：ratio = tau；无 λ
    got = _verify(solve, [[1] * 6], [1.0], loss, 4,
                  context_budget="+0", iterative_size=100)
    # tau = 4/9；int(5*4/9)=int(2.22)=2 → pos1 → 过滤哨兵后降序 [6,4,4,2,1][1] = 4.0
    assert abs(got["tau"] - 4.0 / 9.0) < 1e-9
    assert abs(got["thresholds"][0] - 4.0) < 1e-9
    # region [0, 6)：j0 恒真，j2 6>4 T，j3/j4 等于 4.0 丢弃，j1 哨兵 > 4 保留
    assert got["keep_masks"][0] == [True, True, True, False, False, False], got["keep_masks"][0]


def check_validation(solve):
    """长度、类型、取值范围、枚举、budget 串等非法输入抛 ValueError。"""
    docs3 = [[1, 2], [3], [4, 5, 6]]
    loss9 = [1.0] * 9

    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    bad(lambda: solve([], [1.0], [1.0], 5))
    bad(lambda: solve([[]], [1.0], [], 5))
    bad(lambda: solve([[1, "a"]], [1.0], [1.0, 2.0], 5))
    bad(lambda: solve([[1, 2]], [1.0, 2.0], [1.0, 2.0], 5))
    bad(lambda: solve(docs3, [1.0, 2.0], loss9, 5))
    bad(lambda: solve(docs3, [1.0, float("nan"), 3.0], loss9, 5))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9[:-1], 5))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, True))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], [1.0, float("inf")] + loss9[2:], 5))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, context_budget="100"))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, context_budget="+1e3"))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, reorder_context="reverse"))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, condition_in_question="doc"))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, iterative_size=0))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, iterative_size=True))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, force_context_ids=[3]))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5, force_context_ids=[True]))
    bad(lambda: solve(docs3, [1.0, 2.0, 3.0], loss9, 5,
                      dynamic_context_compression_ratio="0.5"))
