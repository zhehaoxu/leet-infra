def _oracle(p_cmp_heads, L, l, top_n, window, gates, branch_outputs):
    """独立重写的 oracle：rank 用 (score, -idx) 升序排序改写、
    求和用 sum(map(...))、聚合用显式循环——与 solution 不同代码路径，
    同论文 2502.11089 语义。"""
    if not p_cmp_heads:
        raise ValueError("empty")
    n = len(p_cmp_heads[0])
    if any(len(r) != n for r in p_cmp_heads):
        raise ValueError("ragged")
    if n != L // l or top_n < 1 or window < 1 or len(gates) != 3:
        raise ValueError("bad params")
    o_cmp, o_slc, o_win = branch_outputs
    if not o_cmp or len(o_cmp) != len(o_slc) or len(o_slc) != len(o_win):
        raise ValueError("bad outputs")

    # (10) 组内求和（sum + map 改写）
    shared = [sum(map(lambda r: r[i], p_cmp_heads)) for i in range(n)]

    # (11) (-score, idx) 升序 == 分数降序 + 并列小索引优先；
    # rank = 位置 + 1 <= top_n ⇔ 取前 top_n 个
    ranked = sorted(range(n), key=lambda i: (-shared[i], i))
    keep = set(ranked[: min(top_n, n)])

    cmp_len = n * l
    slc_len = len(keep) * l
    win_len = min(window, L)

    g = gates
    out = []
    for j in range(len(o_cmp)):
        out.append(g[0] * o_cmp[j] + g[1] * o_slc[j] + g[2] * o_win[j])

    return {
        "shared_scores": shared,
        "selected_blocks": sorted(keep),
        "cmp_kv_len": cmp_len,
        "slc_kv_len": slc_len,
        "win_kv_len": win_len,
        "n_remapped": cmp_len + slc_len + win_len,
        "output": out,
    }


def _approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


def _compare(solve, p, L, l, top_n, window, gates, branch_outputs):
    exp = _oracle(p, L, l, top_n, window, gates, branch_outputs)
    got = solve(p, L, l, top_n, window, gates, branch_outputs)
    assert set(got) == set(exp), f"键不一致：{sorted(got)} vs {sorted(exp)}"
    assert got["selected_blocks"] == exp["selected_blocks"], (
        f"selected_blocks：期望 {exp['selected_blocks']}，得到 {got['selected_blocks']}"
    )
    for k in ("cmp_kv_len", "slc_kv_len", "win_kv_len", "n_remapped"):
        assert got[k] == exp[k], f"{k}：期望 {exp[k]}，得到 {got[k]}"
    assert len(got["shared_scores"]) == len(exp["shared_scores"])
    for i, (a, b) in enumerate(zip(got["shared_scores"], exp["shared_scores"])):
        assert _approx(a, b), f"shared_scores[{i}]：期望 {b}，得到 {a}"
    for i, (a, b) in enumerate(zip(got["output"], exp["output"])):
        assert _approx(a, b), f"output[{i}]：期望 {b}，得到 {a}"
    return got


def check_basic_single_head(solve):
    # 单 head、4 块、top 2：p = [0.1, 0.4, 0.2, 0.3] → 共享分同 p，
    # rank: 块1(0.4)=1、块3(0.3)=2、块2(0.2)=3、块0(0.1)=4 → 选 {1,3}
    p = [[0.1, 0.4, 0.2, 0.3]]
    led = _compare(solve, p, L=8, l=2, top_n=2, window=4,
                   gates=[0.5, 0.3, 0.2],
                   branch_outputs=[[1.0], [1.0], [1.0]])
    assert led["shared_scores"] == [0.1, 0.4, 0.2, 0.3]
    assert led["selected_blocks"] == [1, 3]
    assert led["cmp_kv_len"] == 8 and led["slc_kv_len"] == 4 and led["win_kv_len"] == 4
    assert led["n_remapped"] == 16
    assert _approx(led["output"][0], 1.0)  # 0.5+0.3+0.2


def check_gqa_group_sum(solve):
    # 式 (10)：组内 H 个 head **求和**共享（不是平均/最大）。
    # head0=[0.5, 0.1], head1=[0.2, 0.3], head2=[0.1, 0.2] → shared=[0.8, 0.6]
    p = [[0.5, 0.1], [0.2, 0.3], [0.1, 0.2]]
    led = _compare(solve, p, L=4, l=2, top_n=1, window=2,
                   gates=[1.0, 0.0, 0.0], branch_outputs=[[1], [1], [1]])
    assert _approx(led["shared_scores"][0], 0.8)
    assert _approx(led["shared_scores"][1], 0.6)
    assert led["selected_blocks"] == [0]
    # 单 head 的分数格局与组和不同：若逐 head 独立选，head1 会选块 1
    # —— 本题强制组内一致，这正是 GQA 下的 KV 加载账


def check_tie_stable_low_index(solve):
    # 并列分数：rank 按块索引升序依次排，top 2 只带走索引小者。
    # p = [0.5, 0.3, 0.3, 0.1] → rank: 块0=1、块1=2、块2=3、块3=4 → 选 {0,1}
    p = [[0.5, 0.3, 0.3, 0.1]]
    led = _compare(solve, p, L=8, l=2, top_n=2, window=1,
                   gates=[0, 1, 0], branch_outputs=[[1], [1], [1]])
    assert led["selected_blocks"] == [0, 1]


def check_top_n_larger_than_blocks(solve):
    # top_n 超过块数：全选（rank 条件对全部块成立），n_sel = n_blocks
    p = [[0.2, 0.5], [0.3, 0.5]]
    led = _compare(solve, p, L=4, l=2, top_n=10, window=1,
                   gates=[0, 1, 0], branch_outputs=[[1], [1], [1]])
    assert led["selected_blocks"] == [0, 1]
    assert led["slc_kv_len"] == 4
    assert led["n_remapped"] == 4 + 4 + 1


def check_window_cap(solve):
    # 滑窗长度 = min(w, L)：窗口大于序列长度时退化全量
    # L=8, l=4：n_blocks=2
    led = _compare(solve, [[0.6, 0.4]], 8, 4, 1, 100,
                   [0, 0, 1], [[1], [1], [1]])
    assert led["cmp_kv_len"] == 8 and led["slc_kv_len"] == 4
    assert led["win_kv_len"] == 8  # min(100, 8)
    assert led["n_remapped"] == 20


def check_window_normal(solve):
    # 正常滑窗：L=10, w=3 → 3 个 token；压缩块 l=2 → 5 块
    p = [[0.1] * 5]
    led = _compare(solve, p, L=10, l=2, top_n=3, window=3,
                   gates=[0, 0, 1], branch_outputs=[[1], [1], [1]])
    assert led["win_kv_len"] == 3
    assert led["n_remapped"] == 10 + 6 + 3


def check_gated_aggregation(solve):
    # 式 (5)：g_cmp*o_cmp + g_slc*o_slc + g_win*o_win，逐维独立
    p = [[1.0]]
    led = _compare(solve, p, L=1, l=1, top_n=1, window=1,
                   gates=[0.2, 0.3, 0.5],
                   branch_outputs=[[2.0, 4.0], [1.0, 1.0], [0.4, 0.0]])
    assert _approx(led["output"][0], 0.2 * 2 + 0.3 * 1 + 0.5 * 0.4)  # 0.9
    assert _approx(led["output"][1], 0.2 * 4 + 0.3 * 1 + 0.5 * 0.0)  # 1.1


def check_zero_gate_branch_silent(solve):
    # 门控为 0 的分支输出不影响聚合（防 shortcut 的独立 KV 设计账）
    p = [[1.0]]
    led = _compare(solve, p, L=1, l=1, top_n=1, window=1,
                   gates=[0.0, 1.0, 0.0],
                   branch_outputs=[[999.0], [2.0], [-999.0]])
    assert _approx(led["output"][0], 2.0)


def check_l_equals_one_degenerate(solve):
    # l=1：每 token 一块，n_blocks = L；块分数即逐 token 分数
    p = [[0.05, 0.15, 0.3, 0.5]]
    led = _compare(solve, p, L=4, l=1, top_n=2, window=2,
                   gates=[1, 0, 0], branch_outputs=[[1], [1], [1]])
    assert led["selected_blocks"] == [2, 3]
    assert led["cmp_kv_len"] == 4 and led["slc_kv_len"] == 2
    assert led["win_kv_len"] == 2 and led["n_remapped"] == 8


def check_selected_sorted_ascending(solve):
    # 输出契约：selected_blocks 升序 —— rank 序 [1,2,0] 必须重排为 [0,1,2]
    p = [[0.1, 0.4, 0.2, 0.05]]
    led = _compare(solve, p, 8, 2, 3, 1, [0, 1, 0], [[1], [1], [1]])
    assert led["selected_blocks"] == [0, 1, 2]
    assert led["slc_kv_len"] == 6


def check_invalid_inputs(solve):
    for bad in [
        ([], 4, 2, 1, 1, [1, 0, 0], [[1], [1], [1]]),          # 空 head 列表
        ([[0.5], [0.5, 0.5]], 4, 2, 1, 1, [1, 0, 0], [[1], [1], [1]]),  # head 长度不一
        ([[0.5, 0.5]], 4, 2, 0, 1, [1, 0, 0], [[1], [1], [1]]),          # top_n < 1
        ([[0.5, 0.5]], 4, 2, 1, 0, [1, 0, 0], [[1], [1], [1]]),          # window < 1
        ([[0.5, 0.5]], 4, 2, 1, 1, [1, 0], [[1], [1], [1]]),             # gates 长度
        ([[0.5, 0.5]], 4, 2, 1, 1, [1, 0, 0], [[1], [1]]),               # 输出长度不一
        ([[0.5, 0.5]], 4, 2, 1, 1, [1, 0, 0], [[], [], []]),             # 空输出
    ]:
        try:
            solve(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应抛 ValueError：{bad}")
