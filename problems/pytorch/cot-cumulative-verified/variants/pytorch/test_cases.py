"""Cumulative Reasoning（Game24）累积态验证 -- 测试用例（期望值由官方 solve() 沙箱实测钉死）."""


def R(chosen, nxt, rem, judge, verify, eq=''):
    return {'chosen': chosen, 'next': nxt, 'remaining': rem,
            'judgement': judge, 'verify': verify, 'equation': eq}


# ---------- mode="dist" ----------

def check_dist(solve):
    assert solve(mode="dist", index=0, n_states=4, trycnt=6) == \
        [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]
    assert solve(mode="dist", index=2, n_states=4, trycnt=6) == \
        [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]  # 2 < 1/2*6=3.0 仍在前半程
    assert solve(mode="dist", index=3, n_states=4, trycnt=6) == [0.25, 0.25, 0.25, 0.25]
    assert solve(mode="dist", index=1, n_states=1, trycnt=6) == [1.0]
    # 奇数 trycnt：1/2*5=2.5，i=2 仍偏置、i=3 均匀
    assert solve(mode="dist", index=2, n_states=3, trycnt=5) == [0.5, 0.25, 0.25]
    assert solve(mode="dist", index=3, n_states=3, trycnt=5) == [1 / 3, 1 / 3, 1 / 3]
    assert solve(mode="dist", index=0, n_states=1, trycnt=0) == [1.0]  # 0 < 0 不成立


# ---------- mode="solve" / "state"：准入与成功路径 ----------

def check_success_basic(solve):
    rounds = [R('2, 4, 6', '  2 * 4 = 8  ', '6, 8', 'Valid', 'likely', '2 * 4 = 8'),
              R('6, 8', '6 * 8 = 48', '48', 'Invalid', 'likely'),
              R('6, 8', '4 * 6 = 24', '24', 'Valid', 'sure', '(2 * 4) * 6 = 24')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=3) == \
        (True, 2, ['2 * 4 = 8 (left 6, 8)', '4 * 6 = 24 (left 24)', '(2 * 4) * 6 = 24'])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=3)
    assert st['records'] == ['2, 4, 6', '6, 8'], st
    assert st['parent'] == {'6, 8': '2, 4, 6'}, st
    assert st['last_step'] == {'6, 8': '2 * 4 = 8'}, st  # 存储的是 strip 后算式
    assert st['forbidden'] == {'2, 4, 6': ['2 * 4 = 8'], '6, 8': []}, st


def check_deep_chain(solve):
    rounds = [R('1, 2, 3, 4', '1 * 2 = 2', '2, 3, 4', 'Valid', 'likely', '1 * 2 = 2'),
              R('2, 3, 4', '2 * 3 = 6', '4, 6', 'Valid', 'likely', '2 * 3 = 6'),
              R('4, 6', '4 * 6 = 24', '24', 'Valid', 'sure', '(1 * 2) * 3 * 4 = 24')]
    assert solve(mode="solve", puzzle='1, 2, 3, 4', rounds=rounds, trycnt=3) == \
        (True, 2, ['1 * 2 = 2 (left 2, 3, 4)', '2 * 3 = 6 (left 4, 6)',
                   '4 * 6 = 24 (left 24)', '(1 * 2) * 3 * 4 = 24'])


def check_gates_no_success(solve):
    rounds = [R('2, 4, 6', '2 * 4 = 8!', '6, 8', 'Valid', 'likely'),   # 门 A：'!' 非法
              R('2, 4, 6', '2 * 4 = 8', '   ', 'Valid', 'likely'),     # 门 B：全空白
              R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),    # 准入
              R('6, 8', '6 * 8 = 48', '48', 'Valid', 'likely'),        # 准入
              R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),    # a 已在 records
              R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'impossible')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=6) == (False, 6, [])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=6)
    assert st['records'] == ['2, 4, 6', '6, 8', '48'], st
    assert st['parent'] == {'6, 8': '2, 4, 6', '48': '6, 8'}, st
    assert st['last_step'] == {'6, 8': '2 * 4 = 8', '48': '6 * 8 = 48'}, st
    assert st['forbidden'] == {'2, 4, 6': ['2 * 4 = 8'],
                               '6, 8': ['6 * 8 = 48'], '48': []}, st


def check_dup_drop(solve):
    # 第二个相同 likely：a 已在 records -> 整轮静默丢弃，forbidden 不追加
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),
              R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),
              R('2, 4, 6', '4 * 6 = 24', '24', 'Valid', 'sure', '4 * 6 = 24')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=3) == \
        (True, 2, ['4 * 6 = 24 (left 24)', '4 * 6 = 24'])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=3)
    assert st['forbidden']['2, 4, 6'] == ['2 * 4 = 8'], st  # 只追加一次
    assert st['records'] == ['2, 4, 6', '6, 8'], st


def check_multiline_edges(solve):
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8  \njunk', 'Valid', 'likely'),
              R('6, 8', '6 * 8 = 48\n6 + 8 = 14', '48', 'Valid', 'likely'),
              R('48', '48 / 2 = 24', '24', 'Valid', 'sure', '6 * 8 = 24  \njunk')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=3) == \
        (True, 2, ['2 * 4 = 8 (left 6, 8)', '6 * 8 = 48\n6 + 8 = 14 (left 48)',
                   '48 / 2 = 24 (left 24)', '6 * 8 = 24'])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=3)
    # a 取 remaining 首行（strip 后 '6, 8  \njunk' -> '6, 8'），junk 不入状态
    assert '6, 8  \njunk' not in st['records'] and '6, 8' in st['records'], st
    # last_step 存完整 strip 后的 next（含内部换行）
    assert st['last_step']['48'] == '6 * 8 = 48\n6 + 8 = 14', st
    # equation 取 strip 后首行（尾随空白被二次 strip 剪掉）
    assert st['forbidden']['6, 8'] == ['6 * 8 = 48\n6 + 8 = 14'], st


def check_forbidden_accumulation(solve):
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),
              R('2, 4, 6', '2 + 4 = 6', '6, 6', 'Valid', 'likely'),
              R('6, 6', '6 * 6 = 36', '36', 'Valid', 'impossible'),
              R('6, 6', '6 + 6 = 12', '12', 'Valid', 'likely'),
              R('12', '12 * 2 = 24', '24', 'Valid', 'sure', '2 * (2 + 4) + 6 + 6 = 24')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=5) == \
        (True, 4, ['2 + 4 = 6 (left 6, 6)', '6 + 6 = 12 (left 12)',
                   '12 * 2 = 24 (left 24)', '2 * (2 + 4) + 6 + 6 = 24'])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=5)
    assert st['forbidden']['2, 4, 6'] == ['2 * 4 = 8', '2 + 4 = 6'], st
    assert st['records'] == ['2, 4, 6', '6, 8', '6, 6', '12'], st
    assert st['parent'] == {'6, 8': '2, 4, 6', '6, 6': '2, 4, 6', '12': '6, 6'}, st


def check_success_early(solve):
    # sure 立即返回：后面的 sure 不覆盖 marker
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),
              R('6, 8', '4 * 6 = 24', '24', 'Valid', 'sure', '4 * 6 = 24'),
              R('6, 8', '6 * 6 = 36', '36', 'Valid', 'sure', '6 * 6 = 36')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=3) == \
        (True, 1, ['2 * 4 = 8 (left 6, 8)', '4 * 6 = 24 (left 24)', '4 * 6 = 24'])


def check_round_boundaries(solve):
    # rounds 超出 trycnt：第三轮被忽略（state 无 '48'）；失败 marker = trycnt
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely'),
              R('2, 4, 6', '2 + 4 = 6', '6, 6', 'Valid', 'impossible'),
              R('6, 8', '6 * 8 = 48', '48', 'Valid', 'likely')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=2) == (False, 2, [])
    st = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=2)
    assert st['records'] == ['2, 4, 6', '6, 8'], st
    # rounds 不足 trycnt：提前按失败收场，marker 仍是 trycnt
    short = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'impossible')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=short, trycnt=9) == (False, 9, [])
    # chosen 不在 records：官方 KeyError 被 except 吞掉 -> 该轮静默跳过
    off = [R('9, 9', '2 * 4 = 8', '6, 8', 'Valid', 'likely')]
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=off, trycnt=1) == (False, 1, [])
    # trycnt=0：一轮都不跑
    assert solve(mode="solve", puzzle='2, 4, 6',
                 rounds=[R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely')],
                 trycnt=0) == (False, 0, [])


# ---------- 纯度 ----------

def check_purity(solve):
    rounds = [R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely')]
    snapshot = [dict(r) for r in rounds]
    r1 = solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=1)
    st1 = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=1)
    r2 = solve(mode="solve", puzzle='2, 4, 6', rounds=rounds, trycnt=1)
    st2 = solve(mode="state", puzzle='2, 4, 6', rounds=rounds, trycnt=1)
    assert r1 == r2 and st1 == st2
    assert rounds == snapshot  # 入参轮脚本未被改动


# ---------- 参数校验 ----------

def check_args_validation(solve):
    ok_round = R('2, 4, 6', '2 * 4 = 8', '6, 8', 'Valid', 'likely')
    for bad in (
        lambda: solve(mode="nope"),
        lambda: solve(mode="solve", puzzle=5, rounds=[], trycnt=1),
        lambda: solve(mode="solve", puzzle=None, rounds=[], trycnt=1),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds='nope', trycnt=1),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds=None, trycnt=1),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds=[], trycnt=-1),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds=[], trycnt='5'),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds=[], trycnt=True),
        lambda: solve(mode="solve", puzzle='2, 4, 6', rounds=[['x']], trycnt=1),
        lambda: solve(mode="solve", puzzle='2, 4, 6',
                      rounds=[{'chosen': 'x'}], trycnt=1),
        lambda: solve(mode="dist", index=-1, n_states=3, trycnt=5),
        lambda: solve(mode="dist", index=0, n_states=0, trycnt=5),
        lambda: solve(mode="dist", index=0, n_states=3, trycnt=-1),
        lambda: solve(mode="dist", index=0.5, n_states=3, trycnt=5),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    # 轮内坏值不抛：静默跳过（官方 except continue 语义）
    bad_round = dict(ok_round, next=None)
    assert solve(mode="solve", puzzle='2, 4, 6', rounds=[bad_round], trycnt=1) == \
        (False, 1, [])


check_dist.example = True
check_success_basic.example = True
check_deep_chain.example = True
check_gates_no_success.example = True
check_dup_drop.example = True
check_multiline_edges.example = True
check_forbidden_accumulation.example = True
check_success_early.example = True
check_round_boundaries.example = True
check_purity.example = True
check_args_validation.example = True
