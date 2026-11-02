import json
import random

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：显式栈递归 + 分段拼串

def _oracle(messages, all_time_message_count, warning_tokens, tok,
            summarize, time):
    if not messages:
        raise ValueError("empty")
    if all_time_message_count is None:
        all_time_message_count = len(messages) - 1

    total = 0
    cutoff = len(messages) - 1
    for m in messages[::-1]:
        total += tok(str(m))
        if total >= warning_tokens * 0.2:
            break
        cutoff -= 1
    pre_min = cutoff
    cutoff = min(len(messages) - 3, cutoff)
    pre_shift = cutoff
    try:
        if messages[cutoff]["role"] == "user":
            shifted = cutoff + 1
            if messages[shifted]["role"] == "user":
                pass                    # 官方 ignoring 注释不生效
            cutoff = shifted
    except IndexError:
        pass

    seq = messages[1:cutoff] if cutoff >= 0 else messages[1:len(messages) - 1]

    # 显式栈递归：栈帧 (seq, 阶段)；inputs 按调用序收集
    inputs = []

    def run(seq):
        s_in = str(seq)
        if tok(s_in) > warning_tokens:
            ratio = (warning_tokens / tok(s_in)) * 0.8
            cut2 = int(len(seq) * ratio)
            sub = run(seq[:cut2])
            s_in = str([sub] + seq[cut2:])
        inputs.append(s_in)
        return summarize(s_in)

    summary = run(seq)

    remaining = len(messages[cutoff:]) if cutoff >= 0 else \
        len(messages[cutoff:])   # 负索引同 Python 语义
    hidden = all_time_message_count - remaining
    smc = len(seq)
    ctx = "".join([
        "Note: prior messages (", str(hidden), " of ",
        str(all_time_message_count),
        " total messages) have been hidden from view due to "
        "conversation memory constraints.\n",
        "The following is a summary of the previous ", str(smc),
        " messages:\n ", summary])
    packed = json.dumps({"type": "system_alert", "message": ctx,
                         "time": time})
    sum_user = {"role": "user", "content": packed}
    new_msgs = [messages[0]] + [sum_user] + messages[cutoff:]
    if cutoff >= 0:
        summarized = list(range(1, cutoff))
    else:
        summarized = list(range(1, len(messages) - 1))
    return {"cutoff_pre_min": pre_min, "cutoff_pre_shift": pre_shift,
            "cutoff": cutoff, "summarized": summarized,
            "summary_inputs": inputs, "summary": summary,
            "summary_message": packed, "summary_message_count": smc,
            "remaining_message_count": remaining,
            "hidden_message_count": hidden,
            "new_messages": new_msgs,
            "messages_total_new": all_time_message_count + 1}


# ---------------------------------------------------------------- compare

def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            _eq(x, y)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), \
            (a.keys() ^ b.keys())
        for k in a:
            _eq(a[k], b[k])
        return
    assert a == b, (a, b)


def _compare(solve, *args):
    got = solve(*[_dc(x) for x in args])
    exp = _oracle(*[_dc(x) for x in args])
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- checks

def _mk(role, content):
    return {"role": role, "content": content}


def _sum(s):
    return "S<" + str(len(s)) + ">"


def _tok(s):
    return len(s)


def check_negative_cutoff(solve):
    """累计跑满 → cutoff 减到 -1；min 后仍 -1；负索引下 messages[-1]
    命中末条 user → 移位到 0（messages[0] 非 user）；seq=messages[1:0]
    为空；trim+prepend 复制 system；'[]' 输入。"""
    msgs = [_mk("system", "S0"), _mk("user", "u1"),
            _mk("assistant", "a1"), _mk("user", "u2")]
    # warning=1000 → 阈值 200 远大于全部 str 长度 → 无 break
    led = _compare(solve, msgs, 10, 1000, _tok, _sum, "T0")
    assert led["cutoff_pre_min"] == -1     # 全部累计 < 200，跑满减到 -1
    assert led["cutoff_pre_shift"] == -1
    assert led["cutoff"] == 0          # -1 负索引命中 user → 移位 0
    assert led["summarized"] == []
    assert led["summary_inputs"] == ["[]"]
    assert led["summary"] == "S<2>"
    assert led["summary_message_count"] == 0
    assert led["remaining_message_count"] == 4
    assert led["hidden_message_count"] == 6
    assert led["messages_total_new"] == 11
    assert led["new_messages"][0] is not None
    assert led["new_messages"][0] == _mk("system", "S0")
    assert led["new_messages"][1]["role"] == "user"
    assert "prior messages (6 of 10 total messages)" in \
        led["new_messages"][1]["content"]
    assert led["new_messages"][2:] == msgs   # trim(0) 保留全表 → m0 重复
    assert led["new_messages"].count(_mk("system", "S0")) == 2
    sm = json.loads(led["summary_message"])
    assert sm["type"] == "system_alert" and sm["time"] == "T0"
    assert sm["message"].endswith("previous 0 messages:\n S<2>")


def check_shift_always_bug(solve):
    """官方 bug：cutoff 处是 user 时移位无条件发生——内层 'still
    user' 只打印 ignoring，cutoff 仍 +1。"""
    msgs = [_mk("system", "S0"),           # 35
            _mk("user", "x" * 8),          # 39
            _mk("user", "y" * 8),          # 39
            _mk("assistant", "z" * 3)]     # 39
    # 阈值 0.2*200=40：末条 39<40、次末 78>=40 → break → pre_min=2；
    # min(1,2)=1；messages[1] user → 移位到 2（即便 messages[2] 也是
    # user）→ cutoff=2，seq=[m1]
    led = _compare(solve, msgs, 6, 200, _tok, _sum, "T")
    assert led["cutoff_pre_min"] == 2
    assert led["cutoff_pre_shift"] == 1
    assert led["cutoff"] == 2              # 若尊重 ignoring 则为 1
    assert led["summarized"] == [1]
    assert led["summary_message_count"] == 1
    assert led["remaining_message_count"] == 2
    assert led["hidden_message_count"] == 4
    assert led["new_messages"] == [msgs[0],
                                   led["new_messages"][1],
                                   msgs[2], msgs[3]]


def check_no_break_small(solve):
    """len<3 的退化：min(len-3, ·) 为负；cutoff=-2 时 messages[-2]
    越界 IndexError → 保持 -2（len=1）；len=2 → min(-1, ·)=-1。"""
    m1 = [_mk("system", "tiny")]
    led = _compare(solve, m1, 1, 1000, _tok, _sum, "T")
    assert led["cutoff_pre_min"] == -1     # 0 减 1（跑满一轮）
    assert led["cutoff_pre_shift"] == -2
    assert led["cutoff"] == -2             # IndexError → 不移位
    assert led["summarized"] == []
    # len=2：cap = -1；messages[-1]=末条
    m2 = [_mk("system", "t"), _mk("user", "u")]
    led = _compare(solve, m2, 4, 1000, _tok, _sum, "T")
    assert led["cutoff_pre_min"] == -1     # 两条都累计完，1-2=-1
    assert led["cutoff_pre_shift"] == -1
    assert led["cutoff"] == 0              # -1 命中 user → 移位 0
    assert led["new_messages"][2:] == m2


def check_recursion(solve):
    """递归摘要：tok(str(seq))>threshold → ratio=(th/tot)*0.8、
    cut2=int(n*ratio) 向零截断；递归返回串与剩余 dict 混装 str()；
    calls 按调用序（内层先）。"""
    msgs = [_mk("system", "S0"),           # 35
            _mk("user", "u" * 20),         # 51
            _mk("assistant", "a" * 20),    # 56
            _mk("assistant", "m4"),
            _mk("user", "m5"),
            _mk("assistant", "z" * 3)]     # 39 → 阈值 12 首轮 break
    # warning=200：cut 阈值 40 → pre_min=5；min(3,5)=3；messages[3]
    # 是 assistant 不移位；seq=[m1,m2]，str 长度 1+52+2+57+1=113>50
    # → 但 gate 用 warning=200？不行——gate 与 cut 阈值同源。
    # 用 warning=60：cut 阈值 12 → 末条 40>=12 break → pre_min=5；
    # min(3,5)=3；seq str=113 > 60 → ratio=48/113*0.8=0.33982…；
    # cut2=int(2*0.33982)=0 → 递归 '[]'（S<2>），外层输入混装。
    led = _compare(solve, msgs, 20, 60, _tok, _sum, "T")
    assert led["cutoff"] == 3
    assert led["summarized"] == [1, 2]
    m1r, m2r = str(msgs[1]), str(msgs[2])
    assert len(m1r) == 51 and len(m2r) == 56
    outer = str(["S<2>", msgs[1], msgs[2]])
    assert led["summary_inputs"] == ["[]", outer]
    assert len(outer) == 1 + 6 + 2 + 51 + 2 + 56 + 1
    assert led["summary"] == "S<" + str(len(outer)) + ">"
    assert led["summary_message_count"] == 2
    assert led["remaining_message_count"] == 3
    assert led["hidden_message_count"] == 17


def check_recursion_cut2_positive(solve):
    """cut2>0 的递归：前 cut2 条递归、后段原样；断言 cut2=1 与
    混装串逐段。"""
    msgs = [_mk("system", "S0"),           # 35
            _mk("user", "u" * 12),         # 43
            _mk("assistant", "a" * 12),    # 48
            _mk("user", "u" * 12),         # 43
            _mk("assistant", "m4"),
            _mk("assistant", "m5"),
            _mk("assistant", "z" * 3)]     # 39
    # warning=100：cut 阈值 20 → 末条 39>=20 break → pre_min=6；
    # min(4,6)=4；messages[4] assistant 不移位；seq=[m1,m2,m3]，
    # str=1+43+2+48+2+43+1=140>100 → ratio=(100/140)*0.8≈0.5714；
    # cut2=int(3*0.5714)=1 → 递归 [m1]（str 45 ≤100 → S<45>）；
    # 外层输入 str(['S<45>', m2, m3]) = 1+7+2+48+2+43+1=104
    led = _compare(solve, msgs, 30, 100, _tok, _sum, "T")
    assert led["cutoff"] == 4
    assert led["summarized"] == [1, 2, 3]
    inner = str([msgs[1]])
    assert len(inner) == 1 + 43 + 1
    outer = str(["S<45>", msgs[2], msgs[3]])
    assert led["summary_inputs"] == [inner, outer]
    assert len(outer) == 1 + 7 + 2 + 48 + 2 + 43 + 1
    assert led["summary"] == "S<104>"
    assert led["hidden_message_count"] == 30 - 3


def check_domain(solve):
    try:
        solve([], 0, 100, _tok, _sum, "T")
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------- random

def check_random_cross(solve):
    """随机交叉验证 oracle（固定种子可复现）。"""
    rng = random.Random(20261102)
    roles = ["system", "user", "assistant", "function"]
    for _ in range(150):
        n = rng.randint(1, 8)
        msgs = []
        for i in range(n):
            role = "system" if i == 0 and rng.random() < 0.8 \
                else rng.choice(roles)
            content = "".join(rng.choice("abcdefxy ")
                              for _ in range(rng.randint(0, 24)))
            msgs.append({"role": role, "content": content})
        warning = rng.choice([30, 50, 80, 120, 200, 400])
        all_time = rng.randint(n - 1, n + 6)
        led = _compare(solve, msgs, all_time, warning, _tok, _sum, "TT")
        c = led["cutoff"]
        expect_len = 2 + (len(msgs) - c if c >= 0
                          else min(abs(c), len(msgs)))
        assert len(led["new_messages"]) == expect_len, (c, len(msgs))
        assert led["summary_inputs"][-1] is not None


def check_example(solve):
    """run 模式样例：负 cutoff + 移位 bug + 空 seq 的组合。"""
    msgs = [_mk("system", "S0"), _mk("user", "hi"),
            _mk("assistant", "ok"), _mk("user", "bye")]
    led = _compare(solve, msgs, 8, 1000, _tok, _sum, "T0")
    assert led["cutoff"] == 0
    assert led["summary_inputs"] == ["[]"]
    assert led["new_messages"].count(_mk("system", "S0")) == 2


check_example.example = True


def check_recursion_exact_gate(solve):
    """递归门恰等：tok(str(seq)) == warning_tokens 时官方 `>` 不触发、
    `>=` 变异触发——此处钉 140==140 恰等不递归之外的场景：
    warning=140 > 140? 不，140==140 → `>` 不触发（门是严格大于）。
    为钉住 `>` 的下界语义，本锚点用 140 vs 139 双档。"""
    msgs = [{"role": "system", "content": "S0"},
            {"role": "user", "content": "u" * 12},          # 43
            {"role": "assistant", "content": "a" * 12},     # 48
            {"role": "user", "content": "u" * 12},          # 43
            {"role": "assistant", "content": "m4"},
            {"role": "assistant", "content": "m5"},
            {"role": "assistant", "content": "zzz"}]        # 39
    # seq str = 1+43+2+48+2+43+1 = 140；warning=141 > 140 → 门不触发
    led = _compare(solve, msgs, 30, 141, _tok, _sum, "T")
    assert len(led["summary_inputs"]) == 1
    assert led["summary"] == "S<140>"
    # warning=140 == 140 → 官方 `>` 仍不触发（恰等证据：>= 变异
    # 在此档触发递归 → len(summary_inputs)==1 断言失败被抓）
    led = _compare(solve, msgs, 30, 140, _tok, _sum, "T")
    assert len(led["summary_inputs"]) == 1
    assert led["summary"] == "S<140>"
    # warning=139 < 140 → 触发；ratio=(139/140)*0.8≈0.7943；
    # cut2=int(3*0.7943)=int(2.3829)=2
    led = _compare(solve, msgs, 30, 139, _tok, _sum, "T")
    inner = str([msgs[1], msgs[2]])           # 1+43+2+48+1 = 95
    outer = str(["S<95>", msgs[3]])           # 1+7+2+43+1 = 54
    assert led["summary_inputs"] == [inner, outer]
    assert len(inner) == 95 and len(outer) == 54
    assert led["summary"] == "S<54>"


def check_accum_exact_break(solve):
    """cut 累计恰等断点：warning=195 → 阈值 39.0；末条 user 8 字符
    str 恰 39 → 官方 >= 首轮即 break（pre_min=len-1）；`>` 变异会
    再吃一条 → pre_min=len-2 被抓。"""
    msgs = [{"role": "system", "content": "S0"},    # 35
            {"role": "user", "content": "a"},       # 32
            {"role": "assistant", "content": "b"},  # 37
            {"role": "user", "content": "c"},       # 32
            {"role": "assistant", "content": "d"},  # 37
            {"role": "user", "content": "abcdefgh"}]  # 39 == 阈值
    led = _compare(solve, msgs, 40, 195, _tok, _sum, "T")
    assert len(str(msgs[5])) == 39
    assert led["cutoff_pre_min"] == 5
    assert led["cutoff_pre_shift"] == 3
    assert led["cutoff"] == 4            # messages[3] user → 移位 +1
    assert led["summarized"] == [1, 2, 3]
    assert led["remaining_message_count"] == 2
