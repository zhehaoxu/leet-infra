import random

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：类版 EnvironmentHistory + 分段 join

_HEAD = ('You will be given the history of a past experience in which '
         'you were placed in an environment and given a task to '
         'complete. You were unsuccessful in completing the task. Do '
         'not summarize your environment, but rather think about the '
         'strategy and path you took to attempt to complete the task. '
         'Devise a concise, new plan of action that accounts for your '
         'mistake with reference to specific actions that you should '
         'have taken. For example, if you tried A and B but forgot C, '
         'then devise a plan to achieve C with environment-specific '
         'actions. You will need this later when you are solving the '
         'same task. Give your plan after "Plan". Here are two examples')


class _EnvHistory:
    """独立写法的 EnvironmentHistory：用列表攒段，最后 join。"""

    def __init__(self, base_query, start_info, memory):
        segs = [base_query]
        if len(memory) > 0:
            segs.append("\n\nYour memory for the task below:")
            segs.extend("\nTrial %d:\n%s" % (i, mm.strip())
                        for i, mm in enumerate(memory))
        segs.append("\nHere is the task:\n" + start_info)
        self.cur_query = "".join(segs)
        self.items = []
        self.last_action = ""
        self.exhausted = False

    def add(self, label, value):
        if label not in ("action", "observation", "human_edit"):
            raise ValueError("bad label")
        self.items.append((label, value))
        if label == "action":
            if value == self.last_action:
                self.exhausted = True
            else:
                self.last_action = value

    def reset(self):
        self.items = []

    def render(self):
        out = [self.cur_query, "\n"]
        for k, (label, value) in enumerate(self.items):
            if k:
                out.append("\n")
            if label == "action":
                out.append("> " + value)
            elif label == "observation":
                out.append(value)
            else:
                out.append("[human edit]: " + value)
        return "".join(out)


def _oracle(base_query, start_info, memory, ops, few_shot,
            env_configs, trial_log, reflections):
    mem_h = memory[-3:] if len(memory) > 3 else list(memory)
    h = _EnvHistory(base_query, start_info, mem_h)
    for label, value in ops:
        if label == "reset":
            h.reset()
        else:
            h.add(label, value)

    env_logs = trial_log.split("#####\n\n#####")
    if len(env_logs) != len(env_configs):
        raise ValueError("mismatch")
    queries = []
    qi = 0
    envs = []
    for i, env in enumerate(env_configs):
        for k in ("name", "memory", "is_success", "skip"):
            if k not in env:
                raise ValueError("missing " + k)
        do = (not env["is_success"]) and (not env["skip"])
        if do:
            mem = list(env["memory"])
            qmem = mem[-3:] if len(mem) > 3 else mem
            scenario = env_logs[i].split("Here is the task:")[-1].strip()
            segs = [_HEAD, ":\n\n", few_shot, "\n\n", scenario]
            if len(qmem) > 0:
                segs.append("\n\nPlans from past attempts:\n")
                segs.extend("Trial #%d: %s\n" % (j, m)
                            for j, m in enumerate(qmem))
            segs.append("\n\nNew plan:")
            queries.append("".join(segs))
            if qi >= len(reflections):
                raise ValueError("queue")
            mem = mem + [reflections[qi]]
            qi += 1
        else:
            mem = list(env["memory"])
        envs.append({"name": env["name"], "memory": mem,
                     "is_success": env["is_success"], "skip": env["skip"]})
    return {"cur_query": h.cur_query, "history_str": h.render(),
            "is_exhausted": h.exhausted, "queries": queries,
            "envs": envs}


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

BQ = "Interact with a household to solve a task."
TASK = "put sponge on counter"
FS = "<few-shot demo>"


def check_base_query_formats(solve):
    """base query 装配：空 memory 无段；len>3 截 [-3:] 且 Trial 编号
    按截断后列表从头数；条目 m.strip()；段头双换行、任务段
    '\\nHere is the task:\\n'；len==3 恰不截断。"""
    ok_log = "\n#####\n\nEnvironment #0: Success\n\n#####\n"
    ok_cfg = [{"name": "e0", "memory": [], "is_success": True,
               "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, ok_cfg, ok_log, [])
    assert led["cur_query"] == BQ + "\nHere is the task:\n" + TASK
    assert led["is_exhausted"] is False
    mem = ["plan one", "  plan two  ", "plan\nthree", "plan four"]
    led = _compare(solve, BQ, TASK, mem, [], FS, ok_cfg, ok_log, [])
    exp = (BQ + "\n\nYour memory for the task below:"
           + "\nTrial 0:\nplan two"
           + "\nTrial 1:\nplan\nthree"
           + "\nTrial 2:\nplan four"
           + "\nHere is the task:\n" + TASK)
    assert led["cur_query"] == exp, led["cur_query"]
    mem3 = ["a", "b", "c"]
    led = _compare(solve, BQ, TASK, mem3, [], FS, ok_cfg, ok_log, [])
    assert led["cur_query"] == (BQ
                                + "\n\nYour memory for the task below:"
                                + "\nTrial 0:\na\nTrial 1:\nb\nTrial 2:\nc"
                                + "\nHere is the task:\n" + TASK)


def check_history_render_and_exhaust(solve):
    """渲染：action '> ' 前缀、observation 原样、human_edit 前缀、
    '\\n' 连接且末条后无换行；熔断是『与 _last_action 相同』——隔
    observation 的重复也触发、只置 True；reset 只清 history。"""
    ok_log = "\n#####\n\nEnvironment #0: Success\n\n#####\n"
    ok_cfg = [{"name": "e0", "memory": [], "is_success": True,
               "skip": False}]
    ops = [("action", "go to shelf 1"),
           ("observation", "You arrive at loc 1."),
           ("action", "take sponge"),
           ("human_edit", "be careful")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    q = BQ + "\nHere is the task:\n" + TASK
    exp = (q + "\n> go to shelf 1" + "\nYou arrive at loc 1."
           + "\n> take sponge" + "\n[human edit]: be careful")
    assert led["history_str"] == exp, led["history_str"]
    assert led["is_exhausted"] is False
    ops = [("action", "A"), ("observation", "o"), ("action", "A")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["is_exhausted"] is True
    ops = [("action", "A"), ("action", "B"), ("action", "B")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["is_exhausted"] is True
    ops = [("action", "A"), ("action", "B"), ("action", "C")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["is_exhausted"] is False
    # reset 清 history 但不清 last_action → reset 后重复 A 仍熔断
    ops = [("action", "A"), ("observation", "o1"), ("reset", ""),
           ("action", "A")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["history_str"] == q + "\n> A"
    assert led["is_exhausted"] is True
    ops = [("action", "A"), ("reset", ""), ("action", "B")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["is_exhausted"] is False
    # A,A,B,A：第二个 A 即触发（A==last A），熔断粘滞
    ops = [("action", "A"), ("action", "A"), ("action", "B"),
           ("action", "A")]
    led = _compare(solve, BQ, TASK, [], ops, FS, ok_cfg, ok_log,
                   [])
    assert led["is_exhausted"] is True
    try:
        solve(BQ, TASK, [], [("think", "x")], FS, ok_cfg,
              ok_log, [])
        assert False
    except ValueError:
        pass


def check_reflection_query(solve):
    """反思 query：HEAD + ':\\n\\n' + few_shot + '\\n\\n' + scenario
    （多次 marker 取最后一段；单条目 log 的 part0 带头 '#####' 残段
    且 scenario 吃到 STATUS/##### 尾巴、strip 收口）；空记忆无 plans
    段；'Trial #{i}: {m}\\n' 不 strip；末尾 '\\n\\nNew plan:'。"""
    log0 = ("\n#####\n\nEnvironment #0:\nHere is the task: decoy\n"
            "Here is the task:\nreal task\n\nSTATUS: FAIL\n\n#####\n")
    cfg = [{"name": "env_0", "memory": [], "is_success": False,
            "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg, log0, ["r0"])
    exp = (_HEAD + ":\n\n" + FS
           + "\n\nreal task\n\nSTATUS: FAIL\n\n#####"
           + "\n\nNew plan:")
    assert led["queries"] == [exp], led["queries"]
    assert led["envs"][0]["memory"] == ["r0"]
    cfg = [{"name": "env_0", "memory": ["p1", " p2 "], "is_success":
            False, "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg, log0, ["r0"])
    exp = (_HEAD + ":\n\n" + FS
           + "\n\nreal task\n\nSTATUS: FAIL\n\n#####"
           + "\n\nPlans from past attempts:\nTrial #0: p1\n"
           + "Trial #1:  p2 \n" + "\n\nNew plan:")
    assert led["queries"] == [exp], led["queries"]
    cfg = [{"name": "env_0", "memory": ["m1", "m2", "m3", "m4"],
            "is_success": False, "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg, log0, ["r0"])
    assert "Trial #0: m2\nTrial #1: m3\nTrial #2: m4\n" in led["queries"][0]
    assert "m1" not in led["queries"][0]
    assert led["envs"][0]["memory"] == ["m1", "m2", "m3", "m4", "r0"]


def check_update_ledger(solve):
    """update_memory 总账：success/skip 门、按 env 序消费队列、
    追加完整 memory、条数不匹配/队列不足/缺键 ValueError；
    末段 env 的 scenario 带尾 '#####' + 汇总行。"""
    log = ("\n#####\n\nEnvironment #0:\nHere is the task:"
           "\nt0\n\nSTATUS: FAIL\n\n#####\n"
           "\n#####\n\nEnvironment #1: Success\n\n#####\n"
           "\n#####\n\nEnvironment #2:\nHere is the task:"
           "\nt2\n\nSTATUS: FAIL\n\n#####\n"
           "-----\nSUCCESS: 1\n-----\n")
    cfg = [{"name": "env_0", "memory": ["old"], "is_success": False,
            "skip": False},
           {"name": "env_1", "memory": [], "is_success": True,
            "skip": False},
           {"name": "env_2", "memory": [], "is_success": False,
            "skip": True}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg, log, ["r0"])
    assert led["queries"][0] == (
        _HEAD + ":\n\n" + FS + "\n\nt0\n\nSTATUS: FAIL"
        + "\n\nPlans from past attempts:\nTrial #0: old\n"
        + "\n\nNew plan:")
    assert led["envs"][0]["memory"] == ["old", "r0"]
    assert led["envs"][1]["memory"] == []        # success 不反思
    assert led["envs"][2]["memory"] == []        # skip 不反思
    assert len(led["queries"]) == 1
    # skip 门：skip=True 拦住失败 env
    cfg = [{"name": "e0", "memory": [], "is_success": True, "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg,
                   "\n#####\n\nEnvironment #0: Success\n\n#####\n", [])
    assert led["queries"] == []
    fail_cfg = [{"name": "e0", "memory": [], "is_success": False,
                 "skip": False}]
    try:
        solve(BQ, TASK, [], [], FS, fail_cfg,
              "\n#####\n\nE\n\n#####\n", [])
        assert False
    except ValueError:
        pass
    try:
        solve(BQ, TASK, [], [], FS, cfg + [dict(cfg[0])], "", [])
        assert False
    except ValueError:
        pass
    try:
        solve(BQ, TASK, [], [], FS, [{"name": "e0", "memory": [],
                                      "is_success": False}],
              "\n#####\n\nE\n\n#####\n", ["r"])
        assert False
    except ValueError:
        pass


def check_log_split_layout(solve):
    """切分结构 quirk：part0 带 '#####\\n\\n' 头残段（无 marker 时
    scenario=整段 strip 含 #####）；中段 '\\n\\n' 包裹；末段带尾
    '#####\\n'。空 log 单 env → scenario=''。"""
    log = ("\n#####\n\nEnvironment #0:\nH\n\nSTATUS: FAIL"
           "\n\n#####\n"
           "\n#####\n\nEnvironment #1:\nHere is the task:"
           "\nx\n\nSTATUS: FAIL\n\n#####\n")
    cfg = [{"name": "e0", "memory": [], "is_success": False,
            "skip": False},
           {"name": "e1", "memory": [], "is_success": False,
            "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg, log,
                   ["r0", "r1"])
    assert led["queries"][0] == (
        _HEAD + ":\n\n" + FS
        + "\n\n#####\n\nEnvironment #0:\nH\n\nSTATUS: FAIL"
        + "\n\nNew plan:")
    assert led["queries"][1] == (
        _HEAD + ":\n\n" + FS
        + "\n\nx\n\nSTATUS: FAIL\n\n#####"
        + "\n\nNew plan:")
    cfg1 = [{"name": "e0", "memory": [], "is_success": False,
             "skip": False}]
    led = _compare(solve, BQ, TASK, [], [], FS, cfg1, "", ["r0"])
    assert led["queries"][0] == (_HEAD + ":\n\n" + FS
                                 + "\n\n" + "\n\nNew plan:")


# ---------------------------------------------------------------- random

_WORDS = ["go to shelf 1", "take sponge", "A", "B", "put", "look",
          "goto 2", "clean", "heat", "x"]


def _rand_str(rng, n):
    return "".join(rng.choice("abcdefgh ") for _ in range(n))


def _rand_ops(rng):
    ops = []
    last = ""
    for _ in range(rng.randint(0, 12)):
        r = rng.random()
        if r < 0.45:
            v = last if (rng.random() < 0.3 and last) else \
                rng.choice(_WORDS)
            ops.append(("action", v))
            last = v
        elif r < 0.85:
            ops.append(("observation", _rand_str(rng, rng.randint(0, 8))))
        elif r < 0.95:
            ops.append(("human_edit", _rand_str(rng, rng.randint(0, 6))))
        else:
            ops.append(("reset", ""))
            last = ""          # 生成器侧约定，与语义判定无关
    return ops


def _rand_log(rng, i):
    body = "Here is the task:\n" + _rand_str(rng, rng.randint(0, 10))
    if rng.random() < 0.5:
        body = "Here is the task: decoy%d\n" % i + body
    return ("\n#####\n\nEnvironment #%d:\n%s\n\nSTATUS: FAIL\n\n#####\n"
            % (i, body))


def check_random_cross(solve):
    """随机交叉验证 oracle（固定种子可复现）。"""
    rng = random.Random(20261102)
    for _ in range(120):
        memory = [_rand_str(rng, rng.randint(0, 9))
                  for _ in range(rng.randint(0, 5))]
        ops = _rand_ops(rng)
        n_env = rng.randint(1, 4)
        cfg = []
        for i in range(n_env):
            cfg.append({"name": "env_%d" % i,
                        "memory": [_rand_str(rng, rng.randint(0, 7))
                                   for _ in range(rng.randint(0, 5))],
                        "is_success": rng.random() < 0.3,
                        "skip": rng.random() < 0.2})
        log = "".join(_rand_log(rng, i) for i in range(n_env))
        n_ref = sum(1 for e in cfg
                    if not e["is_success"] and not e["skip"])
        refs = ["refl%d" % k for k in range(n_ref)]
        led = _compare(solve, BQ, _rand_str(rng, 5), memory, ops,
                       _rand_str(rng, 4), cfg, log, refs)
        assert len(led["queries"]) == n_ref
        assert len(led["envs"]) == n_env


def check_example(solve):
    """run 模式样例：滑窗截断 + 渲染 + 一次反思 query。"""
    mem = ["a", "b", "c", "d"]
    ops = [("action", "A"), ("observation", "o"), ("action", "A")]
    ok_log = "\n#####\n\nEnvironment #0:\nHere is the task:\nT0" \
        "\n\nSTATUS: FAIL\n\n#####\n"
    ok_cfg = [{"name": "e0", "memory": ["p"], "is_success": False,
               "skip": False}]
    led = _compare(solve, BQ, TASK, mem, ops, FS, ok_cfg, ok_log, ["r"])
    assert led["cur_query"] == (BQ
                                + "\n\nYour memory for the task below:"
                                + "\nTrial 0:\nb\nTrial 1:\nc"
                                + "\nTrial 2:\nd"
                                + "\nHere is the task:\n" + TASK)
    assert led["history_str"].endswith("> A")
    assert led["is_exhausted"] is True
    assert led["queries"][0] == (
        _HEAD + ":\n\n" + FS
        + "\n\nT0\n\nSTATUS: FAIL\n\n#####"
        + "\n\nPlans from past attempts:\nTrial #0: p\n"
        + "\n\nNew plan:")


check_example.example = True
