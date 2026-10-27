import re

RE_FINAL = re.compile(r"^\s*final answer:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
RE_ACTION = re.compile(r"^\s*action:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
RE_INPUT = re.compile(r"^\s*action input:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def _oracle(transcripts, tools, max_steps=None):
    """独立 oracle：用逐行正则解析 + 显式循环，与参考实现的「首词前缀匹配」互不共享代码。"""
    observations, steps, answer, prev = [], 0, None, None
    for block in transcripts:
        if max_steps is not None and steps == max_steps:
            break
        steps += 1
        final = RE_FINAL.search(block)
        if final:
            answer = final.group(1).strip()
            break
        action = RE_ACTION.search(block)
        if not action:
            break
        name = action.group(1).strip()
        arg = RE_INPUT.search(block)
        if arg is None:
            raise ValueError("action requires an input")
        arg = arg.group(1).strip()
        if (name, arg) == prev:
            break
        prev = (name, arg)
        if name in tools:
            observations.append(str(tools[name](arg)))
        else:
            observations.append("Error: unknown tool '%s'" % name)
    return observations, steps, answer


_WEATHER = {"weather": lambda city: "22C", "clock": lambda _when: "10:00"}

_SCRIPT = [
    "Thought: 先查天气\nAction: weather\nAction Input: Beijing",
    "Thought: 再查时间\nAction: clock\nAction Input: now",
    "Thought: 信息够了\nFinal Answer: Beijing 22C at 10:00",
]


def check_example(solve):
    obs, steps, answer = solve(_SCRIPT, _WEATHER)
    assert obs == ["22C", "10:00"], f"观测序列不对：{obs}"
    assert steps == 3, f"steps 应为 3，got {steps}"
    assert answer == "Beijing 22C at 10:00", f"答案不对：{answer}"


check_example.example = True


def check_termination(solve):
    # 第一轮就给答案：不执行任何工具
    obs, steps, answer = solve(["Thought: 会了\nFinal Answer: 42"], _WEATHER)
    assert (obs, steps, answer) == ([], 1, "42"), (obs, steps, answer)

    # 空答案也算答案（空串，而不是 None）
    assert solve(["Final Answer:"], _WEATHER) == ([], 1, "")

    # 无 Action 无 Final Answer：模型自己收手，steps 仍计入本轮
    obs, steps, answer = solve(["Thought: 我放弃"], _WEATHER)
    assert (obs, steps, answer) == ([], 1, None), (obs, steps, answer)

    # 脚本提前耗尽：答案保持 None，steps 等于脚本轮数
    obs, steps, answer = solve(_SCRIPT[:2], _WEATHER)
    assert (obs, steps, answer) == (["22C", "10:00"], 2, None)


def check_unknown_tool(solve):
    """未知工具不中断循环，只产生一条错误观测，后续轮继续。"""
    calls = []
    tools = {"weather": lambda city: calls.append(city) or "22C"}
    script = [
        "Action: flights\nAction Input: PEK-SHA",
        "Action: weather\nAction Input: Shanghai",
        "Final Answer: ok",
    ]
    obs, steps, answer = solve(script, tools)
    assert obs == ["Error: unknown tool 'flights'", "22C"], obs
    assert calls == ["Shanghai"], f"只应调用已知工具一次：{calls}"
    assert (steps, answer) == (3, "ok")


def check_max_steps(solve):
    obs, steps, answer = solve(_SCRIPT, _WEATHER, max_steps=1)
    assert (obs, steps, answer) == (["22C"], 1, None), (obs, steps, answer)

    # max_steps=0：一轮都不消费
    assert solve(_SCRIPT, _WEATHER, max_steps=0) == ([], 0, None)

    # 与正则 oracle 对拍（含跨轮重复动作与未知工具）
    script = [
        "Thought: t0\nAction: weather\nAction Input:  A ",
        "Action: missing\nAction Input: x",
        "Action: weather\nAction Input: A",
        "Action: weather\nAction Input: A",
        "Final Answer: unreachable",
    ]
    for limit in (None, 1, 2, 3, 4, 9):
        got = solve(script, _WEATHER, max_steps=limit)
        want = _oracle(script, _WEATHER, max_steps=limit)
        assert got == want, f"max_steps={limit} 与 oracle 不一致：{got} vs {want}"


def check_repeat_guard_and_invalid(solve):
    """连续两次相同 (工具名, 参数) 判定死循环：第二次不再执行。"""
    seen = []
    tools = {"echo": lambda s: seen.append(s) or s.upper()}
    script = [
        "Action: echo\nAction Input: hi",
        "Action: echo\nAction Input: hi",
        "Action: echo\nAction Input: bye",
    ]
    obs, steps, answer = solve(script, tools)
    assert obs == ["HI"], f"重复动作不应再次执行：{obs}"
    assert seen == ["hi"], f"工具只应被调用一次：{seen}"
    assert (steps, answer) == (2, None), (steps, answer)

    # 参数不同则不算重复
    obs, steps, _ = solve(script[:1] + script[2:], tools)
    assert obs == ["HI", "BYE"] and steps == 2, (obs, steps)

    for bad in (
        lambda: solve("not-a-list", tools),
        lambda: solve(_SCRIPT, ["weather"]),
        lambda: solve(["Action: weather"], tools),  # 缺 Action Input
        lambda: solve(_SCRIPT, _WEATHER, max_steps=-1),
        lambda: solve([123], _WEATHER),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
