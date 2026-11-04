"""Tournament-GRPO -- binary format + knockout tournament reward tests."""

"""
EXPECTED values were dumped from the official
datawhalechina/agentic-rl-lab experiments/tournament-grpo/reward.py (see
notes.md). The judge is parameterized as (seed, pick).
"""

EXPECTED = {
    'ans_basic': '42',
    'ans_empty': '',
    'ans_none': '',
    'ans_tagged': '',
    'ans_trailing': '',
    'ans_two': '',
    'fmt_bad_tool': {'format_reward': 0.0, 'failure_reasons': ['tool:unknown_tool_name'], 'rollout_answer': '42'},
    'fmt_empty_call': {'format_reward': 0.0, 'failure_reasons': ['structure:each_segment_must_start_with_google_search_cycle'], 'rollout_answer': '42'},
    'fmt_empty_think': {'format_reward': 0.0, 'failure_reasons': ['structure:must_start_with_non_empty_think'], 'rollout_answer': '42'},
    'fmt_gap': {'format_reward': 1.0, 'failure_reasons': [], 'rollout_answer': '42'},
    'fmt_good': {'format_reward': 1.0, 'failure_reasons': [], 'rollout_answer': '42'},
    'fmt_no_search': {'format_reward': 0.0, 'failure_reasons': ['structure:each_segment_must_start_with_google_search_cycle'], 'rollout_answer': '42'},
    'fmt_two_cycles': {'format_reward': 1.0, 'failure_reasons': [], 'rollout_answer': '42'},
    'normalize_basic': [0.0, 0.3333333333333333, 1.0],
    'normalize_same': [0.5, 0.5, 0.5],
    'parse_answer_attr': [[{'tag': 'think', 'attrs': {}, 'content': 'x'}, {'tag': 'answer', 'attrs': {'k': 'v'}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_attr_dup': [[{'tag': 'think', 'attrs': {'a': '1'}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_attr_junk': [[{'tag': 'think', 'attrs': {'a': '1', 'junk': '2'}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_closeattr': [[{'tag': 'think', 'attrs': {}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_fallback': [[{'tag': 'think', 'attrs': {}, 'content': '1<snippet>2<webpage>3'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:broken_xml']],
    'parse_good': [[{'tag': 'think', 'attrs': {}, 'content': 'plan'}, {'tag': 'call_tool', 'attrs': {'name': 'google_search'}, 'content': 'q1'}, {'tag': 'tool_output', 'attrs': {}, 'content': 'o1'}, {'tag': 'think', 'attrs': {}, 'content': 't1'}, {'tag': 'call_tool', 'attrs': {'name': 'browse_webpage'}, 'content': 'u1'}, {'tag': 'tool_output', 'attrs': {}, 'content': 'o2'}, {'tag': 'think', 'attrs': {}, 'content': 't2'}, {'tag': 'answer', 'attrs': {}, 'content': '42'}], []],
    'parse_nested': [[{'tag': 'think', 'attrs': {}, 'content': 'outer<div>x</div>inner'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:unknown_tags']],
    'parse_stray': [[{'tag': 'think', 'attrs': {}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['structure:stray_text_outside_tags']],
    'parse_tail_stray': [[{'tag': 'think', 'attrs': {}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['structure:stray_text_outside_tags']],
    'parse_tool_dupname': [[{'tag': 'call_tool', 'attrs': {'name': 'g'}, 'content': 'x'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_tool_noname': [[{'tag': 'call_tool', 'attrs': {}, 'content': 'bare'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:invalid_attrs']],
    'parse_unclosed': [[{'tag': 'think', 'attrs': {}, 'content': 'plan'}], ['tag:broken_xml']],
    'parse_unknown': [[{'tag': 'think', 'attrs': {}, 'content': 'x'}, {'tag': 'foo', 'attrs': {}, 'content': 'bar'}, {'tag': 'answer', 'attrs': {}, 'content': 'a'}], ['tag:unknown_tags']],
    'score_first_noanswer': {'scores': [0.0, 1.0], 'tour': [0.0, 0.0], 'raw': [3.0, 1.0]},
    'score_group': {'scores': [2.0, 0.375, 0.125, 0.0], 'fmt': [1.0, 0.0, 0.0, 0.0], 'tour': [1.0, 0.375, 0.125, 0.0], 'raw': [8.0, 3.0, 1.0, 0.0], 'answers': ['42', '5', '4', '']},
    'tour3_first': {'raw': [1.0, 3.0, 0.0], 'rewards': [0.3333333333333333, 1.0, 0.0]},
    'tour3_longest': {'raw': [1.0, 3.0, 0.0], 'rewards': [0.3333333333333333, 1.0, 0.0], 'matches': [{'repeat': 1, 'round': 1, 'candidates': [1, 2, 0], 'winner': 1, 'output': 'stub-output'}, {'repeat': 2, 'round': 1, 'candidates': [0, 2, 1], 'winner': 0, 'output': 'stub-output'}, {'repeat': 3, 'round': 1, 'candidates': [1, 0, 2], 'winner': 1, 'output': 'stub-output'}, {'repeat': 4, 'round': 1, 'candidates': [1, 0, 2], 'winner': 1, 'output': 'stub-output'}]},
    'tour5_longest': {'raw': [6.0, 2.0, 2.0, 0.0, 2.0], 'rewards': [1.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.3333333333333333], 'matches': [{'repeat': 1, 'round': 1, 'candidates': [2, 1, 3], 'winner': 2, 'output': 'stub-output'}, {'repeat': 1, 'round': 1, 'candidates': [0, 4], 'winner': 0, 'output': 'stub-output'}, {'repeat': 1, 'round': 2, 'candidates': [0, 2], 'winner': 0, 'output': 'stub-output'}, {'repeat': 2, 'round': 1, 'candidates': [1, 2, 4], 'winner': 1, 'output': 'stub-output'}, {'repeat': 2, 'round': 1, 'candidates': [0, 3], 'winner': 0, 'output': 'stub-output'}, {'repeat': 2, 'round': 2, 'candidates': [0, 1], 'winner': 0, 'output': 'stub-output'}, {'repeat': 3, 'round': 1, 'candidates': [1, 0, 3], 'winner': 1, 'output': 'stub-output'}, {'repeat': 3, 'round': 1, 'candidates': [4, 2], 'winner': 4, 'output': 'stub-output'}, {'repeat': 3, 'round': 2, 'candidates': [4, 1], 'winner': 4, 'output': 'stub-output'}, {'repeat': 4, 'round': 1, 'candidates': [2, 1, 4], 'winner': 2, 'output': 'stub-output'}, {'repeat': 4, 'round': 1, 'candidates': [0, 3], 'winner': 0, 'output': 'stub-output'}, {'repeat': 4, 'round': 2, 'candidates': [0, 2], 'winner': 0, 'output': 'stub-output'}]},
    'tour8_longest': {'raw': [1.0, 3.0, 4.0, 1.0, 4.0, 5.0, 4.0, 6.0], 'rewards': [0.0, 0.4, 0.6, 0.0, 0.6, 0.8, 0.6, 1.0], 'num_matches': 28, 'first_match': {'repeat': 1, 'round': 1, 'candidates': [6, 0], 'winner': 6, 'output': 'stub-output'}},
}

# ---------- shared fixtures ----------

GOOD = ("<think>plan</think>"
        "<call_tool name=\"google_search\">q1</call_tool>"
        "<tool_output>o1</tool_output>"
        "<think>t1</think>"
        "<call_tool name=\"browse_webpage\">u1</call_tool>"
        "<tool_output>o2</tool_output>"
        "<think>t2</think>"
        "<answer>42</answer>")

# ---------- mode="parse" ----------


def check_parse(solve):
    blocks, errors = solve(mode="parse", text=GOOD)
    # 栈式解析：顶层块按开合配对切出；合法轨迹零错误
    assert [b["tag"] for b in blocks] == \
        ["think", "call_tool", "tool_output", "think", "call_tool",
         "tool_output", "think", "answer"], blocks
    assert errors == EXPECTED["parse_good"][1], errors
    # 未知标签记录错误但块保留
    _, errors = solve(mode="parse",
                      text="<think>x</think><foo>bar</foo><answer>a</answer>")
    assert errors == EXPECTED["parse_unknown"][1], errors
    # 关闭标签与栈顶不匹配：回退到最近同名开标签；<div> 是未知标签
    blocks, errors = solve(
        mode="parse",
        text="<think>outer<div>x</div>inner</think><answer>a</answer>")
    assert errors == EXPECTED["parse_nested"][1], errors
    assert blocks[0]["content"] == "outer<div>x</div>inner", blocks
    # 未闭合标签：栈非空结束 -> broken_xml，块只剩已闭合的
    blocks, errors = solve(mode="parse",
                           text="<think>plan</think>"
                                "<call_tool name=\"google_search\">q")
    assert errors == EXPECTED["parse_unclosed"][1], errors
    assert [b["tag"] for b in blocks] == ["think"], blocks
    # 标签外杂文（开头与结尾各测一次）
    _, errors = solve(mode="parse",
                      text="loose<think>x</think><answer>a</answer>")
    assert errors == EXPECTED["parse_stray"][1], errors
    # 属性账：杂文属性、重复 key、call_tool 无 name、answer 带属性全 invalid
    cases = ["<think a=\"1\" junk=\"2\">x</think><answer>a</answer>",
             "<think a=\"1\" a=\"2\">x</think><answer>a</answer>",
             "<call_tool>bare</call_tool><answer>a</answer>",
             "<think>x</think><answer k=\"v\">a</answer>"]
    for text, key in zip(cases, ["parse_attr_junk", "parse_attr_dup",
                                 "parse_tool_noname",
                                 "parse_answer_attr"]):
        blocks, errors = solve(mode="parse", text=text)
        assert errors == EXPECTED[key][1], (key, errors)
    # 闭标签带属性 -> invalid_attrs
    blocks, errors = solve(mode="parse",
                           text="<think>x</think a=\"v\"><answer>a</answer>")
    assert errors == EXPECTED["parse_closeattr"][1], errors
    # 回退语义：think 闭时栈 [think, snippet, webpage] -> 回退到 think、
    # 中间未闭合标签静默丢弃、think 块仍发射（栈清空）
    blocks, errors = solve(
        mode="parse",
        text="<think>1<snippet>2<webpage>3</think><answer>a</answer>")
    assert errors == EXPECTED["parse_fallback"][1], errors
    assert blocks[0]["content"] == "1<snippet>2<webpage>3", blocks
    assert len(blocks) == 2, blocks
    # 尾部杂文 -> stray
    _, errors = solve(mode="parse",
                      text="<think>x</think><answer>a</answer>tail")
    assert errors == EXPECTED["parse_tail_stray"][1], errors
    # call_tool 的重复 name（allowed 检查兜不住的场景）
    blocks, errors = solve(
        mode="parse",
        text="<call_tool name=\"g\" name=\"h\">x</call_tool>"
             "<answer>a</answer>")
    assert errors == EXPECTED["parse_tool_dupname"][1], errors
    assert blocks[0]["attrs"] == {"name": "g"}, blocks


# ---------- mode="answer" ----------


def check_answer(solve):
    # 唯一、非空、无标签、位于末尾
    assert solve(mode="answer",
                 text="<think>x</think><answer> 42 </answer>") == \
        EXPECTED["ans_basic"]
    for text, key in [("<answer>a</answer><answer>b</answer>", "ans_two"),
                      ("<answer>a</answer>tail", "ans_trailing"),
                      ("<answer><b>x</b></answer>", "ans_tagged"),
                      ("<answer></answer>", "ans_empty"),
                      ("<think>x</think>", "ans_none")]:
        assert solve(mode="answer", text=text) == EXPECTED[key], key


# ---------- mode="format" ----------


def check_format(solve):
    assert solve(mode="format", text=GOOD) == EXPECTED["fmt_good"]
    # 每段必须以 google_search 循环开头：browse 开头判失败
    assert solve(mode="format", text=(
        "<think>plan</think>"
        "<call_tool name=\"browse_webpage\">u</call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_no_search"]
    # 两次 search + 一次 browse 的合法长轨迹
    assert solve(mode="format", text=(
        "<think>plan</think>"
        "<call_tool name=\"google_search\">q1</call_tool>"
        "<tool_output>o1</tool_output><think>t</think>"
        "<call_tool name=\"google_search\">q2</call_tool>"
        "<tool_output>o2</tool_output><think>t</think>"
        "<call_tool name=\"browse_webpage\">u</call_tool>"
        "<tool_output>o3</tool_output><think>t</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_two_cycles"]
    # 未知工具名
    assert solve(mode="format", text=(
        "<think>plan</think><call_tool name=\"bing\">q</call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_bad_tool"]
    # 空 think 开头
    assert solve(mode="format", text=(
        "<think></think><call_tool name=\"google_search\">q</call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_empty_think"]
    # call_tool 内容为空 -> 该轮 cycle 不成立 -> 结构失败
    assert solve(mode="format", text=(
        "<think>plan</think>"
        "<call_tool name=\"google_search\"></call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_empty_call"]
    # search 段 + browse 段交替合法（search 段后的 browse 段不需要再 search）
    assert solve(mode="format", text=(
        "<think>plan</think>"
        "<call_tool name=\"google_search\">q</call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<call_tool name=\"browse_webpage\">u</call_tool>"
        "<tool_output>o</tool_output><think>t</think>"
        "<call_tool name=\"browse_webpage\">u2</call_tool>"
        "<tool_output>o2</tool_output><think>t2</think>"
        "<answer>42</answer>")) == EXPECTED["fmt_gap"]


# ---------- mode="normalize" / "tournament" ----------


def check_normalize(solve):
    # 组内 min-max；全同分全 0.5
    got = solve(mode="normalize", scores=[1.0, 2.0, 4.0])
    assert [round(x, 9) for x in got] == \
        [round(x, 9) for x in EXPECTED["normalize_basic"]], got
    assert solve(mode="normalize", scores=[3.0, 3.0, 3.0]) == \
        EXPECTED["normalize_same"]


def check_tournament(solve):
    # K=8：每次 4+2+1=7 场、4 次共 28 场；random.Random(seed) 决定对阵
    res = solve(mode="tournament", query="q", answers=["a%d" % i
                                                       for i in range(8)],
                seed=7, pick="longest")
    assert res["raw_scores"] == EXPECTED["tour8_longest"]["raw"], res
    assert [round(x, 9) for x in res["rewards"]] == \
        [round(x, 9) for x in EXPECTED["tour8_longest"]["rewards"]]
    assert len(res["matches"]) == EXPECTED["tour8_longest"]["num_matches"]
    # 3 人组：轮转分组 candidates[i::num_groups] 在奇数候选时出 3 人组
    res = solve(mode="tournament", query="q", answers=["x", "y", "z"],
                seed=3, pick="longest")
    assert res["raw_scores"] == EXPECTED["tour3_longest"]["raw"], res
    assert res["matches"][0]["candidates"] == \
        EXPECTED["tour3_longest"]["matches"][0]["candidates"]
    assert res["matches"][0]["winner"] == \
        EXPECTED["tour3_longest"]["matches"][0]["winner"]
    # 5 人：4+? 结构，3 人组出现
    res = solve(mode="tournament", query="q",
                answers=["p", "q", "r", "s", "t"], seed=11, pick="longest")
    assert res["raw_scores"] == EXPECTED["tour5_longest"]["raw"], res
    three = [m for m in res["matches"] if len(m["candidates"]) == 3]
    assert three, "expected a 3-player group"
    # pick="first"：3 人组的胜者 = shuffle 后的组首，随抽签变化
    res = solve(mode="tournament", query="q", answers=["x", "y", "z"],
                seed=3, pick="first")
    assert res["raw_scores"] == EXPECTED["tour3_first"]["raw"], res


# ---------- mode="score" ----------


def check_score(solve):
    res = solve(mode="score", query="  what is 2+2  ",
                solutions=[GOOD,
                           "<think>plan</think><answer>5</answer>",
                           "<answer>4</answer>", "garbage"],
                seed=5, pick="longest")
    results = res["results"]
    # 总分 = 锦标赛 min-max 奖励 + 0/1 格式分；无答案候选参赛但锦标赛分置 0
    assert [r["score"] for r in results] == \
        EXPECTED["score_group"]["scores"], results
    assert [r["format_reward"] for r in results] == \
        EXPECTED["score_group"]["fmt"]
    assert [r["tournament_reward"] for r in results] == \
        EXPECTED["score_group"]["tour"]
    assert [r["tournament_raw_score"] for r in results] == \
        EXPECTED["score_group"]["raw"]
    # judge 只看提取出的答案；query 被 strip
    assert res["log"]["answers"] == EXPECTED["score_group"]["answers"]
    # 无答案候选参赛但归一化后锦标赛分置 0（它赢的场次的分被抹掉）
    res2 = solve(mode="score", query="q", solutions=["garbage", GOOD],
                 seed=9, pick="first")
    got2 = res2["results"]
    assert [r["score"] for r in got2] == \
        EXPECTED["score_first_noanswer"]["scores"], got2
    assert [r["tournament_reward"] for r in got2] == \
        EXPECTED["score_first_noanswer"]["tour"]
    # raw 分里 garbage 确实赢过（3 > 1），归一后 [1.0, 0.0] 再置零
    assert [r["tournament_raw_score"] for r in got2] == \
        EXPECTED["score_first_noanswer"]["raw"]
