"""CoCoGen（arXiv:2210.07128）代码↔图双向解析账 -- 测试用例（期望值由官方
relation.py + utils.py + algo_utils.py exec 沙箱实测钉死）."""

GRAPH1 = ("(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)"
          "(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)")
ROW1 = {"belief": "Marijuana should not be legalized.",
        "argument": "Marijuana is dangerous for society.",
        "stance": "support", "graph": GRAPH1}

# ---------- mode="parse"：get_nodes ----------

def check_parse_basic(solve):
    res = solve(mode="parse", graph_str=GRAPH1)
    assert res["nodes"] == ["marijuana", "recreational drug", "drug addiction",
                            "dangerous for society", "legalized"], res
    assert res["from_to"] == {
        "marijuana": ["recreational drug"],
        "recreational drug": ["drug addiction"],
        "drug addiction": ["dangerous for society"],
        "dangerous for society": ["legalized"]}, res
    assert res["from_to_typed"] == {
        "marijuana": [("is a", "recreational drug")],
        "recreational drug": [("capable of", "drug addiction")],
        "drug addiction": [("is a", "dangerous for society")],
        "dangerous for society": [("not desires", "legalized")]}, res


def check_parse_quirks(solve):
    # 首尾各剥一个字符（不论是否括号）；无 ';' 段整段跳过
    assert solve(mode="parse", graph_str="X(a; rel; b)Y") == {
        "nodes": ["(a", "b)"], "from_to": {"(a": ["b)"]},
        "from_to_typed": {"(a": [("rel", "b)")]}}, None
    assert solve(mode="parse", graph_str="(noise)(a; r; b)") == {
        "nodes": ["a", "b"], "from_to": {"a": ["b"]},
        "from_to_typed": {"a": [("r", "b")]}}, None
    # 空串与 "(a)"（剥首尾后无 ';'）都是空账
    empty = {"nodes": [], "from_to": {}, "from_to_typed": {}}
    assert solve(mode="parse", graph_str="") == empty
    assert solve(mode="parse", graph_str="(a)") == empty
    # 同 from→to 的第二条边（不同关系）整条静默丢弃
    dup = {"nodes": ["a", "b"], "from_to": {"a": ["b"]},
           "from_to_typed": {"a": [("r1", "b")]}}
    assert solve(mode="parse", graph_str="(a; r1; b)(a; r2; b)") == dup
    assert solve(mode="parse", graph_str="(a; r; b)(a; r; b)") == {
        "nodes": ["a", "b"], "from_to": {"a": ["b"]},
        "from_to_typed": {"a": [("r", "b")]}}
    # 段内 ';' 数 != 2 -> 解包 ValueError
    try:
        solve(mode="parse", graph_str="(a; b; c; d)")
    except ValueError:
        pass
    else:
        raise AssertionError("4-field segment must raise ValueError")
    # from 带前后空白被 strip
    assert solve(mode="parse", graph_str="( a ; r ; b )") == {
        "nodes": ["a", "b"], "from_to": {"a": ["b"]},
        "from_to_typed": {"a": [("r", "b")]}}


# ---------- mode="to_python"：graph_to_python（relation 变体） ----------

def check_to_python_docstring(solve):
    assert solve(mode="to_python", belief=ROW1["belief"],
                 argument=ROW1["argument"], stance=ROW1["stance"],
                 graph_str=GRAPH1, prompt_part_only=False) == (
        'class ExplanationDAG:\n'
        '\n'
        '    def __init__(self):\n'
        '        belief = "Marijuana should not be legalized."\n'
        '        argument = "Marijuana is dangerous for society."\n'
        '        stance = "support"\n'
        '\n'
        '        # create a DAG to support belief using argument\n'
        '        begin = ["marijuana"]\n'
        '        add_edge("marijuana", "is a", "recreational drug")\n'
        '        add_edge("recreational drug", "capable of", "drug addiction")\n'
        '        add_edge("drug addiction", "is a", "dangerous for society")\n'
        '        add_edge("dangerous for society", "not desires", "legalized")\n'
        '\n')


def check_to_python_prompt_only(solve):
    assert solve(mode="to_python", belief=ROW1["belief"],
                 argument=ROW1["argument"], stance=ROW1["stance"],
                 graph_str=GRAPH1, prompt_part_only=True) == (
        'class ExplanationDAG:\n'
        '\n'
        '    def __init__(self):\n'
        '        belief = "Marijuana should not be legalized."\n'
        '        argument = "Marijuana is dangerous for society."\n'
        '        stance = "support"\n'
        '\n'
        '        # create a DAG to support belief using argument\n')


def check_to_python_root_order(solve):
    # (x -> y)(w -> x)：唯一 root 是 w（x 有父）
    assert solve(mode="to_python", belief="B.", argument="A", stance="oppose",
                 graph_str="(x; leads to; y)(w; is a; x)").count('begin = ["w"]') == 1
    # 双 root 按节点添加序（a 先于 d：dict 序 src 先入图）
    code = solve(mode="to_python", belief="B", argument="A", stance="support",
                 graph_str="(a; r; b)(b; r2; c)(d; r3; b)")
    assert 'begin = ["a", "d"]' in code, code
    # belief/argument/stance 原样插值不转义
    code2 = solve(mode="to_python", belief='He said "hi"', argument="A",
                  stance="s", graph_str="(a; r; b)", prompt_part_only=True)
    assert 'belief = "He said "hi""' in code2, code2


# ---------- mode="to_graph"：python_to_graph（relation 变体） ----------

def check_to_graph_roundtrip(solve):
    code = solve(mode="to_python", belief=ROW1["belief"],
                 argument=ROW1["argument"], stance=ROW1["stance"],
                 graph_str=GRAPH1)
    res = solve(mode="to_graph", code_str=code)
    assert res == {"belief": "Marijuana should not be legalized.",
                   "argument": "Marijuana is dangerous for society.",
                   "stance": "support", "graph": GRAPH1}, res


def check_to_graph_quirks(solve):
    HEAD = 'belief = "b"\nargument = "a"\nstance = "s"\n'
    # 干净代码 round-trip
    assert solve(mode="to_graph", code_str=(
        HEAD + 'begin = ["x", "w"]\nadd_edge("x", "leads to", "y")\n'
               'add_edge("w", "is a", "x")\n')) == {
        "belief": "b", "argument": "a", "stance": "s",
        "graph": "(x; leads to; y)(w; is a; x)"}, None
    # 无 belief 行 -> 正则 None.group -> AttributeError
    try:
        solve(mode="to_graph", code_str='add_edge("a", "r", "b")')
    except AttributeError:
        pass
    else:
        raise AssertionError("missing belief must raise AttributeError")
    # fields < 3 -> IndexError
    try:
        solve(mode="to_graph", code_str=HEAD + 'add_edge("x")')
    except IndexError:
        pass
    else:
        raise AssertionError("short add_edge must raise IndexError")
    # 双空格使 ', "' 分隔失配 -> 单字段 -> IndexError
    try:
        solve(mode="to_graph", code_str=HEAD + 'add_edge("n1",  "r1",  "n2")')
    except IndexError:
        pass
    else:
        raise AssertionError("double-space separator must raise IndexError")
    # 行尾无 ')'：[:-1] 吃掉尾部引号，仍可解析
    assert solve(mode="to_graph", code_str=(
        HEAD + 'add_edge("n3", "r2", "n4"\n'))["graph"] == "(n3; r2; n4)"
    # 首个 '(' 后的截断：target 内 '(' 之后的内容整体丢失
    assert solve(mode="to_graph", code_str=HEAD + 'add_edge("a", "r", "b (x)")')["graph"] == "(a; r; b)"
    # split("(")[1] 之后的 junk 也进 target，经 '"' 删除 / ';' -> '; ' / 单趟 '  ' -> ' '
    assert solve(mode="to_graph", code_str=HEAD + 'add_edge("n1", "r1", "n2");  junk')["graph"] == "(n1; r1; n2);  jun)"


# ---------- mode="roots"：GraphAlgos.root_nodes over to_nx_graph ----------

def check_roots(solve):
    _, ftd1, _ = None, None, None
    assert solve(mode="roots", src_dest={
        "marijuana": ["recreational drug"],
        "recreational drug": ["drug addiction"],
        "drug addiction": ["dangerous for society"],
        "dangerous for society": ["legalized"]}) == ["marijuana"]
    # 双 root 按添加序
    assert solve(mode="roots", src_dest={"a": ["b"], "b": ["c"], "d": ["b"]}) == ["a", "d"]
    # 纯环图无 root
    assert solve(mode="roots", src_dest={"a": ["b"], "b": ["a"]}) == []
    # 空表
    assert solve(mode="roots", src_dest={}) == []
    # to_nx_graph 只 add_edge：空 dests 的 src 不入图
    assert solve(mode="roots", src_dest={"b": []}) == []
    assert solve(mode="roots", src_dest={"b": [], "a": ["b"]}) == ["a"]


# ---------- mode="case"：utils 三函数 ----------

def check_case_camel(solve):
    assert solve(mode="case", text="travel to the theme park.", case="camel") == \
        "TravelToTheThemePark"
    assert solve(mode="case", text="do A (thing)!", case="camel") == "DoA(thing)!"
    # capitalize 把其余字符转小写（与仅首字母 upper 不同）
    assert solve(mode="case", text="doA Thing.", case="camel") == "DoaThing"
    try:
        solve(mode="case", text="", case="camel")
    except IndexError:
        pass
    else:
        raise AssertionError("empty text must raise IndexError")


def check_case_snake(solve):
    assert solve(mode="case", text="Travel To The Theme Park.", case="snake") == \
        "travel_to_the_theme_park"
    # [\\s\\W] 各换一个 '_'；strip 只剥空白，尾 '_' 保留
    assert solve(mode="case", text="Do A (Thing)!", case="snake") == "do_a__thing__"
    assert solve(mode="case", text="Mixed  CASE  here", case="snake") == "mixed__case__here"
    try:
        solve(mode="case", text="", case="snake")
    except IndexError:
        pass
    else:
        raise AssertionError("empty text must raise IndexError")


def check_case_desnake(solve):
    assert solve(mode="case", text="travel_to_the_theme_park", case="desnake") == \
        "travel to the theme park"
    assert solve(mode="case", text="a__b", case="desnake") == "a  b"


# ---------- 参数校验 ----------

def check_args_validation(solve):
    bad = [("mode", lambda: solve(mode="nope")),
           ("graph_str", lambda: solve(mode="parse", graph_str=123)),
           ("belief", lambda: solve(mode="to_python", belief=None,
                                    argument="a", stance="s", graph_str="(a; r; b)")),
           ("stance", lambda: solve(mode="to_python", belief="b", argument="a",
                                    stance=1, graph_str="(a; r; b)")),
           ("prompt_flag", lambda: solve(mode="to_python", belief="b", argument="a",
                                         stance="s", graph_str="(a; r; b)",
                                         prompt_part_only=1)),
           ("code_str", lambda: solve(mode="to_graph", code_str=None)),
           ("src_dest", lambda: solve(mode="roots", src_dest=["a"])),
           ("src_dest_val", lambda: solve(mode="roots", src_dest={"a": "b"})),
           ("src_dest_item", lambda: solve(mode="roots", src_dest={1: ["b"]})),
           ("text", lambda: solve(mode="case", text=None, case="snake")),
           ("case", lambda: solve(mode="case", text="x", case="kebab"))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")
