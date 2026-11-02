from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：节点用 dict、深度靠父亲链现算、排序
# 用选择法提取最大值（稳定优先取先出现者）。

_FB = 2
_PB = 2
_AE = 10000
_BE = 100000


def _mk(nodes, ntype, desc, dad):
    nodes.append({"type": ntype, "desc": desc, "obs": "", "code": None,
                  "kids": [], "dad": dad, "pruned": False, "term": False,
                  "fin": False, "exp": 0, "score": 0.0})
    idx = len(nodes) - 1
    if dad is not None:
        nodes[dad]["kids"].append(idx)
    return idx


def _dep(nodes, i):
    d, j = 0, i
    while nodes[j]["dad"] is not None:
        j = nodes[j]["dad"]
        d += 1
    return d


def _finish(nodes, i, left):
    nodes[i]["fin"] = True
    if nodes[i]["type"] == "Action Input":
        left -= 1
    if nodes[i]["dad"] is not None and left >= 0:
        _finish(nodes, nodes[i]["dad"], left)


def _pick_a(v):
    s = v.strip().lower()
    if s == "":
        raise ValueError("empty rank vote")
    return s[-1] == "a"


def _oracle(responses, tool_table, pair_votes, single_chain_max_step=5,
            tree_beam_size=2, max_query_count=20, answer=1,
            with_filter=True, give_up_seed=42):
    nodes = []
    root = _mk(nodes, "Action Input", "", None)
    st = {"status": 0, "term": [], "give": [], "nen": 0, "q": 0,
          "rq": 0, "tok": 0, "order": [], "div": {}, "at": 0}

    def walk(i):
        nodes[i]["exp"] = st["nen"]
        st["order"].append(i)
        st["nen"] += 1

        if _dep(nodes, i) >= single_chain_max_step or nodes[i]["pruned"] \
                or nodes[i]["term"]:
            if nodes[i]["term"]:
                st["status"] = 1
                st["term"].append(i)
                return _FB
            nodes[i]["pruned"] = True
            if nodes[i]["code"] == 4:
                st["give"].append(i)
                return _PB
            return 1

        cands = []
        for _ in range(tree_beam_size):
            here = i
            if nodes[here]["kids"]:
                got = []
                for kid in nodes[here]["kids"]:
                    t = kid
                    while (not nodes[t]["term"]
                           and nodes[t]["type"] != "Action Input"
                           and nodes[t]["kids"]):
                        t = nodes[t]["kids"][0]
                    if nodes[t]["type"] == "Action Input":
                        got.append({"name": nodes[nodes[t]["dad"]]["desc"],
                                    "arguments": nodes[t]["desc"],
                                    "function_output": nodes[t]["obs"],
                                    "value": 0.0})
                if got:
                    st["div"][here] = got

            if st["at"] >= len(responses):
                raise ValueError("no response left")
            r = responses[st["at"]]
            st["at"] += 1
            st["q"] += 1
            st["tok"] += r.get("tokens", 0)
            if st["q"] >= max_query_count:
                return _BE

            body = r.get("content")
            call = r.get("function_call")
            bad = r.get("error_code", 0)

            if body is not None:
                here = _mk(nodes, "Thought", body, here)
                if bad != 0:
                    nodes[here]["code"] = bad
                    nodes[here]["pruned"] = True

            if call is not None:
                if "name" not in call or "arguments" not in call:
                    raise ValueError("bad function_call")
                key = (call["name"], call["arguments"])
                act = _mk(nodes, "Action", call["name"], here)
                if key not in tool_table:
                    raise ValueError("no tool step")
                obs, status = tool_table[key]
                here = _mk(nodes, "Action Input", call["arguments"], act)
                nodes[here]["obs"] = obs
                nodes[here]["code"] = status
                if status == 3:
                    nodes[here]["term"] = True
                    _finish(nodes, here, _FB)
                elif status == 4:
                    nodes[here]["pruned"] = True
                else:
                    pass

            if call is None and nodes[here]["type"] == "Action Input":
                raise ValueError("no function_call on Action Input")

            back = None
            if not with_filter:
                rv = walk(here)
                if len(st["term"]) >= answer:
                    back = _AE
                elif rv > 1:
                    back = rv - 1
            else:
                cands.append(here)
            if back is not None:
                return back

        if len(cands) > 1:
            pts = [0.0] * len(cands)
            for a in range(len(cands) - 1):
                for b in range(a + 1, len(cands)):
                    if (a, b) not in pair_votes:
                        raise ValueError("no vote")
                    v = pair_votes[(a, b)]
                    win1 = 1 if _pick_a(v[0]) else 0
                    win2 = 1 if _pick_a(v[1]) else 0
                    earn = [0, 0]
                    earn[1 - win1] += 1
                    earn[win2] += 1
                    if earn[0] > earn[1]:
                        pts[a] += 1
                    elif earn[0] < earn[1]:
                        pts[b] += 1
                    else:
                        pts[a] += 0.5
                        pts[b] += 0.5
                    st["rq"] += 2
                    st["q"] += 2
            for k, nd in enumerate(cands):
                nodes[nd]["score"] = pts[k]
            rest = list(range(len(cands)))
            picked = []
            while rest:                      # 选择法：稳定取最大
                best = rest[0]
                for t in rest[1:]:
                    if nodes[cands[t]]["score"] > nodes[cands[best]]["score"]:
                        best = t
                picked.append(cands[best])
                rest.remove(best)
            cands = picked

        for nd in cands:
            rv = walk(nd)
            if len(st["term"]) >= answer:
                return _AE
            elif rv > 1:
                _finish(nodes, i, 2)
                return rv - 1
        return 1

    rv = walk(root)

    node = None
    ans = None
    kind = None
    for t in st["term"]:
        if not nodes[t]["pruned"]:
            node, ans, kind = t, nodes[t]["desc"], "give_answer"
            break
    if kind is None and st["give"]:
        import random as _random
        g = _random.Random(give_up_seed)
        pos = g.randint(0, len(st["give"]) - 1)
        node = st["give"][pos]
        ans = nodes[node]["desc"]
        kind = "give_up"

    return {"status": st["status"],
            "query_count": st["q"],
            "rank_query_count": st["rq"],
            "total_tokens": st["tok"],
            "return_value": rv,
            "expand_order": list(st["order"]),
            "expand_nums": {k: nodes[k]["exp"] for k in range(len(nodes))},
            "node_types": [nodes[k]["type"] for k in range(len(nodes))],
            "node_depths": [_dep(nodes, k) for k in range(len(nodes))],
            "terminal_nodes": list(st["term"]),
            "give_up_nodes": list(st["give"]),
            "pruned_node_ids": [k for k in range(len(nodes))
                                if nodes[k]["pruned"]],
            "finished_node_ids": [k for k in range(len(nodes))
                                  if nodes[k]["fin"]],
            "prior_scores": {k: nodes[k]["score"] for k in range(len(nodes))},
            "diversity_candidates": {k: [dict(c) for c in v]
                                     for k, v in st["div"].items()},
            "final_answer": ans,
            "finish_type": kind,
            "answer_node": node}


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
    if isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) < 1e-9, (a, b)
        return
    assert a == b, (a, b)


def _compare(solve, *args, **kw):
    got = solve(*[_dc(x) for x in args], **deepcopy(kw))
    exp = _oracle(*[_dc(x) for x in args], **deepcopy(kw))
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- checks

_TOOL = {("get", "A"): ("okA", 0),
         ("get", "B"): ("okB", 0),
         ("hall", "H"): ("bad api", 1),
         ("final", "X"): ("ansX", 3),
         ("bad", "Y"): ("no", 4)}


def _call(name, args):
    return {"name": name, "arguments": args}


def _r(content=None, call=None, err=0, tok=0):
    return {"content": content, "function_call": call, "error_code": err,
            "tokens": tok}


def check_dfsdt_single_chain(solve):
    """DFSDT（with_filter=False）单链出答案：终局返回 2，上层
    result-1 倒计数；answer=1 时改走 10000 提前退出。"""
    resp = [_r("t1", _call("get", "A")), _r("t2", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False)
    assert led["status"] == 1
    assert led["node_types"] == ["Action Input", "Thought", "Action",
                                 "Action Input", "Thought", "Action",
                                 "Action Input"]
    assert led["node_depths"] == [0, 1, 2, 3, 4, 5, 6]
    assert led["expand_order"] == [0, 3, 6]
    assert led["expand_nums"] == {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 2}
    assert led["terminal_nodes"] == [6]
    assert led["final_answer"] == "X"
    assert led["answer_node"] == 6
    assert led["finish_type"] == "give_answer"
    assert led["return_value"] == 10000          # answer=1 提前退出
    # make_finish(2)：从终局向上数满 2 个 Action Input（含自身）
    assert led["finished_node_ids"] == [0, 1, 2, 3, 4, 5, 6]
    assert led["query_count"] == 2 and led["rank_query_count"] == 0
    # answer=99 不提前退出 → 倒计数：终局 2 → 上层 1 → root 返回 1
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, answer=99)
    assert led["terminal_nodes"] == [6]
    assert led["return_value"] == 1
    assert led["status"] == 1


def check_depth_cap(solve):
    """进入即判深度：depth >= single_chain_max_step 直接剪（不是
    give up，返回 1）；但终局判定优先于深度判定。"""
    resp = [_r("t1", _call("get", "A"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, single_chain_max_step=3, answer=99)
    assert led["expand_order"] == [0, 3]
    assert led["pruned_node_ids"] == [3]         # 深度 3 >= 3 → 剪
    assert led["give_up_nodes"] == []
    assert led["return_value"] == 1
    # 终局优先：AI 在 depth==max_step 但 status 3 → 记 terminal 不剪
    resp = [_r("t1", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, single_chain_max_step=3)
    assert led["terminal_nodes"] == [3]
    assert led["pruned_node_ids"] == []
    assert led["status"] == 1
    # 链内建点不看深度：max_step=2 时 Action（depth 2）照样建出来，
    # 只有被递归进入的链末端（AI，depth 3）才判剪
    resp = [_r("t1", _call("get", "A"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, single_chain_max_step=2, answer=99)
    assert led["node_depths"] == [0, 1, 2, 3]
    assert led["expand_order"] == [0, 3]
    assert led["pruned_node_ids"] == [3]
    assert led["return_value"] == 1


def check_give_up_and_error_codes(solve):
    """放弃的三种来源：工具 status 4、LLM error_code 4（Thought 上
    也记 give up）、普通 error_code 只剪不放弃；status 1（幻觉 API
    名）不动树。"""
    # 工具 status 4 → AI 落 pruned+code 4 → 进入即 give up
    led = _compare(solve, [_r("t", _call("bad", "Y"))], _TOOL, {},
                   tree_beam_size=1, with_filter=False, answer=99)
    assert led["give_up_nodes"] == [3]
    assert led["pruned_node_ids"] == [3]
    assert led["finish_type"] == "give_up"
    assert led["final_answer"] == "Y"
    assert led["return_value"] == 1
    assert led["finished_node_ids"] == []        # DFSDT 回溯不补 finish
    # LLM error_code=4（Thought 上）也算 give up
    led = _compare(solve, [_r("t", None, 4)], _TOOL, {},
                   tree_beam_size=1, with_filter=False, answer=99)
    assert led["give_up_nodes"] == [1]
    assert led["final_answer"] == "t"
    # error_code=2：只剪不放弃
    led = _compare(solve, [_r("t", None, 2)], _TOOL, {},
                   tree_beam_size=1, with_filter=False, answer=99)
    assert led["pruned_node_ids"] == [1]
    assert led["give_up_nodes"] == []
    assert led["finish_type"] is None
    assert led["final_answer"] is None
    assert led["answer_node"] is None
    # status=1（幻觉 API 名）：树不变、照常继续展开
    resp = [_r("a", _call("hall", "H")), _r("b", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, single_chain_max_step=6, answer=99)
    assert led["pruned_node_ids"] == []
    assert led["terminal_nodes"] == [6]
    assert led["status"] == 1


def check_normal_dfs_rank(solve):
    """normal DFS（with_filter=True）：先攒 tree_beam_size 个候选，
    两两比对（每对 2 次查询计入 query_count）、和分降序稳定排序，
    再按序展开；平分保生成序。"""
    resp = [_r("a", _call("get", "A")), _r("b", _call("get", "B"))]
    # 票串只看末字符：'b' 表示「第一个候选输了」
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "a"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=99)
    assert led["expand_order"] == [0, 6, 3]       # 第二个候选排第一
    assert led["expand_nums"] == {0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 1}
    assert led["rank_query_count"] == 2
    assert led["query_count"] == 4                # 2 次生成 + 2 次比对
    # 反向票（cand0 两向全胜）→ 仍按生成序展开
    led = _compare(solve, resp, _TOOL, {(0, 1): ["a", "b"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=99)
    assert led["expand_order"] == [0, 3, 6]
    # 双向各赢一次（["b","b"] / ["a","a"]）→ 0.5+0.5 平分 → 保生成序
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "b"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=99)
    assert led["expand_order"] == [0, 3, 6]
    # beam=1 不排序、不花比对查询
    led = _compare(solve, resp[:1], _TOOL, {}, tree_beam_size=1,
                   single_chain_max_step=3, answer=99)
    assert led["rank_query_count"] == 0
    assert led["query_count"] == 1
    # 票只看**末字符**：'a bad choice' 末字符 'e' → 判「第一个输」
    led = _compare(solve, resp, _TOOL,
                   {(0, 1): ["a bad choice", "a"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=99)
    assert led["prior_scores"][6] == 1.0
    assert led["prior_scores"][3] == 0.0
    assert led["expand_order"] == [0, 6, 3]
    # 三候选：平分 0.5 + 和分排序（1.5 / 1.0 / 0.5）
    resp3 = [_r("a", _call("get", "A")), _r("b", _call("get", "B")),
             _r("c", _call("get", "A"))]
    led = _compare(solve, resp3, _TOOL,
                   {(0, 1): ["b", "b"], (0, 2): ["a", "b"],
                    (1, 2): ["b", "a"]},
                   tree_beam_size=3, single_chain_max_step=3, answer=99)
    assert led["prior_scores"] == {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.5,
                                   4: 0.0, 5: 0.0, 6: 0.5, 7: 0.0,
                                   8: 0.0, 9: 1.0}
    assert led["expand_order"] == [0, 3, 9, 6]
    assert led["rank_query_count"] == 6            # 3 对 × 2
    assert led["query_count"] == 9                 # 3 生成 + 6 比对


def check_backtrack_make_finish(solve):
    """normal DFS 展开循环：result > 1（子树终局/放弃）→ 对当前节点
    make_finish(2) 并 return result-1 —— 同层剩余候选**不再展开**；
    即便 answer 未满也一样。"""
    resp = [_r("a", _call("bad", "Y")), _r("b", _call("get", "B"))]
    led = _compare(solve, resp, _TOOL, {(0, 1): ["a", "a"]},
                   tree_beam_size=2, single_chain_max_step=5, answer=99)
    assert led["expand_order"] == [0, 3]          # 候选 2 从未进入
    assert led["expand_nums"][6] == 0
    assert led["give_up_nodes"] == [3]
    assert led["return_value"] == 1
    # root 是 Action Input：make_finish(2) 在 root 上就数满 1 个后停
    assert led["finished_node_ids"] == [0]
    # 终局同理：answer=2 却只拿到 1 个 terminal（同层被放弃）
    resp = [_r("a", _call("final", "X")), _r("b", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "b"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=2)
    assert led["terminal_nodes"] == [3]
    assert led["expand_order"] == [0, 3]
    assert led["return_value"] == 1
    assert led["status"] == 1
    # answer=1 → 同样只到第一个终局，但返回 10000
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "b"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=1)
    assert led["terminal_nodes"] == [3]
    assert led["return_value"] == 10000
    assert led["finished_node_ids"] == [0, 1, 2, 3, 4, 5, 6]


def check_query_budget(solve):
    """max_query_count 判定在查询计数之后、建节点之前：超预算的那条
    响应整个丢弃；返回 100000 逐层 -1 上卷。"""
    resp = [_r("a", _call("get", "A")), _r("b", _call("get", "B"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, max_query_count=2, answer=99)
    assert led["query_count"] == 2
    assert led["node_types"] == ["Action Input", "Thought", "Action",
                                 "Action Input"]          # 第 2 条丢弃
    assert led["return_value"] == 99999
    assert led["status"] == 0
    # 空响应（content/function_call 皆 None）落在 Thought 上 → 自递归
    resp = [_r("t"), _r(None), _r(None)]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, max_query_count=3, answer=99)
    assert led["expand_order"] == [0, 1, 1]       # 同一个节点被进入两次
    assert led["expand_nums"][1] == 2
    assert led["node_types"] == ["Action Input", "Thought"]
    assert led["return_value"] == 99998           # 100000 - 2 层上卷
    assert led["query_count"] == 3
    # tokens 累加（含被丢弃的那条）
    resp = [_r("a", _call("get", "A"), tok=7), _r("b", tok=5)]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, max_query_count=2, answer=99)
    assert led["total_tokens"] == 12


def check_diversity_candidates(solve):
    """第二轮起喂 diversity 候选：沿 children[0] 链走到第一个
    Action Input；name 取 Action 描述、arguments 取 AI 描述、
    value 是 compute_weight() 恒 0.0（ToT Elo 残迹）。"""
    resp = [_r("a", _call("get", "A")), _r("b", _call("get", "B"))]
    led = _compare(solve, resp, _TOOL, {(0, 1): ["a", "a"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=99)
    assert led["diversity_candidates"] == {
        0: [{"name": "get", "arguments": "A",
             "function_output": "okA", "value": 0.0}]}
    # 第一个候选只有 Thought（无 action）→ js 空 → 不喂候选
    led = _compare(solve, [_r("only1"), _r("only2")], _TOOL,
                   {(0, 1): ["a", "a"]}, tree_beam_size=2,
                   single_chain_max_step=1, answer=99)
    assert led["diversity_candidates"] == {}
    assert led["node_types"] == ["Action Input", "Thought", "Thought"]
    assert led["pruned_node_ids"] == [1, 2]
    assert led["return_value"] == 1


def check_two_terminals_and_finish_span(solve):
    """两个终局的登记序（append）+ 第一个未被剪的终局当答案；三层深
    链时 make_finish(2) 只向上数 2 个 Action Input（含自身），到第三
    个就停。"""
    resp = [_r("a", _call("get", "A")), _r("b", _call("get", "B")),
            _r("c", _call("final", "X")), _r("d", _call("get", "A")),
            _r("e", _call("final", "X")), _r("f", _call("get", "B"))]
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "b"]},
                   tree_beam_size=2, single_chain_max_step=4, answer=2)
    assert led["terminal_nodes"] == [9, 15]        # append 序
    assert led["answer_node"] == 9                 # 第一个未剪终局
    assert led["status"] == 1
    assert led["expand_order"] == [0, 3, 9, 6, 15]
    assert led["return_value"] == 10000
    assert led["query_count"] == 12
    assert led["rank_query_count"] == 6
    # 三层深链：终局 AI 在 depth 9，向上只数 2 个 Action Input（自身 +
    # depth 6 那个），depth 3 的第三个 AI 与更上层不标 finished
    resp = [_r("t1", _call("get", "A")), _r("t2", _call("get", "A")),
            _r("t3", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1,
                   with_filter=False, single_chain_max_step=9)
    assert led["node_depths"] == list(range(10))
    assert led["terminal_nodes"] == [9]
    assert led["finished_node_ids"] == [3, 4, 5, 6, 7, 8, 9]
    assert led["return_value"] == 10000


def check_default_args(solve):
    """默认参数（single_chain_max_step=5 / tree_beam_size=2 /
    max_query_count=20 / answer=1 / with_filter=True）：深度 4 的
    Thought 不被剪，继续生成到深度 5 才判；error_code=4 落 give up。"""
    resp = [_r("a", _call("get", "A")), _r("t"), _r("u", None, 4)]
    led = _compare(solve, resp, _TOOL, {}, tree_beam_size=1)
    assert led["node_depths"] == [0, 1, 2, 3, 4, 5]
    assert led["expand_order"] == [0, 3, 4, 5]
    assert led["give_up_nodes"] == [5]
    assert led["pruned_node_ids"] == [5]
    assert led["finish_type"] == "give_up"
    assert led["return_value"] == 1


def check_errors(solve):
    """域约束：响应队列耗尽 / 工具表缺步 / 票缺失 / 票串空 /
    Action Input 上无 function_call（官方 KeyError）/ call 缺键。"""
    tool = {("get", "A"): ("okA", 0)}
    for args, kw in [
        (([_r("a", _call("get", "A"))], tool, {}, ),
         dict(tree_beam_size=1, with_filter=False, max_query_count=5)),
        (([_r("a", _call("get", "Z"))], tool, {}),
         dict(tree_beam_size=1, with_filter=False)),
        (([_r("a", _call("get", "A")), _r("b", _call("get", "A"))],
          tool, {}),
         dict(tree_beam_size=2, single_chain_max_step=3)),
        (([_r(None)], tool, {}), dict(tree_beam_size=1, with_filter=False)),
        (([_r("a", {"name": "get"})], tool, {}),
         dict(tree_beam_size=1, with_filter=False)),
    ]:
        try:
            solve(*[_dc(x) for x in args], **deepcopy(kw))
            assert False, args
        except ValueError:
            pass
    # 票串为空 → ValueError
    try:
        solve([_r("a", _call("get", "A")), _r("b", _call("get", "A"))],
              tool, {(0, 1): ["", "a"]}, tree_beam_size=2,
              single_chain_max_step=3)
        assert False
    except ValueError:
        pass


def check_random_cross(solve):
    """随机脚本对拍 oracle（固定种子）：树形、预算、排序、放弃全随机。"""
    import random

    class _Votes(dict):                    # 任意候选对都给确定的票
        def __contains__(self, k):
            return True

        def __getitem__(self, k):
            i, j = k
            return ["vote %d" % ((i * 3 + j) % 2), "vote %d" % ((i + j) % 3)]

    rng = random.Random(20261102)
    keys = [("t0", "a"), ("t0", "b"), ("t1", "c"), ("t2", "d")]
    tool = {("t0", "a"): ("o0", 0), ("t0", "b"): ("o0b", 1),
            ("t1", "c"): ("o1", 3), ("t2", "d"): ("o2", 4)}
    for _ in range(150):
        resp = []
        for _ in range(rng.randint(0, 14)):
            pick = rng.random()
            if pick < 0.18:
                resp.append(_r(None, None, rng.choice([0, 0, 2, 4]),
                               rng.randint(0, 3)))
            elif pick < 0.40:
                resp.append(_r("t%d" % rng.randint(0, 9), None,
                               rng.choice([0, 0, 0, 4]),
                               rng.randint(0, 3)))
            elif pick < 0.55:
                k = rng.choice(keys)
                resp.append(_r(None, _call(*k), 0, rng.randint(0, 3)))
            else:
                k = rng.choice(keys)
                resp.append(_r("t%d" % rng.randint(0, 9), _call(*k),
                               rng.choice([0, 0, 0, 2]),
                               rng.randint(0, 3)))
        kw = dict(single_chain_max_step=rng.randint(1, 5),
                  tree_beam_size=rng.randint(1, 3),
                  max_query_count=rng.randint(1, 7),
                  answer=rng.randint(1, 2),
                  with_filter=rng.random() < 0.5,
                  give_up_seed=rng.randint(0, 99))
        try:
            _compare(solve, resp, tool, _Votes(), **kw)
        except ValueError:
            try:
                _oracle(*[_dc(resp), _dc(tool), _Votes()], **deepcopy(kw))
            except ValueError:
                continue
            raise


def check_example(solve):
    """run 模式样例：normal DFS 两候选排序 → 展开 → 出答案。"""
    resp = [_r("先试 A", _call("get", "A")),
            _r("再试 B", _call("final", "X"))]
    led = _compare(solve, resp, _TOOL, {(0, 1): ["b", "a"]},
                   tree_beam_size=2, single_chain_max_step=3, answer=1)
    # 排序把 cand1（b/final X）提到首位 → 展开即出答案 → answer=1
    # 提前退出，cand0 不再展开
    assert led["expand_order"] == [0, 6]
    assert led["terminal_nodes"] == [6]
    assert led["final_answer"] == "X"
    assert led["finish_type"] == "give_answer"
    assert led["status"] == 1
    assert led["query_count"] == 4
    assert led["return_value"] == 10000


check_example.example = True
