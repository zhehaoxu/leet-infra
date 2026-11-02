"""
ToolLLM DFS 决策树账：把 ToolLLM（arXiv 2307.16789）的
「基于深度优先搜索的决策树」（DFSDT）搜索无状态化成一棵可复算的
树账。

回源（OpenBMB/ToolBench master，X-PLUG/ToolLLM 已迁此仓）：
- toolbench/inference/Algorithms/DFS.py → DFS_tree_search.DFS
  （回溯倒计数、剪枝、diversity 候选、normal-DFS 排序展开）
- toolbench/inference/Tree/Tree.py → tree_node.make_finish /
  compute_weight / get_depth
- toolbench/inference/LLM_rank/rank_candidate.py → rank2_subfix /
  rank2symmetry / sum_based_rankn

solve(responses, tool_table, pair_votes, single_chain_max_step=5,
      tree_beam_size=2, max_query_count=20, answer=1,
      with_filter=True, give_up_seed=42)

- responses：LLM 响应队列，**一次 LLM 查询消费一条**（先序消费）。
  每项 {"content": str|None, "function_call": {"name","arguments"}|
  None, "error_code": int=0, "tokens": int=0}。content 非 None 建
  Thought 子节点；function_call 非 None 建 Action + Action Input
  两节点（工具真的执行：observation/status 查 tool_table）。
- tool_table：{(action_name, action_input): (observation, status)}；
  status：0 成功 / 1 幻觉 API 名 / 3 给出最终答案 / 4 放弃。
- pair_votes：{(i, j): [vote_ij, vote_ji]}（i<j，i/j 为候选在
  next_tree_split_nodes 里的生成序）；vote 是 rank2_subfix 的 LLM
  输出串，**strip().lower() 后末字符为 'a'** 表示「第一个候选更
  好」（官方只看末字符，不看语义）。
- single_chain_max_step：树深上限（root 深 0，Thought/Action/
  Action Input 各占一层）；tree_beam_size：每层生成几个子节点；
  max_query_count：LLM 查询预算（**含排序查询**）；answer：找到
  n 个 Final Answer 就提前退出；with_filter=True 走 normal DFS
  （先排序再展开），False 走 DFSDT（生成即递归的先序遍历）。

返回账：
{"status": 0/1（有任何 Final Answer 即 1）,
 "query_count", "rank_query_count", "total_tokens",
 "return_value": DFS(root) 的返回值（回溯倒计数诊断）,
 "expand_order": [节点 id ...]（DFS 进入序，重复进入重复记）,
 "expand_nums": {节点 id: 最终 expand_num},
 "node_types": [每 id 的 "Action Input"/"Thought"/"Action"],
 "node_depths": [每 id 的深度],
 "terminal_nodes": [id ...]（append 序）, "give_up_nodes": [id ...],
 "pruned_node_ids": [id ...], "finished_node_ids": [id ...],
 "prior_scores": {节点 id: 排序分（仅 normal DFS 排序时赋值）},
 "diversity_candidates": {node id: [{"name","arguments",
     "function_output","value"}, ...]},
 "final_answer": str|None, "finish_type": "give_answer"/"give_up"/
     None, "answer_node": id|None}

域约束（ValueError）：响应队列耗尽；tool_table 缺 (name, args)；
pair_votes 缺键或票串为空；Action Input 节点上收到无 function_call
的响应（官方此处 KeyError）；function_call 缺 name/arguments。
"""
import random


_FINAL_BACK = 2      # final_answer_back_length
_PRUNE_BACK = 2      # prune_back_length
_ANSWER_EXIT = 10000
_BUDGET_EXIT = 100000


class _Node:
    __slots__ = ("nid", "node_type", "description", "observation",
                 "observation_code", "children", "father", "pruned",
                 "is_terminal", "finished", "expand_num", "prior_score")

    def __init__(self, nid, node_type, description=""):
        self.nid = nid
        self.node_type = node_type
        self.description = description
        self.observation = ""
        self.observation_code = None
        self.children = []
        self.father = None
        self.pruned = False
        self.is_terminal = False
        self.finished = False
        self.expand_num = 0
        self.prior_score = 0.0


def _depth(node):
    d = 0
    while node.father is not None:
        node = node.father
        d += 1
    return d


def _make_finish(node, inter_val):
    """官方 make_finish：标记 finished 并向上传播，直到数满
    inter_val 个 Action Input（含自身）为止。"""
    node.finished = True
    if node.node_type == "Action Input":
        inter_val -= 1
    if node.father is not None and inter_val >= 0:
        _make_finish(node.father, inter_val)


def _vote_win_a(vote):
    s = vote.strip().lower()
    if s == "":
        raise ValueError("empty rank vote")
    return s[-1] == "a"


class _Run:
    def __init__(self, responses, tool_table, pair_votes):
        self.responses = responses
        self.cursor = 0
        self.tool_table = tool_table
        self.pair_votes = pair_votes
        self.nodes = []
        self.status = 0
        self.terminal_nodes = []
        self.give_up_nodes = []
        self.now_expand_num = 0
        self.query_count = 0
        self.rank_query_count = 0
        self.total_tokens = 0
        self.expand_order = []
        self.diversity = {}

    def new_node(self, node_type, description, father):
        n = _Node(len(self.nodes), node_type, description)
        n.father = father
        if father is not None:
            father.children.append(n)
        self.nodes.append(n)
        return n

    def next_response(self):
        if self.cursor >= len(self.responses):
            raise ValueError("LLM response queue exhausted")
        resp = self.responses[self.cursor]
        self.cursor += 1
        return resp


def _sum_based_rankn(run, cands):
    """rank2symmetry（双向各问一次）+ 平分各 0.5 的对和排序分。"""
    n = len(cands)
    scores = [0.0] * n
    for i in range(n - 1):
        for j in range(i + 1, n):
            key = (i, j)
            if key not in run.pair_votes:
                raise ValueError("missing pair vote for candidates %r" % (key,))
            votes = run.pair_votes[key]
            bigger1 = 1 if _vote_win_a(votes[0]) else 0
            bigger2 = 1 if _vote_win_a(votes[1]) else 0
            side = [0, 0]
            side[1 - bigger1] += 1
            side[bigger2] += 1
            if side[0] > side[1]:
                scores[i] += 1
            elif side[0] < side[1]:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5
            run.rank_query_count += 2
            run.query_count += 2
    return scores


def _dfs(run, node, max_step, beam, max_q, answer, with_filter):
    node.expand_num = run.now_expand_num
    run.expand_order.append(node.nid)
    run.now_expand_num += 1

    # 进入即判：深度到顶 / 已剪枝 / 已是终局
    if _depth(node) >= max_step or node.pruned or node.is_terminal:
        if node.is_terminal:                      # final answer
            run.status = 1
            run.terminal_nodes.append(node.nid)
            return _FINAL_BACK
        node.pruned = True
        if node.observation_code == 4:            # give up
            run.give_up_nodes.append(node.nid)
            return _PRUNE_BACK
        return 1

    split_nodes = []
    for _ in range(beam):
        cur = node
        # 已有子节点 → diversity prompt（沿 children[0] 链走到第一个
        # Action Input，作为「已试过的动作」喂回）
        if cur.children:
            js = []
            for child in cur.children:
                t = child
                while (not t.is_terminal and t.node_type != "Action Input"
                       and t.children):
                    t = t.children[0]
                if t.node_type == "Action Input":
                    js.append({"name": t.father.description,
                               "arguments": t.description,
                               "function_output": t.observation,
                               "value": 0.0})
            if js:
                run.diversity[cur.nid] = js

        resp = run.next_response()
        run.query_count += 1
        run.total_tokens += resp.get("tokens", 0)
        if run.query_count >= max_q:
            return _BUDGET_EXIT
        content = resp.get("content")
        fcall = resp.get("function_call")
        err = resp.get("error_code", 0)

        if content is not None:
            cur = run.new_node("Thought", content, cur)
            if err != 0:
                cur.observation_code = err
                cur.pruned = True

        if fcall is not None:
            if "name" not in fcall or "arguments" not in fcall:
                raise ValueError("function_call missing name/arguments")
            name = fcall["name"]
            args = fcall["arguments"]
            cur = run.new_node("Action", name, cur)
            key = (name, args)
            if key not in run.tool_table:
                raise ValueError("tool step not in table: %r" % (key,))
            obs, status = run.tool_table[key]
            cur = run.new_node("Action Input", args, cur)
            cur.observation = obs
            cur.observation_code = status
            if status == 3:                # final answer（check_success）
                cur.is_terminal = True
                _make_finish(cur, _FINAL_BACK)
            elif status == 4:              # give up
                cur.pruned = True
            # status == 1：官方只把 new_message 里 name 改成
            # invalid_hallucination_function_name（消息层），树不变

        # 官方此处对当前节点追加 assistant / function 消息；若当前节点
        # 是 Action Input 而响应里没有 function_call，取 name 直接炸。
        if fcall is None and cur.node_type == "Action Input":
            raise ValueError("function_call missing at Action Input node")

        rv = None
        if not with_filter:                # DFSDT：先序递归
            result = _dfs(run, cur, max_step, beam, max_q, answer,
                          with_filter)
            if len(run.terminal_nodes) >= answer:
                rv = _ANSWER_EXIT
            elif result > 1:
                rv = result - 1
        else:                              # normal DFS：先攒候选
            split_nodes.append(cur)
        if rv is not None:
            return rv

    # normal DFS：候选 >1 时 LLM 两两比对、和分降序（平分保生成序）
    if len(split_nodes) > 1:
        scores = _sum_based_rankn(run, split_nodes)
        for s, nd in zip(scores, split_nodes):
            nd.prior_score = s
        order = sorted(range(len(split_nodes)),
                       key=lambda t: -split_nodes[t].prior_score)
        split_nodes = [split_nodes[t] for t in order]

    for nd in split_nodes:
        result = _dfs(run, nd, max_step, beam, max_q, answer, True)
        if len(run.terminal_nodes) >= answer:
            return _ANSWER_EXIT
        elif result > 1:
            _make_finish(node, 2)
            return result - 1

    return 1


def solve(responses, tool_table, pair_votes, single_chain_max_step=5,
          tree_beam_size=2, max_query_count=20, answer=1,
          with_filter=True, give_up_seed=42):
    run = _Run(responses, tool_table, pair_votes)
    root = run.new_node("Action Input", "", None)
    return_value = _dfs(run, root, single_chain_max_step, tree_beam_size,
                        max_query_count, answer, with_filter)

    answer_node = None
    final_answer = None
    finish_type = None
    for nid in run.terminal_nodes:
        if not run.nodes[nid].pruned:     # 第一个未被剪的终局
            answer_node = nid
            final_answer = run.nodes[nid].description
            finish_type = "give_answer"
            break
    if finish_type is None and run.give_up_nodes:
        rng = random.Random(give_up_seed)
        pos = rng.randint(0, len(run.give_up_nodes) - 1)
        answer_node = run.give_up_nodes[pos]
        final_answer = run.nodes[answer_node].description
        finish_type = "give_up"

    return {"status": run.status,
            "query_count": run.query_count,
            "rank_query_count": run.rank_query_count,
            "total_tokens": run.total_tokens,
            "return_value": return_value,
            "expand_order": list(run.expand_order),
            "expand_nums": {n.nid: n.expand_num for n in run.nodes},
            "node_types": [n.node_type for n in run.nodes],
            "node_depths": [_depth(n) for n in run.nodes],
            "terminal_nodes": list(run.terminal_nodes),
            "give_up_nodes": list(run.give_up_nodes),
            "pruned_node_ids": [n.nid for n in run.nodes if n.pruned],
            "finished_node_ids": [n.nid for n in run.nodes if n.finished],
            "prior_scores": {n.nid: n.prior_score for n in run.nodes},
            "diversity_candidates": {k: [dict(c) for c in v]
                                     for k, v in run.diversity.items()},
            "final_answer": final_answer,
            "finish_type": finish_type,
            "answer_node": answer_node}
