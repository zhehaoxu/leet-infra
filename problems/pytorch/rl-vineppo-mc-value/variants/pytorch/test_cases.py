"""VinePPO（arXiv:2410.01679）步级 Monte-Carlo 回报账 -- 测试用例（期望值由
官方 VinePPO 源码 exec 沙箱实测钉死：stub Registrable（per-base-class 注册
表）/ StubTask（answer -> pred 映射）/ StubEnc（HF BatchEncoding 表面），
torch/Ray/多进程剥除，score/advantage/paths/聚合/过滤/字符级分摊全入账）."""
import json

# ---------- fixtures: a reasoning tree ----------
# root (query) -> step1 -> {step2a(leaf ok), step2b -> {leaf bad}}
# plus an early-stopped leaf directly under root


def make_tree():
    return {
        "text": "Q: 2+2?",
        "children": [
            {
                "text": "Step 1: We add 2 and 2 to get 4.",
                "children": [
                    {"text": "Step 2a: So the answer is 4.",
                     "answer": "So the answer is 4.",
                     "pred": "4"},
                    {
                        "text": "Step 2b: We multiply instead.",
                        "children": [
                            {"text": "Step 3b: The answer is 5.",
                             "answer": "The answer is 5.",
                             "pred": "5"},
                        ],
                    },
                ],
            },
            {
                "text": "Shortcut: The answer is 4.",
                "answer": "The answer is 4.",
                "pred": "4",
            },
        ],
    }


# 官方口径：叶 score 从 node["answer"] 经 task 提取——本映射只认两个
# step 叶的 answer；Shortcut 叶的 answer "The answer is 4." 不在映射 ->
# 提取 None -> score 0（其自带 "pred": "4" 字段完全不参与）。
PRED_MAP = {"So the answer is 4.": "4", "The answer is 5.": "5"}


def scored_tree(solve):
    tree = solve(mode="score", tree=make_tree(), gold="4", pred_map=PRED_MAP)
    return solve(mode="advantage", tree=tree)


# ---------- mode="score"/"advantage"：compute_score / compute_advantage ----------

def check_score_full(solve):
    tree2 = make_tree()
    solve(mode="score", tree=tree2, gold="4", pred_map=PRED_MAP)
    # solve 契约：返回新树、不就地修改入参
    assert tree2 == make_tree(), "input tree was mutated"
    tree = solve(mode="score", tree=make_tree(), gold="4", pred_map=PRED_MAP)
    assert json.dumps(tree, sort_keys=True) == EXPECTED["score_full"], tree
    # node["pred"] 字段不参与：置 None 后 score 仍由 answer 提取决定
    t2 = make_tree()
    t2["children"][0]["children"][0]["pred"] = None
    t2 = solve(mode="score", tree=t2, gold="4", pred_map=PRED_MAP)
    assert json.dumps(t2, sort_keys=True) == EXPECTED["score_none_pred"], t2
    # 全 mismatch：叶全 0、内部均值也 0
    t3 = solve(mode="score", tree=make_tree(), gold="9", pred_map=PRED_MAP)
    assert json.dumps(t3, sort_keys=True) == EXPECTED["score_no_match"], t3


def check_score_childless(solve):
    # 无 answer 且无 children 的节点：走 else 分支 score=0、不写
    # is_correct_answer（叶判定是 "answer" in node，不是无 children）
    tree = {"text": "root", "children": [
        {"text": "a", "answer": "a"},
        {"text": "empty"},
    ]}
    out = solve(mode="score", tree=tree, gold="g", pred_map={})
    assert out["score"] == 0.0 and out["children"][0]["score"] == 0, out
    assert out["children"][1]["score"] == 0, out
    assert "is_correct_answer" not in out["children"][1], out


def check_advantage_full(solve):
    # 期望树里 advantage 全部是「child.score − parent.score」的树形差分
    out = scored_tree(solve)
    assert json.dumps(out, sort_keys=True) == EXPECTED["adv_full"], out
    # root 无 advantage、叶有 advantage（compute_advantage 不跳过叶）
    assert "advantage" not in out, out
    leaf = out["children"][1]
    assert leaf["advantage"] == 0 - 0.25, leaf
    # childless 内部节点：get("children", []) 默认空、无崩溃
    t = {"text": "root", "score": 1.0}
    assert solve(mode="advantage", tree=t) == {"text": "root", "score": 1.0}
EXPECTED = {
    'score_full': '{"children": [{"children": [{"answer": "So the answer is 4.", "is_correct_answer": 1, "pred": "4", "score": 1, "text": "Step 2a: So the answer is 4."}, {"children": [{"answer": "The answer is 5.", "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}], "score": 0.0, "text": "Step 2b: We multiply instead."}], "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"answer": "The answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}], "score": 0.25, "text": "Q: 2+2?"}',
    'score_none_pred': '{"children": [{"children": [{"answer": "So the answer is 4.", "is_correct_answer": 1, "pred": null, "score": 1, "text": "Step 2a: So the answer is 4."}, {"children": [{"answer": "The answer is 5.", "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}], "score": 0.0, "text": "Step 2b: We multiply instead."}], "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"answer": "The answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}], "score": 0.25, "text": "Q: 2+2?"}',
    'score_no_match': '{"children": [{"children": [{"answer": "So the answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Step 2a: So the answer is 4."}, {"children": [{"answer": "The answer is 5.", "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}], "score": 0.0, "text": "Step 2b: We multiply instead."}], "score": 0.0, "text": "Step 1: We add 2 and 2 to get 4."}, {"answer": "The answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}], "score": 0.0, "text": "Q: 2+2?"}',
    'adv_full': '{"children": [{"advantage": 0.25, "children": [{"advantage": 0.5, "answer": "So the answer is 4.", "is_correct_answer": 1, "pred": "4", "score": 1, "text": "Step 2a: So the answer is 4."}, {"advantage": -0.5, "children": [{"advantage": 0.0, "answer": "The answer is 5.", "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}], "score": 0.0, "text": "Step 2b: We multiply instead."}], "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"advantage": -0.25, "answer": "The answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}], "score": 0.25, "text": "Q: 2+2?"}',
    'paths_basic': '[{"node_chain": [{"score": 0.25, "text": "Q: 2+2?"}, {"advantage": 0.25, "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"advantage": 0.5, "answer": "So the answer is 4.", "is_correct_answer": 1, "pred": "4", "score": 1, "text": "Step 2a: So the answer is 4."}]}, {"node_chain": [{"score": 0.25, "text": "Q: 2+2?"}, {"advantage": 0.25, "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"advantage": -0.5, "score": 0.0, "text": "Step 2b: We multiply instead."}, {"advantage": 0.0, "answer": "The answer is 5.", "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}]}, {"node_chain": [{"score": 0.25, "text": "Q: 2+2?"}, {"advantage": -0.25, "answer": "The answer is 4.", "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}]}]',
    'paths_importance': '[{"node_chain": [{"importance_weight": 8, "score": 0.25, "text": "Q: 2+2?"}, {"advantage": 0.25, "importance_weight": 4, "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"advantage": 0.5, "answer": "So the answer is 4.", "importance_weight": 2, "is_correct_answer": 1, "pred": "4", "score": 1, "text": "Step 2a: So the answer is 4."}]}, {"node_chain": [{"importance_weight": 8, "score": 0.25, "text": "Q: 2+2?"}, {"advantage": 0.25, "importance_weight": 4, "score": 0.5, "text": "Step 1: We add 2 and 2 to get 4."}, {"advantage": -0.5, "importance_weight": 2, "score": 0.0, "text": "Step 2b: We multiply instead."}, {"advantage": 0.0, "answer": "The answer is 5.", "importance_weight": 1, "is_correct_answer": 0, "pred": "5", "score": 0, "text": "Step 3b: The answer is 5."}]}, {"node_chain": [{"importance_weight": 8, "score": 0.25, "text": "Q: 2+2?"}, {"advantage": -0.25, "answer": "The answer is 4.", "importance_weight": 4, "is_correct_answer": 0, "pred": "4", "score": 0, "text": "Shortcut: The answer is 4."}]}]',
    'paths_repeat_early': '[3, 3, 4, 2, 2, 2, 2]',
    'agg_topk_third': '[["Q: 2+2?", "Step 1: We add 2 and 2 to get 4.", "Step 2a: So the answer is 4."]]',
    'agg_topk_twothird': '[["Q: 2+2?", "Step 1: We add 2 and 2 to get 4.", "Step 2a: So the answer is 4."], ["Q: 2+2?", "Step 1: We add 2 and 2 to get 4.", "Step 2b: We multiply instead.", "Step 3b: The answer is 5."]]',
    'agg_maxk_members': '[["Q: 2+2?", "Shortcut: The answer is 4."], ["Q: 2+2?", "Step 1: We add 2 and 2 to get 4.", "Step 2a: So the answer is 4."]]',
    'filter_flags': [True, False, False],
    'filter_nonzero_score': [True, True, False],
    'filter_nz_last': [True, False, False],
    'filter_z_last': [False, True, True],
    'filter_nz_adv': [True, False, False],
    'adv_single': {'char': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    'adv_multi': {'char': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    'adv_three_nodes': {'char': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75], 'response_text': '\nAA\nBBB\nCCCC'},
    'adv_multichar_token': {'clean_adv': [0.5, 0.5, 0.5, 1.0, 1.0], 'token_adv_full': [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]},
    'adv_sentinel': {'char': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -777.0, -777.0, -777.0], 'clean_token_adv': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -777.0, -777.0, -777.0]},
    'adv_multi_sentinel': {'char': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -777.0, -777.0, -777.0]},
}


# ---------- mode="paths"：extract_paths_from_tree ----------

def check_paths_basic(solve):
    tree = scored_tree(solve)
    paths = solve(mode="paths", tree=tree)
    assert json.dumps(paths, sort_keys=True) == EXPECTED["paths_basic"], paths
    # 官方在内部 deepcopy：外部树的 children 不被 pop("children") 掏空
    assert "children" in tree, "input tree was mutated"


def check_paths_importance(solve):
    paths = solve(mode="paths", tree=scored_tree(solve),
                  include_importance_weights=True, branch_factor=2,
                  max_depth=3)
    assert json.dumps(paths, sort_keys=True) == EXPECTED["paths_importance"], \
        paths


def check_paths_repeat_early(solve):
    paths = solve(mode="paths", tree=scored_tree(solve),
                  include_importance_weights=True,
                  repeat_early_stopped_paths=True, branch_factor=2,
                  max_depth=3)
    # 3 条常规路径（Step2a 浅叶 depth2<3 也按 weight 2 重复）+ Shortcut 叶
    # （depth 1）按 importance_weight=2^2 重复
    assert json.dumps([len(p["node_chain"]) for p in paths]) == \
        EXPECTED["paths_repeat_early"], paths


# ---------- mode="aggregate"：PathAggregator 注册类 ----------

def jkey(p):
    return json.dumps(p, sort_keys=True)


def check_agg_none_unique(solve):
    paths = json.loads(EXPECTED["paths_basic"])
    # none：原列表原样返回
    assert solve(mode="aggregate", variant="none", paths=paths) is paths
    # unique：3 条互异路径全保留；整条 path sort_keys 序列化进 set 去重
    uniq = solve(mode="aggregate", variant="unique", paths=paths)
    assert len(uniq) == 3, uniq
    dup = paths + [json.loads(jkey(paths[0]))]
    dup_u = solve(mode="aggregate", variant="unique", paths=dup)
    assert sorted(map(jkey, dup_u)) == sorted(map(jkey, paths)), dup_u
    # 键序不同但内容相同的重复也坍缩（sort_keys 序列化的意义所在）
    def rev_keys(d):
        return {k: rev_keys(v) if isinstance(v, dict) else
                ([rev_keys(x) if isinstance(x, dict) else x for x in v]
                 if isinstance(v, list) else v)
                for k, v in reversed(list(d.items()))}
    dup2 = paths + [rev_keys(json.loads(jkey(paths[1])))]
    dup2_u = solve(mode="aggregate", variant="unique", paths=dup2)
    assert sorted(map(jkey, dup2_u)) == sorted(map(jkey, paths)), dup2_u


def check_agg_topk(solve):
    paths = json.loads(EXPECTED["paths_basic"])
    kw = dict(mode="aggregate", variant="top_k_percent")
    # 3 条路径取 50% -> int(1.5)=1（mean 与 sum 同数）
    assert len(solve(paths=paths, top_k_percent=0.5, **kw)) == 1
    assert len(solve(paths=paths, top_k_percent=0.5, reduction="sum",
                     **kw)) == 1
    # 10% -> int(0.3)=0：keep_at_least_one 撑到 1 / False 时空
    assert len(solve(paths=paths, top_k_percent=0.1, **kw)) == 1
    assert len(solve(paths=paths, top_k_percent=0.1, keep_at_least_one=False,
                     **kw)) == 0
    # 空输入原样返回
    assert solve(paths=[], top_k_percent=0.5, **kw) == []
    # 1/3 -> 按路径 advantage 均值降序取 1 条（Step2a 路：mean(0, 0.25, 0.5)）
    kept = solve(paths=paths, top_k_percent=1 / 3, **kw)
    assert json.dumps([[n.get("text", "ROOT") for n in p["node_chain"]]
                       for p in kept], sort_keys=True) == \
        EXPECTED["agg_topk_third"], kept
    # 2/3 -> 取 2 条：根节点 advantage 记 0 且计入 mean 分母（第 2/3 名是
    # 2b 链 p1，均值 -0.0625 > p2 的 -0.125；根记非 0 会翻成 p2）
    kept2 = solve(paths=paths, top_k_percent=2 / 3, **kw)
    assert json.dumps([[n.get("text", "ROOT") for n in p["node_chain"]]
                       for p in kept2], sort_keys=True) == \
        EXPECTED["agg_topk_twothird"], kept2


def check_agg_maxk(solve):
    paths = json.loads(EXPECTED["paths_basic"])
    # k=2 < 3：random.Random(42).sample 确定性采样
    got = solve(mode="aggregate", variant="max_k_paths", paths=paths, k=2)
    assert len(got) == 2, got
    # 采样成员被官方沙箱钉死（rng.Random(42) 每实例重置）
    got_members = json.dumps(
        [[n.get("text", "ROOT") for n in p["node_chain"]] for p in got],
        sort_keys=True)
    assert got_members == EXPECTED["agg_maxk_members"], got_members
    # k=10 >= 3：原样返回（同一列表、顺序不变）
    got_all = solve(mode="aggregate", variant="max_k_paths", paths=paths, k=10)
    assert got_all == paths, got_all
    # k=3 == len：边界走 <= 原样返回；若走 sample 会乱序
    got_k3 = solve(mode="aggregate", variant="max_k_paths", paths=paths, k=3)
    assert got_k3 == paths, got_k3


# ---------- mode="filter"：PathFilter 注册类 ----------

def check_filters(solve):
    paths = json.loads(EXPECTED["paths_basic"])
    cases = [
        ("none", [True, True, True]),
        # successful：末节点 is_correct_answer（路径 0 对、其余错）
        ("successful", EXPECTED["filter_flags"]),
        # non_zero_score：看 node_chain[1]（第一步）——路径 1 末节点 0 但
        # 第一步 0.5 -> True
        ("non_zero_score", EXPECTED["filter_nonzero_score"]),
        ("non_zero_last_step_score", EXPECTED["filter_nz_last"]),
        ("zero_last_step_score", EXPECTED["filter_z_last"]),
        ("non_zero_last_step_advantage", EXPECTED["filter_nz_adv"]),
    ]
    for variant, exp in cases:
        got = [bool(solve(mode="filter", variant=variant, paths=p))
               for p in paths]
        assert got == [bool(x) for x in exp], (variant, got)
    # 末节点无 answer 的路径（childless 无 answer 叶）：successful 必 False
    bare = {"node_chain": [{"text": "root", "score": 0.25},
                           {"text": "bare step", "score": 0}]}
    assert solve(mode="filter", variant="successful", paths=bare) is False
    assert solve(mode="filter", variant="non_zero_last_step_score",
                 paths=bare) is False
    assert solve(mode="filter", variant="zero_last_step_score",
                 paths=bare) is True


# ---------- mode="advantage_split"：字符级分摊 + token 清理 ----------

def enc_for(response_text):
    """StubEnc：首 token 是裸 "▁"（offset [0,0]），之后每字符一 token。"""
    n = len(response_text)
    return (list(range(n + 1)), [[0, 0]] + [[i, i + 1] for i in range(n)],
            ["▁"] + list(response_text))


NODE_CHAIN_SINGLE = [
    {"text": "Q"},
    {"text": "Shortcut: The answer is 4.", "advantage": 1.0},
]
RESP_SINGLE = "\n" + NODE_CHAIN_SINGLE[1]["text"]

NODE_CHAIN_MULTI = [
    {"text": "Q"},
    {"text": "Step 1: We add 2 and 2 to get 4.", "advantage": 0.5},
    {"text": "Step 2a: So the answer is 4.", "advantage": 1.0},
]
RESP_MULTI = ("\n" + NODE_CHAIN_MULTI[1]["text"] + "\n"
              + NODE_CHAIN_MULTI[2]["text"])


def check_adv_split_single(solve):
    ids, offsets, tokens = enc_for(RESP_SINGLE)
    char, tok, clean_ids, clean_adv = solve(
        mode="advantage_split", node_chain=NODE_CHAIN_SINGLE,
        response_text=RESP_SINGLE, input_ids=ids, offset_mapping=offsets,
        tokens=tokens)
    # 单节点：首节点 text = "\n" + text，恰好填满 -> 无 -777 残留
    assert char == EXPECTED["adv_single"]["char"], char
    assert tok == char + [char[0]], tok  # token[i] = char[start_i]
    assert clean_ids == list(range(1, len(RESP_SINGLE) + 1)), clean_ids
    assert clean_adv == char, clean_adv


def check_adv_split_multi(solve):
    ids, offsets, tokens = enc_for(RESP_MULTI)
    char, tok, clean_ids, clean_adv = solve(
        mode="advantage_split", node_chain=NODE_CHAIN_MULTI,
        response_text=RESP_MULTI, input_ids=ids, offset_mapping=offsets,
        tokens=tokens)
    # 多节点：首段 "\n"+text+"\n"（含尾换行）为 0.5 段，末段无换行为 1.0 段
    assert char == EXPECTED["adv_multi"]["char"], char
    # 逐段边界：0.5 段长 = 1 + len(step1) + 1
    n_head = 1 + len(NODE_CHAIN_MULTI[1]["text"]) + 1
    assert char[:n_head] == [0.5] * n_head, char[:n_head]
    assert char[n_head:] == [1.0] * (len(RESP_MULTI) - n_head), char[n_head:]


def check_adv_split_three(solve):
    # 三个响应节点：首 "\n"+t+"\n"、中间 t+"\n"、末 t
    node_chain = [{"text": "Q"},
                  {"text": "AA", "advantage": 0.25},
                  {"text": "BBB", "advantage": 0.5},
                  {"text": "CCCC", "advantage": 0.75}]
    resp = "\nAA\nBBB\nCCCC"
    n = len(resp)
    char, tok, clean_ids, clean_adv = solve(
        mode="advantage_split", node_chain=node_chain, response_text=resp,
        input_ids=list(range(n + 1)),
        offset_mapping=[[0, 0]] + [[i, i + 1] for i in range(n)],
        tokens=["▁"] + list(resp))
    assert char == EXPECTED["adv_three_nodes"]["char"], char
    assert tok == [0.25] + char, tok  # token[i] = char[start_i]，首 offset [0,0]
    assert char[:4] == [0.25] * 4 and char[4:8] == [0.5] * 4 \
        and char[8:] == [0.75] * 4, char


def check_adv_split_multi_sentinel(solve):
    # 多节点 + 3 字符哨兵尾巴：末段是标量广播赋值，写入到恰好 node 拼接
    # 长度为止、不吞哨兵槽（末段若多写 1 字符会吃掉第一个 -777 槽）
    resp = RESP_MULTI + "\n\n\n"
    n = len(resp)
    char, tok, clean_ids, clean_adv = solve(
        mode="advantage_split", node_chain=NODE_CHAIN_MULTI,
        response_text=resp, input_ids=list(range(n + 1)),
        offset_mapping=[[0, 0]] + [[i, i + 1] for i in range(n)],
        tokens=["▁"] + list(resp))
    assert char == EXPECTED["adv_multi_sentinel"]["char"], char
    assert char[-3:] == [-777.0] * 3 and char[-4] == 1.0, char


def check_adv_split_multichar(solve):
    # token 覆盖 [30,36) 横跨 0.5/1.0 边界 -> 只取起始字符 char[30]=0.5；
    # [34,37) 起点恰在 1.0 段 -> 1.0
    offsets = [[0, 0], [0, 4], [4, 5], [30, 36], [34, 37], [40, 42]]
    tok_adv = solve(mode="advantage_split", node_chain=NODE_CHAIN_MULTI,
                    response_text=RESP_MULTI, input_ids=[0, 1, 2, 3, 4, 5],
                    offset_mapping=offsets,
                    tokens=["▁", "ABCD", "E", "FGHIJK", "LMN", "OP"])
    assert tok_adv[1] == EXPECTED["adv_multichar_token"]["token_adv_full"], \
        tok_adv[1]
    assert tok_adv[3] == EXPECTED["adv_multichar_token"]["clean_adv"], \
        tok_adv[3]
    assert tok_adv[2] == [1, 2, 3, 4, 5], tok_adv[2]


def check_adv_split_sentinel(solve):
    # response_text 比 node 拼接多 3 个尾部字符 -> 字符账尾部留 -777，
    # 覆盖这些字符的 token 账同样 -777
    resp = RESP_SINGLE + "\n\n\n"
    ids, offsets, tokens = enc_for(resp)
    char, tok, clean_ids, clean_adv = solve(
        mode="advantage_split", node_chain=NODE_CHAIN_SINGLE,
        response_text=resp, input_ids=ids, offset_mapping=offsets,
        tokens=tokens)
    assert char == EXPECTED["adv_sentinel"]["char"], char
    assert char[-3:] == [-777.0] * 3 and char[:27] == [1.0] * 27, char
    assert clean_adv == EXPECTED["adv_sentinel"]["clean_token_adv"], clean_adv
    assert clean_adv[-3:] == [-777.0] * 3, clean_adv
