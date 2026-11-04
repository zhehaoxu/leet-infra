"""ARPO（arXiv:2507.19849）熵感知 rollout 分支账 -- 测试用例。"""
EXPECTED = {
    'entropy_single': {'entropy': 0.4373631522, 'monitor': 0.4373631522, 'initial': 0.4373631522},
    'entropy_calc_empty': 0.0,
    'beam_prob_rising': 0.6,
    'beam_prob_falling_cap': 0.9,
    'beam_basic': {'branches': [[3, 0], [4, 0]], 'rps': {'0': 4}},
    'beam_slot_cap': {'branches': [[3, 0]], 'rps': {'0': 3}},
    'beam_size1': {'branches': [], 'rps': {'0': 1}},
    'refill_basic': {'refills': [[1, 5], [2, 6]], 'rps': {'0': 4, '1': 2, '2': 4}},
    'round_tool': ['tool', 3, []],
    'round_limit': ['limit_eos', 5, [99]],
    'round_length': ['continue', 0, []],
    'round_length_full': ['stop', 0, []],
    'round_eos': ['stop', 0, []],
    'reorg_basic': {'seqs': [[7, 8], [9], [5, 6], [30]], 'masks': [[1, 1], [1], [1, 1], [1]]},
    'reorg_dup_fill': {'seqs': [[7, 8], [7, 8], [7, 8]], 'masks': [[1, 1], [1, 1], [1, 1]]},
    'entropy_short_fallback': 0.5209891474,
    'branch_prob_eq': {'branches': [[3, 0], [3, 0]], 'rps': {'0': 4}},
    'refill_active_skip': {'refills': [], 'rps': {'0': 2}},
}

"""ARPO（arXiv:2507.19849）熵感知 rollout 分支账 -- 测试用例（期望值由官方
vllm_rollout_with_tools.py exec 沙箱实测钉死：熵监测段逐行抽取 + 自适应
分支/补支/回合控制/输出重组直通 exec，vLLM 引擎/工具执行器/日志剥除、
rollout 输出与随机源全入参）。"""
import math

VOCAB = 100
NORM = math.log(VOCAB)

TOPK_LPS = [
    [-0.1, -2.3, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
    [-0.2, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5],
    [-0.05, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
]


def make_output(token_ids, per_token_topk):
    # 官方表面：output.outputs[0].token_ids / .logprobs[j]（rank -> logprob
    # 的映射）；判题域简化为 dict（token_ids + per_token_topk 值列表）
    return {"token_ids": token_ids, "per_token_topk": per_token_topk}


def topk_dicts(lps):
    return [{r: lp for r, lp in enumerate(l)} for l in lps]


def const_rand(seq):
    it = iter(seq)
    return lambda: next(it)


# ---------- mode="entropy"：top-10 扁平 + 归一化 + 初始熵记录 ----------

def check_entropy_single(solve):
    out = make_output([10, 11, 12], topk_dicts(TOPK_LPS))
    init = {}
    cur, init2 = solve(mode="entropy", outputs=[out], active_indices=[7],
                       initial_entropy_dict=init, vocab_size=VOCAB)
    # 扁平 top-10 的 -Σ p·l / log(vocab_size)；首见即初始熵
    assert round(cur[7], 10) == EXPECTED["entropy_single"]["entropy"], cur
    assert round(init2[7], 10) == EXPECTED["entropy_single"]["initial"], init2
    # 手工核对一遍（独立第二条线）
    flat = [lp for lps in TOPK_LPS for lp in lps]
    man = -sum(math.exp(l) * l for l in flat) / NORM
    assert abs(cur[7] - man) < 1e-12, cur


def check_entropy_initial_frozen(solve):
    # initial_entropy_dict 只在首见时写：再次监测不覆盖（delta 对首次观测）
    out1 = make_output([10, 11, 12], topk_dicts(TOPK_LPS))
    lps2 = [[-0.01, -5.0] * 5, [-0.02, -4.0] * 5]
    out2 = make_output([13, 14], topk_dicts(lps2))
    init = {}
    cur1, init = solve(mode="entropy", outputs=[out1], active_indices=[7],
                       initial_entropy_dict=init, vocab_size=VOCAB)
    e_first = cur1[7]
    cur2, init = solve(mode="entropy", outputs=[out2], active_indices=[7],
                       initial_entropy_dict=init, vocab_size=VOCAB)
    assert cur2[7] != e_first, (cur1, cur2)
    assert init[7] == e_first, init  # 初始熵保持首见值
    # 空 logprobs 的 rollout -> 熵 0（官方 guard）
    out_empty = make_output([], [])
    cur3, _ = solve(mode="entropy", outputs=[out_empty], active_indices=[8],
                    initial_entropy_dict=init, vocab_size=VOCAB)
    assert cur3[8] == 0.0, cur3


def check_entropy_short_fallback(solve):
    # per_token 只有 2 组、tokens 3 个：j=2 取不到 -> 用最后一个（j=1）
    out_fb = make_output([10, 11, 12], topk_dicts(TOPK_LPS[:2]))
    cur, _ = solve(mode="entropy", outputs=[out_fb], active_indices=[7],
                   initial_entropy_dict={}, vocab_size=VOCAB)
    assert round(cur[7], 10) == round(EXPECTED["entropy_short_fallback"], 10), \
        cur
    # 独立核对：flat = 组0 + 组1 + 组1（回退）
    flat = TOPK_LPS[0] + TOPK_LPS[1] + TOPK_LPS[1]
    man = -sum(math.exp(l) * l for l in flat) / NORM
    assert abs(cur[7] - man) < 1e-12, cur


# ---------- mode="branch"：熵 delta 门控的自适应分支 ----------

def check_branch_basic(solve):
    # 样本 0 两个活跃 rollout [3,4]；num_samples=4、rps=2 -> 2 个名额；
    # beam_size=3 -> 每 idx 最多 2 支
    cur_ent = {3: 0.9, 4: 0.5}
    init_ent = {3: 0.5, 4: 0.7}
    # idx3: rand .5 -> prob .3 <= .5 分支；idx3 二次 rand .9 -> .7 > .5 跳；
    # idx4: rand .2 -> delta -0.2 -> prob .3 分支
    branches, rps = solve(mode="branch", active_by_sample={0: [3, 4]},
                          current_entropy_dict=cur_ent,
                          initial_entropy_dict=init_ent,
                          rollouts_per_sample={0: 2}, num_samples=4,
                          beam_size=3, branch_probability=0.5,
                          entropy_weight=0.5, rand_fn=const_rand([0.5, 0.9, 0.2]))
    assert [list(b) for b in branches] == EXPECTED["beam_basic"]["branches"], branches
    
    # 熵升的 idx3 比熵降的 idx4 更早获得分支（delta 语义顺序验证）
    assert branches[0][0] == 3 and branches[1][0] == 4, branches


def check_branch_prob_eq(solve):
    # prob 恰等于 branch_probability：官方严格 > -> 分支（两次 rand 都 0.5）
    branches, rps = solve(mode="branch", active_by_sample={0: [3]},
                          current_entropy_dict={3: 0.7},
                          initial_entropy_dict={3: 0.7},
                          rollouts_per_sample={0: 2}, num_samples=4,
                          beam_size=3, branch_probability=0.5,
                          entropy_weight=0.5, rand_fn=const_rand([0.5, 0.5]))
    assert [list(b) for b in branches] == \
        EXPECTED["branch_prob_eq"]["branches"], branches
    assert {str(k): v for k, v in rps.items()} == \
        EXPECTED["branch_prob_eq"]["rps"], rps


def check_branch_caps(solve):
    cur_ent = {3: 0.9, 4: 0.5}
    init_ent = {3: 0.5, 4: 0.7}
    # remaining_slots=1：idx3 用掉唯一名额后 idx4 break
    branches, rps = solve(mode="branch", active_by_sample={0: [3, 4]},
                          current_entropy_dict=cur_ent,
                          initial_entropy_dict=init_ent,
                          rollouts_per_sample={0: 2}, num_samples=3,
                          beam_size=3, branch_probability=0.5,
                          entropy_weight=0.5, rand_fn=const_rand([0.0, 0.0, 0.0]))
    assert [list(b) for b in branches] == EXPECTED["beam_slot_cap"]["branches"], branches
    assert {str(k): v for k, v in rps.items()} == EXPECTED["beam_slot_cap"]["rps"], rps
    # beam_size=1 -> branches_per_idx = min(0, ...) = 0 -> 无分支
    branches2, rps2 = solve(mode="branch", active_by_sample={0: [3]},
                            current_entropy_dict=cur_ent,
                            initial_entropy_dict=init_ent,
                            rollouts_per_sample={0: 1}, num_samples=4,
                            beam_size=1, branch_probability=0.5,
                            entropy_weight=0.5, rand_fn=const_rand([0.0]))
    assert branches2 == EXPECTED["beam_size1"]["branches"], branches2
    
    # rps 已满 -> remaining_slots <= 0 -> 直接跳过
    branches3, rps3 = solve(mode="branch", active_by_sample={0: [3]},
                            current_entropy_dict=cur_ent,
                            initial_entropy_dict=init_ent,
                            rollouts_per_sample={0: 4}, num_samples=4,
                            beam_size=3, branch_probability=0.5,
                            entropy_weight=0.5, rand_fn=const_rand([0.0]))
    assert branches3 == [] and rps3 == {0: 4}, (branches3, rps3)


# ---------- mode="refill"：不活跃样本每轮补 1 支 ----------

def check_refill(solve):
    stI = {0: [3, 4, 9, 10], 1: [5], 2: [6, 7]}
    refills, rps = solve(mode="refill", active_by_sample={0: [3, 4]},
                         rollouts_per_sample={0: 4, 1: 1, 2: 3},
                         num_samples=4, sample_to_indices=stI)
    # 样本 0 活跃不补；1 未满补 1（模板=首下标 5）；2 未满补 1（模板 6）
    assert [list(b) for b in refills] == EXPECTED["refill_basic"]["refills"], refills
    assert {str(k): v for k, v in rps.items()} == EXPECTED["refill_basic"]["rps"], rps


def check_refill_active_skip(solve):
    # 活跃样本即使未满也不由 refill 补支（refill 只管不在场的样本）
    refills, rps = solve(mode="refill", active_by_sample={0: [3, 4]},
                         rollouts_per_sample={0: 2}, num_samples=4,
                         sample_to_indices={0: [3, 4, 9, 10]})
    assert refills == [], refills
    assert rps == {0: 2}, rps


# ---------- mode="round"：工具限额 / EOS / length ----------

def check_round(solve):
    kw = dict(mode="round", stop_sequences=["</search>"], eos_token_id=99,
              max_len=100)
    # 工具调用：计数器未满 -> tool、counter+1
    assert list(solve(finish_reason="stop", stop_reason="</search>",
                      call_counter=2, tool_call_limit=5, response_len=10,
                      **kw)) == EXPECTED["round_tool"]
    # 满额 -> limit_eos：追加 EOS 终止
    assert list(solve(finish_reason="stop", stop_reason="</search>",
                      call_counter=5, tool_call_limit=5, response_len=10,
                      **kw)) == EXPECTED["round_limit"]
    # length 且未到 max_len -> continue；已到 -> stop
    assert list(solve(finish_reason="length", stop_reason=None,
                      call_counter=0, tool_call_limit=5, response_len=99,
                      **kw)) == EXPECTED["round_length"]
    assert list(solve(finish_reason="length", stop_reason=None,
                      call_counter=0, tool_call_limit=5, response_len=100,
                      **kw)) == EXPECTED["round_length_full"]
    # EOS -> stop
    assert list(solve(finish_reason="stop", stop_reason=None,
                      call_counter=0, tool_call_limit=5, response_len=10,
                      **kw)) == EXPECTED["round_eos"]


# ---------- mode="reorg"：每样本前 num_samples 个 + 复制末位补齐 ----------

def check_reorg(solve):
    curr = {
        0: [1, 2],
        3: [1, 2, 7, 8],
        4: [1, 2, 9],
        5: [3, 4, 5, 6],
        9: [1, 2, 30],
    }
    masks = {3: [1, 1], 4: [1], 5: [1, 1], 9: [1]}
    seqs, omasks = solve(mode="reorg", batch_size=1,
                         sample_to_indices={0: [3, 4, 5, 9]}, num_samples=4,
                         curr_inputs=curr, result_masks=masks,
                         prompt_lens={0: 2})
    assert seqs == EXPECTED["reorg_basic"]["seqs"], seqs
    assert omasks == EXPECTED["reorg_basic"]["masks"], omasks
    # 不足 num_samples -> 复制最后一个补齐
    seqs2, omasks2 = solve(mode="reorg", batch_size=1,
                           sample_to_indices={0: [3]}, num_samples=3,
                           curr_inputs=curr, result_masks=masks,
                           prompt_lens={0: 2})
    assert seqs2 == EXPECTED["reorg_dup_fill"]["seqs"], seqs2
    assert omasks2 == EXPECTED["reorg_dup_fill"]["masks"], omasks2
