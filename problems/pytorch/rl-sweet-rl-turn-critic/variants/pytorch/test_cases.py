"""SWEET-RL -- turn-level critic ledger tests."""

"""
The oracle is a faithful re-execution of the official scripts
(facebookresearch/sweet_rl scripts/: rank_best_of_n.py,
generate_random_pairs_from_ranks.py, sample_best_of_n.py,
evaluate_code.py). Model calls (vLLM logprobs, the judge RM) are
fully parameterized; the random draws are replayed with Random(seed)
and cross-checked against the official global random.seed(42) order.
"""

# ---------- shared fixtures ----------

EOS = "<|eot_id|>"

# judge prompt_logprobs: None first token, 3 real tokens, EOS, 3 real
# tokens after the eos, then 7 filler tokens of which the LAST 4 are
# dropped by the official [:-4] slice
PL = [None,
      {5: (-0.1, "Hello")},
      {6: (-0.2, " world")},
      {7: (-0.3, EOS)},
      {8: (-0.4, " a")},
      {9: (-0.5, " b")},
      {10: (-0.6, " c")},
      ] + [{i: (-9.0 - i, " x")} for i in range(11, 18)]

# EOS after 1 real token + exactly 4 filler tokens -> tail == [-0.4]
PL_SHORT = [None,
            {5: (-1.0, EOS)},
            {6: (-0.2, " p")},
            {7: (-0.3, EOS)},
            {8: (-0.4, " q")},
            {9: (-9.9, " x")},
            {10: (-9.8, " x")},
            {11: (-9.7, " x")},
            {12: (-9.6, " x")},
            ]

JR_A = [round(0.1 * ((i * 7) % 13) - 0.6, 3) for i in range(16)]
JR_B = [round(0.05 * ((i * 11) % 17), 3) for i in range(16)]

PAIR_DATA = [
    {"input": "task A",
     "additional_outputs": ["o%d" % i for i in range(16)],
     "judge_result": JR_A},
    {"input": "task B",
     "additional_outputs": ["p%d" % i for i in range(16)],
     "judge_result": JR_B},
]


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---------- mode="judge_prompt" ----------


def check_judge_prompt(solve):
    # ground truth 注入（training-time info 是 SWEET-RL 的核心）
    p = solve(mode="judge_prompt", turn_input="turn", output="out",
              ground_truth="gt.py")
    assert p == ("In light that the final answer is: gt.py."
                 "turnout<|start_header_id|>assistant<|end_header_id|>\n\n"), p
    # no_use_ground_truth 分支: 无前缀
    p = solve(mode="judge_prompt", turn_input="t", output="o")
    assert p == "t" + "o" + "<|start_header_id|>assistant<|end_header_id|>\n\n"


# ---------- mode="judge_score" ----------


def check_judge_score(solve):
    # 官方 [:-4] 截尾: PL 有 14 项，保留前 10 位（None 跳过） ->
    # tokens [Hello, world, EOS, a, b, c, x, x, x]，最后 EOS 在 idx 2,
    # tail = [-0.4, -0.5, -0.6, -20.0, -21.0, -22.0]
    mean_score = solve(mode="judge_score", prompt_logprobs=PL,
                       eos_token=EOS)
    assert _close(mean_score, -10.75), mean_score
    sum_score = solve(mode="judge_score", prompt_logprobs=PL,
                      eos_token=EOS, use_sum=True)
    assert _close(sum_score, -64.5), sum_score

    # 恰好 EOS 后 1 个 token + 4 个截尾: tail == [-0.4]
    short = solve(mode="judge_score", prompt_logprobs=PL_SHORT,
                  eos_token=EOS)
    assert _close(short, -0.4), short

    # 两个 EOS: 取最后一个之后
    tail_val = solve(mode="judge_score", prompt_logprobs=PL_SHORT,
                     eos_token=EOS, use_sum=True)
    assert _close(tail_val, -0.4)

    # 无 EOS: find_last_index 返回 -1 -> tail 为全部 logprobs
    # 长度 11，[:-4] 保留前 7 位 -> [None, A, B, C, x(i8), x(i9), x(i10)]
    # tokens [A,B,C,x,x,x] 无 EOS；tail = 全部 6 个 logprob
    # sum = -1.0-0.2-0.3-17.0-18.0-19.0 = -55.5
    pl_noeos = [None,
                {5: (-1.0, "A")},
                {6: (-0.2, "B")},
                {7: (-0.3, "C")},
                ] + [{i: (-9.0 - i, " x")} for i in range(8, 15)]
    full = solve(mode="judge_score", prompt_logprobs=pl_noeos,
                 eos_token=EOS, use_sum=True)
    assert _close(full, -55.5), full


# ---------- mode="rank" ----------


def check_rank(solve):
    # 组 0 相减 [0.5, 0.5, 0.0, 0.5] -> 降序稳定序 [0, 1, 3, 2]
    # 组 1 相减 [-0.5, 0.0, 0.5, 0.0] -> [2, 1, 3, 0]
    idx = solve(mode="rank",
                judge_scores=[1.0, 2.0, 0.5, 1.5, 0.0, 1.0, 2.5, 0.5],
                reference_scores=[0.5, 1.5, 0.5, 1.0, 0.5, 1.0, 2.0, 0.5],
                best_of_n=4)
    assert idx == [[0, 1, 3, 2], [2, 1, 3, 0]], idx

    # 无基线（全 0 参照）时退化为原始分数排序
    idx = solve(mode="rank", judge_scores=[0.1, 0.3, 0.2, 0.0],
                reference_scores=[0.0, 0.0, 0.0, 0.0], best_of_n=4)
    assert idx == [[1, 2, 0, 3]], idx


# ---------- mode="flatten" ----------


def check_flatten(solve):
    hist = [
        {"role": "user", "input": "Fix the bug", "output": "here is mine"},
        {"role": "agent", "output": "draft code"},
        {"role": "user", "input": "still failing", "output": "try again"},
    ]
    recs = solve(mode="flatten", dialogue_history=hist,
                 ground_truth="fixed.py")
    # 只展开带 input 的 turn（agent 输出条目跳过）
    assert len(recs) == 2
    assert recs[0]["input"] == "Fix the bug"
    assert recs[0]["input_with_ground_truth"] == \
        "In light that the final answer is: fixed.py.Fix the bug"
    assert recs[1]["older_output"] == "try again"
    # messages 截到当前 turn
    assert recs[0]["messages"] == hist[:1]
    assert recs[1]["messages"] == hist[:3]


# ---------- mode="pairs" ----------


def check_pairs(solve):
    # seed 42 复现官方 random.seed(42) 的抽签序列（沙箱对拍钉死）
    out = solve(mode="pairs", data=PAIR_DATA, num_pairs=3, seed=42,
                no_prompt=True)
    assert len(out) == 6
    # 每对无 prompt 键，input 拼进 response 侧
    for pair in out:
        assert set(pair) == {"chosen", "rejected"}
        assert pair["chosen"].startswith("task ")
        assert pair["chosen"].endswith(EOS)
    # seed 42 固定：第一对来自 task B 的 p7 vs p13（沙箱对拍锚点）
    assert out[0]["chosen"] == "task Bp7" + EOS, out[0]
    assert out[0]["rejected"] == "task Bp13" + EOS, out[0]

    # 带 prompt 分离: chosen/rejected 只有 output+eos
    out = solve(mode="pairs", data=PAIR_DATA[:1], num_pairs=1, seed=42)
    assert set(out[0]) == {"chosen", "rejected", "prompt"}
    assert out[0]["prompt"] == "task A"
    assert out[0]["chosen"].startswith("o")

    # bestworst: 升序 argsort 的最低分做 rejected、最高分做 chosen
    out = solve(mode="pairs", data=PAIR_DATA[:1], num_pairs=1, seed=42,
                bestworst=True)
    lo = JR_A.index(min(JR_A))
    hi = JR_A.index(max(JR_A))
    assert out[0]["rejected"] == "o%d" % lo + EOS
    assert out[0]["chosen"] == "o%d" % hi + EOS

    # get_sft: 官方 chosen==rejected（同一 output 两次）
    out = solve(mode="pairs", data=PAIR_DATA[:1], num_pairs=1, seed=42,
                get_sft=True)
    assert out[0]["chosen"] == out[0]["rejected"]
    assert set(out[0]) == {"chosen", "rejected", "prompt"}


# ---------- mode="gt_pairs" ----------


def check_gt_pairs(solve):
    # k=3 次重复、2 任务；reshape(3,2) = [[1.0, 0.0], [0.5, 1.0], [0.0, 0.5]]
    # i0: argmax=重复0（1.0），重复1 (0.5<1.0) 与重复2 (0.0<1.0) 各出一对
    # i1: argmax=重复1（1.0），重复0 (0.0) 与重复2 (0.5) 各出一对
    pairs = solve(mode="gt_pairs",
                  rewards=[1.0, 0.0, 0.5, 1.0, 0.0, 0.5], k=3)
    assert pairs == [
        {"chosen_index": 0, "rejected_index": 2},
        {"chosen_index": 0, "rejected_index": 4},
        {"chosen_index": 3, "rejected_index": 1},
        {"chosen_index": 3, "rejected_index": 5},
    ], pairs

    # reward == max 的非 argmax 重复**不**出对（官方严格 <）
    # [1,1,1,0,0,0] reshape(3,2) = [[1,1],[1,0],[0,0]]:
    #   i0: [1,1,0] argmax=0, j1 的 1.0 不小于 max -> 不出对; j2 的 0.0 -> (0,4)
    #   i1: [1,0,0] argmax=0, (1,3) 与 (1,5)
    pairs = solve(mode="gt_pairs",
                  rewards=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], k=3)
    assert pairs == [
        {"chosen_index": 0, "rejected_index": 4},
        {"chosen_index": 1, "rejected_index": 3},
        {"chosen_index": 1, "rejected_index": 5},
    ], pairs

    # 任务内全同分（无严格小于）: reshape 后两任务各自全 1 / 全 0 -> 0 对
    assert solve(mode="gt_pairs",
                 rewards=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], k=3) == []

    # k=1: 每任务只有一次重复，无对
    assert solve(mode="gt_pairs", rewards=[1.0, 0.0], k=1) == []


TESTS = [
    check_judge_prompt,
    check_judge_score,
    check_rank,
    check_flatten,
    check_pairs,
    check_gt_pairs,
]
