"""GMPO（arXiv:2507.20673）token 几何平均策略梯度账 -- 测试用例（期望值由
官方 train_zero_math_gmpo.py exec 沙箱实测钉死，torch 2.10.0 float64）."""

R1 = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # seq1: eos reward 1
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # seq2: 0
      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],   # seq3: 0.5
      [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],   # seq4: 2
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # seq5: 0
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]   # seq6: 0

R2 = [[1.0] * 3, [0.0] * 3, [0.5] * 3, [0.25] * 3,
      [2.0] * 3, [0.0] * 3, [1.0] * 3, [0.0] * 3]   # num_samples=4 两组

DIFFS = [0.05, -0.3, 0.5, -0.02, 0.0]   # token logprob diffs; 0.5 超出 clip 0.2
MASK = [1, 1, 1, 1, 0]                   # 末位 padding

BIG = [0.9, -0.9, 0.9, -0.9, 0.9]       # 长序列大 diff：sum vs 逐 token clamp 分歧

STATS_KEYS = ("logprobs_diff_max", "logprobs_diff_min", "zero_pg_loss_count")


# ---------- mode="advantage"：compute_monte_carlo_advantages ----------

def check_advantage_basic(solve):
    # base / drgrpo：纯 dev（同值）；grpo：组内 std（unbiased）+1e-8 归一
    assert solve(mode="advantage", rewards=R1, num_samples=3) == \
        [0.5, -0.5, 0.0, 1.3333333333333335, -0.6666666666666666,
         -0.6666666666666666]
    assert solve(mode="advantage", rewards=R1, num_samples=3,
                 critic_type="drgrpo") == \
        [0.5, -0.5, 0.0, 1.3333333333333335, -0.6666666666666666,
         -0.6666666666666666]
    assert solve(mode="advantage", rewards=R1, num_samples=3,
                 critic_type="grpo") == \
        [0.9999999800000003, -0.9999999800000003, 0.0,
         1.1547005283792517, -0.5773502641896258, -0.5773502641896258]


def check_advantage_two_groups(solve):
    # 8 序列、num_samples=4：组均值 1.3125 / 2.25；grpo 的 std 是组内
    # n-1 分母（第三组元素 std 全同 -> 比值 = dev/1.2651...）
    assert solve(mode="advantage", rewards=R2, num_samples=4) == \
        [1.6875, -1.3125, 0.1875, -0.5625, 3.75, -2.25, 0.75, -2.25]
    assert solve(mode="advantage", rewards=R2, num_samples=4,
                 critic_type="drgrpo") == \
        [1.6875, -1.3125, 0.1875, -0.5625, 3.75, -2.25, 0.75, -2.25]
    assert solve(mode="advantage", rewards=R2, num_samples=4,
                 critic_type="grpo") == \
        [1.3174650881948058, -1.02469506859596, 0.14638500979942284,
         -0.43915502939826856, 1.3055824151222792, -0.7833494490733676,
         0.26111648302445584, -0.7833494490733676]


def check_advantage_validation(solve):
    bad = [("rewards", lambda: solve(mode="advantage", rewards=None,
                                     num_samples=3)),
           ("rewards_1d", lambda: solve(mode="advantage",
                                        rewards=[0.0, 1.0], num_samples=2)),
           ("num_samples", lambda: solve(mode="advantage", rewards=R1,
                                         num_samples=0)),
           ("num_samples_type", lambda: solve(mode="advantage", rewards=R1,
                                              num_samples=2.0)),
           ("critic_type", lambda: solve(mode="advantage", rewards=R1,
                                         num_samples=3, critic_type="ppo"))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")


# ---------- mode="pg"：六路分支 ----------

def check_pg_gmpo(solve):
    # adv=+0.5 -> sgn=-1：diff 翻转 [-0.05,0.3,-0.5,0.02,0] -> clamp
    # [-0.05,0.2,-0.2,0.02,0] -> max [ -0.05,0.3,-0.2,0.02,0] -> 翻回
    # [0.05,-0.3,0.2,-0.02,0] -> 均值 -0.0175 -> ratio=exp(-0.0175)
    res = solve(mode="pg", variant="gmpo", new_logps=[DIFFS],
                old_logps=[[0.0] * 5], response_masks=[MASK],
                advantage=[0.5], loss_masks=[1.0], cliprange=0.2)
    assert res["loss"] == [-0.49132611819860306], res
    assert res["stats"]["logprobs_diff_max"] == [0.5]
    assert res["stats"]["logprobs_diff_min"] == [-0.3]
    assert res["stats"]["zero_pg_loss_count"] == [0]
    # adv=-0.5 -> sgn=+1：max 后 [0.05,-0.2,0.5,-0.02,0] -> 均值 0.0825
    res2 = solve(mode="pg", variant="gmpo", new_logps=[DIFFS],
                 old_logps=[[0.0] * 5], response_masks=[MASK],
                 advantage=[-0.5], loss_masks=[1.0], cliprange=0.2)
    assert res2["loss"] == [0.5429993363149622], res2
    assert set(res2["stats"]) == set(STATS_KEYS)


def check_pg_zero_adv_boundary(solve):
    # adv=0 走 sgn=-1（>= 比较）：翻账后逐位 max 得和为 0 -> ratio=1 ->
    # loss = -0.0（保留 -0.0 的符号性）
    res = solve(mode="pg", variant="gmpo", new_logps=[DIFFS],
                old_logps=[[0.0] * 5], response_masks=[MASK],
                advantage=[0.0], loss_masks=[1.0], cliprange=0.2)
    assert res["loss"] == [-0.0], res


def check_pg_gmpo_variants(solve):
    common = dict(mode="pg", new_logps=[DIFFS], old_logps=[[0.0] * 5],
                  response_masks=[MASK], advantage=[0.5],
                  loss_masks=[1.0], cliprange=0.2)
    # noclip：cliprange 内定 1000（入参 0.2 被忽略）
    res = solve(mode="pg", variant="gmpo_noclip", new_logps=[DIFFS],
                old_logps=[[0.0] * 5], response_masks=[MASK],
                advantage=[0.5], loss_masks=[1.0], cliprange=0.2)
    assert res["loss"] == [-0.5295926353273661], res
    # cliprange 入参换成别的值结果不变（内定 1000 的旁证）
    res_b = solve(mode="pg", variant="gmpo_noclip", new_logps=[DIFFS],
                  old_logps=[[0.0] * 5], response_masks=[MASK],
                  advantage=[0.5], loss_masks=[1.0], cliprange=0.7)
    assert res_b["loss"] == res["loss"]
    # seqclip：diff 先对 mask 求 sum 成标量 0.23 -> stats amax==amin==0.23
    res = solve(variant="gmpo_seqclip", **common)
    assert res["loss"] == [-0.525635548579641], res
    assert res["stats"]["logprobs_diff_max"] == [0.23000000000000004]
    assert res["stats"]["logprobs_diff_min"] == [0.0]
    # without_norm：ratio=exp(sum) 不除 mask.sum()
    res = solve(variant="gmpo_without_norm", **common)
    assert res["loss"] == [-0.4661969113423492], res


def check_pg_grpo_family(solve):
    common = dict(mode="pg", new_logps=[DIFFS], old_logps=[[0.0] * 5],
                  response_masks=[MASK], advantage=[0.5],
                  loss_masks=[1.0], cliprange=0.2)
    # 标准 PPO-clip：token 级 ratio + clamp(0.8, 1.2) + 逐位 max + masked
    # mean + loss_masks 门控 -> 标量
    res = solve(variant="grpo", **common)
    assert res["loss"] == -0.49653599879556215, res
    # grpo_clip_wider：clamp(0.67, 1.49) 硬编码（入参 0.2 不参与 clamp）
    res = solve(variant="grpo_clip_wider", **common)
    assert res["loss"] == -0.5327859987955621, res
    # drgrpo 风格 masked_sum：除 constant_normalizer=1024 而非 mask.sum()
    res = solve(mode="pg", variant="grpo", aggregator="sum",
                constant_normalizer=1024, new_logps=[DIFFS],
                old_logps=[[0.0] * 5], response_masks=[MASK],
                advantage=[0.5], loss_masks=[1.0], cliprange=0.2)
    assert res["loss"] == -0.0019395937452951646, res
    # loss_masks=0 -> (pg_loss * 0).mean() = 0.0
    res = solve(mode="pg", variant="grpo", new_logps=[DIFFS],
                old_logps=[[0.0] * 5], response_masks=[MASK],
                advantage=[0.5], loss_masks=[0.0], cliprange=0.2)
    assert res["loss"] == 0.0, res


def check_pg_big_seq_divergence(solve):
    # BIG 序列 [-0.9,0.9,...] 翻账后：gmpo 逐 token clamp 有 -0.2/0.9 混合，
    # seqclip 先 sum 得 -1.2+... 与 without_norm/noclip 四值互异
    vals = {}
    for variant in ("gmpo", "gmpo_seqclip", "gmpo_without_norm", "gmpo_noclip"):
        res = solve(mode="pg", variant=variant, new_logps=[BIG],
                    old_logps=[[0.0] * 5], response_masks=[[1] * 5],
                    advantage=[0.3], loss_masks=[1.0], cliprange=0.2)
        vals[variant] = res
    assert vals["gmpo"]["loss"] == [-0.23598835874194607]
    assert vals["gmpo_seqclip"]["loss"] == [-0.3122432324438279]
    assert vals["gmpo_without_norm"]["loss"] == [-0.09035826438152646]
    assert vals["gmpo_noclip"]["loss"] == [-0.359165208936543]
    # seqclip 的 stats：diff 是标量 0.9（amax==amin）
    assert vals["gmpo_seqclip"]["stats"]["logprobs_diff_max"] == \
        [0.9000000000000001]
    assert vals["gmpo_seqclip"]["stats"]["logprobs_diff_min"] == \
        [0.9000000000000001]
    assert len({tuple(v["loss"]) for v in vals.values()}) == 4


def check_pg_batch2_runtime_error(solve):
    # 官方怪癖：advantage 是张量，batch>1 时 sgn 布尔化/广播 RuntimeError
    # ——两个 pg 家族都只支持单序列 mini-batch
    for variant in ("gmpo", "grpo"):
        try:
            solve(mode="pg", variant=variant,
                  new_logps=[[0.05, -0.3, 0.5, -0.02, 0.0],
                             [0.1, 0.1, -0.4, 0.0, 0.3]],
                  old_logps=[[0.0] * 5, [0.0] * 5],
                  response_masks=[[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
                  advantage=[0.5, -0.5], loss_masks=[1.0, 1.0],
                  cliprange=0.2)
        except RuntimeError:
            continue
        raise AssertionError(f"batch=2 {variant} must raise RuntimeError")


def check_pg_validation(solve):
    common = dict(new_logps=[DIFFS], old_logps=[[0.0] * 5],
                  response_masks=[MASK], advantage=[0.5],
                  loss_masks=[1.0], cliprange=0.2)
    bad = [("mode", lambda: solve(mode="loss")),
           ("variant", lambda: solve(mode="pg", variant="dapo", **common)),
           ("new_logps", lambda: solve(mode="pg", variant="gmpo",
                                       new_logps=DIFFS,
                                       old_logps=[[0.0] * 5],
                                       response_masks=[MASK],
                                       advantage=[0.5], loss_masks=[1.0])),
           ("advantage", lambda: solve(mode="pg", variant="gmpo",
                                       new_logps=[DIFFS],
                                       old_logps=[[0.0] * 5],
                                       response_masks=[MASK],
                                       advantage="0.5", loss_masks=[1.0])),
           ("aggregator", lambda: solve(mode="pg", variant="grpo",
                                        aggregator="median", **common)),
           ("sum_no_cn", lambda: solve(mode="pg", variant="grpo",
                                       aggregator="sum", **common)),
           ("sum_bad_cn", lambda: solve(mode="pg", variant="grpo",
                                        aggregator="sum",
                                        constant_normalizer=0, **common))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")


# ---------- mode="kl_reward"：token 奖励组装 ----------

def check_kl_reward(solve):
    logps = [[0.1, 0.2, 0.3, 0.0, 0.0], [0.5, 0.1, 0.0, 0.0, 0.0]]
    ref = [[0.15, 0.1, 0.25, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0, 0.0]]
    rmask = [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    fin = [[1.0], [0.0]]
    res = solve(mode="kl_reward", logps=logps, ref_logps=ref,
                response_masks=rmask, final_rewards=fin, coef=0.1)
    assert res == {"rewards":
                   [[0.004999999999999999, -0.010000000000000002, 0.995,
                     -0.0, -0.0],
                    [-0.009999999999999998, 0.02, -0.0, -0.0, -0.0]],
                   "eos_indices": [2, 1]}, res


def check_kl_reward_empty_mask(solve):
    # 第二行全 False -> eos=-1 负索引：final reward 注入该行最后一个位置
    logps = [[0.1, 0.2, 0.3, 0.0, 0.0], [0.5, 0.1, 0.0, 0.0, 0.0]]
    ref = [[0.15, 0.1, 0.25, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0, 0.0]]
    fin = [[1.0], [0.0]]
    res = solve(mode="kl_reward", logps=logps, ref_logps=ref,
                response_masks=[[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                final_rewards=fin, coef=0.1)
    assert res == {"rewards":
                   [[0.004999999999999999, 0.99, -0.0, -0.0, -0.0],
                    [-0.0, 0.0, -0.0, -0.0, 0.0]],
                   "eos_indices": [1, -1]}, res


def check_kl_validation(solve):
    ok = dict(logps=[[0.1, 0.2]], ref_logps=[[0.1, 0.1]],
              response_masks=[[1, 0]], final_rewards=[[1.0]], coef=0.1)
    bad = [("logps", lambda: solve(mode="kl_reward", **{**ok, "logps": None})),
           ("coef", lambda: solve(mode="kl_reward", **{**ok, "coef": None})),
           ("final", lambda: solve(mode="kl_reward",
                                   **{**ok, "final_rewards": [1.0]}))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")
