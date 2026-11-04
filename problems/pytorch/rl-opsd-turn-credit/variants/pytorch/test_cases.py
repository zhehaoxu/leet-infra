"""AgentOPSD -- turn-level credit assignment tests."""

"""
EXPECTED values were dumped from the official
datawhalechina/agentic-rl-lab experiments/agentopsd (advantages.py,
teacher.py::_score_turns, loss.py::build_agentopsd_ppo_datum) in a sandbox;
see notes.md. Float comparisons use a 1e-9 tolerance to absorb
float-noise at the epsilon clip boundaries.
"""

EXPECTED = {
    'assign': {'stats': {'groups': 2, 'degenerate_groups': 1}, 'trajs': [{'group_id': 'g1', 'reward': 1.0, 'sequence_advantage': 0.7069568129996042, 'initial_belief': 0.6666666666666666, 'turn_advantages': [0.7776524942995647, 0.6362611316996438, 0.6888488915254039], 'step_teacher_evidence': [0.8, -0.3, 0.1], 'step_turn_advantage': [0.7776524942995647, 0.6362611316996438, 0.6888488915254039]}, {'group_id': 'g1', 'reward': 0.0, 'sequence_advantage': -1.4139136259992082, 'initial_belief': 0.6666666666666666, 'turn_advantages': [-1.5553049885991292, -1.345230798401383, -1.285822319574403], 'step_teacher_evidence': [-0.5, 0.2, 0.4], 'step_turn_advantage': [-1.5553049885991292, -1.345230798401383, -1.285822319574403]}, {'group_id': 'g1', 'reward': 1.0, 'sequence_advantage': 0.7069568129996042, 'initial_belief': 0.6666666666666666, 'turn_advantages': [0.6658986529283433, 0.6497227727053427, 0.7776524942995647], 'step_teacher_evidence': [0.1, 0.1], 'step_turn_advantage': [0.6658986529283433, 0.6497227727053427]}, {'group_id': 'g2', 'reward': 0.5, 'sequence_advantage': 0.0, 'initial_belief': 0.5, 'turn_advantages': [0.0, 0.0], 'step_teacher_evidence': [1.0, -1.0], 'step_turn_advantage': [0.0, 0.0]}, {'group_id': 'g2', 'reward': 0.5, 'sequence_advantage': 0.0, 'initial_belief': 0.5, 'turn_advantages': [0.0, 0.0], 'step_teacher_evidence': [0.3, 0.3], 'step_turn_advantage': [0.0, 0.0]}]},
    'grp_a': {'advantages': [0.9998000399920016, -0.9998000399920016, 0.9998000399920016, -0.9998000399920016], 'mean_reward': 0.5, 'std': 0.5},
    'grp_b': {'advantages': [0.0, 0.0, 0.0], 'mean_reward': 0.5},
    'grp_c': {'advantages': [0.0], 'mean_reward': 0.0},
    'ppo': {'input_len': 13, 'target_tokens': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], 'old_logprobs': [0.0, -1.0, -0.9, -0.8, -0.7, 0.0, 0.0, -0.5, -0.6, -0.4, -0.3, 0.0, 0.0], 'advantages': [0.0, 1.2, 1.2, 1.2, 1.2, 0.0, 0.0, -0.8, -0.8, -0.8, -0.8, 0.0, 0.0], 'action_tokens': 8},
    'ppo_trunc': {'input_len': 13, 'target_tokens': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], 'old_logprobs': [0.0, -1.0, -0.9, -0.8, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'advantages': [0.0, 1.2, 1.2, 1.2, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'action_tokens': 4},
    'score_turns': {'prefix_offset': 4, 'evidences': [2.1, 1.64, 1.29], 'action_tokens': 9, 'teacher_slices': [[-0.11, -0.12, 0.13], [-0.17, -0.18, 0.19, -0.2], [0.25, -0.26]], 'teacher_lp_window': [-0.11, -0.12, 0.13, -0.14, -0.15, 0.16, -0.17, -0.18, 0.19, -0.2, -0.21, 0.22, -0.23, -0.24, 0.25, -0.26], 'manual': {'t0': [-0.11, -0.12, 0.13], 't1': [-0.17, -0.18, 0.19, -0.2], 't2': [0.25, -0.26]}},
    'score_zip_trunc': {'evidences': [1.27], 'action_tokens': 4, 'teacher_slice': [-0.11, -0.12, 0.13, -0.14], 'manual_expect_evidence': 1.27},
    'traj_cfg2': {'initial_belief': 0.3, 'turns': [{'turn_index': 1, 'evidence': 1.0, 'cumulative_evidence': 1.0, 'belief_before': 0.3, 'belief_after': 0.538101526224449, 'delta_belief': 0.23810152622444897, 'outcome_credit': 0.23810152622444897, 'normalized_credit': 1.1521149611861592, 'weight': 1.5, 'advantage': 2.8000000000000003}, {'turn_index': 2, 'evidence': -0.4, 'cumulative_evidence': 0.09999999999999998, 'belief_before': 0.538101526224449, 'belief_after': 0.32141036836664855, 'delta_belief': -0.2166911578578004, 'outcome_credit': -0.2166911578578004, 'normalized_credit': -0.693634563531104, 'weight': 0.6531827182344481, 'advantage': 1.4450923491751169}, {'turn_index': 3, 'evidence': 0.6, 'cumulative_evidence': 0.6499999999999999, 'belief_before': 0.32141036836664855, 'belief_after': 0.45083491669136255, 'delta_belief': 0.129424548324714, 'outcome_credit': 0.129424548324714, 'normalized_credit': 0.7110557987879631, 'weight': 1.3555278993939814, 'advantage': 2.5688446390303703}, {'turn_index': 4, 'evidence': -1.5, 'cumulative_evidence': -1.175, 'belief_before': 0.45083491669136255, 'belief_after': 0.11688159617674698, 'delta_belief': -0.33395332051461557, 'outcome_credit': -0.33395332051461557, 'normalized_credit': -1.1695361964430186, 'weight': 0.5, 'advantage': 1.2}]},
    'traj_cfg2_single': {'turns': [{'weight': 1.0, 'normalized_credit': 0.0, 'advantage': -1.0, 'outcome_credit': -0.22117415359800607}]},
    'traj_clip_hi': {'sequence_advantage': 0.9998000399920016, 'initial_belief': 0.9999, 'turns': [{'turn_index': 1, 'evidence': 0.0, 'cumulative_evidence': 0.0, 'belief_before': 0.9999, 'belief_after': 0.9999000000000001, 'delta_belief': 1.1102230246251565e-16, 'outcome_credit': 1.1102230246251565e-16, 'normalized_credit': 5.551115123122701e-13, 'weight': 1.000000000000111, 'advantage': 0.9998000399920571}, {'turn_index': 2, 'evidence': 0.0, 'cumulative_evidence': 0.0, 'belief_before': 0.9999000000000001, 'belief_after': 0.9999000000000001, 'delta_belief': 0.0, 'outcome_credit': 0.0, 'normalized_credit': -5.551115123122701e-13, 'weight': 0.999999999999889, 'advantage': 0.9998000399919461}]},
    'traj_clip_lo': {'sequence_advantage': -0.9998000399920016, 'initial_belief': 0.0001, 'turns': [{'turn_index': 1, 'evidence': 0.0, 'cumulative_evidence': 0.0, 'belief_before': 0.0001, 'belief_after': 9.999999999999999e-05, 'delta_belief': -1.3552527156068805e-20, 'outcome_credit': 1.3552527156068805e-20, 'normalized_credit': 6.776263578034401e-17, 'weight': 1.0, 'advantage': -0.9998000399920016}, {'turn_index': 2, 'evidence': 0.0, 'cumulative_evidence': 0.0, 'belief_before': 9.999999999999999e-05, 'belief_after': 9.999999999999999e-05, 'delta_belief': 0.0, 'outcome_credit': -0.0, 'normalized_credit': -6.776263578034401e-17, 'weight': 1.0, 'advantage': -0.9998000399920016}]},
    'traj_neg': {'sequence_advantage': -0.9998000399920016, 'initial_belief': 0.5, 'turns': [{'turn_index': 1, 'evidence': -0.5, 'cumulative_evidence': -0.5, 'belief_before': 0.5, 'belief_after': 0.37754066879814546, 'delta_belief': -0.12245933120185454, 'outcome_credit': 0.12245933120185454, 'normalized_credit': 1.382511457353024, 'weight': 1.2, 'advantage': -1.0997800439912018}, {'turn_index': 2, 'evidence': 0.2, 'cumulative_evidence': -0.27499999999999997, 'belief_before': 0.37754066879814546, 'belief_after': 0.4316800165217519, 'delta_belief': 0.05413934772360646, 'outcome_credit': -0.05413934772360646, 'normalized_credit': -0.43945610390374845, 'weight': 0.9121087792192503, 'advantage': -0.955863216966232}, {'turn_index': 3, 'evidence': 0.4, 'cumulative_evidence': 0.13875000000000004, 'belief_before': 0.4316800165217519, 'belief_after': 0.5346319578769636, 'delta_belief': 0.10295194135521168, 'outcome_credit': -0.10295194135521168, 'normalized_credit': -0.9430553534492755, 'weight': 0.8113889293101448, 'advantage': -0.905513361982676}]},
    'traj_pos': {'sequence_advantage': 0.9998000399920016, 'initial_belief': 0.5, 'turns': [{'turn_index': 1, 'evidence': 0.8, 'cumulative_evidence': 0.8, 'belief_before': 0.5, 'belief_after': 0.6899744811276125, 'delta_belief': 0.1899744811276125, 'outcome_credit': 0.1899744811276125, 'normalized_credit': 1.4125482507404141, 'weight': 1.2, 'advantage': 1.0997800439912018}, {'turn_index': 2, 'evidence': -0.3, 'cumulative_evidence': 0.46, 'belief_before': 0.6899744811276125, 'belief_after': 0.6130141761393355, 'delta_belief': -0.076960304988277, 'outcome_credit': -0.076960304988277, 'normalized_credit': -0.34910884847423873, 'weight': 0.9301782303051522, 'advantage': 0.964896135925391}, {'turn_index': 3, 'evidence': 0.1, 'cumulative_evidence': 0.537, 'belief_before': 0.6130141761393355, 'belief_after': 0.6311142657248102, 'delta_belief': 0.01810008958547471, 'outcome_credit': 0.01810008958547471, 'normalized_credit': 0.2782496962539538, 'weight': 1.0556499392507908, 'advantage': 1.0276194457362482}, {'turn_index': 4, 'evidence': -0.9, 'cumulative_evidence': -0.38985000000000003, 'belief_before': 0.6311142657248102, 'belief_after': 0.4037534106709316, 'delta_belief': -0.2273608550538786, 'outcome_credit': -0.2273608550538786, 'normalized_credit': -1.3416890985201293, 'weight': 0.8, 'advantage': 0.8998200359928015}]},
    'traj_single': {'sequence_advantage': 1.5, 'initial_belief': 0.7, 'turns': [{'turn_index': 1, 'evidence': 2.0, 'cumulative_evidence': 2.0, 'belief_before': 0.7, 'belief_after': 0.945178837561174, 'delta_belief': 0.2451788375611741, 'outcome_credit': 0.2451788375611741, 'normalized_credit': 0.0, 'weight': 1.0, 'advantage': 1.5}]},
    'traj_zero': {'sequence_advantage': 0.0, 'initial_belief': 0.5, 'turns': [{'turn_index': 1, 'evidence': 1.2, 'cumulative_evidence': 1.2, 'belief_before': 0.5, 'belief_after': 0.7685247834990175, 'delta_belief': 0.26852478349901754, 'outcome_credit': 0.0, 'normalized_credit': 0.0, 'weight': 1.0, 'advantage': 0.0}, {'turn_index': 2, 'evidence': -1.2, 'cumulative_evidence': -0.06000000000000005, 'belief_before': 0.7685247834990175, 'belief_after': 0.4850044983805899, 'delta_belief': -0.28352028511842764, 'outcome_credit': -0.0, 'normalized_credit': -0.0, 'weight': 1.0, 'advantage': 0.0}, {'turn_index': 3, 'evidence': 0.5, 'cumulative_evidence': 0.44299999999999995, 'belief_before': 0.4850044983805899, 'belief_after': 0.6089736380692562, 'delta_belief': 0.1239691396886663, 'outcome_credit': 0.0, 'normalized_credit': 0.0, 'weight': 1.0, 'advantage': 0.0}]},
}

# ---------- shared helpers ----------


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _assert_close_list(got, want, tol=1e-9):
    assert len(got) == len(want), (len(got), len(want))
    for i, (g, w) in enumerate(zip(got, want)):
        assert _close(g, w, tol), (i, g, w)


def _assert_turn(got, want):
    """逐字段比较一个 turn 账（solution 返回 dict，dump 键一致）。"""
    for key in [
        "turn_index", "evidence", "cumulative_evidence", "belief_before",
        "belief_after", "delta_belief", "outcome_credit",
        "normalized_credit", "weight", "advantage",
    ]:
        g, w = got[key], want[key]
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            assert _close(g, w), (key, g, w)
        else:
            assert g == w, (key, g, w)


# ---------- mode="group" ----------


def check_group_sequence_advantages(solve):
    advantages, mean_reward = solve(mode="group", rewards=[1.0, 0.0, 1.0, 0.0],
                                    epsilon=1e-4)
    _assert_close_list(advantages, EXPECTED["grp_a"]["advantages"])
    assert _close(mean_reward, EXPECTED["grp_a"]["mean_reward"])

    # 全同分组: std=0 -> advantage 全 0（分母 0+eps），mean 作 initial belief
    advantages, mean_reward = solve(mode="group", rewards=[0.5, 0.5, 0.5],
                                    epsilon=1e-4)
    _assert_close_list(advantages, EXPECTED["grp_b"]["advantages"])
    assert _close(mean_reward, EXPECTED["grp_b"]["mean_reward"])

    # 单条轨迹组
    advantages, mean_reward = solve(mode="group", rewards=[0.0],
                                    epsilon=1e-4)
    _assert_close_list(advantages, EXPECTED["grp_c"]["advantages"])
    assert _close(mean_reward, EXPECTED["grp_c"]["mean_reward"])


# ---------- mode="reshape" ----------


def check_reshape_turn_advantages(solve):
    # 正优势轨迹（evidence 有正有负），默认 config
    b0, turns = solve(mode="reshape",
                      sequence_advantage=EXPECTED["grp_a"]["advantages"][0],
                      initial_belief=EXPECTED["grp_a"]["mean_reward"],
                      evidences=[0.8, -0.3, 0.1, -0.9])
    assert _close(b0, EXPECTED["traj_pos"]["initial_belief"])
    assert len(turns) == 4
    for got, want in zip(turns, EXPECTED["traj_pos"]["turns"]):
        _assert_turn(got, want)
    # 首轮: cumulative = gamma*0 + e0（c0=0 递推起点）
    assert _close(turns[0]["cumulative_evidence"], 0.8)
    # weight 触上界 1.2；A_turn = A_seq * ((1-0.5) + 0.5*1.2) = A_seq * 1.1
    assert _close(turns[0]["weight"], 1.2)
    assert _close(turns[0]["advantage"],
                  EXPECTED["grp_a"]["advantages"][0] * 1.1)

    # 负优势轨迹: sign=-1 把负 delta 翻正（credit 仍可正可负）
    b0, turns = solve(mode="reshape",
                      sequence_advantage=EXPECTED["grp_a"]["advantages"][1],
                      initial_belief=EXPECTED["grp_a"]["mean_reward"],
                      evidences=[-0.5, 0.2, 0.4])
    for got, want in zip(turns, EXPECTED["traj_neg"]["turns"]):
        _assert_turn(got, want)

    # 零优势: sign=0 -> credit 全 0 -> z 全 0 -> weight 全 1 -> A_turn 全 0
    b0, turns = solve(mode="reshape", sequence_advantage=0.0,
                      initial_belief=EXPECTED["grp_b"]["mean_reward"],
                      evidences=[1.2, -1.2, 0.5])
    for got, want in zip(turns, EXPECTED["traj_zero"]["turns"]):
        _assert_turn(got, want)


def check_reshape_clip_and_config(solve):
    # clip 上界: initial_belief=1.0 -> b0 = 1-eps
    b0, turns = solve(mode="reshape",
                      sequence_advantage=EXPECTED["grp_a"]["advantages"][2],
                      initial_belief=1.0, evidences=[0.0, 0.0])
    assert _close(b0, EXPECTED["traj_clip_hi"]["initial_belief"])
    for got, want in zip(turns, EXPECTED["traj_clip_hi"]["turns"]):
        _assert_turn(got, want)

    # clip 下界: initial_belief=0.0 -> b0 = eps
    b0, turns = solve(mode="reshape",
                      sequence_advantage=EXPECTED["grp_a"]["advantages"][3],
                      initial_belief=0.0, evidences=[0.0, 0.0])
    assert _close(b0, EXPECTED["traj_clip_lo"]["initial_belief"])
    for got, want in zip(turns, EXPECTED["traj_clip_lo"]["turns"]):
        _assert_turn(got, want)

    # 自定义 config: gamma=0.5, lambda=0.8, b=0.5, eps=0.01
    b0, turns = solve(mode="reshape", sequence_advantage=2.0,
                      initial_belief=0.3, evidences=[1.0, -0.4, 0.6, -1.5],
                      gamma=0.5, reshape_lambda=0.8, weight_bound=0.5,
                      epsilon=0.01)
    assert _close(b0, EXPECTED["traj_cfg2"]["initial_belief"])
    for got, want in zip(turns, EXPECTED["traj_cfg2"]["turns"]):
        _assert_turn(got, want)

    # 单 turn: credit std=0 -> z=0 -> weight=1
    b0, turns = solve(mode="reshape", sequence_advantage=-1.0,
                      initial_belief=0.4, evidences=[0.9],
                      gamma=0.5, reshape_lambda=0.8, weight_bound=0.5,
                      epsilon=0.01)
    got, want = turns[0], EXPECTED["traj_cfg2_single"]["turns"][0]
    for key in ("weight", "normalized_credit", "advantage", "outcome_credit"):
        assert _close(got[key], want[key]), (key, got[key], want[key])


# ---------- mode="score" ----------


def check_score_turns(solve):
    # student_initial=6, teacher_initial=10 -> prefix_offset=+4
    steps = [
        (6, 9, [-1.1, -0.4, -0.7]),
        (12, 16, [-0.2, -0.9, -0.3, -0.6]),
        (20, 22, [-0.8, -0.5]),
    ]
    window = EXPECTED["score_turns"]["teacher_lp_window"]
    full = ([0.0] * 10) + list(window) + ([0.0] * 4)
    evidences, action_tokens, slices = solve(
        mode="score", steps=steps, teacher_logprobs=full,
        student_initial_length=6, teacher_initial_length=10)
    _assert_close_list(evidences, EXPECTED["score_turns"]["evidences"])
    assert action_tokens == EXPECTED["score_turns"]["action_tokens"] == 9
    assert len(slices) == 3
    for got, want in zip(slices, EXPECTED["score_turns"]["teacher_slices"]):
        _assert_close_list(got, want)
    # 手工核对第一轮切片: teacher 区间 = [6+4, 9+4)
    _assert_close_list(slices[0], full[10:13])


def check_score_turns_zip_truncation(solve):
    # teacher 切片 4 个 token 但 student logprobs 只有 2 个 -> zip 截断
    window = EXPECTED["score_turns"]["teacher_lp_window"]
    full = ([0.0] * 10) + list(window) + ([0.0] * 4)
    evidences, action_tokens, slices = solve(
        mode="score", steps=[(6, 10, [-1.1, -0.4])], teacher_logprobs=full,
        student_initial_length=6, teacher_initial_length=10)
    _assert_close_list(evidences, EXPECTED["score_zip_trunc"]["evidences"])
    assert action_tokens == EXPECTED["score_zip_trunc"]["action_tokens"] == 4
    _assert_close_list(slices[0], EXPECTED["score_zip_trunc"]["teacher_slice"])
    # evidence 只用了前 2 个 teacher 值
    assert _close(evidences[0],
                  (full[10] - (-1.1)) + (full[11] - (-0.4)))


# ---------- mode="assign" ----------


def check_assign_agentopsd_advantages(solve):
    trajs = [
        ("g1", 1.0, [0.8, -0.3, 0.1], 3),
        ("g1", 0.0, [-0.5, 0.2, 0.4], 3),
        ("g1", 1.0, [0.1, 0.1, 0.2], 2),
        ("g2", 0.5, [1.0, -1.0], 2),
        ("g2", 0.5, [0.3, 0.3], 2),
    ]
    results, groups, degenerate = solve(mode="assign", trajectories=trajs)
    want = EXPECTED["assign"]
    assert groups == want["stats"]["groups"]
    assert degenerate == want["stats"]["degenerate_groups"]
    assert len(results) == 5
    for got, w in zip(results, want["trajs"]):
        assert got["group_id"] == w["group_id"]
        assert _close(got["sequence_advantage"], w["sequence_advantage"])
        assert _close(got["initial_belief"], w["initial_belief"])
        _assert_close_list(got["turn_advantages"], w["turn_advantages"])
        # zip(steps, turns) 截断写回
        _assert_close_list(got["step_evidences"], w["step_teacher_evidence"])
        _assert_close_list(got["step_turn_advantages"],
                           w["step_turn_advantage"])

    # g2 是退化组: sequence_advantage 全 0 -> turn advantage 全 0
    for got in results[3:]:
        assert got["sequence_advantage"] == 0.0
        for value in got["turn_advantages"]:
            assert value == 0.0

    # build_training_datums 过滤语义: sequence_advantage != 0.0 才进训练
    kept = [t[1] for got, t in zip(results, trajs)
            if got["sequence_advantage"] != 0.0]
    assert kept == [1.0, 0.0, 1.0]


# ---------- mode="ppo" ----------


def check_build_ppo_arrays(solve):
    full_tokens = list(range(100, 114))
    steps = [(2, [-1.0, -0.9, -0.8, -0.7]), (8, [-0.5, -0.6, -0.4, -0.3])]
    out = solve(mode="ppo", full_tokens=full_tokens, steps=steps,
                turn_advantages=[1.2, -0.8])
    assert out["action_tokens"] == EXPECTED["ppo"]["action_tokens"] == 8
    assert out["target_tokens"] == EXPECTED["ppo"]["target_tokens"]
    assert out["input_tokens"] == list(range(100, 113))
    _assert_close_list(out["old_logprobs"], EXPECTED["ppo"]["old_logprobs"])
    _assert_close_list(out["advantages"], EXPECTED["ppo"]["advantages"])

    # steps 多于 turn_advantages -> zip 截断
    out = solve(mode="ppo", full_tokens=full_tokens,
                steps=steps + [(12, [-0.2, -0.1])], turn_advantages=[1.2])
    assert out["action_tokens"] == EXPECTED["ppo_trunc"]["action_tokens"] == 4
    _assert_close_list(out["old_logprobs"],
                       EXPECTED["ppo_trunc"]["old_logprobs"])
    _assert_close_list(out["advantages"], EXPECTED["ppo_trunc"]["advantages"])

    # 非动作 token 全 0
    assert out["old_logprobs"][0] == 0.0
    assert out["advantages"][0] == 0.0


TESTS = [
    check_group_sequence_advantages,
    check_reshape_turn_advantages,
    check_reshape_clip_and_config,
    check_score_turns,
    check_score_turns_zip_truncation,
    check_assign_agentopsd_advantages,
    check_build_ppo_arrays,
]
