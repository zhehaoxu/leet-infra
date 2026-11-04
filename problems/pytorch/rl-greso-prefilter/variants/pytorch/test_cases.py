"""GRESO (arXiv:2506.02177) pre-rollout selective filtering ledger -- test cases."""

"""
EXPECTED values were dumped from the official Infini-AI-Lab/GRESO code
(presampling_selector.py DataProfiler + ray_trainer adaptive-sampling tail;
see notes.md). Random sources are parameterized as explicit values.
"""

EXPECTED = {
    'adapt_clamp_hi': [0.95, 0.94],
    'adapt_clamp_lo': [0.05, 0.05],
    'adapt_epoch0': [0.5, 0.5],
    'adapt_mixed': [0.49, 0.49],
    'adapt_over': [0.49, 0.49],
    'adapt_target_gap': [0.51, 0.49],
    'adapt_under': [0.51, 0.49],
    'filter_counts': [1, 1],
    'filter_log': [['a_0', 2], ['a_1', 0], ['a_2', 1], ['a_3', 2], ['a_4', 2]],
    'filter_sel': [0, 3, 4],
    'ids_basic': ['dapo_3', 'dapo_17', 'math_5'],
    'prof_len': 3,
    'refill_basic': [64, True, False],
    'refill_clamp': [384, False, False],
    'refill_est_eq': [384, True, False],
    'refill_factor': [375, True, False],
    'refill_full': [384, False, True],
    'refill_last': [64, True, False],
    'skip_easy_backoff_break': False,
    'skip_easy_backoff_count0': False,
    'skip_easy_backoff_eq': True,
    'skip_easy_backoff_hi': True,
    'skip_easy_backoff_lo': False,
    'skip_easy_example_n1': False,
    'skip_easy_example_n2': True,
    'skip_easy_example_n_inf': True,
    'skip_easy_example_never': False,
    'skip_easy_prob_c0': False,
    'skip_easy_prob_c2': True,
    'skip_easy_prob_c2b': False,
    'skip_easy_prob_floor': True,
    'skip_hard_backoff_c1': False,
    'skip_hard_backoff_c1b': False,
    'skip_hard_backoff_c3': False,
    'skip_hard_backoff_c3lo': True,
    'skip_hard_gate': False,
    'skip_hard_prob_c1': False,
    'skip_hard_prob_c3': False,
    'skip_hard_prob_c3b': True,
    'skip_hard_prob_gate': False,
    'trace_dapo17': [[0, 0.25], [1, 0.05]],
    'trace_dapo3': [[0, 1.0], [1, 0.9], [2, 1.0]],
    'trace_math5': [[0, 0.0]],
    'trace_missing': [],
    'zva_flags_keep': [[True, True, True, True, False], [8, 9]],
}

# ---------- shared fixtures ----------

# dapo_3 trace: [(0, 1.0), (1, 0.9), (2, 1.0)] -- an example that was solved,
# dipped, solved again
OPS_MAIN = [
    ["add", 0, ["dapo_3", "dapo_17", "math_5"], [1.0, 0.25, 0.0]],
    ["add", 1, ["dapo_3", "dapo_17"], [0.9, 0.05]],
    ["add", 2, ["dapo_3"], [1.0]],
]


def make_filter_prof_ops():
    # a_0: two trailing >=0.98 (count 2); a_1: one; a_2: all-low (hard,
    # count 3); a_3: trailing run of 2; a_4: history high then low --
    # easy run broken but hard-skippable
    return [
        ["add", 0, ["a_0", "a_1", "a_2", "a_3", "a_4"],
         [1.0, 1.0, 0.03, 1.0, 1.0]],
        ["add", 1, ["a_0", "a_2", "a_3", "a_4"], [0.99, 0.02, 1.0, 0.05]],
        ["add", 2, ["a_2"], [0.05]],
    ]


FILTER_BATCH = [{"data_source": "a", "index": i} for i in range(5)]

# rands consumed lazily in order by the filter's easy/hard checks
FILTER_RANDS = [0.9, 0.05, 0.5, 0.5, 0.9, 0.05, 0.5]

# ---------- mode="id" / "trace" ----------


def check_id(solve):
    ids = solve(mode="id", data_sources=["dapo", "dapo", "math"],
                indices=[3, 17, 5])
    # data id = f"{data_source}_{index}"
    assert ids == EXPECTED["ids_basic"], ids


def check_trace(solve):
    tr = solve(mode="trace", ops=OPS_MAIN, data_id="dapo_3")
    # (epoch, reward) pairs appended in call order
    assert [list(t) for t in tr] == EXPECTED["trace_dapo3"], tr
    assert [list(t) for t in solve(mode="trace", ops=OPS_MAIN,
                                   data_id="dapo_17")] == \
        EXPECTED["trace_dapo17"]
    assert [list(t) for t in solve(mode="trace", ops=OPS_MAIN,
                                   data_id="math_5")] == \
        EXPECTED["trace_math5"]
    # unknown id -> empty trace (official .get default)
    assert solve(mode="trace", ops=OPS_MAIN,
                 data_id="nope_1") == EXPECTED["trace_missing"]


# ---------- mode="skip" ----------


def check_skip_easy_example(solve):
    # reward exactly == 1 anywhere in the trace (n=-1 -> forever)
    assert solve(mode="skip", variant="easy_example", ops=OPS_MAIN,
                 current_epoch=100, data_id="dapo_3") == \
        EXPECTED["skip_easy_example_n_inf"]
    # windowed: epoch(2) + n(2) >= current(4) -> True; n=1 -> False
    assert solve(mode="skip", variant="easy_example", ops=OPS_MAIN,
                 current_epoch=4, data_id="dapo_3", n=2) == \
        EXPECTED["skip_easy_example_n2"]
    assert solve(mode="skip", variant="easy_example", ops=OPS_MAIN,
                 current_epoch=4, data_id="dapo_3", n=1) == \
        EXPECTED["skip_easy_example_n1"]
    # math_5 has no reward==1 -> never skipped
    assert solve(mode="skip", variant="easy_example", ops=OPS_MAIN,
                 current_epoch=999, data_id="math_5") == \
        EXPECTED["skip_easy_example_never"]


def check_skip_backoff(solve):
    # dapo_17 tail run >= 0.98 is empty -> count 0 -> False regardless
    assert solve(mode="skip", variant="easy_backoff", ops=OPS_MAIN,
                 current_epoch=50, data_id="dapo_17", k=2, backoff=1) == \
        EXPECTED["skip_easy_backoff_count0"]
    # q_x: 4 consecutive 1.0s -> count 4, max_epoch 3; k*count=8
    OPS_Q = [["add", 0, "q_x", 1.0], ["add", 1, "q_x", 1.0],
             ["add", 2, "q_x", 1.0], ["add", 3, "q_x", 1.0]]
    assert solve(mode="skip", variant="easy_backoff", ops=OPS_Q,
                 current_epoch=50, data_id="q_x", k=2, backoff=5) == \
        EXPECTED["skip_easy_backoff_lo"]
    assert solve(mode="skip", variant="easy_backoff", ops=OPS_Q,
                 current_epoch=7, data_id="q_x", k=2, backoff=7) == \
        EXPECTED["skip_easy_backoff_hi"]
    # a trailing 0.97 (< 0.98) breaks the run -> count 0
    OPS_QB = OPS_Q + [["add", 4, "q_x", 0.97]]
    assert solve(mode="skip", variant="easy_backoff", ops=OPS_QB,
                 current_epoch=50, data_id="q_x", k=2, backoff=7) == \
        EXPECTED["skip_easy_backoff_break"]
    # boundary: max_epoch(3) + backoff(7) == current(10) -> >= keeps True
    OPS_Q4 = [["add", i, "q_x", 1.0] for i in range(4)]
    assert solve(mode="skip", variant="easy_backoff", ops=OPS_Q4,
                 current_epoch=10, data_id="q_x", k=2, backoff=7) == \
        EXPECTED["skip_easy_backoff_eq"]
    # hard: latest reward > 0.11 means solvable -> never hard-skipped
    assert solve(mode="skip", variant="hard_backoff", ops=OPS_MAIN,
                 current_epoch=50, data_id="dapo_3", k=2, backoff=1) == \
        EXPECTED["skip_hard_gate"]
    # dapo_17: latest 0.05 passes the gate; tail run <= 0.11 count=1;
    # max_epoch 1 + backoff 1 = 2 < 3 and < 50
    assert solve(mode="skip", variant="hard_backoff", ops=OPS_MAIN,
                 current_epoch=3, data_id="dapo_17", k=2, backoff=1) == \
        EXPECTED["skip_hard_backoff_c1"]
    assert solve(mode="skip", variant="hard_backoff", ops=OPS_MAIN,
                 current_epoch=50, data_id="dapo_17", k=2, backoff=1) == \
        EXPECTED["skip_hard_backoff_c1b"]
    # h: three trailing <= 0.11 entries; backoff 5 -> 2+5=7 < 10
    OPS_H = [["add", 0, "h", 0.05], ["add", 1, "h", 0.02],
             ["add", 2, "h", 0.1]]
    assert solve(mode="skip", variant="hard_backoff", ops=OPS_H,
                 current_epoch=10, data_id="h", k=2, backoff=5) == \
        EXPECTED["skip_hard_backoff_c3"]
    # backoff clamped to 2 -> 2+2=4 >= 4 -> True
    assert solve(mode="skip", variant="hard_backoff", ops=OPS_H,
                 current_epoch=4, data_id="h", k=2, backoff=2) == \
        EXPECTED["skip_hard_backoff_c3lo"]


def check_skip_probabilistic(solve):
    OPS_E = [["add", 0, "e", 1.0], ["add", 1, "e", 1.0]]
    # count=2, p=0.5 -> skip_prob = 1 - 0.25 = 0.75
    assert solve(mode="skip", variant="easy_probabilistic", ops=OPS_E,
                 current_epoch=10, data_id="e", p=0.5,
                 rand_values=[0.1]) == EXPECTED["skip_easy_prob_c2"]
    assert solve(mode="skip", variant="easy_probabilistic", ops=OPS_E,
                 current_epoch=10, data_id="e", p=0.5,
                 rand_values=[0.8]) == EXPECTED["skip_easy_prob_c2b"]
    # count=0 -> skip_prob = 1 - max(p^0, 0.01) = 0 -> never skips
    assert solve(mode="skip", variant="easy_probabilistic", ops=OPS_MAIN,
                 current_epoch=10, data_id="dapo_17", p=0.5,
                 rand_values=[0.0]) == EXPECTED["skip_easy_prob_c0"]
    # hard probabilistic: the solvable gate runs first
    assert solve(mode="skip", variant="hard_probabilistic", ops=OPS_MAIN,
                 current_epoch=10, data_id="dapo_3", p=0.5,
                 rand_values=[0.0]) == EXPECTED["skip_hard_prob_gate"]
    assert solve(mode="skip", variant="hard_probabilistic", ops=OPS_MAIN,
                 current_epoch=10, data_id="dapo_17", p=0.5,
                 rand_values=[0.5]) == EXPECTED["skip_hard_prob_c1"]
    OPS_Z = [["add", 0, "z", 0.0], ["add", 1, "z", 0.0],
             ["add", 2, "z", 0.0]]
    assert solve(mode="skip", variant="hard_probabilistic", ops=OPS_Z,
                 current_epoch=10, data_id="z", p=0.5,
                 rand_values=[0.9]) == EXPECTED["skip_hard_prob_c3"]
    # p=0.9: skip_prob = 1 - 0.9^3 = 0.271; rand 0.15 < 0.271 -> True
    assert solve(mode="skip", variant="hard_probabilistic", ops=OPS_Z,
                 current_epoch=10, data_id="z", p=0.9,
                 rand_values=[0.15]) == EXPECTED["skip_hard_prob_c3b"]
    # skip_prob floor: count=2/p=0.5 -> 0.75 (floor 0.01); rand 0.6 in
    # [0.5, 0.75) separates floor 0.01 from a floor of 0.5
    assert solve(mode="skip", variant="easy_probabilistic", ops=OPS_E,
                 current_epoch=10, data_id="e", p=0.5,
                 rand_values=[0.6]) == EXPECTED["skip_easy_prob_floor"]


# ---------- mode="filter" ----------


def check_filter(solve):
    sel, log, counts = solve(mode="filter", ops=make_filter_prof_ops(),
                             current_epoch=5, batch=FILTER_BATCH,
                             p_easy=0.5, p_hard=0.5, rand_values=FILTER_RANDS)
    # a_0 kept (easy prob 0.75 vs rand 0.9; hard gate: latest 0.99 solvable)
    assert sel == EXPECTED["filter_sel"], sel
    # log codes: 0 = easy skip, 1 = hard skip, 2 = kept
    assert log == EXPECTED["filter_log"], log
    assert counts == EXPECTED["filter_counts"], counts
    # a_4 (history high then low): easy run broken (rand 0.9 fail) then
    # hard skip (rand 0.05) -- swapping easy/hard order would reassign the
    # draws and change the verdict, so the order is observable here
    a4 = [entry for entry in log if entry[0] == "a_4"]
    assert a4 == [["a_4", 2]], a4


# ---------- mode="refill" ----------


def check_refill(solve):
    # estimate = int(100 / 0.8 / 8 * 1.25) = 19 -> last=True, clamp to 64
    assert list(solve(mode="refill", reminder=100, zero_variance_ratio=0.2,
                      n=8, default_bs=384)) == EXPECTED["refill_basic"]
    # estimate 19 <= default -> last_sampling True even without filtering
    assert list(solve(mode="refill", reminder=50, zero_variance_ratio=0.0,
                      n=8, default_bs=384)) == EXPECTED["refill_last"]
    # estimate 15625 > default -> bs = default, last False
    assert list(solve(mode="refill", reminder=1000, zero_variance_ratio=0.9,
                      n=8, default_bs=384)) == EXPECTED["refill_clamp"]
    # full_sampling_bs short-circuits everything
    assert list(solve(mode="refill", reminder=100, zero_variance_ratio=0.2,
                      n=8, default_bs=384, full_sampling_bs=True)) == \
        EXPECTED["refill_full"]
    # estimate exactly equals default (384): <= keeps last_sampling True
    assert list(solve(mode="refill", reminder=1970, zero_variance_ratio=0.2,
                      n=8, default_bs=384)) == EXPECTED["refill_est_eq"]
    # factor split: estimate(1.25)=375 <= 400 < estimate(1.5)=450
    assert list(solve(mode="refill", reminder=1920, zero_variance_ratio=0.2,
                      n=8, default_bs=400)) == EXPECTED["refill_factor"]


# ---------- mode="adapt" ----------


def check_adapt(solve):
    # epoch 0 freezes both p's
    assert list(solve(mode="adapt", epoch=0, hard_ratio=0.9, easy_ratio=0.9,
                      p_hard=0.5, p_easy=0.5)) == EXPECTED["adapt_epoch0"]
    # targets: easy = 0.25/3 ~= 0.0833, hard = 0.25*2/3 ~= 0.1667
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.9, easy_ratio=0.9,
                      p_hard=0.5, p_easy=0.5)) == EXPECTED["adapt_over"]
    # 0.1 sits between the two targets: hard adapts down (-0.01),
    # easy adapts up (+0.01)
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.1, easy_ratio=0.1,
                      p_hard=0.5, p_easy=0.5)) == EXPECTED["adapt_under"]
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.9, easy_ratio=0.1,
                      p_hard=0.5, p_easy=0.5)) == EXPECTED["adapt_mixed"]
    # clamp bounds
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.9, easy_ratio=0.9,
                      p_hard=0.05, p_easy=0.05)) == \
        EXPECTED["adapt_clamp_lo"]
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.1, easy_ratio=0.1,
                      p_hard=0.95, p_easy=0.95)) == \
        EXPECTED["adapt_clamp_hi"]
    # 0.14 sits between 1/2*t (0.125) and 2/3*t (0.1667): hard adapts up,
    # easy (0.14 >= 1/3*t) adapts down
    assert list(solve(mode="adapt", epoch=1, hard_ratio=0.14,
                      easy_ratio=0.14, p_hard=0.5, p_easy=0.5)) == \
        EXPECTED["adapt_target_gap"]


# ---------- mode="zero_variance" ----------


def check_zero_variance(solve):
    flags, keep = solve(mode="zero_variance",
                        rewards=[1, 1, 0, 0, 0.5, 0.5, 0, 0, 1, 0], n=2)
    # a group with max == min carries zero advantage and is dropped whole
    assert flags == EXPECTED["zva_flags_keep"][0], flags
    assert keep == EXPECTED["zva_flags_keep"][1], keep
