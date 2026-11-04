"""OPD（datawhalechina/agentic-rl-lab）反向 KL 蒸馏账 -- 测试用例。"""
EXPECTED = {
    'teacher_basic': [-1.1, -0.9, -1.3, -0.7],
    'teacher_none': 'ValueError',
    'teacher_short': 'ValueError',
    'datum_basic': {'input_ids': [101, 102, 103, 104, 105, 201, 202, 203], 'target_tokens': [0, 0, 0, 0, 201, 202, 203, 204], 'logprobs': [0.0, 0.0, 0.0, 0.0, -0.5, -0.6, -0.4, -0.8], 'advantages': [0.0, 0.0, 0.0, 0.0, -0.2, 0.1, -0.3, 0.4]},
    'datum_len_mismatch': 'ValueError',
    'step_datum_advantages': [[0.0, 0.0, 0.0, 0.0, 0.3, 0.3, -0.15, 0.3], [0.0, 0.9, -0.15]],
    'step_reverse_kls': [-0.2, -0.2, 0.1, -0.2, -0.6, 0.1],
    'step_counts': [4, 2],
    'step_metrics': {'opd/reverse_kl_mean': -0.166667, 'opd/reverse_kl_std': 0.235702},
    'step_datum_logprobs': [[0.0, 0.0, 0.0, 0.0, -0.5, -0.6, -0.4, -0.8], [0.0, -0.9, -0.3]],
    'step_datum_targets': [[0, 0, 0, 0, 201, 202, 203, 204], [0, 205, 206]],
    'step_all_empty': 'RuntimeError',
    'sched_build_2_1': ['medical', 'medical', 'ceval'],
    'sched_build_1_2': ['medical', 'ceval', 'ceval'],
    'sched_count_7': {'medical': 5, 'ceval': 2},
    'sched_count_exact': {'medical': 4, 'ceval': 2},
    'sched_count_8': {'medical': 6, 'ceval': 2},
    'sched_resolve_steps0': 6,
    'sched_resolve_steps_pos': 10,
    'sched_resolve_max': 12,
    'sched_batch_step0': [{'id': 'm0'}, {'id': 'm1'}],
    'sched_batch_step2': [{'id': 'm4'}, {'id': 'm0'}],
    'sched_batch_wrap': [{'id': 'm1'}, {'id': 'm2'}, {'id': 'm3'}],
}

"""OPD（datawhalechina/agentic-rl-lab）反向 KL 蒸馏账 -- 测试用例（期望值由
官方 experiments 源码 exec 沙箱实测钉死：01-demo-sync.py 的 completion_
teacher_logprobs / build_opd_datum / step 内核逐行抽取 + 05-interleaved-
multi-teacher-opd.py 的调度四函数直通 exec，PyTRIO/datasets/swanlab 剥除、
rollout 与 teacher 打分全入参）."""

PROMPT_IDS = [101, 102, 103, 104, 105]
COMPLETION_IDS = [201, 202, 203, 204]
STUDENT_LPS = [-0.5, -0.6, -0.4, -0.8]
ADVANTAGES = [-0.2, 0.1, -0.3, 0.4]


class Seq:
    """rollout 序列的入参表面（tokens/logprobs）。"""

    def __init__(self, tokens, logprobs):
        self.tokens = tokens
        self.logprobs = logprobs


def teacher_by_pos(pids, ids):
    """确定性 teacher 打分：-(0.1 * 位置 + 0.3)。"""
    return [-(0.1 * k + 0.3) for k in range(len(ids))]


def rnd(seq):
    return [round(x, 6) for x in seq]


# ---------- mode="teacher_logprobs"：completion 区间切片 + 双重校验 ----------

def check_teacher_basic(solve):
    all_lps = [0.1, 0.2, 0.3, 0.4, 0.5, -1.1, -0.9, -1.3, -0.7]
    got = solve(mode="teacher_logprobs", all_logprobs=all_lps,
                prompt_ids=PROMPT_IDS, completion_ids=COMPLETION_IDS)
    # 从 len(prompt_ids) 切到结尾——prompt 区间的 logprob 不进蒸馏账
    assert got == EXPECTED["teacher_basic"], got


def check_teacher_guards(solve):
    # completion 区间含 None -> ValueError
    bad = [0.1, 0.2, 0.3, 0.4, 0.5, -1.1, None, -1.3, -0.7]
    try:
        solve(mode="teacher_logprobs", all_logprobs=bad,
              prompt_ids=PROMPT_IDS, completion_ids=COMPLETION_IDS)
        assert False, "expected ValueError"
    except ValueError:
        pass
    # all_logprobs 过短 -> 切片后长度失配 -> 同一 ValueError（不 IndexError）
    try:
        solve(mode="teacher_logprobs", all_logprobs=[0.1, 0.2, 0.3],
              prompt_ids=PROMPT_IDS, completion_ids=COMPLETION_IDS)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------- mode="datum"：右移四字段账 ----------

def check_datum_basic(solve):
    d = solve(mode="datum", prompt_ids=PROMPT_IDS,
              completion_ids=COMPLETION_IDS, old_logprobs=STUDENT_LPS,
              advantages=ADVANTAGES)
    # input = prompt + completion[:-1]；target = [0]*4 + completion
    assert d["input_ids"] == EXPECTED["datum_basic"]["input_ids"], d
    assert d["target_tokens"] == EXPECTED["datum_basic"]["target_tokens"], d
    assert rnd(d["logprobs"]) == EXPECTED["datum_basic"]["logprobs"], d
    assert rnd(d["advantages"]) == EXPECTED["datum_basic"]["advantages"], d
    # dtype 契约：target int64、logprobs/advantages float32（round 后比对
    # 已隐含 float32 噪声容差）
    pl = len(PROMPT_IDS) - 1
    # prompt 区间（除最后一位）logprobs/advantages 全 0.0 占位
    assert d["logprobs"][:pl] == [0.0] * pl, d
    assert d["advantages"][:pl] == [0.0] * pl, d


def check_datum_len_mismatch(solve):
    try:
        solve(mode="datum", prompt_ids=PROMPT_IDS,
              completion_ids=COMPLETION_IDS, old_logprobs=STUDENT_LPS[:3],
              advantages=ADVANTAGES)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------- mode="step"：批账（空序列跳过 / reverse KL / metrics） ----------

def make_batches():
    return [
        (PROMPT_IDS, [Seq([201, 202, 203, 204], STUDENT_LPS), Seq([], [])]),
        ([106, 107], [Seq([205, 206], [-0.9, -0.3])]),
    ]


def check_step_basic(solve):
    datums, rkls, counts, metrics = solve(
        mode="step", batches=make_batches(), teacher_fn=teacher_by_pos,
        kl_penalty_coef=1.5)
    # 空序列跳过：2 个 datum、6 个 token 的 reverse_kl
    assert len(datums) == 2, datums
    assert [round(x, 6) for x in rkls] == EXPECTED["step_reverse_kls"], rkls
    assert counts == EXPECTED["step_counts"], counts
    # advantages = -coef * reverse_kl 逐 token 进 datum（prompt 区 0 占位）
    for d, exp in zip(datums, EXPECTED["step_datum_advantages"]):
        assert rnd(d["advantages"]) == exp, d
    # logprobs 字段透传 student rollout 旧策略值（右移占位后）
    for d, exp in zip(datums, EXPECTED["step_datum_logprobs"]):
        assert rnd(d["logprobs"]) == exp, d
    for d, exp in zip(datums, EXPECTED["step_datum_targets"]):
        assert d["target_tokens"] == exp, d
    assert round(metrics["opd/reverse_kl_mean"], 6) == \
        round(EXPECTED["step_metrics"]["opd/reverse_kl_mean"], 6), metrics
    assert round(metrics["opd/reverse_kl_std"], 6) == \
        round(EXPECTED["step_metrics"]["opd/reverse_kl_std"], 6), metrics


def check_step_all_empty(solve):
    try:
        solve(mode="step", batches=[([1, 2], [Seq([], [])])],
              teacher_fn=teacher_by_pos, kl_penalty_coef=1.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# ---------- mode="schedule"：IDT 交替调度账 ----------

def check_schedule_build_count(solve):
    kw = dict(mode="schedule")
    assert solve(variant="build", medical_steps_per_cycle=2,
                 ceval_steps_per_cycle=1, **kw) == EXPECTED["sched_build_2_1"]
    assert solve(variant="build", medical_steps_per_cycle=1,
                 ceval_steps_per_cycle=2, **kw) == EXPECTED["sched_build_1_2"]
    sched = EXPECTED["sched_build_2_1"]
    # 7 步 = 2 个整周期（m4 c2）+ 第 7 步回绕到 medical
    assert solve(variant="count", schedule=sched, total_steps=7,
                 **kw) == EXPECTED["sched_count_7"]
    # 8 步：窗口端点语义可辨（平移窗口会把多出的步翻成 ceval）
    assert solve(variant="count", schedule=sched, total_steps=8,
                 **kw) == EXPECTED["sched_count_8"]
    assert solve(variant="count", schedule=sched, total_steps=6,
                 **kw) == EXPECTED["sched_count_exact"]


def check_schedule_resolve_batch(solve):
    kw = dict(mode="schedule")
    # steps=0：按两类数据 ceil(size/batch) 取 max 周期数乘回周期长度
    got = solve(variant="resolve", steps=0, batch_size=2,
                medical_steps_per_cycle=2, ceval_steps_per_cycle=1,
                medical_size=7, ceval_size=4, **kw)
    assert got == EXPECTED["sched_resolve_steps0"], got
    assert solve(variant="resolve", steps=10, batch_size=2,
                 medical_steps_per_cycle=2, ceval_steps_per_cycle=1,
                 medical_size=7, ceval_size=4,
                 **kw) == EXPECTED["sched_resolve_steps_pos"]
    # max-周期数：medical 只需 1 个周期、ceval 需 4 个 -> 取 4 -> 12 步
    got_max = solve(variant="resolve", steps=0, batch_size=2,
                    medical_steps_per_cycle=2, ceval_steps_per_cycle=1,
                    medical_size=4, ceval_size=8, **kw)
    assert got_max == EXPECTED["sched_resolve_max"], got_max
    rows = [{"id": "m%d" % k} for k in range(5)]
    assert solve(variant="batch", rows=rows, task_step=0, batch_size=2,
                 **kw) == EXPECTED["sched_batch_step0"]
    # 任务自己的 step 下标回绕：step2 从 rows[4] 起跨尾回绕
    assert solve(variant="batch", rows=rows, task_step=2, batch_size=2,
                 **kw) == EXPECTED["sched_batch_step2"]
    assert solve(variant="batch", rows=rows, task_step=2, batch_size=3,
                 **kw) == EXPECTED["sched_batch_wrap"]
