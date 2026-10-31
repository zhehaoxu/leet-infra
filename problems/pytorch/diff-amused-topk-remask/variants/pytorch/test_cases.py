import math

import torch

LOG_FLOAT64_MAX = 709.782712893384  # math.log(torch.finfo(torch.float64).max)


def _softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def _gumbel(u):
    return -math.log(max(-math.log(max(u, 1e-20)), 1e-20))


def _linspace(lo, hi, steps):
    if steps == 1:
        return [float(lo)]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def _oracle(logits, sample, drawn, uniforms, timestep, num_steps, mask_id, schedule, temperature, cap):
    """纯 Python 重算：逐行做 softmax、置信度排序与 top-k 截断，不使用任何张量算子。"""
    batch, seq, _ = len(logits), len(logits[0]), len(logits[0][0])
    probs = [[_softmax(row) for row in batch_logits] for batch_logits in logits]
    if isinstance(temperature, (tuple, list)):
        temps = _linspace(float(temperature[0]), float(temperature[1]), num_steps)
    else:
        temps = _linspace(float(temperature), 0.01, num_steps)

    committed = [[d if s == mask_id else s for d, s in zip(drow, srow)] for drow, srow in zip(drawn, sample)]
    unknown = [[s == mask_id for s in srow] for srow in sample]

    if timestep == 0:
        return committed, committed, [[False] * seq for _ in range(batch)], [[0] for _ in range(batch)]

    step_idx = num_steps - 1 - timestep
    ratio = (step_idx + 1) / num_steps
    mask_ratio = math.cos(ratio * math.pi / 2) if schedule == "cosine" else 1.0 - ratio
    mask_ratio *= cap

    temperature_step = temps[step_idx]
    prev, masking, mask_lens = [], [], []
    for b in range(batch):
        count = sum(1 for u in unknown[b] if u)
        mask_len = int(math.floor(seq * mask_ratio))
        mask_len = min(count - 1, mask_len)
        mask_len = max(1, mask_len)
        conf = []
        for j in range(seq):
            if unknown[b][j]:
                log_p = math.log(max(probs[b][j][committed[b][j]], 1e-20))
            else:
                log_p = LOG_FLOAT64_MAX
            conf.append(log_p + temperature_step * _gumbel(uniforms[b][j]))
        cut_off = sorted(conf)[mask_len]
        row_mask = [c < cut_off for c in conf]
        masking.append(row_mask)
        mask_lens.append([mask_len])
        prev.append([mask_id if m else committed[b][j] for j, m in enumerate(row_mask)])
    return prev, committed, masking, mask_lens


def _tensors(batch=1, seq=6, vocab=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.rand(batch, seq, vocab, generator=g, dtype=torch.float64) * 4.0 - 2.0
    sample = torch.randint(0, vocab, (batch, seq), generator=g, dtype=torch.int64)
    drawn = torch.randint(0, vocab, (batch, seq), generator=g, dtype=torch.int64)
    uniforms = torch.rand(batch, seq, generator=g, dtype=torch.float64)
    return logits, sample, drawn, uniforms


def _as_lists(*tensors):
    return [t.tolist() for t in tensors]


_EXAMPLE_LOGITS = [
    [[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
]
_EXAMPLE_SAMPLE = [[2, 0, 2, 2, 2, 2]]
_EXAMPLE_DRAWN = [[1, 0, 0, 2, 2, 0]]
_EXAMPLE_UNIFORMS = [[0.2, 0.5, 0.9, 0.1, 0.75, 0.4]]


def _example_inputs():
    return (
        torch.tensor(_EXAMPLE_LOGITS, dtype=torch.float64),
        torch.tensor(_EXAMPLE_SAMPLE, dtype=torch.int64),
        torch.tensor(_EXAMPLE_DRAWN, dtype=torch.int64),
        torch.tensor(_EXAMPLE_UNIFORMS, dtype=torch.float64),
    )


def check_example(solve):
    logits, sample, drawn, uniforms = _example_inputs()
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 2, "cosine", (2.0, 0.0), 1.0)
    assert committed.tolist() == [[1, 0, 0, 2, 2, 0]], committed
    assert mask_len.tolist() == [[4]], mask_len
    assert masking.tolist() == [[True, False, False, True, True, True]], masking
    assert prev.tolist() == [[2, 0, 0, 2, 2, 2]], prev
    assert prev.dtype == torch.int64 and masking.dtype == torch.bool and mask_len.dtype == torch.int64
    assert prev.shape == (1, 6) and mask_len.shape == (1, 1)


def check_example_variants(solve):
    logits, sample, drawn, uniforms = _example_inputs()
    # step_idx = 0（第一个去噪步）：cos(0.25*pi/2)*6 = 5.54 -> 5，但只允许比「已掩位置数 - 1」还少
    _, _, masking, mask_len = solve(logits, sample, drawn, uniforms, 3, 4, 2, "cosine", (2.0, 0.0), 1.0)
    assert mask_len.tolist() == [[4]], mask_len
    assert masking.tolist() == [[True, False, False, True, True, True]]
    # step_idx = 2：cos(0.75*pi/2)*6 = 2.30 -> 2
    prev, _, masking, mask_len = solve(logits, sample, drawn, uniforms, 1, 4, 2, "cosine", (2.0, 0.0), 1.0)
    assert mask_len.tolist() == [[2]]
    assert masking.tolist() == [[True, False, False, True, False, False]]
    assert prev.tolist() == [[2, 0, 0, 2, 2, 0]]
    # linear 型：1 - 0.5 = 0.5，floor(6*0.5) = 3
    prev, _, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 2, "linear", (2.0, 0.0), 1.0)
    assert mask_len.tolist() == [[3]]
    assert masking.tolist() == [[True, False, False, True, False, True]]
    assert prev.tolist() == [[2, 0, 0, 2, 2, 2]]
    # 末步收束：timestep 0 不重掩
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 0, 4, 2, "cosine", (2.0, 0.0), 1.0)
    assert prev.tolist() == committed.tolist() == [[1, 0, 0, 2, 2, 0]]
    assert masking.tolist() == [[False] * 6]
    assert mask_len.tolist() == [[0]]


def check_schedule_tables(solve):
    # seq = 8、N = 4、全部位置都是掩码：逐 step 核对 mask_len 的调度表
    seq, steps = 8, 4
    logits = torch.zeros(1, seq, 2, dtype=torch.float64)
    sample = torch.full((1, seq), 1, dtype=torch.int64)
    drawn = torch.zeros(1, seq, dtype=torch.int64)
    uniforms = torch.full((1, seq), 0.5, dtype=torch.float64)
    cos_expected = [int(math.floor(seq * math.cos(((i + 1) / steps) * math.pi / 2))) for i in range(steps)]
    lin_expected = [int(math.floor(seq * (1.0 - (i + 1) / steps))) for i in range(steps)]
    for step in range(steps):
        step_idx = steps - 1 - step
        if step == 0:
            assert int(solve(logits, sample, drawn, uniforms, 0, steps, 1, "cosine", (2.0, 0.0), 1.0)[3]) == 0
            continue
        got = int(solve(logits, sample, drawn, uniforms, step, steps, 1, "cosine", (2.0, 0.0), 1.0)[3])
        assert got == cos_expected[step_idx], (step, step_idx, got, cos_expected)
        got = int(solve(logits, sample, drawn, uniforms, step, steps, 1, "linear", (2.0, 0.0), 1.0)[3])
        assert got == lin_expected[step_idx], (step, step_idx, got, lin_expected)
    # 余弦型比线性型掩得更多（中段起）
    assert cos_expected[1] == 5 and lin_expected[1] == 4, (cos_expected, lin_expected)


def check_mask_forced_to_at_least_one(solve):
    logits, sample, drawn, uniforms = _example_inputs()
    # starting_mask_ratio = 0 -> floor 出 0，被 max(1, .) 抬回 1：仍然恰好重掩一个位置
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 2, "linear", (2.0, 0.0), 0.0)
    assert mask_len.tolist() == [[1]]
    assert int(masking.sum()) == 1, masking
    # 被掩的是置信度最低的那个未知位置
    assert masking.tolist() == [[False, False, False, True, False, False]], masking
    assert prev.tolist() == [[1, 0, 0, 2, 2, 0]], prev


def check_clamped_by_known_count(solve):
    # 只剩两个未知位置：floor(6*cos(pi/4)) = 4，但被 min(unknown - 1, .) 压到 1
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 2.0]]], dtype=torch.float64
    )
    sample = torch.tensor([[0, 1, 1, 0, 0, 0]], dtype=torch.int64)
    drawn = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.int64)
    uniforms = torch.tensor([[0.3, 0.6, 0.1, 0.9, 0.5, 0.2]], dtype=torch.float64)
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 1, "cosine", (2.0, 0.0), 1.0)
    assert mask_len.tolist() == [[1]], mask_len
    assert committed.tolist() == [[0, 1, 0, 0, 0, 0]], committed
    assert masking.tolist() == [[False, False, True, False, False, False]], masking
    assert prev.tolist() == [[0, 1, 1, 0, 0, 0]], prev


def check_no_unknown_still_masks_one(solve):
    # sample 里一个掩码都没有：unknown.sum() - 1 = -1 -> min 出负数 -> max(1, .) 抬回 1
    # 于是所有位置都是「已提交」的，其中置信度最低的那个会被重新掩成 mask_token_id
    logits = torch.tensor([[[3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0], [2.0, 0.0, 1.0]]], dtype=torch.float64)
    sample = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    drawn = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)
    uniforms = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 1, 4, 2, "cosine", (1.0, 1.0), 1.0)
    assert mask_len.tolist() == [[1]], mask_len
    assert committed.tolist() == [[0, 1, 0, 1]], committed
    # 四个位置的概率各不相同但都被 finfo(max) 顶掉，置信度只由 Gumbel 噪声决定：u 最小者被掩
    assert masking.tolist() == [[True, False, False, False]], masking
    assert prev.tolist() == [[2, 1, 0, 1]], prev


def check_known_positions_never_masked_while_unknown_exist(solve):
    logits, sample, drawn, uniforms = _tensors(batch=2, seq=9, vocab=4, seed=3)
    sample = sample.clone()
    sample[:, :3] = 3  # 前三个位置已知（mask_token_id 是 0）
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 5, 0, "cosine", (2.0, 0.0), 1.0)
    assert bool(masking[:, :3].any()) is False, masking
    assert torch.equal(prev[:, :3], committed[:, :3])
    assert torch.equal(masking.sum(dim=-1, keepdim=True).to(torch.int64), mask_len)


def check_temperature_zero_picks_lowest_confidence(solve):
    # 温度 (0, 0)：置信度退化成 log p，掩码就是「概率最低的 mask_len 个未知位置」
    logits = torch.tensor(
        [[[0.0, 0.0, 5.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0], [2.0, 2.0, 0.0]]], dtype=torch.float64
    )
    sample = torch.tensor([[2, 2, 2, 2, 2]], dtype=torch.int64)
    drawn = torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.int64)
    uniforms = torch.full((1, 5), 0.5, dtype=torch.float64)
    # 未知位置的选中概率（T = 0，置信度就是 log p）：pos0=-5.013, pos1=-0.036, pos2=-0.095, pos3=-0.240, pos4=-2.759
    # N=4、timestep=2 -> step_idx=1 -> cos(pi/4)*5 = 3.53 -> 3
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 2, "cosine", (0.0, 0.0), 1.0)
    assert mask_len.tolist() == [[3]], mask_len
    assert masking.tolist() == [[True, False, False, True, True]], masking
    assert prev.tolist() == [[2, 0, 1, 2, 2]], prev
    assert committed.tolist() == [[0, 0, 1, 1, 2]], committed


def check_gumbel_endpoints(solve):
    # 均匀分布 + 均匀 logits：置信度完全由 Gumbel 噪声决定。
    # u = 0 -> gumbel = -log(46.05) = -3.829（最小），u = 1 -> gumbel = 46.05（最大）
    seq = 6
    logits = torch.zeros(1, seq, 3, dtype=torch.float64)
    sample = torch.full((1, seq), 0, dtype=torch.int64)
    drawn = torch.arange(seq, dtype=torch.int64).reshape(1, seq) % 3
    uniforms = torch.tensor([[0.0, 1.0, 0.2, 0.8, 0.5, 0.35]], dtype=torch.float64)
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 1, 4, 0, "linear", (1.0, 1.0), 1.0)
    # step_idx = 2 -> 1 - 0.75 = 0.25 -> floor(6*0.25) = 1
    assert mask_len.tolist() == [[1]], mask_len
    assert masking.tolist() == [[True, False, False, False, False, False]], masking
    assert prev.tolist() == [[0, 1, 2, 0, 1, 2]], prev
    assert committed.tolist() == [[0, 1, 2, 0, 1, 2]], committed
    assert prev.dtype == torch.int64


def check_uniform_clamp(solve):
    # 双端 clamp：p 极小（logits 相差 200）时按 1e-20 截断；u = 0 时同样截断
    logits = torch.tensor([[[0.0, -200.0], [0.0, 200.0], [0.0, 0.0]]], dtype=torch.float64)
    sample = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    drawn = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    uniforms = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float64)
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 0, "linear", (1.0, 1.0), 1.0)
    assert mask_len.tolist() == [[1]], mask_len
    # pos0 选中概率 = softmax(0,-200)[1] = 0 -> clamp 到 1e-20 -> log = -46.05，gumbel(0) = -3.83 -> -49.88
    # pos1 选中概率 = 1 -> log = 0，gumbel(1) = 46.05 -> 46.05
    # pos2 选中概率 = 0.5 -> log = -0.693，gumbel(0.5) = 0.3665 -> -0.327
    # 升序第 1 小的是 pos2，因此只有 pos0 被掩
    assert masking.tolist() == [[True, False, False]], masking
    assert prev.tolist() == [[0, 1, 1]], prev
    assert committed.tolist() == [[1, 1, 1]], committed


def check_scalar_temperature_and_batch_rows(solve):
    logits, sample, drawn, uniforms = _tensors(batch=3, seq=7, vocab=4, seed=11)
    for schedule in ("cosine", "linear"):
        for step in range(1, 5):
            out = solve(logits, sample, drawn, uniforms, step, 5, 2, schedule, 2, 1.0)
            ref = _oracle(*_as_lists(logits, sample, drawn, uniforms), step, 5, 2, schedule, 2, 1.0)
            assert out[0].tolist() == ref[0], (schedule, step)
            assert out[1].tolist() == ref[1]
            assert out[2].tolist() == ref[2]
            assert out[3].tolist() == ref[3]


def check_ties_mask_fewer(solve):
    # 严格小于 + 并列：门槛本身落在并列值上时，埋回个数少于 mask_len
    logits = torch.zeros(1, 4, 3, dtype=torch.float64)
    sample = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
    drawn = torch.tensor([[1, 2, 0, 1]], dtype=torch.int64)
    uniforms = torch.tensor([[0.5, 0.5, 0.5, 0.1]], dtype=torch.float64)
    # linear、timestep=2 -> mask_ratio = 0.5 -> mask_len = min(3, 2) = 2
    # 置信度：三个位置同为 log(1/3) + 0.3665，第四个为 log(1/3) - 0.8340
    # 升序下标 2 是并列值 -> 只有那个孤立低点被掩
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 0, "linear", (1.0, 1.0), 1.0)
    assert mask_len.tolist() == [[2]], mask_len
    assert masking.tolist() == [[False, False, False, True]], masking
    assert int(masking.sum()) < int(mask_len)
    assert prev.tolist() == [[1, 2, 0, 0]], prev


def check_all_tied_nothing_masked(solve):
    # 整行已知且 uniforms 完全相同：四个哨兵置信度全等 -> 一个都不掩
    logits = torch.zeros(1, 4, 3, dtype=torch.float64)
    sample = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)
    drawn = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64)
    uniforms = torch.full((1, 4), 0.5, dtype=torch.float64)
    prev, committed, masking, mask_len = solve(logits, sample, drawn, uniforms, 2, 4, 0, "cosine", (2.0, 0.0), 1.0)
    assert mask_len.tolist() == [[1]], mask_len
    assert masking.tolist() == [[False, False, False, False]], masking
    assert prev.tolist() == committed.tolist() == [[1, 1, 1, 1]], prev


def check_against_oracle(solve):
    configs = [
        (1, 6, 3, 4, "cosine", (2.0, 0.0), 1.0, 2),
        (2, 8, 4, 5, "cosine", (2.0, 0.0), 1.0, 3),
        (3, 9, 5, 6, "linear", (1.5, 0.5), 1.0, 4),
        (2, 7, 2, 8, "cosine", (3.0, 0.0), 0.6, 1),
        (4, 12, 4, 4, "linear", (0.5, 0.0), 1.0, 2),
        (1, 5, 6, 3, "cosine", (2.0, 0.0), 0.0, 1),
        (2, 10, 3, 7, "cosine", (1.0, 1.0), 1.0, 5),
    ]
    for seed, (batch, seq, vocab, steps, schedule, temp, cap, step) in enumerate(configs):
        logits, sample, drawn, uniforms = _tensors(batch, seq, vocab, seed=100 + seed)
        for st in range(steps):
            got = solve(logits, sample, drawn, uniforms, st, steps, 0, schedule, temp, cap)
            ref = _oracle(*_as_lists(logits, sample, drawn, uniforms), st, steps, 0, schedule, temp, cap)
            assert got[0].tolist() == ref[0], (seed, st, schedule)
            assert got[1].tolist() == ref[1]
            assert got[2].tolist() == ref[2]
            assert got[3].tolist() == ref[3]


def check_dtypes_and_float32_input(solve):
    logits, sample, drawn, uniforms = _tensors(batch=2, seq=6, vocab=4, seed=21)
    out = solve(logits, sample, drawn, uniforms, 2, 4, 1, "cosine", (2.0, 0.0), 1.0)
    prev, committed, masking, mask_len = out
    assert prev.dtype == torch.int64 and committed.dtype == torch.int64
    assert masking.dtype == torch.bool and mask_len.dtype == torch.int64
    assert prev.shape == (2, 6) and mask_len.shape == (2, 1)
    ref = _oracle(*_as_lists(logits, sample, drawn, uniforms), 2, 4, 1, "cosine", (2.0, 0.0), 1.0)
    assert prev.tolist() == ref[0]
    # 同样的数值换成 float32 输入，结果必须一致（内部一律 float64）
    out32 = solve(logits.to(torch.float32), sample, drawn, uniforms.to(torch.float32), 2, 4, 1, "cosine", (2.0, 0.0), 1.0)
    assert torch.equal(out32[0], prev) and torch.equal(out32[2], masking) and torch.equal(out32[3], mask_len)
    # int32 的 token 张量同样接受
    outi = solve(logits, sample.to(torch.int32), drawn.to(torch.int32), uniforms, 2, 4, 1, "cosine", (2.0, 0.0), 1.0)
    assert torch.equal(outi[0], prev) and torch.equal(outi[1], committed)


def check_monotone_unmasking(solve):
    # 温度取 0，则掩码数随 step 单调不增，且被保留的恰是概率最高的若干未知位置
    logits = torch.tensor(
        [
            [
                [0.0, 4.0],
                [4.0, 0.0],
                [0.0, 3.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    sample = torch.full((1, 8), 0, dtype=torch.int64)
    drawn = torch.zeros(1, 8, dtype=torch.int64)
    uniforms = torch.full((1, 8), 0.5, dtype=torch.float64)
    counts = []
    for step in range(4):
        _, _, masking, mask_len = solve(logits, sample, drawn, uniforms, step, 4, 0, "cosine", (0.0, 0.0), 1.0)
        counts.append(int(masking.sum()))
        assert int(mask_len) == counts[-1] or step == 0, (step, mask_len)
    assert counts == [0, 3, 5, 7], counts


def _poison(tensor, value):
    out = tensor.clone()
    out.index_put_((torch.tensor([0]),) * out.dim(), torch.tensor(value, dtype=out.dtype))
    return out


def check_invalid_inputs(solve):
    logits, sample, drawn, uniforms = _tensors(batch=1, seq=6, vocab=3, seed=5)
    bad = [
        ("logits-list", lambda: solve(logits.tolist(), sample, drawn, uniforms, 1, 4, 0)),
        ("logits-2d", lambda: solve(logits[0], sample, drawn, uniforms, 1, 4, 0)),
        ("logits-4d", lambda: solve(logits.unsqueeze(0), sample, drawn, uniforms, 1, 4, 0)),
        ("logits-int", lambda: solve(logits.to(torch.int64), sample, drawn, uniforms, 1, 4, 0)),
        ("logits-bool", lambda: solve(logits.to(torch.bool), sample, drawn, uniforms, 1, 4, 0)),
        ("logits-nan", lambda: solve(_poison(logits, float("nan")), sample, drawn, uniforms, 1, 4, 0)),
        ("logits-inf", lambda: solve(_poison(logits, float("inf")), sample, drawn, uniforms, 1, 4, 0)),
        ("seq-len-1", lambda: solve(torch.zeros(1, 1, 3, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.int64),
                                    torch.zeros(1, 1, dtype=torch.int64), torch.zeros(1, 1, dtype=torch.float64), 1, 4, 0)),
        ("batch-0", lambda: solve(torch.zeros(0, 6, 3, dtype=torch.float64), torch.zeros(0, 6, dtype=torch.int64),
                                  torch.zeros(0, 6, dtype=torch.int64), torch.zeros(0, 6, dtype=torch.float64), 1, 4, 0)),
        ("vocab-0", lambda: solve(torch.zeros(1, 6, 0, dtype=torch.float64), sample, drawn, uniforms, 1, 4, 0)),
        ("sample-list", lambda: solve(logits, sample.tolist(), drawn, uniforms, 1, 4, 0)),
        ("sample-float", lambda: solve(logits, sample.to(torch.float32), drawn, uniforms, 1, 4, 0)),
        ("sample-shape", lambda: solve(logits, sample[:, :5], drawn, uniforms, 1, 4, 0)),
        ("sample-oob", lambda: solve(logits, sample.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(3)),
                                     drawn, uniforms, 1, 4, 0)),
        ("sample-negative", lambda: solve(logits, sample.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(-1)),
                                          drawn, uniforms, 1, 4, 0)),
        ("drawn-list", lambda: solve(logits, sample, drawn.tolist(), uniforms, 1, 4, 0)),
        ("drawn-bool", lambda: solve(logits, sample, drawn.to(torch.bool), uniforms, 1, 4, 0)),
        ("drawn-oob", lambda: solve(logits, sample, drawn.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(7)),
                                    uniforms, 1, 4, 0)),
        ("uniforms-list", lambda: solve(logits, sample, drawn, uniforms.tolist(), 1, 4, 0)),
        ("uniforms-int", lambda: solve(logits, sample, drawn, uniforms.to(torch.int64), 1, 4, 0)),
        ("uniforms-shape", lambda: solve(logits, sample, drawn, uniforms[:, :4], 1, 4, 0)),
        ("uniforms-below-0", lambda: solve(logits, sample, drawn, uniforms - 0.5, 1, 4, 0)),
        ("uniforms-above-1", lambda: solve(logits, sample, drawn, uniforms + 0.5, 1, 4, 0)),
        ("uniforms-nan", lambda: solve(logits, sample, drawn, _poison(uniforms, float("nan")), 1, 4, 0)),
        ("timestep-float", lambda: solve(logits, sample, drawn, uniforms, 1.0, 4, 0)),
        ("timestep-bool", lambda: solve(logits, sample, drawn, uniforms, True, 4, 0)),
        ("timestep-negative", lambda: solve(logits, sample, drawn, uniforms, -1, 4, 0)),
        ("timestep-oob", lambda: solve(logits, sample, drawn, uniforms, 4, 4, 0)),
        ("steps-zero", lambda: solve(logits, sample, drawn, uniforms, 0, 0, 0)),
        ("steps-float", lambda: solve(logits, sample, drawn, uniforms, 0, 4.0, 0)),
        ("mask-id-float", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 1.0)),
        ("mask-id-oob", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 3)),
        ("mask-id-negative", lambda: solve(logits, sample, drawn, uniforms, 1, 4, -1)),
        ("schedule-none", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, None)),
        ("schedule-typo", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "Cosine")),
        ("schedule-unknown", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "squared-res"),),
        ("temperature-string", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", "2")),
        ("temperature-triple", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (2.0, 1.0, 0.0))),
        ("temperature-nan", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (float("nan"), 0.0))),
        ("temperature-inf", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (2.0, float("inf")))),
        ("temperature-huge", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (12.0, 0.0))),
        ("temperature-neg-huge", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (0.0, -12.0))),
        ("ratio-negative", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (2.0, 0.0), -0.1)),
        ("ratio-above-one", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (2.0, 0.0), 1.5)),
        ("ratio-nan", lambda: solve(logits, sample, drawn, uniforms, 1, 4, 0, "cosine", (2.0, 0.0), float("nan"))),
    ]
    for name, call in bad:
        try:
            call()
        except ValueError:
            continue
        except RuntimeError:
            raise AssertionError(f"{name} 应抛 ValueError，却抛了 RuntimeError")
        raise AssertionError(f"{name} 应抛 ValueError")
