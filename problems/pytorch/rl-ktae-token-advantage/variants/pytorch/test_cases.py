"""KTAE (arXiv:2505.16826) token-level advantage ledger -- test cases."""

"""
EXPECTED values were dumped from the official ZNLP/KTAE torch code
(compute_key_tokens.py + core_algos.py::compute_ktae_outcome_advantage_and_keytokens;
see notes.md). Long vectors (~151645) are stored sparsely; float32 lgamma/sigmoid
may differ in low bits across torch versions, hence the rounded comparisons.
"""
import numpy as np
import torch

EXPECTED = {
    'dist_correct': {'10': 2, '11': 2, '12': 2, '13': 2, '14': 1, '15': 1},
    'dist_wrong': {'10': 1, '12': 2, '13': 1, '16': 1, '17': 1, '18': 1, '19': 1, '21': 1, '151644': 1},
    'table_data': {'10': [2.0, 1.0, 1.0, 2.0], '11': [2.0, 0.0, 1.0, 3.0], '12': [2.0, 2.0, 1.0, 1.0], '13': [2.0, 1.0, 1.0, 2.0], '14': [1.0, 0.0, 2.0, 3.0], '15': [1.0, 0.0, 2.0, 3.0], '16': [0.0, 1.0, 3.0, 2.0], '17': [0.0, 1.0, 3.0, 2.0], '18': [0.0, 1.0, 3.0, 2.0], '19': [0.0, 1.0, 3.0, 2.0], '21': [0.0, 1.0, 3.0, 2.0], '151644': [0.0, 1.0, 3.0, 2.0]},
    'fisher_fmt_len': 151645,
    'fisher_fmt_nonzero': {'11': 0.44932899},
    'ig_nonzero': {'10': 0.081704, '11': 0.459148, '13': 0.081704, '14': 0.190875, '15': 0.190875, '16': 0.190875, '17': 0.190875, '18': 0.190875, '19': 0.190875, '21': 0.190875, '151644': 0.190875},
    'tf_data': {'10': [0.2446602, 0.1461717], '11': [0.352612, 0.0], '12': [0.2446602, 0.2787611], '13': [0.2446602, 0.1461717], '14': [0.1275304, 0.0], '15': [0.1275304, 0.0], '16': [0.0, 0.1461717], '17': [0.0, 0.1461717], '18': [0.0, 0.1461717], '19': [0.0, 0.1461717], '21': [0.0, 0.1461717], '151644': [0.0, 0.1461717]},
    'cohen_len': 151645,
    'cohen_nonzero': {'10': 0.339837, '11': 0.955317, '13': 0.339837, '14': 0.61548, '15': 0.61548, '16': -0.61548, '17': -0.61548, '18': -0.61548, '19': -0.61548, '21': -0.61548, '151644': -0.61548},
    'key_tokens_len': 151645,
    'key_tokens_nonzero': {'10': 0.02892, '11': 1.0, '13': 0.02892, '14': 1.0, '15': 1.0, '16': -1.0, '17': -1.0, '18': -1.0, '19': -1.0, '21': -1.0, '151644': -1.0},
    'special_id_len': 151644,
    'special_id_values': {'151643': 0.0, '10': 0.0, '11': -1.0, '12': 0.0},
    'no_tf_nonzero': {'10': 0.00694, '11': 0.21363, '13': 0.00694, '14': 0.02936, '15': 0.02936, '16': -0.02936, '17': -0.02936, '18': -0.02936, '19': -0.02936, '21': -0.02936, '151644': -0.02936},
    'empty_wrong_nonzero': {},
    'adv_mixed': [[-0.218218, -0.218218, -0.218218, -0.218218], [-0.87287, -0.87287, -0.87287, 0.0], [1.091088, 1.091088, 1.091088, 1.091088], [0.707106, 0.707106, 0.0, 0.707106], [-0.707106, -0.707106, -0.707106, -0.707106], [-0.5, 0.0, -0.5, -0.5]],
    'adv_all_multi': [[-0.218218, -0.218218, -0.218218, -0.218218], [-0.87287, -0.87287, -0.87287, 0.0], [1.091088, 1.091088, 1.091088, 1.091088], [0.999998, 0.999998, 0.0, 0.999998], [-0.999998, -0.999998, -0.999998, -0.999998], [0.0, 0.0, 0.0, 0.0]],
    'adv_mixed_single': [[1.707106, 1.707106, 1.707106, 0.707106], [1.707106, 1.707106, 0.707106, 0.0], [1.999998, 1.999998, 1.999998, 1.999998], [0.0, 0.0, 0.0, 0.0], [-1.707106, -0.707106, -1.707106, -1.707106], [-1.707106, 0.0, -1.707106, -0.707106]],
    'single_maxid': 'IndexError',
    'small_vocab': 'IndexError',
}

# ---------- mode="distribution" / appear_table ----------


def make_inputs():
    responses = torch.tensor([[10, 11, 12, 13],
                              [10, 14, 12, 151644],
                              [11, 11, 15, 13],
                              [10, 12, 14, 16],
                              [17, 12, 18, 151644],
                              [19, 20, 21, 13]])
    mask = torch.tensor([[1, 1, 1, 1],
                         [1, 1, 1, 0],
                         [1, 1, 1, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 1],
                         [1, 0, 1, 1]])
    rewards = torch.tensor([1.0, 0.5, 2.0, 0.0, -1.0, -0.5])
    return responses, mask, rewards


def make_table():
    table = torch.zeros((151645, 4), dtype=torch.float32)
    for k, v in EXPECTED["table_data"].items():
        table[int(k)] = torch.tensor(v, dtype=torch.float32)
    return table


def nonzero_map(vec, digits=5):
    return {str(i): round(v, digits) for i, v in enumerate(vec) if v != 0}


def check_distribution(solve):
    responses, mask, rewards = make_inputs()
    cd = solve(mode="distribution", tensor=responses[rewards > 0.0],
               mask=mask[rewards > 0.0])
    # 官方 get_distribution：token id -> 出现该 token 的独立行数
    assert {str(k): v for k, v in cd.items()} == EXPECTED["dist_correct"], cd
    wd = solve(mode="distribution", tensor=responses[rewards <= 0.0],
               mask=mask[rewards <= 0.0])
    assert {str(k): v for k, v in wd.items()} == EXPECTED["dist_wrong"], wd


def check_appear_table(solve):
    responses, mask, rewards = make_inputs()
    table = solve(mode="appear_table", responses_ids=responses, mask=mask,
                  rewards=rewards)
    # 列联表 [correct_cnt, wrong_cnt, tc-cnt, tw-cnt]；cc+wc==0 的行全 0
    data = {str(i): [round(x, 5) for x in row.tolist()]
            for i, row in enumerate(table) if any(row)}
    assert data == EXPECTED["table_data"], data
    # 正确/错误组按 rewards 符号切分（>0 correct、<=0 wrong）
    assert len(table) == 151645


# ---------- mode="fisher" / "ig" / "tf" / "cohen" ----------


def check_fisher(solve):
    fmt = solve(mode="fisher", table=make_table())
    assert len(fmt) == EXPECTED["fisher_fmt_len"]
    # p==1 压成 0；只有 token 11（[2,0;1,3] 列联表）的 p=0.4 -> exp(-0.8)
    got = nonzero_map(fmt.tolist(), 8)
    exp = {k: round(v, 8) for k, v in EXPECTED["fisher_fmt_nonzero"].items()}
    assert got == exp, (got, exp)


def check_ig(solve):
    ig = solve(mode="ig", table=make_table())
    got = nonzero_map(ig.tolist(), 6)
    exp = {k: round(v, 6) for k, v in EXPECTED["ig_nonzero"].items()}
    assert got == exp, (got, exp)


def check_tf(solve):
    responses, mask, rewards = make_inputs()
    tf = solve(mode="tf", correct=responses[rewards > 0.0],
               correct_mask=mask[rewards > 0.0],
               wrong=responses[rewards <= 0.0],
               wrong_mask=mask[rewards <= 0.0], table=make_table())
    data = {str(i): [round(x, 7) for x in row.tolist()]
            for i, row in enumerate(tf) if any(row)}
    exp = {k: [round(x, 7) for x in v]
           for k, v in EXPECTED["tf_data"].items()}
    assert data == exp, data


def check_cohen(solve):
    co = solve(mode="cohen", table=make_table()).tolist()
    assert len(co) == EXPECTED["cohen_len"]
    got = nonzero_map(co, 6)
    exp = {k: round(v, 6) for k, v in EXPECTED["cohen_nonzero"].items()}
    assert got == exp, (got, exp)
    # 两组频率完全相同的 token 12 -> arcsine 差恰 0
    assert round(co[12], 9) == 0.0
    assert co[11] > 0 and co[16] < 0


# ---------- mode="key_tokens" ----------


def check_key_tokens(solve):
    responses, mask, rewards = make_inputs()
    kt = solve(mode="key_tokens", responses_ids=responses, mask=mask,
               rewards=rewards)
    assert len(kt) == EXPECTED["key_tokens_len"]
    got = nonzero_map(kt.tolist())
    exp = {k: round(v, 5) for k, v in EXPECTED["key_tokens_nonzero"].items()}
    assert got == exp, (got, exp)
    # 只在 correct 组 -> sigmoid 饱和 +1；只在 wrong 组 -> -1；同频 -> 0
    assert got["11"] == 1.0 and got["16"] == -1.0 and "12" not in got, got


def check_key_tokens_quirks(solve):
    # 151643 硬编码清零；表大小 = max token id + 1（多行组路径）
    kt3 = solve(mode="key_tokens",
                responses_ids=torch.tensor([[10, 151643, 12], [10, 11, 12]]),
                mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
                rewards=torch.tensor([1.0, 0.0]))
    assert len(kt3) == EXPECTED["special_id_len"]
    assert kt3[151643].item() == 0.0
    for tid, key in [(10, "10"), (11, "11"), (12, "12")]:
        assert round(kt3[tid].item(), 5) == \
            EXPECTED["special_id_values"][key], tid
    # 小词表 -> 硬编码行越界（官方原生 IndexError）
    try:
        solve(mode="key_tokens",
              responses_ids=torch.tensor([[10, 151643, 12], [10, 11, 12]]),
              mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
              rewards=torch.tensor([1.0, 0.0]), max_token_num=12)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    # 空 wrong 组 -> 全 0 早退
    responses, mask, rewards = make_inputs()
    kt2 = solve(mode="key_tokens", responses_ids=responses, mask=mask,
                rewards=torch.tensor([1.0, 0.5, 2.0, 1.0, 0.5, 2.0]))
    assert nonzero_map(kt2.tolist()) == EXPECTED["empty_wrong_nonzero"]
    # gamma_tf=0: x 是纯 Cohen，tf 差项不注入
    kt4 = solve(mode="key_tokens", responses_ids=responses, mask=mask,
                rewards=rewards, gamma_tf=0.0)
    got = nonzero_map(kt4.tolist())
    exp = {k: round(v, 5) for k, v in EXPECTED["no_tf_nonzero"].items()}
    assert got == exp, (got, exp)


# ---------- mode="advantage" ----------


def make_tlr():
    return torch.tensor([[0., 0., 1., 0.], [0., .5, 0., 0.],
                         [2., 0., 0., 0.], [0., 0., 0., 0.],
                         [0., -1., 0., 0.], [0., 0., -.5, 0.]])


def check_advantage(solve):
    responses, mask, _ = make_inputs()
    # group0: 3 条（scores [1,.5,2] 全 >0 -> 空 wrong 组、权重全 0、纯标准化
    # 回报 tile）；group1: 2 条（scores [0,-1] 全 <=0 -> 空 correct 组）；
    # group2: 单条（mean 0 / std 1、零权重表）
    adv = solve(mode="advantage", token_level_rewards=make_tlr(),
                responses=responses, eos_mask=mask,
                index=np.array([0, 0, 0, 1, 1, 2]))
    got = [[round(x, 6) for x in row] for row in adv.tolist()]
    exp = [[round(x, 6) for x in row] for row in EXPECTED["adv_mixed"]]
    assert got == exp, (got, exp)
    # 单条组：标准化回报 = raw/(1+1e-6)、零权重
    assert abs(got[5][0] + 0.5) < 1e-5 and got[5][1] == 0.0, got[5]


def check_advantage_all_multi(solve):
    responses, mask, _ = make_inputs()
    adv = solve(mode="advantage", token_level_rewards=make_tlr(),
                responses=responses, eos_mask=mask,
                index=np.array([0, 0, 0, 1, 1, 1]))
    got = [[round(x, 6) for x in row] for row in adv.tolist()]
    exp = [[round(x, 6) for x in row] for row in EXPECTED["adv_all_multi"]]
    assert got == exp, (got, exp)
    # group1 两行 score 0 与 -1：unbiased std=sqrt(0.5)、(0.5)/std=0.7071
    # tile + key token 权重（row4 的 16/17/18 wrong 独占 -> 权重 -1）
    assert got[3][0] == 0.999998 and got[4][0] == -0.999998, (got[3], got[4])


def check_advantage_mixed_single(solve):
    responses, mask, _ = make_inputs()
    # g0=rows0+5、g3=rows1+4（多行，含 max id）、g1=row2 单条、g2=row3 单条
    adv = solve(mode="advantage", token_level_rewards=make_tlr(),
                responses=responses, eos_mask=mask,
                index=np.array([0, 3, 1, 2, 3, 0]))
    got = [[round(x, 6) for x in row] for row in adv.tolist()]
    exp = [[round(x, 6) for x in row]
           for row in EXPECTED["adv_mixed_single"]]
    assert got == exp, (got, exp)
    # 单条组行是纯 (raw-mean0)/(std1+1e-6) 的 eos tile：row2 score=2
    assert abs(got[2][0] - 2.0) < 1e-5, got[2]


def check_advantage_single_maxid(solve):
    responses, mask, _ = make_inputs()
    # 单条组含全局最大 token id（row1 独占一组）：官方零表大小
    # responses.max()（无 +1）-> IndexError（原生怪癖）
    try:
        solve(mode="advantage", token_level_rewards=make_tlr(),
              responses=responses, eos_mask=mask,
              index=np.array([0, 1, 0, 2, 2, 2]))
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
