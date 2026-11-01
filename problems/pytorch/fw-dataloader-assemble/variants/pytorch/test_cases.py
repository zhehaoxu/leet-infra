import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

N, BS = 7, 3
KEYS = [
    "seq_drop_false", "seq_drop_true",
    "shuf_e0_flat", "shuf_e1_flat", "shuf_e0_batches", "shuf_e1_batches",
    "shuf_drop_true_batches", "shuf_replay_flat", "shuf_seed1_flat",
    "repl_n7", "repl_n100", "randperm_mod0_flat", "quirk_probe_perm",
    "len_drop_true", "len_drop_false", "len_bs_gt_n_true", "len_bs_gt_n_false",
]


def _batches(idx, drop_last):
    return [[int(x) for x in b]
            for b in DataLoader(idx, batch_size=BS, drop_last=drop_last)]


def _py():
    """oracle：直接用真 DataLoader / RandomSampler（官方类本身作锚点）。"""
    ds = range(N)
    out = {}
    out["seq_drop_false"] = _batches(ds, False)
    out["seq_drop_true"] = _batches(ds, True)

    g = torch.Generator().manual_seed(0)
    loader = DataLoader(ds, batch_size=BS, shuffle=True,
                        drop_last=False, generator=g)
    e0 = [list(b) for b in loader]     # epoch0 消耗 g
    e1 = [list(b) for b in loader]     # epoch1 同 generator 延续
    e0 = [[int(x) for x in b] for b in e0]
    e1 = [[int(x) for x in b] for b in e1]
    out["shuf_e0_flat"] = [i for b in e0 for i in b]
    out["shuf_e1_flat"] = [i for b in e1 for i in b]
    out["shuf_e0_batches"] = e0
    out["shuf_e1_batches"] = e1

    gd = torch.Generator().manual_seed(0)
    ldt = DataLoader(ds, batch_size=BS, shuffle=True,
                     drop_last=True, generator=gd)
    out["shuf_drop_true_batches"] = [[int(x) for x in b] for b in ldt]

    g2 = torch.Generator().manual_seed(0)
    out["shuf_replay_flat"] = list(RandomSampler(ds, generator=g2))
    g3 = torch.Generator().manual_seed(1)
    out["shuf_seed1_flat"] = list(RandomSampler(ds, generator=g3))

    gr = torch.Generator().manual_seed(0)
    out["repl_n7"] = list(RandomSampler(ds, replacement=True,
                                        num_samples=7, generator=gr))
    gr2 = torch.Generator().manual_seed(1)
    out["repl_n100"] = list(RandomSampler(ds, replacement=True,
                                          num_samples=100, generator=gr2))

    gq = torch.Generator().manual_seed(0)
    s = list(RandomSampler(ds, replacement=False, num_samples=14, generator=gq))
    out["randperm_mod0_flat"] = s
    out["quirk_probe_perm"] = torch.randperm(N, generator=gq).tolist()

    out["len_drop_true"] = len(DataLoader(ds, batch_size=BS, drop_last=True))
    out["len_drop_false"] = len(DataLoader(ds, batch_size=BS, drop_last=False))
    out["len_bs_gt_n_true"] = len(DataLoader(ds, batch_size=10, drop_last=True))
    out["len_bs_gt_n_false"] = len(DataLoader(ds, batch_size=10, drop_last=False))
    return out


def _compare(solve):
    led = solve()
    ref = _py()
    for k in KEYS:
        assert k in led, k
    for k in KEYS:
        assert led[k] == ref[k], (k, led[k], ref[k])
    return led


def check_sequential(solve):
    led = _compare(solve)
    assert led["seq_drop_false"] == [[0, 1, 2], [3, 4, 5], [6]]
    assert led["seq_drop_true"] == [[0, 1, 2], [3, 4, 5]]


def check_shuffle_epochs(solve):
    led = _compare(solve)
    # 同 generator 延续：e1 != e0
    assert led["shuf_e0_flat"] != led["shuf_e1_flat"]
    assert sorted(led["shuf_e0_flat"]) == list(range(N))
    assert sorted(led["shuf_e1_flat"]) == list(range(N))
    assert led["shuf_replay_flat"] != led["shuf_e0_flat"]  # _base_seed 消耗差
    assert led["shuf_replay_flat"] == torch.randperm(
        N, generator=torch.Generator().manual_seed(0)).tolist()
    assert led["shuf_seed1_flat"] != led["shuf_e0_flat"]


def check_replacement(solve):
    led = _compare(solve)
    assert len(led["repl_n7"]) == 7
    assert len(led["repl_n100"]) == 100
    assert all(0 <= i < N for i in led["repl_n100"])
    # 有放回必然有重复
    assert len(set(led["repl_n100"])) < 100


def check_quirk(solve):
    led = _compare(solve)
    assert len(led["randperm_mod0_flat"]) == 14
    assert sorted(led["randperm_mod0_flat"]) == sorted(list(range(N)) * 2)
    # quirk 锚点：第三次空消耗后第四次置换 != 无空消耗情形的第三次置换
    g_no_quirk = torch.Generator().manual_seed(0)
    _ = torch.randperm(N, generator=g_no_quirk).tolist()
    _ = torch.randperm(N, generator=g_no_quirk).tolist()
    assert led["quirk_probe_perm"] != torch.randperm(N, generator=g_no_quirk).tolist()


def check_len(solve):
    led = _compare(solve)
    assert (led["len_drop_true"], led["len_drop_false"]) == (2, 3)
    assert (led["len_bs_gt_n_true"], led["len_bs_gt_n_false"]) == (0, 1)


def check_invalid_batch_size(solve):
    """官方 BatchSampler 校验：batch_size<=0 或 bool -> ValueError，
    drop_last 非 bool -> ValueError（BatchSampler 层）。"""
    from torch.utils.data.sampler import BatchSampler, SequentialSampler
    for bs in (0, -1, True, 2.5, "3"):
        try:
            list(BatchSampler(SequentialSampler(range(5)), bs, False))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for bs={bs}")
    try:
        list(BatchSampler(SequentialSampler(range(5)), 2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for drop_last=1")


def check_invalid_num_samples(solve):
    """官方 RandomSampler 校验：num_samples 非 int 或 <=0 -> ValueError，
    replacement 非 bool -> TypeError（两类异常不同，题面钉官方）。
    quirk：bool num_samples（True）官方**放行**（是 int 子类且 >0，
    被当 1 用）——与 BatchSampler 显式拒 bool batch_size 不同。"""
    ds = range(5)
    for ns in (0, -1, 2.5):
        try:
            list(RandomSampler(ds, replacement=True, num_samples=ns,
                               generator=torch.Generator()))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for ns={ns}")
    try:
        list(RandomSampler(ds, replacement=1, generator=torch.Generator()))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for replacement=1")
    # bool num_samples 放行：True 被当 1，恰出一个索引
    got = list(RandomSampler(ds, replacement=True, num_samples=True,
                             generator=torch.Generator()))
    assert len(got) == 1
