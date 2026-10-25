import torch
from torch.utils.data.distributed import DistributedSampler

N = 7
WORLD = 3


def _oracle_perm(epoch: int) -> list[int]:
    """独立重算：只用 Generator+randperm 拿基础置换，其余（补齐/切片）纯 Python 手写，
    完全不碰 DistributedSampler。"""
    g = torch.Generator()
    g.manual_seed(0 + epoch)  # seed 参数为 0，引擎种子 = seed + epoch
    return torch.randperm(N, generator=g).tolist()


def _oracle_pad(perm: list[int]) -> list[int]:
    total = -(-N // WORLD) * WORLD  # ceil(N/W)*W = 9
    return perm + perm[: total - N]


def _oracle_ranks(full: list[int]) -> list[list[int]]:
    return [full[r::WORLD] for r in range(WORLD)]


def _oracle(shuffle: bool, drop_last: bool, epoch: int):
    if shuffle:
        base = _oracle_perm(epoch)
    else:
        base = list(range(N))
    if drop_last:
        total = (N // WORLD) * WORLD
        return _oracle_ranks(base[:total]), N // WORLD, total
    total = -(-N // WORLD) * WORLD
    return _oracle_ranks(base + base[: total - N]), -(-N // WORLD), total


def check_example(solve):
    out = solve()
    (ranks, ns, ts) = _oracle(True, False, 0)
    assert out["num_samples"] == ns == 3 and out["total_size"] == ts == 9
    assert out["epoch0_ranks"] == ranks, f"epoch0 账不符: {out['epoch0_ranks']}"
    (r1, _, _) = _oracle(True, False, 1)
    assert out["epoch1_ranks"] == r1
    (rd, nsd, tsd) = _oracle(True, True, 0)
    assert out["drop_last_epoch0_ranks"] == rd and nsd == 2 and tsd == 6
    (rn, _, _) = _oracle(False, False, 3)
    assert out["no_shuffle_ranks"] == rn == [[0, 3, 6], [1, 4, 0], [2, 5, 1]]


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    for name, args in [("epoch0_ranks", (True, False, 0)),
                       ("epoch1_ranks", (True, False, 1)),
                       ("drop_last_epoch0_ranks", (True, True, 0)),
                       ("no_shuffle_ranks", (False, False, 3))]:
        (ranks, _, _) = _oracle(*args)
        assert out[name] == ranks, f"{name} 与手写重算不符"


def check_epoch_determinism(solve):
    # 同一 epoch 重放必须逐位一致；不同 epoch 必须换置换
    out = solve()
    assert out["epoch0_again"] == out["epoch0_ranks"], "set_epoch(0) 重跑应完全一致"
    assert out["epoch1_ranks"] != out["epoch0_ranks"], "epoch 变了 shuffle 结果必须变"


def check_padding_duplicates(solve):
    # drop_last=False：三个 rank 的并比数据集多出 2 个重复样本，且重复项正是补齐位
    out = solve()
    all_idx = [i for rank in out["epoch0_ranks"] for i in rank]
    assert len(all_idx) == out["total_size"] == 9
    assert all(len(rank) == out["num_samples"] for rank in out["epoch0_ranks"])
    perm = _oracle_perm(0)
    padded = _oracle_pad(perm)
    dup = padded[N:]
    # padded 序列按 stride 切三份，还原后必须等于 padded
    assert [padded[r::WORLD] for r in range(WORLD)] == out["epoch0_ranks"]
    # 补齐位就是置换开头两个元素
    assert dup == perm[:2]


def check_drop_last_truncates(solve):
    # drop_last=True：砍尾而非补齐，每 rank 恰 2 个，三 rank 并集 = perm 前 6 项
    out = solve()
    perm = _oracle_perm(0)
    dl = out["drop_last_epoch0_ranks"]
    assert all(len(r) == 2 for r in dl)
    rebuilt = [None] * 6
    for r, rank in enumerate(dl):
        for k, v in enumerate(rank):
            rebuilt[r + 3 * k] = v
    assert rebuilt == perm[:6], f"drop_last 截尾账不符: {rebuilt} vs {perm[:6]}"


def check_no_shuffle_is_range_slice(solve):
    out = solve()
    rn = out["no_shuffle_ranks"]
    full = list(range(N)) + [0, 1]
    assert [full[r::WORLD] for r in range(WORLD)] == rn
