"""参考答案：DistributedSampler 单机模拟的完整索引账——epoch 决定置换、补齐重复、rank 切片。"""

from torch.utils.data.distributed import DistributedSampler

N = 7
WORLD = 3


class _DS:
    def __len__(self):
        return N


def solve():
    def ranks(shuffle: bool, drop_last: bool, epoch: int):
        out = []
        for r in range(WORLD):
            s = DistributedSampler(_DS(), num_replicas=WORLD, rank=r,
                                   shuffle=shuffle, drop_last=drop_last, seed=0)
            s.set_epoch(epoch)
            out.append(list(s))
        return out

    s_ref = DistributedSampler(_DS(), num_replicas=WORLD, rank=0, shuffle=True, seed=0)
    return {
        "num_samples": s_ref.num_samples,
        "total_size": s_ref.total_size,
        "epoch0_ranks": ranks(True, False, 0),
        "epoch1_ranks": ranks(True, False, 1),
        "epoch0_again": ranks(True, False, 0),
        "drop_last_epoch0_ranks": ranks(True, True, 0),
        "no_shuffle_ranks": ranks(False, False, 3),
    }
