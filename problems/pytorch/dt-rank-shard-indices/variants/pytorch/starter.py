import torch


def solve(order, num_replicas, rank, drop_last):
    """中文契约 / English contract
    order 是 CPU int64 `[N]`、N >= 1，内容必须是 0..N-1 的一个排列（每个下标恰好出现一次，
    否则 ValueError）——它表示 shuffle 之后已经定好的全局样本顺序，本题不碰任何随机数 /
        order is a CPU int64 [N] tensor holding a permutation of 0..N-1; no RNG is involved.
    num_replicas、rank 是 python int（bool 不算），num_replicas >= 1、0 <= rank < num_replicas；
    drop_last 是 python bool。
    令 R = num_replicas、N = len(order)：只有 drop_last 为真且 N mod R != 0 时
    num_samples = floor(N/R)，其余一律 ceil(N/R)；total_size = num_samples * R。
    effective 序列：drop_last 为真时取 order[:total_size]（砍掉尾巴）；否则在 order 后面接
    padding = total_size - N 个元素——padding <= N 时接 order[:padding]，padding > N 时把整个
    order 重复若干遍再截 padding 个 /
        pad from the head, repeating the whole list when padding exceeds N.
    本 rank 的分片是 effective 的跨步切片 effective[rank:total_size:R]（含左不含右、从 rank 起、
    步长 R），长度恰为 num_samples / strided slice starting at rank with step R.
    coverage[i] 是样本下标 i 在**所有 rank 的 effective 序列**里出现的次数，长度 N /
        coverage counts occurrences over the whole effective sequence, not just this rank.
    不修改输入；非法输入抛 ValueError / never mutate inputs; raise ValueError on invalid input.
    返回 / Returns (shard int64 [num_samples], num_samples int, total_size int, coverage int64 [N])。
    """
    raise NotImplementedError
