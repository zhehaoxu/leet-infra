import torch


def solve(routing, bytes_per_token, bandwidth):
    """Single EP dispatch accounting / 单次 EP dispatch 通信量账本。

    Valid-input contract / 合法输入约定（不要求校验非法输入）:
        routing: CPU torch.int64 tensor, shape (W, E), W >= 1, E >= 0, nonnegative
            counts; routing[r, e] counts tokens from rank r to expert e.
            CPU 非负 token 数张量；专家 e 属于 rank e % W，E 不必被 W 整除。
        bytes_per_token: positive Python int, excluding bool / 每 token 字节数，正整数。
        bandwidth: finite positive Python int or float, excluding bool; each
            rank's AGGREGATE network send+receive bytes/second for ONE dispatch.
            每 rank 单次 dispatch 的网络发送加接收总带宽（字节/秒）。
        Counts and intermediate integer sums/products must fit int64; it is
            sufficient that 2 * routing.sum() * bytes_per_token <= 2**63 - 1.
            计数及中间整数运算须在 int64 范围内，上式为充分条件。

    Returns / 返回:
        (send_counts, recv_counts, per_rank_bytes, max_load_bytes, imbalance, wall_clock):
        send_counts: int64 (W, W), tokens r -> q, INCLUDING local diagonal routes.
            发送路由计数，包含本地对角项。
        recv_counts: int64 (W, W), tokens r <- q; transpose of send_counts / 发送矩阵转置。
        per_rank_bytes: int64 (W,), OFF-DIAGONAL send+receive network bytes / 排除本地的收发字节。
        All output tensors share routing.device / 所有输出张量与 routing 同设备。
        max_load_bytes: Python int, largest per-rank network byte count / 最大网络负载。
        imbalance: Python float, max/mean - 1; zero for all-zero network load.
            最大值/平均值 - 1；全部网络负载为零时返回 0.0；均值不可降为 float32。
        wall_clock: Python float, max_load_bytes/bandwidth, in seconds / 秒。
        Only one dispatch is counted: no reverse combine, computation latency,
            metadata, or token dropping; not a full-duplex max(send, receive) model.
            仅计单次 dispatch，不计反向 combine、计算延迟、元数据或 token 丢弃，
            不使用全双工 max(发送, 接收) 带宽模型。
    """
    raise NotImplementedError
