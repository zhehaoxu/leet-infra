import torch


def solve(requests, max_batch_size):
    """FCFS continuous batching。requests (N,2) long: [arrival_step, gen_len]。
    返回 finish (N,) long：每请求完成 step。
    """
    # TODO
    raise NotImplementedError
