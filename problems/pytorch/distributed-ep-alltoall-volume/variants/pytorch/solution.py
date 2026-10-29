import torch


def solve(routing, bytes_per_token, bandwidth):
    """MoE EP all-to-all 通信量与 straggler 账。"""
    W, E = routing.shape
    expert_rank = torch.arange(E, device=routing.device) % W
    send_counts = torch.zeros(W, W, dtype=torch.int64, device=routing.device)
    for r in range(W):
        for q in range(W):
            mask = expert_rank == q
            send_counts[r, q] = int(routing[r][mask].sum())
    recv_counts = send_counts.T
    per_rank_total = (
        send_counts.sum(dim=1) + send_counts.sum(dim=0) - 2 * send_counts.diagonal()
    ) * bytes_per_token
    max_load = int(per_rank_total.max())
    avg = float(per_rank_total.to(torch.float64).mean())
    imbalance = float(max_load) / avg - 1.0 if avg > 0 else 0.0
    wall_clock = float(max_load) / float(bandwidth)
    return send_counts, recv_counts, per_rank_total, max_load, imbalance, wall_clock
