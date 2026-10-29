import torch


def solve(x, gate_w, bias, group_count, top_groups, k, gamma, update_bias=False):
    experts = gate_w.shape[0]
    if type(group_count) is not int or type(top_groups) is not int or type(k) is not int:
        raise ValueError("group_count, top_groups and k must be integers")
    if group_count < 1 or experts % group_count:
        raise ValueError("experts must be divisible by group_count")
    group_size = experts // group_count
    if group_size < 2:
        raise ValueError("each group needs at least two experts")
    if not 1 <= top_groups <= group_count:
        raise ValueError("top_groups out of range")
    if not 1 <= k <= top_groups * group_size:
        raise ValueError("k out of range")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")

    scores = torch.sigmoid(x @ gate_w.T)
    choice = scores + bias
    grouped = choice.reshape(x.shape[0], group_count, group_size)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.argsort(
        group_scores, dim=-1, descending=True, stable=True
    )[:, :top_groups]
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.repeat_interleave(group_size, dim=1)
    masked = choice.masked_fill(~expert_mask, -torch.inf)
    indices = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :k]
    selected = scores.gather(1, indices)
    weights = selected / selected.sum(dim=-1, keepdim=True)
    load = torch.bincount(indices.reshape(-1), minlength=experts).to(torch.int64)
    if update_bias:
        average = x.shape[0] * k / experts
        new_bias = bias + gamma * torch.sign(torch.as_tensor(average, dtype=bias.dtype) - load)
    else:
        new_bias = bias.clone()
    return weights, indices, scores, group_scores, selected_groups, load, new_bias
