import torch


def solve(W, tp_size, mode, x, targets=None):
    """Simulate TP linear operations and vocab-parallel cross entropy on CPU."""
    if mode not in ('column', 'row', 'vocab'):
        raise ValueError("mode must be 'column', 'row', or 'vocab'")
    dim = 1 if mode == 'row' else 0
    if W.shape[dim] % tp_size != 0:
        raise ValueError("the split dimension must be divisible by tp_size")

    shards = list(W.chunk(tp_size, dim=dim))
    label_map = {}
    losses = None
    if mode == 'row':
        output = x.new_zeros((x.shape[0], W.shape[0]))
        for x_shard, weight_shard in zip(x.chunk(tp_size, dim=1), shards):
            output = output + x_shard @ weight_shard.T
    else:
        local_logits = [x @ shard.T for shard in shards]
        if mode == 'vocab':
            width = W.shape[0] // tp_size
            label_map = {v: (v // width, v % width) for v in range(W.shape[0])}
            labels = targets.to(dtype=torch.long)
            owners = labels // width

            # Per-sample MAX reduction over ranks, not a batch-wide maximum.
            global_max = torch.stack(
                [logits.amax(dim=1) for logits in local_logits]
            ).amax(dim=0)
            exp_sum = torch.zeros_like(global_max)
            target_logit = torch.zeros_like(global_max)
            for rank, logits in enumerate(local_logits):
                shifted = logits - global_max[:, None]
                exp_sum = exp_sum + shifted.exp().sum(dim=1)

                # Nonowners must contribute zero. Clamp only to make their
                # unused gather indices safe; the owner uses the exact index.
                local_ids = (labels - rank * width).clamp(0, width - 1)
                selected = shifted.gather(1, local_ids[:, None]).squeeze(1)
                target_logit = target_logit + torch.where(
                    owners == rank, selected, torch.zeros_like(selected)
                )
            # Both quantities use shifted logits, avoiding cancellation from
            # adding a huge common offset back before subtracting the target.
            losses = exp_sum.log() - target_logit

        # Gathered logits are returned for inspection, not used to compute CE.
        output = torch.cat(local_logits, dim=1)

    return {
        'shards': shards,
        'label_map': label_map,
        'output': output,
        'losses': losses,
    }
