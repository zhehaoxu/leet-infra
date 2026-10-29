import torch


def solve(W, tp_size, mode, x, targets=None):
    """Compute a bias-free tensor-parallel linear layer on CPU.

    Inputs:
        W: CPU floating tensor [out_features, in_features].
        x: CPU floating tensor [batch, in_features], same dtype as W.
           Dimensions are positive; tests use float32/float64 and finite logits.
        tp_size: positive integer; the split dimension must divide evenly.
        mode: 'column', 'row', or 'vocab'.
        targets: CPU integer tensor [batch], required only for 'vocab'.
                 Its values are guaranteed in [0, out_features). Ignored in
                 other modes; defaults to None. In vocab mode out_features=V.

    Return a dict with exactly these keys:
        shards: list of tp_size tensors, in rank order, preserving W's dtype
                and values. Split contiguous index ranges of W on axis 0 for
                column/vocab, axis 1 for row. Views or copies are both allowed.
        label_map: {global_id: (rank, local_id)} for all V vocabulary IDs in
                   vocab mode; {} otherwise. IDs use the current row order.
        output: tensor [batch, out_features], same dtype as W, equal to x @ W.T.
                Column: concatenate x @ shard.T along output axis 1.
                Row: split x on axis 1 too, then sum partial matrix products.
                Vocab: concatenate the rank-local logits along axis 1.
        losses: tensor [batch], same dtype as W, of unreduced cross entropy
                in vocab mode; None otherwise. Compute from local logits via
                per-sample global MAX, SUM of local shifted exponent sums,
                and SUM of masked target-logit contributions from label owners.
                Do not compute CE from dense concatenated logits.

    Do not mutate W, x, or targets. No bias, real distributed processes,
    collectives, or backward pass are required. Raise ValueError for an invalid
    mode or a nondivisible split dimension; other inputs satisfy this contract.
    """
    raise NotImplementedError
