import torch


def solve(float_batches, quant_batches, channel_axis, init_bias=None, post_bias=True,
          channel_size=None):
    if float_batches.shape != quant_batches.shape:
        raise ValueError("float_batches and quant_batches must share a shape")
    if float_batches.dtype != quant_batches.dtype:
        raise ValueError("float_batches and quant_batches must share a dtype")
    if float_batches.dim() < 3:
        raise ValueError("the single-forward rank must be >= 2")
    if float_batches.shape[0] == 0:
        raise ValueError("at least one calibration output pair is required")

    d = float_batches.dim() - 1
    if isinstance(channel_axis, (tuple, list)):
        raw = tuple(channel_axis)
    elif isinstance(channel_axis, int):
        raw = (channel_axis,)
    else:
        raise ValueError("channel_axis must be an int or a tuple/list of ints")
    if len(raw) == 0:
        raise ValueError("channel_axis must not be empty")
    if any(not isinstance(a, int) or isinstance(a, bool) for a in raw):
        raise ValueError("channel_axis entries must be ints")
    norm = tuple(a + d if a < 0 else a for a in raw)
    if any(a < 0 or a >= d for a in norm):
        raise ValueError(f"channel_axis {raw} is out of range for rank {d}")
    if len(set(norm)) != len(norm):
        raise ValueError("channel_axis entries must be distinct")

    spatial = tuple(float_batches.shape[1:])
    bias_shape_list = []
    mean_axis = []
    for i, size in enumerate(spatial):
        if i in norm:
            bias_shape_list.append(size)
        else:
            bias_shape_list.append(1)
            mean_axis.append(i)
    bias_shape = tuple(bias_shape_list)
    mean_axis = tuple(mean_axis)
    channels = 1
    for size in bias_shape:
        channels *= size
    if channel_size is not None and int(channel_size) != channels:
        raise ValueError(f"channel_size {channel_size} != prod(bias_shape) {channels}")

    work = torch.float64
    Fb = float_batches.to(work)
    Qb = quant_batches.to(work)
    T = Fb.shape[0]

    if init_bias is None:
        bias = torch.zeros(channels, dtype=work)
    else:
        if init_bias.numel() != channels:
            raise ValueError(f"init_bias has {init_bias.numel()} elements, expected {channels}")
        bias = init_bias.to(work).reshape(-1).clone()

    updates = []
    snapshot = []
    for t in range(T):
        delta = Fb[t] - Qb[t]
        if post_bias:
            delta = delta + bias.reshape(bias_shape)
        if len(mean_axis) > 0:
            reduced = delta.mean(dim=mean_axis)
        else:
            reduced = delta
        u = reduced.reshape(-1)
        updates.append(u)
        bias = (bias * t + u) / (t + 1)
        snapshot.append(bias.clone())

    bias_updates = torch.stack(updates, dim=0)
    bias_after = torch.stack(snapshot, dim=0)
    final = bias.reshape((1,) + bias_shape)
    corrected = Qb + final
    residual_mse = ((Fb - corrected) ** 2).mean()

    out_dtype = float_batches.dtype
    return {
        "channel_axis_norm": norm,
        "bias_shape": bias_shape,
        "mean_axis": mean_axis,
        "bias_updates": bias_updates.to(out_dtype),
        "bias_after": bias_after.to(out_dtype),
        "bias": bias.to(out_dtype),
        "corrected_output": corrected.to(out_dtype),
        "residual_mse": residual_mse.to(out_dtype),
    }
