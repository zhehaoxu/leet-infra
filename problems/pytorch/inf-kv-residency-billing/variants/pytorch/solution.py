import torch


def _cum_blocks(last_token, block_size):
    """Σ_{t=0..last_token} ceil(t / block_size)，闭式；last_token < 0 时返回 0。"""
    if last_token < 0:
        return 0
    whole, rest = divmod(last_token, block_size)
    return block_size * whole * (whole + 1) // 2 + rest * (whole + 1)


def solve(prompt_len, start, out_tokens, block_size, price_out, price_block_step):
    if block_size < 1:
        raise ValueError("block_size 必须为正")
    if price_out < 0 or price_block_step < 0:
        raise ValueError("单价不能为负")
    for name, t in (("prompt_len", prompt_len), ("start", start), ("out_tokens", out_tokens)):
        if not isinstance(t, torch.Tensor) or t.dim() != 1:
            raise ValueError(f"{name} 必须是一维张量")
    n = int(prompt_len.numel())
    if int(start.numel()) != n or int(out_tokens.numel()) != n:
        raise ValueError("三个输入长度必须一致")
    ps = prompt_len.to(torch.int64).tolist()
    ss = start.to(torch.int64).tolist()
    os_ = out_tokens.to(torch.int64).tolist()
    if n:
        if min(ps) < 1:
            raise ValueError("prompt_len 至少为 1")
        if min(ss) < 0:
            raise ValueError("start 不能为负")
        if min(os_) < 0:
            raise ValueError("out_tokens 不能为负")

    residency = [0] * n
    per_step = {}
    for r in range(n):
        # 区间 [prompt, prompt + out - 1] 的 ceil(t / block_size) 求和
        hi = ps[r] + os_[r] - 1
        residency[r] = _cum_blocks(hi, block_size) - _cum_blocks(ps[r] - 1, block_size)
        for k in range(os_[r]):
            t = ss[r] + k
            per_step[t] = per_step.get(t, 0) + (ps[r] + k + block_size - 1) // block_size

    charge = [
        float(price_out) * os_[r] + float(price_block_step) * residency[r] for r in range(n)
    ]
    return (
        torch.tensor(residency, dtype=torch.int64),
        torch.tensor(charge, dtype=torch.float32),
        torch.tensor(max(per_step.values(), default=0), dtype=torch.int64),
        torch.tensor(sum(charge), dtype=torch.float32),
    )
