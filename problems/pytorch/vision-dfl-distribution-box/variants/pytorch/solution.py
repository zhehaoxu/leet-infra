import torch


def solve(logits, targets=None, stride=1.0, reg_max=7, weights=None):
    if not isinstance(reg_max, int) or isinstance(reg_max, bool) or reg_max < 1:
        raise ValueError("reg_max 必须是 >=1 的整数")
    if logits.dim() != 3:
        raise ValueError("logits 形状必须是 (N, 4, reg_max+1)")
    if logits.shape[2] != reg_max + 1:
        raise ValueError(f"logits 最后一维应为 {reg_max + 1}，实际 {logits.shape[2]}")
    if not (stride > 0.0) or not torch.isfinite(torch.tensor(float(stride))):
        raise ValueError("stride 必须是有限正数")

    x = logits.to(torch.float64)
    probs = torch.softmax(x, dim=-1)
    bins = torch.arange(reg_max + 1, dtype=torch.float64)
    dist = (probs * bins).sum(dim=-1) * float(stride)

    out = {"probs": probs, "dist": dist, "bin_index": bins.clone()}
    if targets is None:
        return out

    t = targets.to(torch.float64)
    if t.shape != dist.shape:
        raise ValueError(f"targets 形状应为 {tuple(dist.shape)}，实际 {tuple(t.shape)}")
    if bool((t < 0).any()) or bool((t > reg_max - 1).any()):
        raise ValueError(f"targets 必须落在 [0, {reg_max - 1}]（右侧 bin 要可索引）")

    left = t.long()
    right = left + 1
    w_left = right.to(torch.float64) - t
    w_right = t - left.to(torch.float64)
    logp = torch.log_softmax(x, dim=-1)
    ce_left = -logp.gather(-1, left.unsqueeze(-1)).squeeze(-1)
    ce_right = -logp.gather(-1, right.unsqueeze(-1)).squeeze(-1)
    loss = ce_left * w_left + ce_right * w_right
    if weights is not None:
        w = weights.to(torch.float64)
        if w.shape != dist.shape:
            raise ValueError(f"weights 形状应为 {tuple(dist.shape)}，实际 {tuple(w.shape)}")
        loss = loss * w
    out.update({
        "left": left,
        "right": right,
        "w_left": w_left,
        "w_right": w_right,
        "loss": loss,
        "loss_mean": loss.sum() / loss.numel(),
    })
    return out
