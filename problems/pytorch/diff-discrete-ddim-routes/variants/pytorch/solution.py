import math

import torch


def _integer(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    value = int(value)
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def solve(logits, tokens, step_index, num_inference_steps, use_loo=False):
    """离散 DDIM（均匀腐蚀）单步的三路路由质量。

    logits 形如 (B, L, K)：每个位置的 K 类词表 logits；
    tokens 形如 (B, L) 的整数张量：当前 block 的 token，取值 [0, K)。

    时间账（alpha 越大越接近干净端）：N = num_inference_steps，
      alpha_t = step_index / N， alpha_s = (step_index + 1) / N
      a = alpha_t / alpha_s   （从 s 存活到 t 的概率）
      b = alpha_s

    先取贪心去噪结果 x0 = argmax(logits, -1)。use_loo=True 时先做 leave-one-out 修正：
    当 0 < alpha_t < 1，把「已观测 token」位置的 logit 减去
      delta = log1p(K * alpha_t / (1 - alpha_t))
    再取 argmax（alpha_t = 0 时不修正）。

    记 same = (tokens == x0)，三路质量为
      clean = b * (1 - a) / K + a * b * same
      stay  = a * (1 - b) / K
      noise = (1 - a) * (1 - b) / K
    再按三者之和归一化（注意这三元组本身并不总是和为 1：
    same 为假时和为 (1 - a*b)/K，所以这个除法不是摆设）。

    返回 (route_probs, pred_tokens)：
      route_probs 为 float32、形状 (B, L, 3)，最后一维依次是 clean / stay / noise；
      pred_tokens 为 int64、形状 (B, L)，即（可能经 LOO 修正后的）argmax。
    """
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits 必须是 torch.Tensor")
    if logits.dim() != 3:
        raise ValueError("logits 必须是三维张量 (B, L, K)")
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("tokens 必须是 torch.Tensor")
    if tokens.dim() != 2:
        raise ValueError("tokens 必须是二维张量 (B, L)")
    if tuple(tokens.shape) != tuple(logits.shape[:2]):
        raise ValueError("tokens 的形状必须与 logits 的前两维一致")
    if tokens.is_floating_point() or tokens.is_complex():
        raise ValueError("tokens 必须是整数张量")
    if logits.shape[-1] < 2:
        raise ValueError("词表大小 K 至少为 2")

    current = _integer("step_index", step_index, 0)
    steps = _integer("num_inference_steps", num_inference_steps, 1)
    if current >= steps:
        raise ValueError("step_index 必须落在 [0, num_inference_steps) 内")

    vocab = logits.shape[-1]
    token_ids = tokens.to(torch.int64)
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab:
        raise ValueError("tokens 的取值必须落在 [0, K) 内")

    alpha_t = current / steps
    alpha_s = (current + 1) / steps

    shifted = logits.to(torch.float64)
    if use_loo and 0.0 < alpha_t < 1.0:
        delta = math.log1p(vocab * alpha_t / (1.0 - alpha_t))
        damp = torch.full((*token_ids.shape, 1), -delta, dtype=torch.float64)
        shifted = shifted.scatter_add(-1, token_ids.unsqueeze(-1), damp)

    pred = shifted.argmax(dim=-1)

    survival = alpha_t / alpha_s
    same = (token_ids == pred).to(torch.float64)
    clean = alpha_s * (1.0 - survival) / vocab + survival * alpha_s * same
    stay = survival * (1.0 - alpha_s) / vocab * torch.ones_like(same)
    noise = (1.0 - survival) * (1.0 - alpha_s) / vocab * torch.ones_like(same)

    routes = torch.stack([clean, stay, noise], dim=-1)
    routes = routes / routes.sum(dim=-1, keepdim=True)
    return routes.to(torch.float32), pred.to(torch.int64)
