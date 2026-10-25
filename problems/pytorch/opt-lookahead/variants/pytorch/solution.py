"""参考答案：Lookahead 优化器封装器（k steps forward, 1 step back）——继承 torch.optim.Optimizer。

wrapper 内部实例化一个**真正的** base 优化器（base="sgd" → 动量 SGD，base="adam" → Adam），
每次 `step()` 严格按三步走：
  1) fast 步计数先不加：先让 base 走一步（训练权重 θ 就地更新，base 自己的动量/二阶矩照常累积，
     同步**绝不**重置或重建这些状态）；
  2) 计数 t ← t + 1（构造时 t=0）；
  3) 若 t 是 k 的正整数倍 → 慢更新 s ← s + α·(θ − s) 在先，重置 θ ← s 在后（顺序不可颠倒）。
因此第一次同步发生在第 k 次 `step()` 之后；慢权重初值是构造时 θ 的 clone。

α=1 时同步对 θ 是恒等（s 被抬到 θ，再写回同一个 θ）→ 轨迹与裸 base 完全一致；
α=0 时慢权重永远不收新东西，同步等于把 θ 硬回滚到上一次同步点；
k=1 时每步同步，等价于把 base 的每一步增量整体缩放 α（θ_T = θ_0 + α·Σ_t d_t）。
"""

import torch


class _Lookahead(torch.optim.Optimizer):
    def __init__(self, params, base, lr, momentum, k, alpha, betas, eps):
        super().__init__(params, dict(lr=lr, momentum=momentum, betas=betas, eps=eps))
        if base == "sgd":
            self.base_optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=0.0, dampening=0.0, nesterov=False
            )
        elif base == "adam":
            self.base_optimizer = torch.optim.Adam(
                params, lr=lr, betas=betas, eps=eps, weight_decay=0.0, amsgrad=False
            )
        else:
            raise ValueError(f"未知 base：{base}（只支持 'sgd' / 'adam'）")
        self.k = k
        self.alpha = alpha
        self._t = 0
        self.slow_weights = [p.detach().clone() for group in self.param_groups for p in group["params"]]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.base_optimizer.step()  # fast 权重走一步；base 状态照常演化
        self._t += 1
        if self.k > 0 and self._t % self.k == 0:
            for fast, slow in zip((p for g in self.param_groups for p in g["params"]), self.slow_weights):
                slow.add_(fast - slow, alpha=self.alpha)  # s ← s + α(θ − s)
                fast.copy_(slow)                          # θ ← s（必须在慢更新之后）
        return loss


def solve(params, base="sgd", lr=0.05, momentum=0.9, k=2, alpha=0.5, betas=(0.9, 0.999), eps=1e-8):
    return _Lookahead(params, base=base, lr=lr, momentum=momentum, k=k, alpha=alpha, betas=betas, eps=eps)
