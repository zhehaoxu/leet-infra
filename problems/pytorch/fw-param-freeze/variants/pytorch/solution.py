"""参考答案：全冻结 -> 换 head -> 优化器只收可训参数 -> 逐步核对冻结/更新账。"""

import torch
import torch.nn as nn


def _explicit(shape: tuple[int, ...], start: int) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (torch.arange(n, dtype=torch.float32) + start).reshape(shape) / 100


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 6))
        self.head = nn.Linear(6, 4)
        with torch.no_grad():
            self.encoder[0].weight.copy_(_explicit((6, 8), 0))
            self.encoder[0].bias.zero_()
            self.encoder[2].weight.copy_(_explicit((6, 6), 48))
            self.encoder[2].bias.zero_()
            self.head.weight.copy_(_explicit((4, 6), 84))
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(self.encoder(x))


def solve():
    model = Net()
    total_before = sum(p.numel() for p in model.parameters())

    for p in model.parameters():
        p.requires_grad_(False)
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 换 head：6->5 输出，新参数可训，显式赋值消除随机初始化
    new_head = nn.Linear(6, 5)
    with torch.no_grad():
        new_head.weight.copy_(_explicit((5, 6), 0))
        new_head.bias.zero_()
    model.head = new_head

    total_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flags = [[name, p.requires_grad] for name, p in model.named_parameters()]

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.1, momentum=0.9)

    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters()
                     if not p.requires_grad}

    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    y = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10

    losses: list[float] = []
    for _ in range(3):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    frozen_ok = all(torch.equal(frozen_before[n], p.detach())
                    for n, p in model.named_parameters() if not p.requires_grad)
    grad_none = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    n_opt_state = len(opt.state)

    return {
        "total_before": total_before,
        "total_after": total_after,
        "trainable_before": trainable_before,
        "trainable_after": trainable_after,
        "flags": flags,
        "losses": losses,
        "final_head_weight_flat": model.head.weight.detach().reshape(-1).tolist(),
        "final_head_bias_flat": model.head.bias.detach().reshape(-1).tolist(),
        "frozen_bitwise_identical": frozen_ok,
        "frozen_grad_is_none": grad_none,
        "n_opt_state_params": n_opt_state,
    }
