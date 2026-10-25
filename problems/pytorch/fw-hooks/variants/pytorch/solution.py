"""参考答案：forward hook 抓激活并替换中间输出，full_backward_hook 改写梯度，
legacy backward_hook 记账，全部行为与解析重算对齐。"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(2, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.tensor([[0.3, 0.6]]))
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))


def solve():
    model = Net()
    acts: dict[str, list[float]] = {}
    fired: dict[str, int] = {"fc1_fwd": 0, "tanh_fwd": 0, "fc2_fwd": 0,
                             "fc1_bwd": 0, "fc2_bwd": 0}

    def fc1_fwd_hook(m, inp, out):
        fired["fc1_fwd"] += 1
        acts["fc1"] = out.detach().reshape(-1).tolist()

    def tanh_fwd_hook(m, inp, out):
        fired["tanh_fwd"] += 1
        acts["tanh"] = out.detach().reshape(-1).tolist()
        return out * 2.0  # 非 None 返回值替换模块输出（翻倍进下游）

    fc2_input_holder: list[list[float]] = []

    def fc2_fwd_hook(m, inp, out):
        fired["fc2_fwd"] += 1
        acts["fc2"] = out.detach().reshape(-1).tolist()
        fc2_input_holder.append(inp[0].detach().reshape(-1).tolist())

    def fc1_bwd_hook(m, gin, gout):
        fired["fc1_bwd"] += 1
        new_gin = tuple(g * 2.0 + 1.0 for g in gin)
        return new_gin  # 改写后向上传播的 grad_input

    def fc2_bwd_hook(m, gin, gout):
        fired["fc2_bwd"] += 1
        return None  # 只记账不改写

    model.fc1.register_forward_hook(fc1_fwd_hook)
    model.tanh.register_forward_hook(tanh_fwd_hook)
    model.fc2.register_forward_hook(fc2_fwd_hook)
    model.fc1.register_full_backward_hook(fc1_bwd_hook)
    model.fc2.register_backward_hook(fc2_bwd_hook)

    x = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    y = torch.tensor([1.0])

    out = model(x.unsqueeze(0)).squeeze()
    loss = (out - y) ** 2
    loss = loss.sum()
    loss.backward()

    return {
        "loss": loss.item(),
        "acts": acts,
        "fc2_input": fc2_input_holder[0],  # fc2 实际看到的输入（已被替换为 2*tanh）
        "fired": fired,
        "x_grad": x.grad.reshape(-1).tolist(),
        "fc1_weight_grad": model.fc1.weight.grad.reshape(-1).tolist(),
        "fc2_weight_grad": model.fc2.weight.grad.reshape(-1).tolist(),
        "fc1_bias_grad": model.fc1.bias.grad.tolist(),
    }
