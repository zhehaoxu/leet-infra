"""参考答案：组装 MLP Block —— nn.Module 子类 + nn.Linear/nn.ReLU。"""

import torch


class MLPLayer(torch.nn.Module):
    def __init__(self, in_features: int, hidden_multiplier: int, out_features: int):
        super().__init__()
        hidden = in_features * hidden_multiplier
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def solve(in_features: int, hidden_multiplier: int, out_features: int) -> torch.nn.Module:
    if hidden_multiplier < 1:
        raise ValueError(f"hidden_multiplier must be >= 1, got {hidden_multiplier}")
    return MLPLayer(in_features, hidden_multiplier, out_features)
