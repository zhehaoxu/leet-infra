import torch


def _close(a, b):
    a = a.float(); b = b.float()
    return torch.allclose(a, b, rtol=1e-5, atol=1e-6)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[1., 2., 3., 4.]])
    w = torch.ones(4)
    out = solve(x, w, 1e-6)
    exp = torch.tensor([[0.3651483655, 0.7302967310, 1.0954451561, 1.4605934620]])
    assert out.shape == x.shape, out.shape
    assert _close(out, exp), out.tolist()


def check_matches_transformers_oracle(solve):
    # 判题环境装有 transformers（离线）：以 Qwen2RMSNorm 同权重作为 ground truth 对拍
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    d = 32
    x = torch.randn(2, 5, d)
    w = torch.randn(d)
    hf = Qwen2RMSNorm(d, eps=1e-6)
    with torch.no_grad():
        hf.weight.copy_(w)
    ref = hf(x)
    out = solve(x, w, 1e-6)
    assert out.shape == x.shape
    assert _close(out, ref), "须与 transformers Qwen2RMSNorm 逐元素一致"


def check_definition_independent(solve):
    # 不走 HF，用独立公式重算：均方（不减均值）、eps 在根号内、float32 归一后乘权重
    d = 16
    x = torch.randn(3, d)
    w = torch.randn(d)
    ms = (x.double() ** 2).mean(-1, keepdim=True)
    ref = (w.double() * (x.double() / torch.sqrt(ms + 1e-6))).float()
    assert _close(solve(x, w, 1e-6), ref)


def check_nondefault_eps_random_weight(solve):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    d = 24
    x = torch.randn(4, 6, d)
    w = torch.randn(d) * 0.5 + 1.0
    eps = 1e-5
    hf = Qwen2RMSNorm(d, eps=eps)
    with torch.no_grad():
        hf.weight.copy_(w)
    assert _close(solve(x, w, eps), hf(x))


def check_input_not_mutated(solve):
    x = torch.randn(2, 8)
    w = torch.randn(8)
    snap = x.clone()
    solve(x, w, 1e-6)
    assert torch.equal(x, snap), "不应修改输入张量"


check_example.example = True
