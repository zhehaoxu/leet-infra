import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-6)


def _hf_mlp(x, gw, uw, dw):
    import transformers as tr
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    hidden, inter = x.shape[-1], gw.shape[0]
    m = Qwen2MLP(tr.Qwen2Config(hidden_size=hidden, intermediate_size=inter, hidden_act="silu"))
    with torch.no_grad():
        m.gate_proj.weight.copy_(gw)
        m.up_proj.weight.copy_(uw)
        m.down_proj.weight.copy_(dw)
    return m(x)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[1., 2.]])
    I = torch.eye(2)
    out = solve(x, I, I, I)
    assert _close(out.flatten(), torch.tensor([0.7310586, 3.5231881])), out.tolist()


def check_glu_gate_path(solve):
    # 只激活 gate、up 全 0 → 输出 0（验证门控是逐元素乘、不是相加）
    torch.manual_seed(42)
    hidden, inter = 6, 8
    x = torch.randn(2, 3, hidden)
    gw = torch.randn(inter, hidden)
    dw = torch.randn(hidden, inter)
    zero = torch.zeros(inter, hidden)
    out = solve(x, gw, zero, dw)
    assert _close(out, torch.zeros_like(out)), "up=0 应使 SwiGLU 输出为 0"


def check_vs_transformers_oracle(solve):
    torch.manual_seed(42)
    hidden, inter = 12, 32
    x = torch.randn(2, 5, hidden)
    gw = torch.randn(inter, hidden) * 0.3
    uw = torch.randn(inter, hidden) * 0.3
    dw = torch.randn(hidden, inter) * 0.3
    assert _close(solve(x, gw, uw, dw), _hf_mlp(x, gw, uw, dw))


def check_2d_input(solve):
    torch.manual_seed(42)
    hidden, inter = 4, 16
    x = torch.randn(7, hidden)
    gw = torch.randn(inter, hidden)
    uw = torch.randn(inter, hidden)
    dw = torch.randn(hidden, inter)
    out = solve(x, gw, uw, dw)
    assert out.shape == (7, hidden)
    assert _close(out, _hf_mlp(x, gw, uw, dw))


def check_input_not_mutated(solve):
    x = torch.randn(3, 4)
    ws = [torch.randn(8, 4) for _ in range(2)] + [torch.randn(4, 8)]
    snap = [t.clone() for t in [x] + ws]
    solve(x, ws[0], ws[1], ws[2])
    for a, b in zip([x] + ws, snap):
        assert torch.equal(a, b)


check_example.example = True
