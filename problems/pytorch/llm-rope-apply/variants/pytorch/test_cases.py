import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-6)


def _hf_rotated(q, k, position_ids, head_dim, base):
    import transformers as tr
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb
    cfg = tr.Qwen2Config(hidden_size=head_dim * 4, num_attention_heads=4,
                         head_dim=head_dim, max_position_embeddings=4096, rope_theta=base)
    rotary = Qwen2RotaryEmbedding(cfg)
    carrier = torch.zeros(q.shape[0], q.shape[2], head_dim, dtype=q.dtype)
    cos, sin = rotary(carrier, position_ids)
    return apply_rotary_pos_emb(q, k, cos, sin)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    q = torch.tensor([[[[1., 2., 3., 4.]]]])
    k = torch.tensor([[[[5., 6., 7., 8.]]]])
    pos = torch.tensor([[1]])
    qr, kr = solve(q, k, pos, head_dim=4, base=100.0)
    assert _close(qr.flatten(), torch.tensor([-1.984111, 1.590675, 2.462378, 4.179684]))
    assert _close(kr.flatten(), torch.tensor([-3.188785, 5.171358, 7.989471, 8.559034]))


def check_position_zero_is_identity(solve):
    # pos=0 → cos=1, sin=0 → 旋转不改变向量
    q = torch.randn(1, 3, 5, 16)
    k = torch.randn(1, 2, 5, 16)
    pos = torch.zeros(1, 5, dtype=torch.long)
    qr, kr = solve(q, k, pos, head_dim=16, base=10000.0)
    assert _close(qr, q) and _close(kr, k)


def check_vs_transformers_oracle_gqa(solve):
    torch.manual_seed(42)
    B, Hq, Hk, S, D, base = 2, 4, 2, 7, 32, 10000.0
    q = torch.randn(B, Hq, S, D)
    k = torch.randn(B, Hk, S, D)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    hq, hk = _hf_rotated(q, k, pos, D, base)
    mq, mk = solve(q, k, pos, head_dim=D, base=base)
    assert _close(mq, hq) and _close(mk, hk), "GQA 下 q/k head 数不同仍须各自正确广播"


def check_varied_positions_and_base(solve):
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 10, 8
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    pos = torch.randint(0, 512, (B, S))
    hq, hk = _hf_rotated(q, k, pos, D, 500000.0)
    mq, mk = solve(q, k, pos, head_dim=D, base=500000.0)
    assert _close(mq, hq) and _close(mk, hk)


def check_input_not_mutated(solve):
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    pos = torch.arange(3).unsqueeze(0)
    qs, ks = q.clone(), k.clone()
    solve(q, k, pos, head_dim=8, base=10000.0)
    assert torch.equal(q, qs) and torch.equal(k, ks)


check_example.example = True
