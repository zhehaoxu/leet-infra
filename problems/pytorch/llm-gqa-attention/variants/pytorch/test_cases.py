import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5)


def _hf_attention(hs, position_ids, nh, nkv, hd, base=10000.0):
    """构造 HF Qwen2Attention(eager) 并抽出其权重张量，返回 (solve 入参..., hf_out)。"""
    import transformers as tr
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding
    hidden = hs.shape[-1]
    cfg = tr.Qwen2Config(hidden_size=hidden, num_attention_heads=nh, num_key_value_heads=nkv,
                         head_dim=hd, max_position_embeddings=4096, attention_dropout=0.0,
                         rope_theta=base, attn_implementation="eager")
    att = Qwen2Attention(cfg, layer_idx=0).eval()
    rotary = Qwen2RotaryEmbedding(cfg)
    cos, sin = rotary(hs, position_ids)
    S = hs.shape[1]
    causal = torch.full((1, 1, S, S), float("-inf")).triu(1)
    qw, qb = att.q_proj.weight.detach(), att.q_proj.bias.detach()
    kw, kb = att.k_proj.weight.detach(), att.k_proj.bias.detach()
    vw, vb = att.v_proj.weight.detach(), att.v_proj.bias.detach()
    ow = att.o_proj.weight.detach()
    with torch.no_grad():
        ref, _ = att(hs, position_embeddings=(cos, sin), attention_mask=causal)
    return (cos, sin, qw, qb, kw, kb, vw, vb, ow), ref


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    hd, S = 4, 3
    I, Z = torch.eye(hd), torch.zeros(hd)
    hs = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]])
    cos, sin = torch.ones(1, S, hd), torch.zeros(1, S, hd)
    out = solve(hs, cos, sin, I, Z, I, Z, I, Z, I)
    exp = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                         [0.377541, 0.622459, 0.0, 0.0],
                         [0.274069, 0.274069, 0.451863, 0.0]]])
    assert _close(out, exp), out.tolist()


def check_first_row_only_self(solve):
    # 因果性：第 0 行只看到自己 → 输出等于其 v（投影为单位阵时等于第 0 个 token）
    hd = 8
    I, Z = torch.eye(hd), torch.zeros(hd)
    hs = torch.randn(1, 5, hd)
    cos, sin = torch.ones(1, 5, hd), torch.zeros(1, 5, hd)
    out = solve(hs, cos, sin, I, Z, I, Z, I, Z, I)
    assert _close(out[0, 0], hs[0, 0]), "首 token 只能attend到自己"


def check_vs_transformers_oracle_gqa(solve):
    torch.manual_seed(42)
    B, S, nh, nkv, hd = 2, 7, 4, 2, 32
    hs = torch.randn(B, S, nh * hd)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    (cos, sin, qw, qb, kw, kb, vw, vb, ow), ref = _hf_attention(hs, pos, nh, nkv, hd)
    got = solve(hs, cos, sin, qw, qb, kw, kb, vw, vb, ow)
    assert got.shape == ref.shape
    assert _close(got, ref), "GQA 须与 Qwen2Attention(eager) 一致"


def check_vs_transformers_oracle_mqa(solve):
    # 极端 GQA：单 KV 头（MQA）
    torch.manual_seed(42)
    B, S, nh, nkv, hd = 1, 6, 4, 1, 16
    hs = torch.randn(B, S, nh * hd)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    (cos, sin, qw, qb, kw, kb, vw, vb, ow), ref = _hf_attention(hs, pos, nh, nkv, hd)
    assert _close(solve(hs, cos, sin, qw, qb, kw, kb, vw, vb, ow), ref)


def check_input_not_mutated(solve):
    hd = 8
    I, Z = torch.eye(hd), torch.zeros(hd)
    hs = torch.randn(1, 4, hd)
    cos, sin = torch.ones(1, 4, hd), torch.zeros(1, 4, hd)
    snap = hs.clone()
    solve(hs, cos, sin, I, Z, I, Z, I, Z, I)
    assert torch.equal(hs, snap)


check_example.example = True
