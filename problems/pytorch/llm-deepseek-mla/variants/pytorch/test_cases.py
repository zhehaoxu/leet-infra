import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5)


def _ex_inputs():
    """固定 seed 的确定性示例输入（不依赖 transformers，供「运行」样例用）。"""
    B, S, hidden = 1, 3, 4
    nh, nope, rope, v = 2, 2, 2, 2
    q_lora, kv_lora = 3, 4
    torch.manual_seed(11)
    hs = torch.randn(B, S, hidden)
    cos = torch.ones(B, S, rope)
    sin = torch.zeros(B, S, rope)
    q_a_w = torch.randn(q_lora, hidden)
    q_a_norm = torch.ones(q_lora)
    q_b_w = torch.randn(nh * (nope + rope), q_lora)
    kv_a_w = torch.randn(kv_lora + rope, hidden)
    kv_a_norm = torch.ones(kv_lora)
    kv_b_w = torch.randn(nh * (nope + v), kv_lora)
    o_w = torch.randn(hidden, nh * v)
    params = (cos, sin, q_a_w, q_a_norm, q_b_w, kv_a_w, kv_a_norm, kv_b_w, o_w, nh, nope, rope, v)
    return hs, params


def _hf_mla(hs, position_ids, nh, nope, rope, v, q_lora, kv_lora, base=10000.0):
    """构造 HF DeepseekV3Attention(eager, rope_interleave=False)，抽出 solve 所需权重，返回 (params, hf_out)。"""
    import transformers as tr
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3Attention,
        DeepseekV3RotaryEmbedding,
    )
    hidden = hs.shape[-1]
    cfg = tr.DeepseekV3Config(
        hidden_size=hidden, num_attention_heads=nh, num_key_value_heads=nh,
        head_dim=rope, q_lora_rank=q_lora, kv_lora_rank=kv_lora,
        qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=v,
        max_position_embeddings=64, attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": base},
        rope_interleave=False, attn_implementation="eager",
    )
    att = DeepseekV3Attention(cfg, layer_idx=0).eval()
    rotary = DeepseekV3RotaryEmbedding(cfg)
    cos, sin = rotary(hs, position_ids)
    S = hs.shape[1]
    causal = torch.full((1, 1, S, S), float("-inf")).triu(1)
    with torch.no_grad():
        ref, _ = att(hs, position_embeddings=(cos, sin), attention_mask=causal)
    params = (
        cos, sin,
        att.q_a_proj.weight.detach(), att.q_a_layernorm.weight.detach(),
        att.q_b_proj.weight.detach(), att.kv_a_proj_with_mqa.weight.detach(),
        att.kv_a_layernorm.weight.detach(), att.kv_b_proj.weight.detach(),
        att.o_proj.weight.detach(), nh, nope, rope, v,
    )
    return params, ref


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    hs, params = _ex_inputs()
    out = solve(hs, *params)
    exp = torch.tensor([[[0.877777, 1.533879, 1.298111, 0.869258],
                         [-1.622245, -1.653416, -3.259178, -2.022986],
                         [-2.708383, -1.197325, -0.054776, -0.642115]]])
    assert out.shape == exp.shape, out.shape
    assert _close(out, exp), out.tolist()


def check_vs_transformers_oracle_mla(solve):
    torch.manual_seed(42)
    B, S, nh, nope, rope, v, q_lora, kv_lora = 2, 7, 3, 5, 4, 5, 10, 7
    hs = torch.randn(B, S, 16)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    p, ref = _hf_mla(hs, pos, nh, nope, rope, v, q_lora, kv_lora)
    got = solve(hs, *p)
    assert got.shape == ref.shape
    assert _close(got, ref), "MLA 须与 DeepseekV3Attention(eager) 一致"


def check_vs_transformers_oracle_mla2(solve):
    # 不同秩/头配置，另解耦 rope 维较小
    torch.manual_seed(7)
    B, S, nh, nope, rope, v, q_lora, kv_lora = 1, 6, 2, 8, 2, 6, 6, 5
    hs = torch.randn(B, S, 12)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    p, ref = _hf_mla(hs, pos, nh, nope, rope, v, q_lora, kv_lora)
    assert _close(solve(hs, *p), ref)


def check_input_not_mutated(solve):
    hs, params = _ex_inputs()
    snap = hs.clone()
    solve(hs, *params)
    assert torch.equal(hs, snap)


check_example.example = True
