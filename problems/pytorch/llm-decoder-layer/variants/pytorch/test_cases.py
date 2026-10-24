import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5)


def _hf_decoder_layer(hs, position_ids, nh, nkv, hd, inter, base=10000.0, eps=1e-6):
    """构造 HF Qwen2DecoderLayer(eager)，抽出 solve 所需的全部权重，返回 (params 元组, hf_out)。"""
    import transformers as tr
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
    hidden = hs.shape[-1]
    cfg = tr.Qwen2Config(hidden_size=hidden, num_attention_heads=nh, num_key_value_heads=nkv,
                         head_dim=hd, intermediate_size=inter, max_position_embeddings=4096,
                         attention_dropout=0.0, rope_theta=base, rms_norm_eps=eps,
                         hidden_act="silu", attn_implementation="eager",
                         num_hidden_layers=1, vocab_size=100)
    layer = Qwen2DecoderLayer(cfg, layer_idx=0).eval()
    rotary = Qwen2RotaryEmbedding(cfg)
    cos, sin = rotary(hs, position_ids)
    S = hs.shape[1]
    causal = torch.full((1, 1, S, S), float("-inf")).triu(1)
    sa, mlp = layer.self_attn, layer.mlp
    params = (
        cos, sin,
        layer.input_layernorm.weight.detach(),
        layer.post_attention_layernorm.weight.detach(),
        sa.q_proj.weight.detach(), sa.q_proj.bias.detach(),
        sa.k_proj.weight.detach(), sa.k_proj.bias.detach(),
        sa.v_proj.weight.detach(), sa.v_proj.bias.detach(),
        sa.o_proj.weight.detach(),
        mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach(),
    )
    with torch.no_grad():
        ref = layer(hs, attention_mask=causal, position_embeddings=(cos, sin))
    return params, ref


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    hd, S = 4, 3
    I, Z = torch.eye(hd), torch.zeros(hd)
    ones = torch.ones(hd)
    hs = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]])
    cos, sin = torch.ones(1, S, hd), torch.zeros(1, S, hd)
    out = solve(hs, cos, sin, ones, ones, I, Z, I, Z, I, Z, I, I, I, I)
    exp = torch.tensor([[[6.523182, 0.0, 0.0, 0.0],
                         [0.254472, 6.255613, 0.0, 0.0],
                         [0.227636, 0.227636, 6.043899, 0.0]]])
    assert _close(out, exp), out.tolist()


def check_residual_stream_identity(solve):
    # o_weight=0 且 gate_weight=0 → 注意力与 MLP 都输出 0，层退化为纯残差恒等
    hd = 8
    eye = torch.eye(hd)
    zM = torch.zeros(hd, hd)      # 2D 权重（矩阵）
    zb = torch.zeros(hd)         # 1D bias
    ones = torch.ones(hd)        # norm gain
    hs = torch.randn(1, 5, hd)
    cos, sin = torch.ones(1, 5, hd), torch.zeros(1, 5, hd)
    # q/k/v=identity, o=0, gate=0, up/down=identity
    out = solve(hs, cos, sin, ones, ones, eye, zb, eye, zb, eye, zb, zM, zM, eye, eye)
    assert _close(out, hs), "子层全 0 时应等于输入（pre-norm 加法残差）"


def check_vs_transformers_oracle_gqa(solve):
    torch.manual_seed(42)
    B, S, nh, nkv, hd, inter = 2, 7, 4, 2, 32, 128
    hs = torch.randn(B, S, nh * hd)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    p, ref = _hf_decoder_layer(hs, pos, nh, nkv, hd, inter)
    got = solve(hs, *p)
    assert got.shape == ref.shape
    assert _close(got, ref), "DecoderLayer 须与 Qwen2DecoderLayer(eager) 一致"


def check_vs_transformers_oracle_mqa(solve):
    torch.manual_seed(42)
    B, S, nh, nkv, hd, inter = 1, 6, 4, 1, 16, 64
    hs = torch.randn(B, S, nh * hd)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    p, ref = _hf_decoder_layer(hs, pos, nh, nkv, hd, inter)
    assert _close(solve(hs, *p), ref)


def check_input_not_mutated(solve):
    hd, inter = 8, 16
    I, Z = torch.eye(hd), torch.zeros(hd)
    hs = torch.randn(1, 4, hd)
    cos, sin = torch.ones(1, 4, hd), torch.zeros(1, 4, hd)
    ones = torch.ones(hd)
    gW = torch.randn(inter, hd); uW = torch.randn(inter, hd); dW = torch.randn(hd, inter)
    snap = hs.clone()
    solve(hs, cos, sin, ones, ones, I, Z, I, Z, I, Z, I, gW, uW, dW)
    assert torch.equal(hs, snap)


check_example.example = True
