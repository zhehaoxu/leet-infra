import torch


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5)


def _build_oracle(vocab, nh, nkv, hd, inter, nlayers, ids, base=10000.0, eps=1e-6):
    """构造 HF Qwen2ForCausalLM(eager)，抽出 solve 需要的权重包，返回 (weights 元组, hf_logits)。"""
    import transformers as tr
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    hidden = nh * hd
    cfg = tr.Qwen2Config(vocab_size=vocab, hidden_size=hidden, num_attention_heads=nh,
                         num_key_value_heads=nkv, head_dim=hd, intermediate_size=inter,
                         num_hidden_layers=nlayers, max_position_embeddings=4096,
                         attention_dropout=0.0, rope_theta=base, rms_norm_eps=eps,
                         hidden_act="silu", attn_implementation="eager",
                         tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    layers = []
    for lyr in model.model.layers:
        sa, mlp = lyr.self_attn, lyr.mlp
        layers.append(dict(
            input_ln_weight=lyr.input_layernorm.weight.detach(),
            post_attention_ln_weight=lyr.post_attention_layernorm.weight.detach(),
            q_weight=sa.q_proj.weight.detach(), q_bias=sa.q_proj.bias.detach(),
            k_weight=sa.k_proj.weight.detach(), k_bias=sa.k_proj.bias.detach(),
            v_weight=sa.v_proj.weight.detach(), v_bias=sa.v_proj.bias.detach(),
            o_weight=sa.o_proj.weight.detach(),
            gate_weight=mlp.gate_proj.weight.detach(),
            up_weight=mlp.up_proj.weight.detach(),
            down_weight=mlp.down_proj.weight.detach(),
        ))
    S = ids.shape[1]
    causal = torch.full((1, 1, S, S), float("-inf")).triu(1)
    with torch.no_grad():
        hf_logits = model(input_ids=ids, attention_mask={"full_attention": causal}).logits
    weights = (
        model.model.embed_tokens.weight.detach(),
        model.model.norm.weight.detach(),
        model.lm_head.weight.detach(),
        layers,
        hd,
    )
    return weights, hf_logits


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    torch.manual_seed(42)
    ids = torch.tensor([[0, 1, 2]])
    w, ref = _build_oracle(4, 1, 1, 4, 4, 2, ids)
    got = solve(ids, *w)
    assert got.shape == (1, 3, 4)
    assert _close(got, ref), got.tolist()


def check_shape_and_argmax(solve):
    torch.manual_seed(0)
    ids = torch.tensor([[5, 3, 1, 4, 2], [0, 6, 2, 5, 1]])
    w, ref = _build_oracle(32, 4, 2, 32, 128, 4, ids)
    got = solve(ids, *w)
    assert got.shape == (2, 5, 32)
    assert torch.equal(got.argmax(-1), ref.argmax(-1))


def check_vs_transformers_oracle_multilayer(solve):
    torch.manual_seed(1)
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2], [2, 7, 1, 8, 2, 8, 1]])
    w, ref = _build_oracle(32, 4, 2, 32, 128, 4, ids)
    assert _close(solve(ids, *w), ref), "整模型须与 Qwen2ForCausalLM(eager) 一致"


def check_vs_transformers_oracle_mqa(solve):
    torch.manual_seed(2)
    ids = torch.arange(6).reshape(1, 6) % 8
    w, ref = _build_oracle(16, 2, 1, 16, 48, 3, ids)
    assert _close(solve(ids, *w), ref)


def check_weights_not_mutated(solve):
    torch.manual_seed(3)
    ids = torch.tensor([[1, 2, 3]])
    w, _ = _build_oracle(8, 1, 1, 8, 16, 2, ids)
    embed, final_norm, lm_head, layers, hd = w
    snap = (embed.clone(), final_norm.clone(), lm_head.clone(),
            layers[0]["q_weight"].clone())
    solve(ids, *w)
    assert torch.equal(embed, snap[0]) and torch.equal(final_norm, snap[1])
    assert torch.equal(lm_head, snap[2]) and torch.equal(layers[0]["q_weight"], snap[3])


check_example.example = True
