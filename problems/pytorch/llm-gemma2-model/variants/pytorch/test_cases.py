import math

import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _rand(seed, *shape, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, generator=g, dtype=torch.float64) - 0.5) * scale


def _make_model(seed, vocab, hidden, heads, kv_heads, head_dim, n_layers, inter, untied=False):
    embed = _rand(seed, vocab, hidden)
    layers = []
    for i in range(n_layers):
        layers.append(
            {
                "input_ln_weight": _rand(seed + 1 + i, hidden),
                "post_attention_ln_weight": _rand(seed + 2 + i, hidden),
                "pre_feedforward_ln_weight": _rand(seed + 3 + i, hidden),
                "post_feedforward_ln_weight": _rand(seed + 4 + i, hidden),
                "q_weight": _rand(seed + 5 + i, heads * head_dim, hidden),
                "k_weight": _rand(seed + 6 + i, kv_heads * head_dim, hidden),
                "v_weight": _rand(seed + 7 + i, kv_heads * head_dim, hidden),
                "o_weight": _rand(seed + 8 + i, hidden, heads * head_dim),
                "gate_weight": _rand(seed + 9 + i, inter, hidden),
                "up_weight": _rand(seed + 10 + i, inter, hidden),
                "down_weight": _rand(seed + 11 + i, hidden, inter),
            }
        )
    final = _rand(seed + 99, hidden)
    head = _rand(seed + 98, vocab, hidden) if untied else None
    return embed, layers, final, head


def _rms(x, weight, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * (1.0 + np.asarray(weight))


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _oracle(ids, embed, final_norm, layers, config, lm_head=None):
    """numpy 逐 batch/头/位置重算整模型，不复用 torch 的任何向量化路径。"""
    ids = np.asarray(ids, dtype=np.int64)
    emb = np.asarray(embed, dtype=np.float64)
    batch, seq = ids.shape
    hidden = emb.shape[1]
    vocab = emb.shape[0]
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = hidden // heads
    kv_heads = config.get("num_key_value_heads")
    if kv_heads is None:
        kv_heads = heads
    n_layers = len(layers)
    layer_types = config.get("layer_types")
    if layer_types is None:
        layer_types = ["sliding_attention" if (i + 1) % 2 else "full_attention" for i in range(n_layers)]
    window = config.get("sliding_window", 4096)
    pre_attn_scalar = config.get("query_pre_attn_scalar", 256.0)
    attn_cap = config.get("attn_logit_softcapping", 50.0)
    final_cap = config.get("final_logit_softcapping", 30.0)
    eps = config.get("rms_norm_eps", 1e-6)
    theta = config.get("rope_theta", 10000.0)
    tied = config.get("tie_word_embeddings", True)
    repeat = heads // kv_heads

    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    angles = np.arange(seq)[:, None] * inv_freq[None, :]
    cos = np.cos(np.concatenate([angles, angles], axis=-1))[None, :, None, :]
    sin = np.sin(np.concatenate([angles, angles], axis=-1))[None, :, None, :]
    half = head_dim // 2

    def rotate(t):
        return t * cos + np.concatenate([-t[..., half:], t[..., :half]], axis=-1) * sin

    x = emb[ids] * math.sqrt(hidden)
    for index in range(n_layers):
        ws = layers[index]
        sliding = layer_types[index] == "sliding_attention"
        h = _rms(x, ws["input_ln_weight"], eps)
        q = rotate((h @ np.asarray(ws["q_weight"]).T).reshape(batch, seq, heads, head_dim))
        k = rotate((h @ np.asarray(ws["k_weight"]).T).reshape(batch, seq, kv_heads, head_dim))
        v = (h @ np.asarray(ws["v_weight"]).T).reshape(batch, seq, kv_heads, head_dim)

        mixed = np.zeros((batch, seq, heads * head_dim))
        for b in range(batch):
            for head in range(heads):
                kvh = head // repeat
                for t in range(seq):
                    allowed = [j for j in range(seq) if j <= t and (not sliding or j > t - window)]
                    scores = q[b, t, head] @ k[b, allowed, kvh].T * pre_attn_scalar**-0.5
                    if attn_cap is not None:
                        scores = attn_cap * np.tanh(scores / attn_cap)
                    shifted = np.exp(scores - scores.max())
                    weights = shifted / shifted.sum()
                    mixed[b, t, head * head_dim : (head + 1) * head_dim] = weights @ v[b, allowed, kvh]
        x = x + _rms(mixed @ np.asarray(ws["o_weight"]).T, ws["post_attention_ln_weight"], eps)

        normed = _rms(x, ws["pre_feedforward_ln_weight"], eps)
        ff = _gelu(normed @ np.asarray(ws["gate_weight"]).T) * (normed @ np.asarray(ws["up_weight"]).T)
        x = x + _rms(ff @ np.asarray(ws["down_weight"]).T, ws["post_feedforward_ln_weight"], eps)

    x = _rms(x, final_norm, eps)
    head_weight = emb if (tied or lm_head is None) else np.asarray(lm_head, dtype=np.float64)
    logits = x @ head_weight.T
    if final_cap is not None:
        logits = final_cap * np.tanh(logits / final_cap)
    return logits


def check_example(solve):
    embed = torch.tensor([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    zeros = torch.zeros(2, dtype=torch.float64)
    layer = {
        "input_ln_weight": zeros,
        "post_attention_ln_weight": zeros,
        "pre_feedforward_ln_weight": zeros,
        "post_feedforward_ln_weight": zeros,
        "q_weight": identity,
        "k_weight": identity,
        "v_weight": identity,
        "o_weight": identity,
        "gate_weight": identity,
        "up_weight": identity,
        "down_weight": identity,
    }
    config = {
        "num_attention_heads": 1,
        "head_dim": 2,
        "sliding_window": 1,
        "query_pre_attn_scalar": 1.0,
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention"],
    }
    ids = torch.tensor([[0, 1, 2]])
    logits = solve(ids, embed, zeros, [layer], config)
    expect = torch.tensor(
        [
            [2.000000, 0.000000, 2.000000],
            [0.534581, 1.927232, 2.461812],
            [1.414214, 1.414214, 2.828427],
        ],
        dtype=torch.float64,
    )
    assert logits.shape == (1, 3, 3) and logits.dtype == torch.float32
    assert _close(logits, expect, 1e-5), logits
    # 与独立 oracle 同步（同一组权重）
    ref = _oracle(ids.numpy(), embed.numpy(), zeros.numpy(), [layer], config)
    assert _close(logits, ref, 1e-6), (logits, ref)


check_example.example = True


def check_default_derivations(solve):
    vocab, hidden, heads, kv, inter = 6, 8, 2, 2, 10
    head_dim = hidden // heads
    embed, layers, final, _ = _make_model(1, vocab, hidden, heads, kv, head_dim, 2, inter)
    ids = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2]])

    explicit = {
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 2,
        "sliding_window": 4096,
        "query_pre_attn_scalar": 256.0,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    base = solve(ids, embed, final, layers, explicit)
    # 只留 num_attention_heads：head_dim / n_head_kv / 层数 / 层型 / tie / 窗口全部走缺省
    minimal = {"num_attention_heads": heads}
    assert torch.equal(solve(ids, embed, final, layers, minimal), base), "缺省推导必须与显式配置逐位一致"

    # 缺省的 layer_types 是"首层 sliding、次层 full"交替（窗口取 3 时才能从输出区分）
    narrow = {"num_attention_heads": heads, "sliding_window": 3}
    explicit_types = {**narrow, "layer_types": ["sliding_attention", "full_attention"]}
    assert torch.equal(solve(ids, embed, final, layers, narrow), solve(ids, embed, final, layers, explicit_types))
    assert not _close(
        solve(ids, embed, final, layers, narrow),
        solve(ids, embed, final, layers, {**narrow, "layer_types": ["full_attention"] * 2}),
        1e-3,
    )
    # 缺省窗口 4096 > S，滑窗层退化为全注意力
    assert torch.equal(solve(ids, embed, final, layers, {**explicit, "sliding_window": 4096}), base)

    # 显式 head_dim 优先于 hidden // n_heads：12/3=4，但这里压到 2
    embed2, layers2, final2, _ = _make_model(2, vocab, 12, 3, 1, 2, 2, inter)
    ids2 = torch.tensor([[0, 1, 2]])
    override = {"num_attention_heads": 3, "num_key_value_heads": 1, "head_dim": 2}
    assert solve(ids2, embed2, final2, layers2, override).shape == (1, 3, vocab)
    assert _close(
        solve(ids2, embed2, final2, layers2, override),
        _oracle(ids2.numpy(), embed2.numpy(), final2.numpy(), layers2, override),
        1e-6,
    )
    # 省略 head_dim 时推导为 12//3=4，与按 2 建的权重对不上 → 必须报错而不是静默用错维度
    try:
        solve(ids2, embed2, final2, layers2, {"num_attention_heads": 3, "num_key_value_heads": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("derived head_dim 与 q_weight 形状不符时应报 ValueError")
    # hidden 不能整除 n_heads 且未给 head_dim → ValueError
    try:
        solve(ids, embed, final, layers, {"num_attention_heads": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("hidden_size 不能被 n_heads 整除且缺 head_dim 时应报 ValueError")


def check_tied_head(solve):
    vocab, hidden, heads, kv, head_dim, inter = 5, 8, 2, 2, 4, 12
    embed, layers, final, head = _make_model(3, vocab, hidden, heads, kv, head_dim, 2, inter, untied=True)
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    config = {"num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim}

    tied_logits = solve(ids, embed, final, layers, config)
    untied_same = solve(ids, embed, final, layers, {**config, "tie_word_embeddings": False}, head)
    assert not _close(tied_logits, untied_same, 1e-4), "独立词表头权重不同，结果必须不同"
    assert _close(
        tied_logits,
        solve(ids, embed, final, layers, {**config, "tie_word_embeddings": False}, embed),
        1e-12,
    ), "untied 且 head=embed 时必须与 tied 完全等价"
    ref = _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config)
    assert _close(tied_logits, ref, 1e-6), (tied_logits, ref)
    # 绑定后梯度意义上就是一个矩阵：logits 的列数 = vocab 行数
    assert tied_logits.shape == (1, 5, vocab)


def check_sliding_window(solve):
    vocab, hidden, heads, kv, head_dim, inter = 6, 8, 2, 1, 4, 10
    embed, layers, final, _ = _make_model(4, vocab, hidden, heads, kv, head_dim, 2, inter)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    base = {
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "query_pre_attn_scalar": 4.0,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
    }
    full = solve(ids, embed, final, layers, {**base, "sliding_window": 6})
    wider = solve(ids, embed, final, layers, {**base, "sliding_window": 4096})
    assert torch.equal(full, wider), "窗口 >= S 时滑窗退化为因果全注意力"
    narrow = solve(ids, embed, final, layers, {**base, "sliding_window": 2})
    assert not _close(narrow, full, 1e-3), "窗口变窄必须改变输出"
    single = solve(ids, embed, final, layers, {**base, "sliding_window": 1})
    assert not _close(single, narrow, 1e-3)
    for config in (base | {"sliding_window": 1}, base | {"sliding_window": 2}):
        ref = _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config)
        assert _close(solve(ids, embed, final, layers, config), ref, 1e-6), config["sliding_window"]


def check_causal_prefix(solve):
    vocab, hidden, heads, kv, head_dim, inter = 5, 8, 2, 1, 4, 9
    embed, layers, final, _ = _make_model(5, vocab, hidden, heads, kv, head_dim, 2, inter)
    config = {
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "sliding_window": 3,
        "query_pre_attn_scalar": 4.0,
    }
    long_ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    short = solve(long_ids[:, :3], embed, final, layers, config)
    long = solve(long_ids, embed, final, layers, config)
    # 因果 + 位置从 0 起算：更长的序列不会改写前缀 token 的 logits
    assert _close(long[:, :3], short, 1e-6), (long[:, :3], short)
    # 位置编码确实生效：交换两个 token 不等于交换两行输出
    swapped = solve(torch.tensor([[1, 0, 2, 3, 4]]), embed, final, layers, config)
    assert not _close(swapped, long[:1], 1e-3)


def check_softcaps(solve):
    vocab, hidden, heads, kv, head_dim, inter = 6, 8, 2, 2, 2, 10
    embed, layers, final, _ = _make_model(6, vocab, hidden, heads, kv, head_dim, 2, inter)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    base = {"num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim,
            "query_pre_attn_scalar": 0.5}
    raw = solve(ids, embed, final, layers, {**base, "final_logit_softcapping": None})
    assert float(raw.abs().max()) > 0.3, "缩放不能太小，否则下面的 tanh 断言没有区分度"
    capped = solve(ids, embed, final, layers, {**base, "final_logit_softcapping": 0.25})
    assert float(capped.abs().max()) < 0.25, "tanh softcap 把 logits 严格压进 (-cap, cap)"
    want = 0.25 * torch.tanh(raw.double() / 0.25)
    assert _close(capped, want, 1e-6), (capped, want)
    # attention 级 softcap：把 logits 压扁会改变权重，权重再改写输出
    plain = solve(ids, embed, final, layers, {**base, "attn_logit_softcapping": None})
    squashed = solve(ids, embed, final, layers, {**base, "attn_logit_softcapping": 0.05})
    assert not _close(plain, squashed, 1e-3)
    ref = _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers,
                  {**base, "attn_logit_softcapping": 0.05, "final_logit_softcapping": 0.25})
    squashed_capped = solve(ids, embed, final, layers,
                            {**base, "attn_logit_softcapping": 0.05, "final_logit_softcapping": 0.25})
    assert _close(squashed_capped, ref, 1e-6), squashed_capped


def check_gqa_grouping(solve):
    vocab, hidden, heads, kv, head_dim, inter = 5, 8, 4, 2, 2, 8
    embed, layers, final, _ = _make_model(7, vocab, hidden, heads, kv, head_dim, 2, inter)
    ids = torch.tensor([[0, 1, 2, 3]])
    config = {"num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim}
    grouped = solve(ids, embed, final, layers, config)

    # 把每个 kv 头按 blocked 顺序复制成 4 个头，等价于 n_kv = n_heads
    expanded = []
    for layer in layers:
        duplicated = dict(layer)
        for key in ("k_weight", "v_weight"):
            head_blocks = [layer[key][h * head_dim : (h + 1) * head_dim] for h in range(kv)]
            duplicated[key] = torch.cat([block for block in head_blocks for _ in range(2)], dim=0)
        expanded.append(duplicated)
    dense = solve(ids, embed, final, expanded, {**config, "num_key_value_heads": heads})
    assert _close(grouped, dense, 1e-6), (grouped, dense)
    assert expanded[0]["k_weight"].shape == (heads * head_dim, hidden)
    ref = _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config)
    assert _close(grouped, ref, 1e-6), (grouped, ref)


def check_against_oracle(solve):
    cases = [
        (11, 6, 8, 2, 2, 4, 2, 10, {"sliding_window": 3, "query_pre_attn_scalar": 4.0}),
        (12, 7, 8, 2, 1, 2, 3, 9, {"sliding_window": 2, "attn_logit_softcapping": 3.0}),
        (13, 5, 12, 3, 1, 2, 2, 7, {"layer_types": ["full_attention", "full_attention"],
                                    "final_logit_softcapping": None}),
        (14, 4, 16, 2, 2, 8, 2, 20, {"attn_logit_softcapping": None, "sliding_window": 4096}),
    ]
    for seed, vocab, hidden, heads, kv, head_dim, n_layers, inter, extra in cases:
        embed, layers, final, _ = _make_model(seed, vocab, hidden, heads, kv, head_dim, n_layers, inter)
        config = {"num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim, **extra}
        ids = torch.tensor(
            [
                [i % vocab for i in range(6)],
                [(6 - i) % vocab for i in range(6)],
            ]
        )
        logits = solve(ids, embed, final, layers, config)
        ref = _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config)
        assert _close(logits, ref, 1e-6), (seed, logits, ref)
    # 独立词表头
    embed, layers, final, head = _make_model(15, 6, 8, 2, 2, 4, 2, 10, untied=True)
    config = {"num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4, "tie_word_embeddings": False}
    ids = torch.tensor([[0, 1, 2, 3]])
    assert _close(
        solve(ids, embed, final, layers, config, head),
        _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config, head.numpy()),
        1e-6,
    )
    # float32 输入可跑，输出仍是 float32
    embed32 = embed.float()
    layers32 = [{k: v.float() for k, v in layer.items()} for layer in layers]
    logits32 = solve(ids, embed32, final.float(), layers32, config, head.float())
    assert logits32.dtype == torch.float32
    assert _close(logits32, _oracle(ids.numpy(), embed.numpy(), final.numpy(), layers, config, head.numpy()), 1e-4)


def _hf_logits(ids, embed, final_norm, layers, config, lm_head=None):
    import transformers as tr
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim") or embed.shape[1] // heads
    kwargs = {
        "vocab_size": embed.shape[0],
        "hidden_size": embed.shape[1],
        "intermediate_size": layers[0]["gate_weight"].shape[0],
        "num_hidden_layers": len(layers),
        "num_attention_heads": heads,
        "head_dim": head_dim,
        "sliding_window": config.get("sliding_window", 4096),
        "query_pre_attn_scalar": int(config.get("query_pre_attn_scalar", 256.0)),
        "attn_logit_softcapping": config.get("attn_logit_softcapping", 50.0),
        "final_logit_softcapping": config.get("final_logit_softcapping", 30.0),
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
    }
    if "num_key_value_heads" in config:
        kwargs["num_key_value_heads"] = config["num_key_value_heads"]
    if "layer_types" in config:
        kwargs["layer_types"] = list(config["layer_types"])
    hf_config = tr.Gemma2Config(**kwargs)
    hf_config._attn_implementation = "eager"
    model = Gemma2ForCausalLM(hf_config).eval()
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(embed.float())
        model.model.norm.weight.copy_(final_norm.float())
        for index, weights in enumerate(layers):
            block = model.model.layers[index]
            block.self_attn.q_proj.weight.copy_(weights["q_weight"].float())
            block.self_attn.k_proj.weight.copy_(weights["k_weight"].float())
            block.self_attn.v_proj.weight.copy_(weights["v_weight"].float())
            block.self_attn.o_proj.weight.copy_(weights["o_weight"].float())
            block.mlp.gate_proj.weight.copy_(weights["gate_weight"].float())
            block.mlp.up_proj.weight.copy_(weights["up_weight"].float())
            block.mlp.down_proj.weight.copy_(weights["down_weight"].float())
            block.input_layernorm.weight.copy_(weights["input_ln_weight"].float())
            block.post_attention_layernorm.weight.copy_(weights["post_attention_ln_weight"].float())
            block.pre_feedforward_layernorm.weight.copy_(weights["pre_feedforward_ln_weight"].float())
            block.post_feedforward_layernorm.weight.copy_(weights["post_feedforward_ln_weight"].float())
        if lm_head is not None:
            model.lm_head.weight.copy_(lm_head.float())
        return model(input_ids=ids).logits


def check_against_transformers(solve):
    cases = [
        # (seed, vocab, hidden, heads, kv, head_dim, layers, inter, extra)
        (21, 6, 8, 2, 1, 4, 3, 10, {"sliding_window": 3, "query_pre_attn_scalar": 4,
                                    "attn_logit_softcapping": 5.0, "final_logit_softcapping": 2.0}),
        (22, 5, 8, 2, 2, 2, 2, 9, {"sliding_window": 4096, "query_pre_attn_scalar": 8,
                                   "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0}),
        (23, 4, 12, 3, 1, 2, 2, 7, {"layer_types": ["full_attention", "full_attention"],
                                    "attn_logit_softcapping": None, "final_logit_softcapping": None,
                                    "query_pre_attn_scalar": 3}),
    ]
    for seed, vocab, hidden, heads, kv, head_dim, n_layers, inter, extra in cases:
        embed, layers, final, _ = _make_model(seed, vocab, hidden, heads, kv, head_dim, n_layers, inter)
        config = {"num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim, **extra}
        ids = torch.tensor(
            [[i % vocab for i in range(5)], [(4 - i) % vocab for i in range(5)]]
        )
        assert _close(solve(ids, embed, final, layers, config),
                      _hf_logits(ids, embed, final, layers, config), 2e-5), (seed, extra)
    # 缺省推导必须与 HF 自己的缺省一致：layer_types 交替 + n_kv = n_heads + tie
    embed, layers, final, _ = _make_model(24, 6, 8, 4, 4, 2, 3, 10)
    config = {"num_attention_heads": 4, "head_dim": 2, "sliding_window": 2, "query_pre_attn_scalar": 4}
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    assert _close(solve(ids, embed, final, layers, config),
                  _hf_logits(ids, embed, final, layers, config), 2e-5), "default derivations"
    # 独立词表头（tie 关闭）
    embed, layers, final, head = _make_model(25, 5, 8, 2, 1, 4, 2, 9, untied=True)
    config = {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
              "tie_word_embeddings": False, "query_pre_attn_scalar": 4}
    ids = torch.tensor([[0, 1, 2, 3]])
    assert _close(solve(ids, embed, final, layers, config, head),
                  _hf_logits(ids, embed, final, layers, config, head), 2e-5), "untied head"


def check_invalid_inputs(solve):
    vocab, hidden, heads, kv, head_dim, inter = 5, 8, 2, 1, 2, 6
    embed, layers, final, _ = _make_model(31, vocab, hidden, heads, kv, head_dim, 2, inter)
    ids = torch.tensor([[0, 1, 2]])
    config = {"num_attention_heads": heads, "head_dim": head_dim}
    good_layer = dict(layers[0])

    def layer_with(**changes):
        changed = dict(good_layer)
        changed.update(changes)
        return changed

    bad_calls = [
        ("config-not-dict", lambda: solve(ids, embed, final, layers, [("num_attention_heads", heads)])),
        ("config-unknown-key", lambda: solve(ids, embed, final, layers, {**config, "num_hidden_head": 2})),
        ("config-missing-heads", lambda: solve(ids, embed, final, layers, {"head_dim": head_dim})),
        ("hidden-mismatch", lambda: solve(ids, embed, final, layers, {**config, "hidden_size": hidden + 2})),
        ("vocab-mismatch", lambda: solve(ids, embed, final, layers, {**config, "vocab_size": vocab + 1})),
        ("heads-zero", lambda: solve(ids, embed, final, layers, {"num_attention_heads": 0})),
        ("heads-bool", lambda: solve(ids, embed, final, layers, {"num_attention_heads": True})),
        ("heads-float", lambda: solve(ids, embed, final, layers, {"num_attention_heads": 2.0})),
        ("head-dim-odd", lambda: solve(ids, embed, final, layers, {**config, "head_dim": 3})),
        ("head-dim-zero", lambda: solve(ids, embed, final, layers, {**config, "head_dim": 0})),
        ("kv-not-divisor", lambda: solve(ids, embed, final, layers, {**config, "num_key_value_heads": 3})),
        ("layers-mismatch", lambda: solve(ids, embed, final, layers, {**config, "num_hidden_layers": 3})),
        ("layer-types-length", lambda: solve(ids, embed, final, layers, {**config, "layer_types": ["full_attention"]})),
        ("layer-types-value", lambda: solve(ids, embed, final, layers, {**config, "layer_types": ["global", "full_attention"]})),
        ("layer-types-not-list", lambda: solve(ids, embed, final, layers, {**config, "layer_types": "full_attention"})),
        ("window-zero", lambda: solve(ids, embed, final, layers, {**config, "sliding_window": 0})),
        ("window-bool", lambda: solve(ids, embed, final, layers, {**config, "sliding_window": True})),
        ("window-float", lambda: solve(ids, embed, final, layers, {**config, "sliding_window": 2.5})),
        ("scalar-zero", lambda: solve(ids, embed, final, layers, {**config, "query_pre_attn_scalar": 0.0})),
        ("scalar-nan", lambda: solve(ids, embed, final, layers, {**config, "query_pre_attn_scalar": float("nan")})),
        ("scalar-bool", lambda: solve(ids, embed, final, layers, {**config, "query_pre_attn_scalar": True})),
        ("attn-cap-zero", lambda: solve(ids, embed, final, layers, {**config, "attn_logit_softcapping": 0.0})),
        ("attn-cap-negative", lambda: solve(ids, embed, final, layers, {**config, "attn_logit_softcapping": -1.0})),
        ("attn-cap-string", lambda: solve(ids, embed, final, layers, {**config, "attn_logit_softcapping": "50"})),
        ("final-cap-inf", lambda: solve(ids, embed, final, layers, {**config, "final_logit_softcapping": float("inf")})),
        ("eps-zero", lambda: solve(ids, embed, final, layers, {**config, "rms_norm_eps": 0.0})),
        ("theta-zero", lambda: solve(ids, embed, final, layers, {**config, "rope_theta": 0.0})),
        ("tie-not-bool", lambda: solve(ids, embed, final, layers, {**config, "tie_word_embeddings": 1})),
        ("tied-with-head", lambda: solve(ids, embed, final, layers, config, embed)),
        ("untied-without-head", lambda: solve(ids, embed, final, layers, {**config, "tie_word_embeddings": False})),
        ("head-shape", lambda: solve(ids, embed, final, layers, {**config, "tie_word_embeddings": False},
                                     torch.zeros(vocab, hidden + 1, dtype=torch.float64))),
        ("head-int", lambda: solve(ids, embed, final, layers, {**config, "tie_word_embeddings": False},
                                   torch.zeros(vocab, hidden, dtype=torch.int64))),
        ("embed-int", lambda: solve(ids, embed.to(torch.int64), final, layers, config)),
        ("embed-1d", lambda: solve(ids, embed[0], final, layers, config)),
        ("embed-nan", lambda: solve(ids, torch.full_like(embed, float("nan")), final, layers, config)),
        ("final-norm-shape", lambda: solve(ids, embed, final[:4], layers, config)),
        ("layers-not-list", lambda: solve(ids, embed, final, tuple(), config)),
        ("layer-not-dict", lambda: solve(ids, embed, final, [list(good_layer.values())], config)),
        ("layer-missing-key", lambda: solve(ids, embed, final, [{k: v for k, v in good_layer.items() if k != "q_weight"}], config)),
        ("layer-extra-key", lambda: solve(ids, embed, final, [layer_with(q_bias=torch.zeros(4, dtype=torch.float64))], config)),
        ("q-shape", lambda: solve(ids, embed, final, [layer_with(q_weight=torch.zeros(heads * head_dim + 2, hidden, dtype=torch.float64))], config)),
        ("k-shape", lambda: solve(ids, embed, final, [layer_with(k_weight=torch.zeros(heads * head_dim, hidden, dtype=torch.float64))], config)),
        ("o-shape", lambda: solve(ids, embed, final, [layer_with(o_weight=torch.zeros(hidden, hidden, dtype=torch.float64))], config)),
        ("gate-shape", lambda: solve(ids, embed, final, [layer_with(gate_weight=torch.zeros(inter, hidden + 1, dtype=torch.float64))], config)),
        ("up-shape", lambda: solve(ids, embed, final, [layer_with(up_weight=torch.zeros(inter + 1, hidden, dtype=torch.float64))], config)),
        ("down-shape", lambda: solve(ids, embed, final, [layer_with(down_weight=torch.zeros(hidden, inter + 1, dtype=torch.float64))], config)),
        ("norm-shape", lambda: solve(ids, embed, final, [layer_with(input_ln_weight=torch.zeros(hidden - 1, dtype=torch.float64))], config)),
        ("ids-not-tensor", lambda: solve([[0, 1, 2]], embed, final, layers, config)),
        ("ids-float", lambda: solve(ids.to(torch.float64), embed, final, layers, config)),
        ("ids-1d", lambda: solve(torch.tensor([0, 1, 2]), embed, final, layers, config)),
        ("ids-empty", lambda: solve(torch.zeros(1, 0, dtype=torch.int64), embed, final, layers, config)),
        ("ids-negative", lambda: solve(torch.tensor([[0, -1, 2]]), embed, final, layers, config)),
        ("ids-out-of-range", lambda: solve(torch.tensor([[0, 1, vocab]]), embed, final, layers, config)),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
