"""Tests for peft-cartridge-kv-prefix (judge-style bare asserts, check(solve))."""

import torch

CHECKS = ["check_pack_layout", "check_pack_validation", "check_encoder_segments",
          "check_load_and_forward", "check_compose"]


def _close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=0, atol=1e-6)


def _mk_cache(L, H, S, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    pkv = []
    for _ in range(L):
        k = torch.randn(1, H, S, D, generator=g)
        v = torch.randn(1, H, S, D, generator=g)
        pkv.append((k, v))
    return pkv


def _expect_value_error(fn, tag):
    try:
        fn()
        raise AssertionError(f"expected ValueError for {tag}")
    except ValueError:
        pass


def check_pack_layout(solve):
    L, H, D, S, nvt = 3, 4, 5, 10, 6
    pkv = _mk_cache(L, H, S, D, seed=7)
    pe = solve(mode="pack", past_key_values=pkv, num_virtual_tokens=nvt)
    assert pe.shape == (nvt, L * 2 * H * D)
    # element ledger: (t, 2l*H*D + h*D + d) == k_l[0,h,t,d]
    for l in range(L):
        k, v = pkv[l]
        for t in (0, nvt - 1):
            for h in (0, H - 1):
                for d in (0, D - 1):
                    assert pe[t, (2 * l) * H * D + h * D + d] == k[0, h, t, d]
                    assert pe[t, (2 * l + 1) * H * D + h * D + d] == v[0, h, t, d]
    # truncation: only the first nvt cached tokens are taken
    expect_t = torch.stack([pkv[l][kv][0, :, 5, :] for l in range(L) for kv in (0, 1)])  # [L*2, H, D]
    assert _close(pe[5].view(L * 2, H, D), expect_t)


def check_pack_validation(solve):
    L, H, D, S, nvt = 2, 2, 3, 5, 4
    ok = _mk_cache(L, H, S, D, seed=1)

    def pack(pkv, n):
        return solve(mode="pack", past_key_values=pkv, num_virtual_tokens=n)

    _expect_value_error(lambda: pack([], nvt), "empty cache")
    _expect_value_error(lambda: pack([(ok[0][0].squeeze(0), ok[0][1])], nvt), "rank-3 key")
    bad_batch = [(ok[0][0].repeat(2, 1, 1, 1), ok[0][1].repeat(2, 1, 1, 1))]
    _expect_value_error(lambda: pack(bad_batch, nvt), "batch=2")
    _expect_value_error(lambda: pack(ok, S + 1), "short cache")
    # inconsistent head shapes: k differs while v matches (v-only check must not suffice)
    bad_heads_kv = [(ok[0][0], ok[0][1]), (torch.randn(1, H + 1, S, D), torch.randn(1, H, S, D))]
    _expect_value_error(lambda: pack(bad_heads_kv, nvt), "k head mismatch with matching v")
    # inconsistent heads (both differ)
    bad_heads = [(ok[0][0], ok[0][1]), (torch.randn(1, H + 1, S, D), torch.randn(1, H + 1, S, D))]
    _expect_value_error(lambda: pack(bad_heads, nvt), "inconsistent heads")
    # k long but v short (k-only length check must not suffice)
    bad_vlen = [(torch.randn(1, H, 6, D), torch.randn(1, H, 3, D))]
    _expect_value_error(lambda: pack(bad_vlen, 4), "v shorter than nvt")
    # inconsistent head_dim
    bad_dim = [(ok[0][0], ok[0][1]), (torch.randn(1, H, S, D + 1), torch.randn(1, H, S, D + 1))]
    _expect_value_error(lambda: pack(bad_dim, nvt), "inconsistent head_dim")
    # per-layer short cache (k shorter than nvt)
    bad_len = [(torch.randn(1, H, 3, D), torch.randn(1, H, S, D))]
    _expect_value_error(lambda: pack(bad_len, 4), "k shorter than nvt")


def check_encoder_segments(solve):
    # RNG anchor: official path — frozen drawn BEFORE trainable, both normal_(std=1)
    L, H, D, nvt, nft = 2, 3, 4, 6, 1
    hidden = L * 2 * H * D
    enc = solve(mode="encoder_new", seed=42, num_virtual_tokens=nvt,
                num_layers=L, token_dim=H * D, num_frozen_tokens=nft)
    torch.manual_seed(42)
    f = torch.empty(nft, hidden).normal_()
    t = torch.empty(nvt - nft, hidden).normal_()
    frozen, trainable = solve(mode="encoder_params", encoder=enc)
    assert _close(frozen.detach(), f)
    assert _close(trainable.detach(), t)
    assert frozen.requires_grad is False
    assert trainable.requires_grad is True
    assert enc.num_trainable_tokens == nvt - nft
    # weight = cat([frozen, trainable], dim=0)
    w = solve(mode="encoder_weight", encoder=enc)
    assert w.shape == (nvt, hidden)
    assert _close(w[:nft], f) and _close(w[nft:], t)
    # nft=0: no frozen parameter, weight is the trainable tensor itself
    enc0 = solve(mode="encoder_new", seed=1, num_virtual_tokens=nvt,
                 num_layers=L, token_dim=H * D, num_frozen_tokens=0)
    assert enc0.frozen_embedding is None
    assert solve(mode="encoder_weight", encoder=enc0) is enc0.trainable_embedding
    # nft=nvt: all frozen
    encF = solve(mode="encoder_new", seed=2, num_virtual_tokens=nvt,
                 num_layers=L, token_dim=H * D, num_frozen_tokens=nvt)
    assert encF.num_trainable_tokens == 0
    assert encF.trainable_embedding.shape == (0, hidden)
    _expect_value_error(
        lambda: solve(mode="encoder_new", seed=3, num_virtual_tokens=nvt,
                      num_layers=L, token_dim=H * D, num_frozen_tokens=nvt + 1),
        "frozen > nvt",
    )
    _expect_value_error(
        lambda: solve(mode="encoder_new", seed=3, num_virtual_tokens=nvt,
                      num_layers=L, token_dim=H * D, num_frozen_tokens=-1),
        "negative frozen",
    )


def check_load_and_forward(solve):
    L, H, D, nvt, nft = 3, 2, 4, 5, 2
    hidden = L * 2 * H * D
    enc = solve(mode="encoder_new", seed=3, num_virtual_tokens=nvt,
                num_layers=L, token_dim=H * D, num_frozen_tokens=nft)
    pe = torch.randn(nvt, hidden)
    solve(mode="encoder_load", encoder=enc, prompt_embeddings=pe)
    frozen, trainable = solve(mode="encoder_params", encoder=enc)
    assert _close(frozen.detach(), pe[:nft])
    assert _close(trainable.detach(), pe[nft:])
    assert _close(solve(mode="encoder_weight", encoder=enc), pe)
    # invalid load shapes
    _expect_value_error(
        lambda: solve(mode="encoder_load", encoder=enc, prompt_embeddings=torch.randn(nvt + 1, hidden)),
        "wrong row count",
    )
    _expect_value_error(
        lambda: solve(mode="encoder_load", encoder=enc, prompt_embeddings=torch.randn(hidden, nvt)),
        "swapped shape",
    )
    # forward ignores token ids and expands batch
    out = solve(mode="encoder_forward", encoder=enc, prefix_tokens=torch.zeros(7, nvt, dtype=torch.long))
    assert out.shape == (7, nvt, hidden)
    assert _close(out[4, 2], solve(mode="encoder_weight", encoder=enc)[2])
    assert _close(out[0], solve(mode="encoder_weight", encoder=enc))
    out1 = solve(mode="encoder_forward", encoder=enc, prefix_tokens=torch.zeros(1, nvt, dtype=torch.long))
    assert out1.shape == (1, nvt, hidden)


def check_compose(solve):
    L, H, D, nvt = 2, 2, 3, 4
    hidden = L * 2 * H * D
    cfg_a = {
        "peft_type": "CARTRIDGE", "task_type": "CAUSAL_LM", "token_dim": H * D,
        "num_layers": L, "num_attention_heads": H, "num_transformer_submodules": 2,
        "num_frozen_tokens": 1,
    }
    cfg_b = dict(cfg_a)
    pa, pb = torch.randn(4, hidden), torch.randn(4, hidden)

    def compose(adapters):
        return solve(mode="compose", adapters=adapters)

    composed, nft = compose([{"prompt_embeddings": pa, "config": cfg_a},
                             {"prompt_embeddings": pb, "config": cfg_b}])
    assert composed.shape == (8, hidden)
    assert _close(composed[:4], pa) and _close(composed[4:], pb)
    assert nft == 1  # frozen count taken from the FIRST adapter only
    # frozen count differing between adapters is allowed; first wins
    cfg_c = dict(cfg_a, num_frozen_tokens=2)
    _, nft2 = compose([{"prompt_embeddings": pa, "config": cfg_a},
                       {"prompt_embeddings": pb, "config": cfg_c}])
    assert nft2 == 1
    # single adapter rejected
    _expect_value_error(lambda: compose([{"prompt_embeddings": pa, "config": cfg_a}]), "len<2")
    # wrong peft_type
    _expect_value_error(
        lambda: compose([{"prompt_embeddings": pa, "config": cfg_a},
                         {"prompt_embeddings": pb, "config": dict(cfg_b, peft_type="PREFIX_TUNING")}]),
        "non-CARTRIDGE",
    )
    # incompatible compat attribute
    _expect_value_error(
        lambda: compose([{"prompt_embeddings": pa, "config": cfg_a},
                         {"prompt_embeddings": pb, "config": dict(cfg_b, num_layers=L + 1)}]),
        "incompatible config",
    )
    # task_type mismatch is the FIRST checked attribute
    _expect_value_error(
        lambda: compose([{"prompt_embeddings": pa, "config": cfg_a},
                         {"prompt_embeddings": pb, "config": dict(cfg_b, task_type="SEQ_2_SEQ_LM")}]),
        "task_type mismatch",
    )
    # three-way composition keeps order
    pc = torch.randn(4, hidden)
    comp3, _ = compose([{"prompt_embeddings": pa, "config": cfg_a},
                        {"prompt_embeddings": pb, "config": cfg_b},
                        {"prompt_embeddings": pc, "config": cfg_b}])
    assert comp3.shape == (12, hidden)
    assert _close(comp3[8:], pc)
