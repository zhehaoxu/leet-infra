import math

import numpy as np
import torch


def _close(a, b, tol=1e-5):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _mscale(scale, mscale=1.0):
    return 1.0 if scale <= 1.0 else 0.1 * mscale * math.log(scale) + 1.0


def _oracle(
    head_dim,
    base,
    factor,
    original_max_position_embeddings,
    positions,
    beta_fast=32.0,
    beta_slow=1.0,
    truncate=True,
    mscale=None,
    mscale_all_dim=None,
    attention_factor=None,
    rope_type="yarn",
    seq_len=None,
):
    """独立 numpy 重算：波长 ramp、γ 与 cos/sin 全走标量公式。"""
    dim = head_dim
    orig = original_max_position_embeddings
    half = dim // 2
    pos_freqs = np.array([base ** (2.0 * i / dim) for i in range(half)], dtype=np.float64)
    extrap = 1.0 / pos_freqs
    interp = extrap / factor

    if rope_type == "linear":
        inv, gamma = interp, 1.0
    elif rope_type == "dynamic_ntk":
        length = orig if seq_len is None else max(seq_len, orig)
        scaled = base * ((factor * length / orig) - (factor - 1.0)) ** (dim / (dim - 2.0))
        inv = np.array([scaled ** (-2.0 * i / dim) for i in range(half)], dtype=np.float64)
        gamma = 1.0
    else:
        if attention_factor is not None:
            gamma = float(attention_factor)
        elif mscale is not None and mscale_all_dim is not None:
            gamma = _mscale(factor, mscale) / _mscale(factor, mscale_all_dim)
        else:
            gamma = _mscale(factor)

        def correction(num_rotations):
            return (dim * math.log(orig / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))

        low = correction(beta_fast)
        high = correction(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low = max(low, 0.0)
        high = min(high, dim - 1.0)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(half, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
        inv = interp * ramp + extrap * (1.0 - ramp)

    freqs = np.asarray(positions, dtype=np.float64)[:, :, None] * inv[None, None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return inv, gamma, np.cos(emb) * gamma, np.sin(emb) * gamma


def _rotation(q, cos, sin):
    half = q.shape[-1] // 2
    x1, x2 = q[..., :half], q[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return q * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def check_example(solve):
    pos = torch.tensor([[0, 1, 2, 3]])
    inv_freq, gamma, cos, sin = solve(8, 10000.0, 4.0, 2048, pos)
    assert inv_freq.dtype == torch.float32 and inv_freq.shape == (4,)
    assert cos.shape == sin.shape == (1, 4, 8) and cos.dtype == torch.float32
    assert _close(inv_freq, [1.0, 0.1, 0.00625, 0.00025]), inv_freq.tolist()
    assert abs(gamma - 1.138629436111989) < 1e-9, gamma
    assert _close(cos[0, 3], [-1.127235, 1.087774, 1.138429, 1.138629] * 2), cos[0, 3].tolist()
    assert _close(sin[0, 3], [0.160683, 0.336488, 0.021348, 0.000854] * 2), sin[0, 3].tolist()
    assert _close(cos[0, 2], [-0.473837, 1.115933, 1.138541, 1.138629] * 2)


check_example.example = True


def check_ramp_structure(solve):
    pos = torch.arange(5).unsqueeze(0)
    head_dim, base, factor, orig = 64, 10000.0, 8.0, 4096
    inv, _, _, _ = solve(head_dim, base, factor, orig, pos)

    def correction(num_rotations):
        return (head_dim * math.log(orig / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))

    low = max(math.floor(correction(32.0)), 0)
    high = min(math.ceil(correction(1.0)), head_dim - 1)
    assert 0 < low < high < head_dim // 2, (low, high)

    vals = inv.tolist()
    extrap = [base ** (-2.0 * i / head_dim) for i in range(head_dim // 2)]
    interp = [v / factor for v in extrap]
    for i in range(low):
        assert abs(vals[i] - extrap[i]) < 1e-6 * extrap[i], f"i={i} 应完整外推"
    for i in range(high + 1, head_dim // 2):
        assert abs(vals[i] - interp[i]) < 1e-6 * interp[i], f"i={i} 应完整插值"
    for i in range(low + 1, high):
        assert interp[i] < vals[i] < extrap[i], f"i={i} 应落在插值与外推之间"
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), "inv_freq 应单调不增"


def check_truncate_and_betas(solve):
    pos = torch.tensor([[0, 1, 2, 3]])
    sharp, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, truncate=True)
    smooth, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, truncate=False)
    assert abs(float(sharp[0]) - 1.0) < 1e-7 and abs(float(smooth[0]) - 1.0) < 1e-7
    assert abs(float(sharp[3]) - 0.00025) < 1e-9 and abs(float(smooth[3]) - 0.00025) < 1e-9
    # 只有 ramp 区的维度受 floor/ceil 影响：i=2 在 1.008..2.513 的区间内
    assert abs(float(sharp[2]) - 0.00625) < 1e-8
    assert float(smooth[2]) < float(sharp[2]) - 1e-4, (float(smooth[2]), float(sharp[2]))

    # beta 越大越往外推：beta_fast 变小时低频端更早开始插值
    wide, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, beta_fast=64.0, beta_slow=1.0)
    narrow, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, beta_fast=16.0, beta_slow=1.0)
    assert float(narrow[1]) > float(wide[1]), "beta_fast 变小应让 i=1 更晚进入插值（保持外推）"


def check_attention_scaling(solve):
    pos = torch.tensor([[0, 1, 2]])
    _, gamma, cos, sin = solve(8, 10000.0, 4.0, 2048, pos)
    assert abs(gamma - (0.1 * math.log(4.0) + 1.0)) < 1e-12
    assert abs(float(cos[0, 0, 0]) - gamma) < 1e-6, "位置 0 的 cos 应等于 γ"
    assert abs(float(sin[0, 0, 0])) < 1e-9, "位置 0 的 sin 应为 0"

    _, gamma1, cos1, _ = solve(8, 10000.0, 1.0, 2048, pos)
    assert gamma1 == 1.0 and abs(float(cos1[0, 0, 0]) - 1.0) < 1e-6

    _, gamma_ratio, _, _ = solve(8, 10000.0, 4.0, 2048, pos, mscale=1.0, mscale_all_dim=2.0)
    want = (0.1 * math.log(4.0) + 1.0) / (0.2 * math.log(4.0) + 1.0)
    assert abs(gamma_ratio - want) < 1e-12, (gamma_ratio, want)

    _, gamma_explicit, cos_e, _ = solve(8, 10000.0, 4.0, 2048, pos, attention_factor=2.5)
    assert gamma_explicit == 2.5 and abs(float(cos_e[0, 0, 0]) - 2.5) < 1e-6


def check_cos_sin_amplitude(solve):
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 16, dtype=torch.float64)
    pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
    _, gamma, cos, sin = solve(16, 10000.0, 8.0, 4096, pos)
    assert abs(gamma - 1.0) > 1e-3, "该配置的 γ 必须不为 1，否则测不出幅度缩放"

    rotated = _rotation(q, cos.double(), sin.double())
    base_norm = q.norm(dim=-1)
    assert torch.allclose(rotated.norm(dim=-1), gamma * base_norm, atol=1e-5), "γ 只缩放幅度、不改旋转角"
    unscaled = _rotation(q, (cos / gamma).double(), (sin / gamma).double())
    assert torch.allclose(unscaled.norm(dim=-1), base_norm, atol=1e-5)


def check_linear_mode(solve):
    pos = torch.arange(4).unsqueeze(0)
    inv, gamma, cos, sin = solve(16, 500000.0, 4.0, 4096, pos, rope_type="linear")
    extrap = torch.tensor([500000.0 ** (-2.0 * i / 16) for i in range(8)], dtype=torch.float64)
    assert _close(inv.double(), extrap / 4.0, tol=1e-6), inv.tolist()
    assert gamma == 1.0, "linear 模式没有注意力温度"
    assert abs(float(cos[0, 1, 0]) - math.cos(0.25)) < 1e-6, "位置 1 的 cos 应取插值角且不乘 γ"

    factor1, _, cos1, _ = solve(16, 500000.0, 1.0, 4096, pos, rope_type="linear")
    plain, _, cos_p, _ = solve(16, 500000.0, 1.0, 4096, pos)
    assert _close(factor1, plain) and _close(cos1, cos_p), "factor=1 时 linear 应退化为未缩放 RoPE"


def check_dynamic_ntk(solve):
    pos = torch.arange(4).unsqueeze(0)
    inv, gamma, _, _ = solve(8, 10000.0, 4.0, 2048, pos, rope_type="dynamic_ntk", seq_len=8192)
    scaled = 10000.0 * ((4.0 * 8192 / 2048) - (4.0 - 1.0)) ** (8.0 / 6.0)
    want = [scaled ** (-2.0 * i / 8) for i in range(4)]
    assert _close(inv.double(), want, tol=1e-6), (inv.tolist(), want)
    assert gamma == 1.0

    short, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, rope_type="dynamic_ntk", seq_len=512)
    at_cap, _, _, _ = solve(8, 10000.0, 4.0, 2048, pos, rope_type="dynamic_ntk")
    plain, _, _, _ = solve(8, 10000.0, 1.0, 2048, pos)
    assert _close(short, at_cap), "seq_len 不足训练长度时应被抬到训练长度"
    assert _close(at_cap, plain), "未超出训练长度时 base 不应被改写"
    assert not _close(inv, plain), "超出训练长度后 inv_freq 必须变"


def check_against_oracle(solve):
    cases = [
        dict(head_dim=8, base=10000.0, factor=4.0, original_max_position_embeddings=2048),
        dict(head_dim=64, base=10000.0, factor=8.0, original_max_position_embeddings=4096, beta_fast=64.0, beta_slow=3.0),
        dict(head_dim=32, base=500000.0, factor=2.0, original_max_position_embeddings=1024, truncate=False),
        dict(head_dim=16, base=10000.0, factor=16.0, original_max_position_embeddings=8192, mscale=1.0, mscale_all_dim=1.0),
        dict(head_dim=128, base=10000.0, factor=4.0, original_max_position_embeddings=2048, attention_factor=1.7),
    ]
    positions = np.array([[0, 1, 2, 5, 17], [3, 40, 100, 511, 4095]])
    pos_t = torch.from_numpy(positions)
    for cfg in cases:
        inv, gamma, cos, sin = solve(position_ids=pos_t, rope_type="yarn", **cfg)
        w_inv, w_gamma, w_cos, w_sin = _oracle(positions=positions, rope_type="yarn", **cfg)
        assert _close(inv.double(), w_inv, tol=1e-6), cfg
        assert abs(gamma - w_gamma) < 1e-9, cfg
        assert _close(cos.double(), w_cos, tol=1e-6), cfg
        assert _close(sin.double(), w_sin, tol=1e-6), cfg

    for cfg in (dict(rope_type="linear"), dict(rope_type="dynamic_ntk", seq_len=16384)):
        full = dict(head_dim=16, base=10000.0, factor=8.0, original_max_position_embeddings=4096, **cfg)
        inv, gamma, cos, sin = solve(position_ids=pos_t, **full)
        w_inv, w_gamma, w_cos, w_sin = _oracle(positions=positions, **full)
        assert _close(inv.double(), w_inv, tol=1e-6), full
        assert abs(gamma - w_gamma) < 1e-12 and _close(cos.double(), w_cos, tol=1e-6), full
        assert _close(sin.double(), w_sin, tol=1e-6), full


def _hf_rope(head_dim, base, factor, orig, rope_type, seq_len=None, **extra):
    import transformers as tr
    from transformers.modeling_rope_utils import _compute_dynamic_ntk_parameters
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rope_parameters = {
        "rope_type": rope_type,
        "rope_theta": base,
        "factor": factor,
        "original_max_position_embeddings": orig,
        "truncate": True,
    }
    rope_parameters.update(extra)
    cfg = tr.LlamaConfig(
        hidden_size=head_dim * 4,
        num_attention_heads=4,
        head_dim=head_dim,
        max_position_embeddings=orig,
        rope_parameters=rope_parameters,
    )
    if rope_type == "dynamic_ntk":
        return _compute_dynamic_ntk_parameters(cfg, seq_len=seq_len)
    rotary = LlamaRotaryEmbedding(cfg)
    return rotary.inv_freq, rotary.attention_scaling


def check_against_transformers(solve):
    positions = torch.tensor([[0, 1, 2, 7, 63], [9, 100, 177, 247, 300]])
    carrier = torch.zeros(positions.shape[0], positions.shape[1], 1)
    for head_dim, base, factor, orig, extra in (
        (8, 10000.0, 4.0, 2048, {}),
        (64, 10000.0, 8.0, 4096, {}),
        (32, 10000.0, 4.0, 2048, {"beta_fast": 64.0, "beta_slow": 3.0}),
        (16, 500000.0, 16.0, 8192, {"mscale": 1.0, "mscale_all_dim": 2.0}),
    ):
        inv, gamma, cos, sin = solve(head_dim, base, factor, orig, positions, **extra)
        hf_inv, hf_gamma = _hf_rope(head_dim, base, factor, orig, "yarn", **extra)
        assert _close(inv.double(), hf_inv.double(), tol=1e-5), (head_dim, extra)
        assert abs(gamma - hf_gamma) < 1e-9
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        import transformers as tr

        cfg = tr.LlamaConfig(
            hidden_size=head_dim * 4,
            num_attention_heads=4,
            head_dim=head_dim,
            max_position_embeddings=orig,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": base,
                "factor": factor,
                "original_max_position_embeddings": orig,
                **extra,
            },
        )
        hf_cos, hf_sin = LlamaRotaryEmbedding(cfg)(carrier, positions)
        assert _close(cos, hf_cos, tol=1e-4), (head_dim, extra)
        assert _close(sin, hf_sin, tol=1e-4), (head_dim, extra)

    for head_dim, base, factor, orig, rope_type, seq_len in (
        (16, 10000.0, 8.0, 4096, "linear", None),
        (32, 10000.0, 4.0, 2048, "dynamic_ntk", 8192),
        (8, 500000.0, 2.0, 1024, "dynamic_ntk", None),
    ):
        inv, gamma, _, _ = solve(head_dim, base, factor, orig, positions, rope_type=rope_type, seq_len=seq_len)
        hf_inv, hf_gamma = _hf_rope(head_dim, base, factor, orig, rope_type, seq_len=seq_len)
        assert _close(inv.double(), hf_inv.double(), tol=1e-5), (rope_type, seq_len)
        assert abs(gamma - hf_gamma) < 1e-12


def check_invalid(solve):
    pos = torch.tensor([[0, 1]])
    for bad in (
        lambda: solve(7, 10000.0, 4.0, 2048, pos),
        lambda: solve(2, 10000.0, 4.0, 2048, pos),
        lambda: solve(8.0, 10000.0, 4.0, 2048, pos),
        lambda: solve(True, 10000.0, 4.0, 2048, pos),
        lambda: solve(8, 1.0, 4.0, 2048, pos),
        lambda: solve(8, 0.0, 4.0, 2048, pos),
        lambda: solve(8, float("nan"), 4.0, 2048, pos),
        lambda: solve(8, float("inf"), 4.0, 2048, pos),
        lambda: solve(8, 10000.0, 0.0, 2048, pos),
        lambda: solve(8, 10000.0, -2.0, 2048, pos),
        lambda: solve(8, 10000.0, 4.0, 0, pos),
        lambda: solve(8, 10000.0, 4.0, 2048.0, pos),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, beta_fast=1.0, beta_slow=8.0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, beta_fast=float("nan")),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, truncate="yes"),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, rope_type="ntk"),
        lambda: solve(8, 10000.0, 4.0, 2048, [[0, 1]]),
        lambda: solve(8, 10000.0, 4.0, 2048, torch.tensor([0, 1])),
        lambda: solve(8, 10000.0, 4.0, 2048, torch.tensor([[0.0, 1.0]])),
        lambda: solve(8, 10000.0, 4.0, 2048, torch.tensor([[-1, 1]])),
        lambda: solve(8, 10000.0, 4.0, 2048, torch.zeros(0, 2, dtype=torch.long)),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, rope_type="linear", attention_factor=2.0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, rope_type="linear", mscale=1.0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, attention_factor=0.0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, attention_factor=-1.0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, rope_type="dynamic_ntk", seq_len=0),
        lambda: solve(8, 10000.0, 4.0, 2048, pos, rope_type="dynamic_ntk", seq_len=4.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
