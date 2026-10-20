"""Test cases for multi-head-attention-forward."""

import torch
import torch.nn.functional as F


def _split(t, num_heads):
    B, L, D = t.shape
    d = D // num_heads
    return t.view(B, L, num_heads, d).transpose(1, 2)  # (B, H, L, d)


def _manual(q, k, v, num_heads, causal):
    """按题面规则的参考实现（float64 提升精度）。"""
    qh, kh, vh = (_split(t.double(), num_heads) for t in (q, k, v))
    d = qh.shape[-1]
    scores = qh @ kh.transpose(-1, -2) / d**0.5
    if causal:
        L = scores.shape[-1]
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    att = F.softmax(scores, dim=-1)
    out = att @ vh  # (B, H, L, d)
    B, H, L, _ = out.shape
    return out.transpose(1, 2).reshape(B, L, H * d)


def check_output_shape_and_dtype(solve):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 5, 8) for _ in range(3))
    out = solve(q, k, v, num_heads=2)
    assert out.shape == (2, 5, 8), f"shape {tuple(out.shape)} != (2, 5, 8)"
    assert out.dtype == torch.float32, f"dtype {out.dtype} != float32"


def check_single_head_matches_manual(solve):
    torch.manual_seed(1)
    q, k, v = (torch.randn(3, 6, 4) for _ in range(3))
    out = solve(q, k, v, num_heads=1)
    assert torch.allclose(out.double(), _manual(q, k, v, 1, False), atol=1e-5), (
        "单头结果与手工参考不一致"
    )


def check_head_split_convention(solve):
    """特征维连续切分约定：头间数值必须互相独立。"""
    torch.manual_seed(2)
    B, L, D = 1, 4, 6
    q, k, v = (torch.randn(B, L, D) for _ in range(3))
    out = solve(q, k, v, num_heads=3)
    ref = _manual(q, k, v, 3, False)
    assert torch.allclose(out.double(), ref, atol=1e-5), (
        "多头切分方式与题面不符（应按特征维连续切分）"
    )


def check_matches_sdpa_bidirectional(solve):
    torch.manual_seed(3)
    B, L, D, H = 2, 7, 8, 4
    q, k, v = (torch.randn(B, L, D) for _ in range(3))
    out = solve(q, k, v, num_heads=H)
    ref = F.scaled_dot_product_attention(
        _split(q, H), _split(k, H), _split(v, H)
    )
    ref = ref.transpose(1, 2).reshape(B, L, D)
    assert torch.allclose(out, ref, atol=1e-5), (
        f"与 SDPA 不一致，最大差 {(out - ref).abs().max()}"
    )


def check_matches_sdpa_causal(solve):
    torch.manual_seed(4)
    B, L, D, H = 2, 7, 8, 4
    q, k, v = (torch.randn(B, L, D) for _ in range(3))
    out = solve(q, k, v, num_heads=H, causal=True)
    ref = F.scaled_dot_product_attention(
        _split(q, H), _split(k, H), _split(v, H), is_causal=True
    )
    ref = ref.transpose(1, 2).reshape(B, L, D)
    assert torch.allclose(out, ref, atol=1e-5), (
        f"causal 与 SDPA 不一致，最大差 {(out - ref).abs().max()}"
    )


def check_causal_future_independence(solve):
    torch.manual_seed(5)
    B, L, D, H = 1, 8, 4, 2
    q, k, v = (torch.randn(B, L, D) for _ in range(3))
    out1 = solve(q, k, v, num_heads=H, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[:, 4:] = torch.randn_like(k2[:, 4:]) * 10
    v2[:, 4:] = torch.randn_like(v2[:, 4:]) * 10
    out2 = solve(q, k2, v2, num_heads=H, causal=True)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-6), (
        "篡改未来 key/value 影响了早期输出——causal mask 未生效"
    )


def check_indivisible_dim_raises(solve):
    q, k, v = (torch.ones(1, 3, 7) for _ in range(3))
    try:
        solve(q, k, v, num_heads=2)
    except ValueError:
        return
    raise AssertionError("D 不能被 num_heads 整除时应抛出 ValueError")


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_output_shape_and_dtype.example = True
check_single_head_matches_manual.example = True
