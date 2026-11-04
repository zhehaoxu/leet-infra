"""AdaptionPrompt test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)
import torch.nn.functional as F


def check_init_params(solve):
    f = solve(mode="init_params", adapter_len=4, hidden=8, seed=0)
    prompt, gate = f["prompt"], f["gate"]
    assert prompt.shape == (1, 4, 8), prompt.shape
    assert gate.shape == (1,), gate.shape
    assert torch.count_nonzero(gate).item() == 0, "gate zero-init"
    # prompt is standard-normal drawn via normal_(generator=): replay the stream
    g = torch.Generator().manual_seed(0)
    ref = torch.empty(1, 4, 8, dtype=torch.float64).normal_(generator=g)
    assert torch.equal(prompt, ref)
    # gate drawn AFTER nothing: separate stream, but prompt consumed the generator
    f2 = solve(mode="init_params", adapter_len=4, hidden=8, seed=0)
    assert torch.equal(f2["prompt"], prompt) and torch.equal(f2["gate"], gate)


def check_adapter_kv(solve):
    torch.manual_seed(0)
    H, hd, kv, L, hidden = 4, 2, 2, 3, 8
    factor = H // kv
    prompt = torch.randn(1, L, hidden)
    k_w = torch.randn(kv * hd, hidden)
    v_w = torch.randn(kv * hd, hidden)
    out = solve(mode="adapter_kv", prompt=prompt, k_w=k_w, v_w=v_w,
                num_heads=H, head_dim=hd, factor=factor)
    ak, av = out["adapter_k"], out["adapter_v"]
    assert ak.shape == (1, H, L, hd), ak.shape
    assert av.shape == (1, H, L, hd), av.shape

    key = F.linear(prompt, k_w).view(1, L, kv, hd)
    value = F.linear(prompt, v_w).view(1, L, kv, hd)
    # official order: transpose first, then repeat_interleave along the KV-head dim
    want_k = key.transpose(1, 2).repeat_interleave(factor, dim=1)
    assert torch.equal(ak, want_k)
    want_v = value.transpose(1, 2).repeat_interleave(factor, dim=1)
    assert torch.equal(av, want_v)
    # v_w (not k_w) feeds the value projection
    assert abs(av[0, 0, 0, 0].item() - 2.7933989571164926) < 1e-9, av[0, 0, 0, 0].item()
    # repeat_interleave groups: [g0, g0, g1, g1] — heads h and h+1 share KV (kv=2);
    # torch.repeat would interleave to [g0, g1, g0, g1] instead.
    assert torch.equal(ak[0, 0], ak[0, 1])
    assert torch.equal(ak[0, 2], ak[0, 3])
    assert not torch.equal(ak[0, 0], ak[0, 2])
    # repeat (interleaved) would NOT match
    wrong = key.transpose(1, 2).repeat(factor, 1, 1, 1)
    assert not torch.equal(ak, wrong)

    # anchor
    assert abs(ak[0, 0, 0, 0].item() - (-3.9983327813446534)) < 1e-9, ak[0, 0, 0, 0].item()

    # second shape family with kv != hd (GQA group count differs from head_dim):
    # view(1, L, kv, hd) mis-sharded into (hd, kv) changes the value layout.
    H2, hd2, kv2, L2 = 2, 4, 1, 2
    prompt2 = torch.randn(1, L2, hidden)
    k_w2 = torch.randn(kv2 * hd2, hidden)
    v_w2 = torch.randn(kv2 * hd2, hidden)
    out2 = solve(mode="adapter_kv", prompt=prompt2, k_w=k_w2, v_w=v_w2,
                 num_heads=H2, head_dim=hd2, factor=H2 // kv2)
    ak2, av2 = out2["adapter_k"], out2["adapter_v"]
    assert ak2.shape == (1, H2, L2, hd2) and av2.shape == (1, H2, L2, hd2)
    key2 = F.linear(prompt2, k_w2).view(1, L2, kv2, hd2).transpose(1, 2).repeat_interleave(2, dim=1)
    value2 = F.linear(prompt2, v_w2).view(1, L2, kv2, hd2).transpose(1, 2).repeat_interleave(2, dim=1)
    assert torch.equal(ak2, key2)
    assert torch.equal(av2, value2)


def check_adapter_attn(solve):
    torch.manual_seed(1)
    bsz, H, q_len, hd, L, hidden = 2, 4, 5, 2, 3, 8
    q = torch.randn(bsz, H, q_len, hd)
    ak = torch.randn(1, H, L, hd)
    av = torch.randn(1, H, L, hd)
    gate = torch.tensor([0.25])
    o_w = torch.randn(hidden, hidden)
    out = solve(mode="adapter_attn", q=q, adapter_k=ak, adapter_v=av,
                gate=gate, head_dim=hd, o_w=o_w)
    probs, adapter_output = out["probs"], out["adapter_output"]
    assert probs.shape == (bsz, H, q_len, L), probs.shape
    assert adapter_output.shape == (bsz, q_len, hidden), adapter_output.shape

    scores = torch.matmul(q, ak.transpose(2, 3)) / math.sqrt(hd)
    want_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    assert torch.allclose(probs, want_probs, atol=1e-12, rtol=0)
    # rows sum to one (softmax over adapter dim)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-9)
    # gate multiplies AFTER softmax (not before: g*softmax(s) != softmax(g*s))
    scores_gated = gate * scores
    wrong = F.softmax(scores_gated, dim=-1, dtype=torch.float32).to(q.dtype)
    assert not torch.allclose(probs, wrong, atol=1e-9)
    want = torch.matmul(gate * want_probs, av).transpose(1, 2).reshape(bsz, q_len, hidden)
    want = F.linear(want, o_w)
    assert torch.allclose(adapter_output, want, atol=1e-12, rtol=0)

    # zero gate (training start) => zero adapter contribution
    out0 = solve(mode="adapter_attn", q=q, adapter_k=ak, adapter_v=av,
                 gate=torch.zeros(1), head_dim=hd, o_w=o_w)
    assert torch.count_nonzero(out0["adapter_output"]).item() == 0

    # anchor
    assert abs(adapter_output[0, 0, 0].item() - 0.04744630390235611) < 1e-9, adapter_output[0, 0, 0].item()


def check_combine(solve):
    torch.manual_seed(2)
    attn_output = torch.randn(2, 5, 8)
    adapter_output = torch.randn(2, 5, 8)
    out = solve(mode="combine", attn_output=attn_output, adapter_output=adapter_output)
    assert torch.equal(out, attn_output + adapter_output)
    # zero-init gate invariance: with adapter_output zero, combine is identity
    assert torch.equal(solve(mode="combine", attn_output=attn_output,
                             adapter_output=torch.zeros_like(attn_output)), attn_output)


def check_param_count(solve):
    c = solve(mode="param_count", adapter_len=16, hidden=4096)
    assert c["trainable"] == 16 * 4096 + 1, c
    assert c["full_prefix"] == 2 * 16 * 4096, c
    assert c["saving"] == 16 * 4096 - 1, c
    # adapter_len=1: prompt contributes hidden params, saving hidden - 1
    c1 = solve(mode="param_count", adapter_len=1, hidden=8)
    assert c1["trainable"] == 9 and c1["saving"] == 7


CHECKS = [
    "check_init_params",
    "check_adapter_kv",
    "check_adapter_attn",
    "check_combine",
    "check_param_count",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
