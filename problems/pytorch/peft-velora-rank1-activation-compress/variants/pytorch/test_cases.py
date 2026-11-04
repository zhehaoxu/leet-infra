"""VeLoRA test cases. Judging convention: string CHECKS + check(solve)."""
import torch

torch.set_default_dtype(torch.float64)


def _unit(rows, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(rows, 1, generator=g), p=2, dim=-1)


def check_project(solve):
    torch.manual_seed(0)
    B, N, C, G = 2, 3, 8, 4
    x = torch.randn(B, N, C)
    v_raw = torch.randn(1, C // G)
    embed = torch.nn.functional.normalize(v_raw, p=2, dim=-1)
    zp = solve(mode="project", x=x, embed=embed, num_groups=G)
    assert zp.shape == (B * N * G,), zp.shape
    assert zp.dtype == torch.float64
    # flatten order: (B, N, G) rows, each row C//G wide
    flat = x.reshape(B * N * G, C // G)
    want = (flat * embed[0]).sum(-1)
    assert torch.allclose(zp, want, atol=1e-9), "flatten+projection mismatch"
    # unit-norm embed bounds |zp| by the chunk norm
    assert (zp.abs() <= flat.norm(dim=-1) + 1e-9).all()
    # anchors
    assert abs(zp[0].item() - (-2.3036448720441434)) < 1e-9, zp[0].item()
    assert abs(zp[-1].item() - (-1.3527679700105146)) < 1e-9, zp[-1].item()
    # sub-token group order matters: swapping the (N, G) axes changes zp
    xt = (x.reshape(B, N, G, C // G).transpose(1, 2)
          .reshape(B * N * G, C // G)).view(B, N, C)
    zp_t = solve(mode="project", x=xt, embed=embed, num_groups=G)
    assert not torch.allclose(zp_t, zp, atol=1e-6), "group order has no effect"
    # a different num_groups re-chunks the features
    embed1 = torch.nn.functional.normalize(torch.randn(1, 1), p=2, dim=-1)
    zp1 = solve(mode="project", x=x, embed=embed1, num_groups=8)
    assert zp1.shape == (B * N * 8,)
    assert torch.allclose(zp1, x.reshape(B * N * 8, 1).squeeze(-1) * embed1[0, 0], atol=1e-9)


def check_init_embed(solve):
    torch.manual_seed(1)
    data = torch.randn(7, 5)
    e = solve(mode="init_embed", data=data)
    assert e.shape == (5,), e.shape
    assert abs(e.norm().item() - 1.0) < 1e-9, "embed must be L2-normalized"
    want = torch.nn.functional.normalize(data.mean(0), p=2, dim=-1)
    assert torch.allclose(e, want, atol=1e-9), "batch_average_once mismatch"
    # the direction follows the batch MEAN, not a single row
    assert torch.dot(e, data.mean(0)).item() > 0
    assert not torch.allclose(e, torch.nn.functional.normalize(data[0], p=2, dim=-1), atol=1e-6)
    # anchors
    assert abs(e[0].item() - (-0.7252327797613423)) < 1e-9, e[0].item()
    # a single-row batch degenerates to the normalized row
    d1 = torch.randn(1, 5)
    e1 = solve(mode="init_embed", data=d1)
    assert torch.allclose(e1, torch.nn.functional.normalize(d1[0], p=2, dim=-1), atol=1e-9)
    # scale invariance: normalizing a scaled mean changes nothing
    e2 = solve(mode="init_embed", data=data * 100.0)
    assert torch.allclose(e2, e, atol=1e-9)


def check_reconstruct(solve):
    torch.manual_seed(2)
    B, N, C, G = 2, 4, 8, 4
    x = torch.randn(B, N, C)
    embed = torch.nn.functional.normalize(torch.randn(1, C // G), p=2, dim=-1)
    zp = solve(mode="project", x=x, embed=embed, num_groups=G)
    hat = solve(mode="reconstruct", zp=zp, v=embed, shape=(B, N, C))
    assert hat.shape == (B, N, C), hat.shape
    want = (zp.unsqueeze(-1) * embed[0]).view(B, N, C)
    assert torch.allclose(hat, want, atol=1e-9), "rank-1 rebuild mismatch"
    # anchors
    assert abs(hat[0, 0, 0].item() - (-0.6753214360032626)) < 1e-9, hat[0, 0, 0].item()
    # lossy: the rebuild differs from the original activations
    assert (hat - x).norm().item() > 1e-3, "reconstruction is unexpectedly exact"
    # the residual is orthogonal to embed inside every chunk
    res = (hat - x).reshape(B * N * G, C // G)
    assert torch.allclose((res * embed[0]).sum(-1), torch.zeros(B * N * G), atol=1e-9)
    # round trip: reconstruct(project(x)) re-projects onto itself
    zp2 = solve(mode="project", x=hat, embed=embed, num_groups=G)
    assert torch.allclose(zp2, zp, atol=1e-9), "round trip broken"


def check_grads(solve):
    torch.manual_seed(3)
    B, N, C, M = 2, 5, 6, 4
    inp_hat = torch.randn(B, N, C)
    weight = torch.randn(M, C)
    grad_output = torch.randn(B, N, M)
    out = solve(mode="grads", input_hat=inp_hat, weight=weight,
                grad_output=grad_output, velora_scale=0.1, use_bias=True)
    gi, gw, gb = out["grad_input"], out["grad_weight"], out["grad_bias"]
    assert gi.shape == (B, N, C) and gw.shape == (M, C) and gb.shape == (B, M)
    # autograd sums the leading batch dim of the per-sample (B, M, C) term
    per = torch.matmul(grad_output.transpose(1, 2), inp_hat)
    assert torch.allclose(gw, 0.1 * per.sum(0), atol=1e-9), "grad_weight account"
    assert torch.allclose(gi, grad_output @ weight, atol=1e-9), "grad_input account"
    assert torch.allclose(gb, grad_output.sum(1), atol=1e-9), "grad_bias sums over seq"
    # anchors
    assert abs(gw[0, 0].item() - 0.22305672787004252) < 1e-9, gw[0, 0].item()
    assert abs(gi[0, 0, 0].item() - (-0.0783169401206513)) < 1e-9, gi[0, 0, 0].item()
    # no bias -> grad_bias is None
    out1 = solve(mode="grads", input_hat=inp_hat, weight=weight,
                 grad_output=grad_output, velora_scale=1.0, use_bias=False)
    assert out1["grad_bias"] is None
    assert torch.allclose(out1["grad_weight"], per.sum(0), atol=1e-9)
    # velora_scale multiplies grad_weight ONLY, grad_input is untouched
    out2 = solve(mode="grads", input_hat=inp_hat, weight=weight,
                 grad_output=grad_output, velora_scale=2.0, use_bias=False)
    assert torch.allclose(out2["grad_weight"], 2.0 * per.sum(0), atol=1e-9)
    assert torch.allclose(out2["grad_input"], gi, atol=1e-9)


def check_lora_block(solve):
    torch.manual_seed(4)
    B, N, C, G, M, r = 2, 4, 12, 4, 5, 3
    x = torch.randn(B, N, C)
    weight = torch.randn(M, C)
    bias = torch.randn(M)
    lora_a = torch.randn(r, C)
    lora_b = torch.randn(M, r)
    scaling, vscale = 0.5, 0.1
    embed = torch.nn.functional.normalize(torch.randn(1, C // G), p=2, dim=-1)
    grad_out_m = torch.randn(B, N, M)
    grad_out_r = torch.randn(B, N, r)
    out = solve(mode="lora_block", x=x, weight=weight, bias=bias,
                lora_a=lora_a, lora_b=lora_b, scaling=scaling,
                num_groups=G, embed=embed, velora_scale=vscale,
                grad_out_m=grad_out_m, grad_out_r=grad_out_r)
    res, ga, gb, gi = out["result"], out["grad_a"], out["grad_b"], out["grad_input"]
    # forward is the plain LoRA path, independent of the velora machinery
    base = torch.nn.functional.linear(x, weight, bias)
    a_out = x @ lora_a.t()
    assert torch.allclose(res, base + (a_out @ lora_b.t()) * scaling, atol=1e-9)
    # grad_a flows through the RECONSTRUCTED activations (input_hat)
    flat = x.reshape(B * N * G, C // G)
    zp = (flat * embed[0]).sum(-1)
    hat = (zp.unsqueeze(-1) * embed[0]).view(B, N, C)
    want_ga = vscale * torch.einsum("b n r, b n c -> r c", grad_out_r, hat)
    assert torch.allclose(ga, want_ga, atol=1e-9), "grad_a account"
    # grad_b uses the REAL lora_a outputs, scaled by the lora scaling
    want_gb = scaling * torch.einsum("b n m, b n r -> m r", grad_out_m, a_out)
    assert torch.allclose(gb, want_gb, atol=1e-9), "grad_b account"
    # grad_input: base term + the lora chain scaling*(g@B)@A
    want_gi = grad_out_m @ weight + scaling * ((grad_out_m @ lora_b) @ lora_a)
    assert torch.allclose(gi, want_gi, atol=1e-9), "grad_input account"
    # anchors
    assert abs(ga[0, 0].item() - (-0.3717755681072157)) < 1e-9, ga[0, 0].item()
    assert abs(gb[0, 0].item() - 1.490616682369623) < 1e-9, gb[0, 0].item()
    # reconstruction is lossy: grad_a differs from the plain-activation gradient
    plain_ga = vscale * torch.einsum("b n r, b n c -> r c", grad_out_r, x)
    assert not torch.allclose(ga, plain_ga, atol=1e-6), "input_hat collapsed to x"
    # a different velora_scale rescales grad_a but nothing else
    out3 = solve(mode="lora_block", x=x, weight=weight, bias=bias,
                 lora_a=lora_a, lora_b=lora_b, scaling=scaling,
                 num_groups=G, embed=embed, velora_scale=0.3,
                 grad_out_m=grad_out_m, grad_out_r=grad_out_r)
    assert torch.allclose(out3["grad_a"], 3.0 * ga, atol=1e-9)
    assert torch.allclose(out3["grad_b"], gb, atol=1e-9)
    assert torch.allclose(out3["result"], res, atol=1e-9)


CHECKS = [
    "check_project",
    "check_init_embed",
    "check_reconstruct",
    "check_grads",
    "check_lora_block",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
