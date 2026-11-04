"""CPT test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

torch.set_default_dtype(torch.float64)
import torch.nn.functional as F


def check_build_masks(solve):
    # type ids: 0 unused, 1 input-template, 2 input, 3 output-template,
    # 4/8/12 label groups (m == 0), 5/6/7 also map via % 4
    types = [1, 2, 3, 4, 8, 0, 6, 5, 7]
    out = solve(mode="build_masks", cpt_tokens_type_mask=types)
    mask = out["mask"]
    assert mask.shape == (len(types), 1), mask.shape
    want = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.bool).view(-1, 1)
    assert torch.equal(mask, want)
    # label group ids (4, 8) are excluded from updates
    assert not bool(mask[3]) and not bool(mask[4])
    # anchors are exact booleans; spot-check the % 4 mapping
    assert bool(mask[7]) and bool(mask[8])


def check_get_epsilon(solve):
    types = [1, 2, 3, 4, 8, 0]
    eps = solve(mode="get_epsilon", cpt_tokens_type_mask=types, token_dim=2048,
                opt_projection_epsilon=0.1, opt_projection_format_epsilon=0.05)
    assert eps.shape == (6,), eps.shape
    assert eps.dtype == torch.float32
    # fp32 representation of 0.05 / 0.1
    assert abs(eps[0].item() - 0.05000000074505806) < 1e-12, eps[0].item()      # m==1 format
    assert abs(eps[1].item() - 0.10000000149011612) < 1e-12, eps[1].item()      # m==2 input
    assert abs(eps[2].item() - 0.05000000074505806) < 1e-12, eps[2].item()      # m==3 format
    assert abs(eps[3].item() - 1.000000013351432e-10) < 1e-18, eps[3].item()     # label group frozen
    assert abs(eps[4].item() - 1.000000013351432e-10) < 1e-18, eps[4].item()
    assert abs(eps[5].item() - 1.000000013351432e-10) < 1e-18, eps[5].item()     # mask 0 frozen
    # dim scaling: sqrt(dim / 2048)
    eps2 = solve(mode="get_epsilon", cpt_tokens_type_mask=types, token_dim=512,
                 opt_projection_epsilon=0.1, opt_projection_format_epsilon=0.05)
    assert abs(eps2[1].item() - 0.05000000074505806) < 1e-9, eps2[1].item()


def check_project(solve):
    torch.manual_seed(0)
    n, d = 5, 4
    delta = torch.randn(n, d)
    delta[4] = 0.0  # one exactly-zero row: projection_mask selects k < n rows
    eps = torch.tensor([0.5, 0.1, 1e-10, 0.3, 0.7])
    proj = solve(mode="project", delta=delta, epsilon=eps)
    assert proj.shape == delta.shape
    # norms are clamped to the ball; never grown
    norms = torch.norm(proj, p=2, dim=1)
    ref = torch.clamp(torch.norm(delta, p=2, dim=1), max=eps.double() + 0 * norms)
    assert torch.allclose(norms, ref, atol=1e-9), (norms, torch.norm(delta, p=2, dim=1))
    # zero row stays zero (official projection_mask = norm > 0)
    assert torch.equal(proj[4], torch.zeros(d))
    # row with norm < eps unchanged
    small = torch.tensor([[0.01, 0.0, 0.0, 0.0]])
    p1 = solve(mode="project", delta=small, epsilon=torch.tensor([0.5]))
    assert torch.equal(p1, small)
    # row with norm > eps shrunk to exactly eps
    big = torch.tensor([[3.0, 4.0, 0.0, 0.0]])  # norm 5
    p2 = solve(mode="project", delta=big, epsilon=torch.tensor([1.0]))
    assert abs(torch.norm(p2, p=2, dim=1)[0].item() - 1.0) < 1e-12
    assert torch.allclose(p2, big / 5.0, atol=1e-12, rtol=0)
    # zero row stays zero (official projection_mask = norm > 0)
    zero = torch.zeros(1, d)
    p3 = solve(mode="project", delta=zero, epsilon=torch.tensor([0.5]))
    assert torch.equal(p3, zero)
    # frozen row (eps=1e-10) shrinks to ~1e-10
    n4 = torch.norm(proj[2], p=2).item()
    assert abs(n4 - 1e-10) < 1e-14, n4


def check_embed(solve):
    torch.manual_seed(1)
    n, d = 3, 4
    emb_w = torch.randn(n, d)
    delta_w = torch.randn(n, d)
    idx = torch.tensor([0, 2, 1])
    out = solve(mode="embed", indices=idx, embedding_w=emb_w, delta_w=delta_w)
    assert torch.equal(out, F.embedding(idx, emb_w) + F.embedding(idx, delta_w))
    # indices may repeat
    idx2 = torch.tensor([1, 1, 0])
    out2 = solve(mode="embed", indices=idx2, embedding_w=emb_w, delta_w=delta_w)
    assert torch.allclose(out2[0], out2[1], atol=0, rtol=0)
    # anchor
    assert abs(out[0, 0].item() - 0.4658425941708275) < 1e-9, out[0, 0].item()


def check_decay_loss(solve):
    torch.manual_seed(2)
    b, t, v = 1, 6, 11
    logits = torch.randn(b, t, v)
    labels = torch.tensor([[1, 2, 3, 4, 5, -100]]).view(b, t)
    # token types: 1 template, 2 input, 4/8 label groups, 3 output template
    types = torch.tensor([[1, 2, 4, 8, 3, 0]]).view(b, t)

    # no weighting: plain mean over valid (non -100) shifted tokens.
    # labels [1,2,3,4,5,-100] -> shift [2,3,4,5,-100]: 4 valid positions.
    loss0 = solve(mode="decay_loss", logits=logits, labels=labels, cpt_type_mask=types,
                  opt_loss_decay_factor=0.5, opt_weighted_loss_type="none")
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    per = F.cross_entropy(shift_logits.reshape(-1, v), shift_labels.reshape(-1),
                          reduction="none", ignore_index=-100)
    want0 = per[:4].mean()
    assert abs(loss0.item() - want0.item()) < 1e-9

    # decay weighting: shifted types are [2, 4, 8, 3, 0]; groups (%4==0, >0): {4, 8}
    # reversed unique ids [8, 4]: group 8 (shift pos 2) weight 1, group 4 (pos 1) 0.5.
    lossd = solve(mode="decay_loss", logits=logits, labels=labels, cpt_type_mask=types,
                  opt_loss_decay_factor=0.5, opt_weighted_loss_type="decay")
    w = torch.tensor([1.0, 0.5, 1.0, 1.0])
    wantd = (per[:4] * w).mean()
    assert abs(lossd.item() - wantd.item()) < 1e-9, (lossd.item(), wantd.item())

    # -100 tokens are excluded entirely (weights apply to valid tokens only)
    labels2 = torch.tensor([[1, 2, 3, 4, -100, -100]]).view(b, t)
    loss2 = solve(mode="decay_loss", logits=logits, labels=labels2, cpt_type_mask=types,
                  opt_loss_decay_factor=0.5, opt_weighted_loss_type="decay")
    per2 = F.cross_entropy(shift_logits.reshape(-1, v),
                           labels2[..., 1:].reshape(-1), reduction="none", ignore_index=-100)
    w2 = torch.tensor([1.0, 0.5, 1.0])
    want2 = (per2[:3] * w2).mean()
    assert abs(loss2.item() - want2.item()) < 1e-9

    # anchor
    assert abs(lossd.item() - 2.5720460018143934) < 1e-9, lossd.item()


CHECKS = [
    "check_build_masks",
    "check_get_epsilon",
    "check_project",
    "check_embed",
    "check_decay_loss",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
