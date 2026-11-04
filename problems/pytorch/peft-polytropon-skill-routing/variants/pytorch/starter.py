"""Polytropon (Poly) — starter template.

Implement `solve(mode, **kw)` reproducing the pure-function core of HuggingFace PEFT's
poly tuner (peft/tuners/poly: layer.py PolyLinear + router.py PolyRouter):

- init_params: A (n_splits, n_skills, in//ns, r) kaiming-initialized per (skill,
  split) — skill-outer, split-inner loops, kaiming_uniform_ on a (r, d) tensor with
  a=sqrt(5) then transposed; B zeros when init_weights else kaiming on (d, r)
  (fan_in = r) transposed; router module_logits (n_tasks, n_splits*n_skills)
  uniform_(-1e-3, 1e-3).
- route: sigmoid(logits[task_ids].view(-1, ns, sk)) then normalize across skills:
  s / (s.sum(-1, keepdim=True) + 1e-12). NOT softmax. Out-of-range task id raises.
- combine: A = einsum("bqs,qsdr->bqdr", w, A).reshape(bs, in, r);
  B = einsum("bqs,qsrd->bqrd", w, B).transpose(1,2).reshape(bs, r, out).
- forward_delta: x.bmm(A_mixed).bmm(B_mixed) / r — the scale is exactly 1/r.
- param_count: lora + router budgets.

See description.md for the full contract.
"""

MODES = ["init_params", "route", "combine", "forward_delta", "param_count"]


def solve(mode, **kw):
    raise NotImplementedError
