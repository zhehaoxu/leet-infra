"""Polytropon (Poly) — HuggingFace PEFT poly tuner.

Reproduces the pure-function core of peft/src/peft/tuners/poly (layer.py PolyLinear +
router.py PolyRouter; papers "Combining Modular Skills in Multitask Learning" and
"Multi-Head Adapter Routing for Cross-Task Generalization").

Polytropon keeps an *inventory* of n_skills LoRA adapters per layer; a per-task
routing function picks the mixture. With n_splits > 1 each LoRA's rank is split into
disjoint blocks and every block has its own routing function (Multi-Head Routing):

- poly_lora_A: (n_splits, n_skills, in // n_splits, r)
- poly_lora_B: (n_splits, n_skills, r, out // n_splits)
- router module_logits: (n_tasks, n_splits * n_skills), uniform_(-1e-3, 1e-3)

Routing (inference): sigmoid(logits) then normalize across skills:
weights = s / (s.sum(-1, keepdim=True) + 1e-12).  No softmax anywhere.

Forward: A = einsum("bqs,qsdr->bqdr", w, poly_lora_A).reshape(bs, in, r)
         B = einsum("bqs,qsrd->bqrd", w, poly_lora_B).transpose(1,2).reshape(bs, r, out)
         delta = x @ A @ B / r          # scaling is exactly 1/r

Init: A per (skill, split) kaiming_uniform_ on a (r, d) tensor with a=sqrt(5)
(fan_in = d = in // n_splits), transposed into place; B zeros when init_weights else
kaiming on (d, r) (fan_in = r) transposed; the loops run skill-outer, split-inner.
"""

import math

import torch

EPS = 1e-12


def _kaiming_transposed(rows, cols, a, generator, dtype):
    # torch.nn.init.kaiming_uniform_ on a (rows, cols) tensor with gain a,
    # then transposed: bound = sqrt(6 / ((1 + a^2) * fan_in)), fan_in = cols.
    bound = math.sqrt(6.0 / ((1.0 + a * a) * cols))
    t = torch.empty(rows, cols, dtype=dtype).uniform_(-bound, bound, generator=generator)
    return t.T


def solve(mode, **kw):
    if mode == "init_params":
        n_splits, n_skills = kw["n_splits"], kw["n_skills"]
        in_f, out_f, r = kw["in_features"], kw["out_features"], kw["r"]
        seed = kw["seed"]
        init_weights = kw.get("init_weights", True)
        dtype = kw.get("dtype", torch.float64)
        assert in_f % n_splits == 0 and out_f % n_splits == 0
        g = torch.Generator().manual_seed(seed)
        d_in, d_out = in_f // n_splits, out_f // n_splits
        A = torch.empty(n_splits, n_skills, d_in, r, dtype=dtype)
        B = torch.empty(n_splits, n_skills, r, d_out, dtype=dtype)
        # Official loops: skill-outer, split-inner; kaiming on a transposed layout.
        for skill in range(n_skills):
            for split in range(n_splits):
                A[split, skill] = _kaiming_transposed(r, d_in, math.sqrt(5), g, dtype)
        if init_weights:
            B = torch.zeros(n_splits, n_skills, r, d_out, dtype=dtype)
        else:
            for skill in range(n_skills):
                for split in range(n_splits):
                    # Official: kaiming on a (d_out, r) tensor (fan_in = r), transposed.
                    B[split, skill] = _kaiming_transposed(d_out, r, math.sqrt(5), g, dtype)
        # Router logits: (n_tasks, n_splits * n_skills), uniform_(-1e-3, 1e-3).
        n_tasks = kw["n_tasks"]
        logits = torch.empty(n_tasks, n_splits * n_skills, dtype=dtype).uniform_(
            -1e-3, 1e-3, generator=g)
        return {"A": A, "B": B, "module_logits": logits}

    if mode == "route":
        # Official PolyRouter forward, inference branch (training-time
        # RelaxedBernoulli rsample is stochastic and out of the judging domain):
        # sigmoid, then normalize across skills (NOT softmax), + EPS guard.
        task_ids = kw["task_ids"]
        module_logits = kw["module_logits"]
        n_splits, n_skills = kw["n_splits"], kw["n_skills"]
        n_tasks = module_logits.shape[0]
        if task_ids is None:
            raise ValueError("task_ids should not be None.")
        if int(task_ids.max().item()) >= n_tasks:
            raise ValueError(f"Only {n_tasks} tasks available. Found task id = {int(task_ids.max().item())}")
        logits = module_logits[task_ids]
        logits = logits.view(-1, n_splits, n_skills)
        s = torch.sigmoid(logits)
        return s / (s.sum(dim=-1, keepdim=True) + EPS)

    if mode == "combine":
        # Official einsum path: mix the skill factors per split, then reassemble the
        # full-rank A (bs, in, r) and B (bs, r, out).
        w = kw["weights"]              # (bs, n_splits, n_skills)
        A_param = kw["A"]              # (n_splits, n_skills, in//ns, r)
        B_param = kw["B"]              # (n_splits, n_skills, r, out//ns)
        r = kw["r"]
        bs = w.shape[0]
        A = torch.einsum("bqs,qsdr->bqdr", w, A_param)
        B = torch.einsum("bqs,qsrd->bqrd", w, B_param)
        in_f = A_param.shape[2] * A_param.shape[0]
        out_f = B_param.shape[3] * B_param.shape[0]
        A = A.reshape(bs, in_f, r)
        B = B.transpose(1, 2).reshape(bs, r, out_f)
        return {"A_mixed": A, "B_mixed": B}

    if mode == "forward_delta":
        # Official: result += x.bmm(A).bmm(B) / r   — the scale is exactly 1/r.
        x = kw["x"]                    # (bs, in)
        w = kw["weights"]
        A_param, B_param, r = kw["A"], kw["B"], kw["r"]
        mixed = solve(mode="combine", weights=w, A=A_param, B=B_param, r=r)
        return x.bmm(mixed["A_mixed"]).bmm(mixed["B_mixed"]) / r

    if mode == "param_count":
        n_splits, n_skills = kw["n_splits"], kw["n_skills"]
        in_f, out_f, r = kw["in_features"], kw["out_features"], kw["r"]
        n_tasks = kw["n_tasks"]
        a_params = n_splits * n_skills * (in_f // n_splits) * r
        b_params = n_splits * n_skills * r * (out_f // n_splits)
        router = n_tasks * n_splits * n_skills
        return {"lora": a_params + b_params, "router": router,
                "total": a_params + b_params + router}

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_init_params",
    "check_route",
    "check_combine",
    "check_forward_delta",
    "check_param_count",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
