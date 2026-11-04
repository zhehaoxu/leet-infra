"""GraLoRA (official SqueezeBits/GraLoRA, now in HF PEFT tuners/gralora) --
granular sub-block LoRA ledger.

Dispatch by the ``mode`` keyword:

- mode="split_dims": the update_layer size/validation account
  (kwargs: in_features, out_features, r, k, hybrid_r, alpha) ->
  {"gralora_rank", "sub_rank", "sub_in", "sub_out", "scaling"};
  r <= 0 raises ValueError; (r - hybrid_r) % k != 0 raises ValueError;
  scaling = alpha / r uses the TOTAL rank r (hybrid included), the
  same convention as plain LoRA
- mode="param_layout": the parameter layout account
  (kwargs: in_features, out_features, r, k, hybrid_r) -> shapes for
  gralora_A (k, in//k, r-hybrid_r) and gralora_B (k, r-hybrid_r, out//k),
  plus the hybrid bypass shapes when hybrid_r > 0; zero_init lists the
  all-zero-initialized parameters ("gralora_B", and
  "gralora_B_general" when hybrid) -- A/A_general are kaiming (nonzero)
- mode="forward_delta": the official double-einsum chain
  (kwargs: x (B,L,in), gralora_A (k,in//k,rank), gralora_B
  (k,rank,out//k), scaling, k) -> scaling * delta, shape (B,L,k*(out//k)):
  input segment n goes through A_n up to gralora_rank dims, the rank
  axis is split into k segments, permuted (segment axis <-> input-block
  axis) and compound-reshaped, then contracted with B row segment
- mode="block_delta": the equivalent explicit block formula
  (kwargs: gralora_A, gralora_B, k) -> dense dW (out, in) where
  dW[j*so:(j+1)*so, n*si:(n+1)*si] =
      B[j][n*sub:(n+1)*sub, :].T @ A[n][:, j*sub:(j+1)*sub]
  each block has rank = sub = (r - hybrid_r) // k (granular), while the
  parameter count equals a same-rank plain LoRA: rank * (in + out)
- mode="full_forward": the gralora branch plus the hybrid full-rank
  LoRA bypass (kwargs: x, gralora_A, gralora_B, scaling, k, hybrid_r,
  gralora_A_general, gralora_B_general) -> forward_delta + scaling *
  (x @ A_general.T @ B_general.T); hybrid_r > 0 requires both general
  tensors, hybrid_r == 0 rejects them

Dropout is fixed at 0 (the Identity branch) and every model call is an
explicit input; only the deterministic layer accounts are replicated.
"""

import torch


def split_dims(in_features, out_features, r, k, hybrid_r, alpha):
    raise NotImplementedError


def param_layout(in_features, out_features, r, k, hybrid_r):
    raise NotImplementedError


def forward_delta(x, gralora_A, gralora_B, scaling, k):
    raise NotImplementedError


def block_delta(gralora_A, gralora_B, k):
    raise NotImplementedError


def full_forward(x, gralora_A, gralora_B, scaling, k, hybrid_r,
                 gralora_A_general=None, gralora_B_general=None):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "split_dims":
        return split_dims(kw["in_features"], kw["out_features"], kw["r"],
                          kw["k"], kw["hybrid_r"], kw["alpha"])
    if mode == "param_layout":
        return param_layout(kw["in_features"], kw["out_features"], kw["r"],
                            kw["k"], kw["hybrid_r"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["gralora_A"], kw["gralora_B"],
                             kw["scaling"], kw["k"])
    if mode == "block_delta":
        return block_delta(kw["gralora_A"], kw["gralora_B"], kw["k"])
    if mode == "full_forward":
        return full_forward(kw["x"], kw["gralora_A"], kw["gralora_B"],
                            kw["scaling"], kw["k"], kw["hybrid_r"],
                            gralora_A_general=kw.get("gralora_A_general"),
                            gralora_B_general=kw.get("gralora_B_general"))
    raise ValueError("unknown mode: %s" % mode)
