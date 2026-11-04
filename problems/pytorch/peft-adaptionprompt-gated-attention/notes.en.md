# Solution notes — AdaptionPrompt

## Official account

- HF PEFT `peft/tuners/adaption_prompt/layer.py` (`AdaptedAttention`, Llama branch);
  paper LLaMA-Adapter v2 (arXiv 2303.16199), official repo ZrrSkywalker/LLA-Adapter.
- The judging domain covers the pure-function parts: prompt/gate init, GQA K/V
  replication, the gated attention branch, residual combine, and the parameter budget.
  transformers glue (rotary re-computation of q, masks) is out of scope — q is given.

## Pitfalls

- **Transpose order**: official is `view(1, L, kv, hd)` → `transpose(1, 2)` →
  `repeat_interleave(factor, dim=1)`. Skipping the transpose repeats along the L dim
  instead and the shape is wrong outright ((1, 6, 2, 2)).
- **repeat vs repeat_interleave**: dim=1 is the KV-head dim; `repeat_interleave`
  yields `[g0, g0, g1, g1]` (adjacent groups) while `repeat` interleaves to
  `[g0, g1, g0, g1]`. The group identity asserts adjacent heads are equal
  (ak[0, 0] == ak[0, 1]), not h vs h+factor.
- **Gate multiplies after the softmax**: `softmax(g·s)` is pre-gating and numerically
  different from the official `g·softmax(s)` — a good mutation target.
- **fp32 softmax upcast**: `dtype=torch.float32` applies even under a float64 global
  default — the fp64 softmax differs from the fp32-domain softmax cast back; tests
  must anchor the upcast behavior.
- **Tests must `set_default_dtype(float64)` at the top**: otherwise the default
  float32 stream produces wrong anchors (root cause of the first all-red run).
- **kv == hd makes the view transpose a true equivalence**: the shape family must
  include kv != hd (kv=1, hd=4), otherwise the (kv,hd)→(hd,kv) view mutant survives.
- **Do not skip value assertions on av**: with only ak compared, both the v_w→k_w and
  the av view-transpose mutants survive.

## Mutant list (17)

init: normal std / gate ones; kv: missing transpose / interleaved repeat / v_w→k_w /
view transpose; attn: scale /√hd→/hd / fp32 upcast / pre-gate / o_proj coefficient /
softmax dim / missing transpose(1,2); combine: add→subtract / dropped branch;
param_count: full_prefix coefficient / saving sign / trainable constant.
17/17 CAUGHT.
