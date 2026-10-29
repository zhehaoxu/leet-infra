# Deep Dive: Assembling the Gemma2 Decoder Layer

## Where it comes from

`transformers==5.20.2`, `models/gemma2/modeling_gemma2.py`:

- `Gemma2RMSNorm.forward`: `output * (1.0 + self.weight.float())`, with the weight initialized to zero;
- `Gemma2Attention.__init__`: `scaling = query_pre_attn_scalar**-0.5`, and `sliding_window = config.sliding_window if layer_type == "sliding_attention" else None`;
- `Gemma2Attention.forward`: `self.attn_logit_softcapping` is forwarded to the attention interface;
- `eager_attention_forward`: `matmul * scaling`, then `tanh(w/softcap)*softcap`, and **only then** `attn_weights + attention_mask`;
- `Gemma2DecoderLayer.forward`: the two post-norms, `residual + post_attention_layernorm(attn)` and `residual + post_feedforward_layernorm(mlp)`;
- `Gemma2MLP`: `down_proj(act_fn(gate_proj(x)) * up_proj(x))` with `hidden_activation="gelu_pytorch_tanh"` (GeGLU);
- `masking_utils.sliding_window_overlay`: `kv_idx > q_idx - sliding_window`, intersected with the causal condition `kv_idx <= q_idx`.

This is a whole-layer assembly problem: the sliding-window mask and the softcap each have their own single-point problem in the operators collection (`sliding-window-attention` / `attention-softcap`); what is tested here is the **data flow** once they are combined with double post-norm and \((1+w)\) normalization.

## Core points and pitfalls

1. **The softcap runs before the mask.** This is a numerical correctness issue, not a style choice. The mask writes \(-\infty\); masking first and then computing \(\text{softcap}\cdot\tanh(L/\text{softcap})\) leaves every masked entry at \(\text{softcap}\cdot\tanh(-\infty)=-\text{softcap}\) — a **finite** value. The smaller the softcap, the closer that fake logit sits to 0 and the more weight softmax gives the masked key. `check_softcap_before_mask` amplifies this with `softcap=0.5` and also runs a "mask first" oracle variant, asserting the two results **must differ**.
2. **Both post-norms act on the sublayer outputs, not on the residual stream.** Writing `x = RMSNorm(residual) + attn` keeps every shape identical and never errors, but it silently turns the layer into "normalize the residual", and Gemma2's pre+post hybrid disappears. The oracle's `norm_target="residual"` variant exists precisely to prove that such a version is judged wrong.
3. **\((1+w)\), not \(w\).** Gemma2 initializes the gain to 0 so that "zero init = identity". Implementing \(x\cdot w\) would zero the whole layer at init. `check_norm_uses_one_plus_weight` uses gain \([1,0]\) on `post_attention`: the correct implementation gives 5.242639, while the \(w\) version collapses to the gain-0 result 3.828426.
4. **The scaling is `query_pre_attn_scalar**-0.5`, not `head_dim**-0.5`.** In Gemma2-2B both equal 256, so this bug is invisible on real weights; the test config uses `query_pre_attn_scalar=8` with `head_dim=2`, so any head_dim-based scaling shows up immediately.
5. **The MLP is a GeGLU, not a SwiGLU**: `gelu(gate, approximate="tanh")` versus `silu(gate)`. Both approach \(x\) for \(x>3\), but they differ by more than 10% around 1–2; the wrong activation shifts every example row.
6. **The sliding window includes the current token**: the condition is \(k>q-w\), so the width is \(w\) and \(k=q\) is always visible. Writing \(k\ge q-w\) widens it to \(w+1\) tokens by admitting token \(q-w\).
7. **GQA expansion is contiguous grouping**: `repeat_kv` uses `expand(...).reshape(...)`, so query head \(h\) maps to kv head \(\lfloor h/r\rfloor\). Using \(h\bmod r\), or `torch.cat` tiling (head order `[k0,k1,k0,k1]`), changes the result whenever \(H_{kv}<H_q\).
8. **What the softcap actually does is compress the logit gaps**: softmax is invariant to a constant shift along a row, so the softcap changes the extremes only. In the example the diagonal logit 2.8284 is squeezed to 2.7554 and the weight drops from 0.9442 to 0.9402; once the logits are far above the softcap the row tends toward uniform.
9. **Positional encoding enters only as precomputed cos/sin**: RoPE is broadcast over heads (`unsqueeze(1)`) on \((B,H,S,d_h)\) and is decoupled from the rest of the layer, so this problem does not duplicate `llm-rope-apply`.
10. **Precision**: the reference solution computes the whole layer in float64 and casts to float32 at the end; HF computes in float32, hence the `2e-5` threshold against HF and `1e-5` against the numpy oracle (also float64).

## How it is judged

Judging happens on three layers:

- **Example values**: the hand-computable identity + all-zero-gain configuration, with all four rows of `output` and all four rows of `attn_weights[0,0]` hard-coded to six decimals;
- **Structural assertions**: (a) the sliding support set under `window=3`, `window=1` and `full_attention` — visible weights strictly positive, masked ones exactly zero, each `(batch, head)` row summing to 1; (b) the softcap must precede the mask (compared against a `mask_first` oracle variant and required to differ); (c) \((1+w)\) normalization (two hand-computed values that must differ from each other); (d) the post-norms must act on the sublayer outputs (compared against the `norm_target="residual"` variant); (e) the scaling must use `query_pre_attn_scalar` (compared against a head_dim-scaled oracle); (f) the GQA head mapping (with \(H_q=4,H_{kv}=2\), both output and weights must match);
- **Two oracles**: (a) an independent numpy `_oracle` where RMSNorm, masking, softmax and GeGLU are written out explicitly and projections use `@`, covering 5 shape × layer-type × softcap configurations plus two eps values, comparing both the output and `attn_weights` element-wise; (b) `transformers`' `Gemma2DecoderLayer` (eager, same weights, the same 4-D additive mask and `position_embeddings`) end-to-end over 4 configurations at `2e-5`.

Another 20 invalid inputs must raise `ValueError`: non-3-D `hidden_states`/`cos`, mismatched `cos`/`sin`, wrong gain length, q/k/o/down row or column counts not divisible by `head_dim`, unknown `layer_type`, `sliding_window` of 0 or a float, non-positive or NaN `query_pre_attn_scalar`, non-positive `softcap`, non-positive or infinite `rms_eps`, and an odd `head_dim`.

## Complexity

Attention is \(O(B\cdot H_q\cdot S^2\cdot d_h)\) (GQA stores only \(H_{kv}\) copies on the kv side); projections plus MLP are \(O\big(BS(d\cdot H_qd_h + d\cdot H_{kv}d_h + 2dI)\big)\). Besides the output and the weights, only the current hidden state is kept: \(O(BSd + B H_q S^2)\).
