# Gemma2 Decoder Layer: Sliding/Global Masks and Double Post-Norm

## Problem

The Gemma2 decoder layer stacks four of its own changes on top of a standard pre-norm Transformer. Individually each is small; together they decide the layer's numerics:

1. **Alternating masks**: `layer_types` makes adjacent layers alternate between `sliding_attention` (only the most recent `sliding_window` tokens, current one included) and `full_attention` (global causal). This problem implements **one** layer, selected by `layer_type`.
2. **Attention-level tanh softcap**: `logits = softcap · tanh(logits / softcap)` squeezes the logits into \([-softcap, softcap]\). It must happen **before** the mask is added — the mask writes \(-\infty\), so masking first and then applying tanh turns every masked entry into the **finite** value \(-\text{softcap}\), and softmax then hands real weight to masked keys.
3. **Double post-norm**: `input_layernorm` and `pre_feedforward_layernorm` are the usual pre-norms; in addition `post_attention_layernorm` acts on the **attention output** and `post_feedforward_layernorm` on the **MLP output**, both before being added back to the residual.
4. **\((1+w)\) RMSNorm**: the gain is initialized to zero and the effective scale is \(1+w\), so "zero init" means identity.

Two more easy-to-miss constants: the scaling uses `query_pre_attn_scalar**-0.5` rather than `head_dim**-0.5` (they often coincide in Gemma2, but the semantics differ), and the MLP is a **GeGLU** (`gelu_pytorch_tanh`), not LLaMA's SwiGLU.

## Rules

Inputs are `hidden_states` \((B,S,d)\), precomputed RoPE `cos`/`sin` \((B,S,d_h)\), the gains of four norms, and seven bias-free weight matrices for q/k/v/o and gate/up/down (linear layers use the \([\text{out},\text{in}]\) layout, `F.linear(x,W)=x@Wᵀ`). Let \(H_q\) be the query head count, \(H_{kv}\) the kv head count, and \(r=H_q/H_{kv}\).

**RMSNorm** (same form in all four places):

\[ \mathrm{RMSNorm}(x,w)=\frac{x}{\sqrt{\frac1d\sum_j x_j^2+\varepsilon}}\odot(1+w) \]

**Attention**:

1. \(h=\mathrm{RMSNorm}(x,w_{\text{in}})\); \(Q=hW_q^\top,\ K=hW_k^\top,\ V=hW_v^\top\), reshaped to \((B,H,S,d_h)\) for \((H_q,d_h)\) / \((H_{kv},d_h)\).
2. Apply RoPE to \(Q,K\): \(\;Q'=Q\odot\cos+\mathrm{rot}(Q)\odot\sin\), where \(\mathrm{rot}\) swaps the two halves and negates one.
3. GQA expansion: query head \(h\) uses kv head \(\lfloor h/r\rfloor\).
4. \(L=Q'K'^\top\cdot \text{query\_pre\_attn\_scalar}^{-1/2}\).
5. **softcap** (when `softcap` is not `None`): \(L\leftarrow\text{softcap}\cdot\tanh(L/\text{softcap})\).
6. **Mask**: an entry is allowed when \(k\le q\); under `layer_type == "sliding_attention"` it additionally needs \(k>q-\text{sliding\_window}\). All others become \(-\infty\), then a float64 softmax along the last axis yields `attn_weights`.
7. \(A=\mathrm{attn\_weights}\cdot V\), reshaped back to \((B,S,H_q d_h)\) and passed through the \(o\) projection.

**Residual and MLP**:

\[ x\leftarrow x+\mathrm{RMSNorm}(A\,W_o^\top,\ w_{\text{post-attn}}) \]

\[ x\leftarrow x+\mathrm{RMSNorm}\big(\mathrm{down}\big(\mathrm{gelu}_{\tanh}(\mathrm{gate}(h_2))\odot \mathrm{up}(h_2)\big),\ w_{\text{post-ff}}\big),\qquad h_2=\mathrm{RMSNorm}(x,w_{\text{pre-ff}}) \]

Return `(output, attn_weights)`: `output` of shape \((B,S,d)\) and `attn_weights` of shape \((B,H_q,S,S)\), both float32.

## Example

Take \(B=1\), \(S=4\), \(d=4\), \(H_q=H_{kv}=2\), \(d_h=2\), MLP width 4; let `hidden_states` be the identity (row \(i\) is \(e_i\)), `cos=1, sin=0` (rotation degenerates to identity), all four gains zero (scale 1), all seven projections identity, `sliding_window=2`, `query_pre_attn_scalar=2`, `softcap=10`, `eps=1e-6`:

```
output =
[ 4.999999, 0.0,      0.0,      0.0      ]
[ 0.128880, 4.995966, 0.0,      0.0      ]
[ 0.0,      1.114066, 4.758161, 0.0      ]
[ 0.0,      0.0,      0.128880, 4.995966 ]

attn_weights[0, 0] =
[ 1.0,      0.0,      0.0, 0.0 ]
[ 0.059786, 0.940214, 0.0, 0.0 ]
[ 0.0,      0.5,      0.5, 0.0 ]
[ 0.0,      0.0,      0.5, 0.5 ]
```

How to read it: `hidden = e_i` becomes \(2e_i\) after RMSNorm. With `sliding_window=2`, row \(i\) only sees \(k\in\{i-1,i\}\), so each row of the weight table has exactly two non-zeros. Rows 1 and 3 have a diagonal logit of \(4/\sqrt2=2.8284\), squeezed by the softcap to \(10\tanh(0.28284)=2.7554\), which softmax maps to \(0.940214\) (without the softcap it would be \(0.9442\)); the partner sits at logit 0 and gets \(0.059786\). In row 2, head 0 sees both visible tokens as zero on dimensions \(0,1\), so all logits are 0 and the weight splits evenly \(0.5/0.5\).

The diagonal of the output is about 5 because: residual \(e_i\) + post-normalized attention \(2e_i\) = \(3e_i\); one more "RMSNorm → GeGLU → RMSNorm" round contributes exactly another \(2e_i\) (the fixed point of GeGLU at 2). Off-diagonal entries come only from the normalized attention output being spread over dimensions.

## Constraints

- `hidden_states` is \((B,S,d)\); `cos`/`sin` share the shape \((B,S,d_h)\) with \(d_h\) a positive even number (RoPE needs paired dimensions).
- All four gains are \((d,)\); `q_weight` is \((H_q d_h,d)\), `k_weight`/`v_weight` are \((H_{kv}d_h,d)\); `o_weight` is \((d,H_qd_h)\); `gate`/`up` are both \((I,d)\); `down` is \((d,I)\). Beyond matching shapes, \(H_{kv}\mid H_q\) is required.
- `layer_type` accepts only `"sliding_attention"` / `"full_attention"`; `sliding_window` is an integer \(\ge1\) (`bool` does not count).
- `query_pre_attn_scalar` and `rms_eps` must be finite positive numbers; `softcap` is either `None` or a finite positive number (0 and negatives raise `ValueError`).
- The softcap must be applied before the mask; both post-norms must act on the sublayer outputs; RMSNorm must use \(1+w\).
- Compute internally in float64 and return float32.
