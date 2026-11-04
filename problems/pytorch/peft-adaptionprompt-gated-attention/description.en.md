# AdaptionPrompt gated attention branch

LLaMA-Adapter v2 (arXiv 2303.16199) ships in HF PEFT as the `adaption_prompt` tuner
(`peft/tuners/adaption_prompt/layer.py`, `AdaptedAttention`). It injects trainable
"adaption prompts" into the attention branch: `adapter_len` learnable tokens feed the
key/value side of attention, and a **zero-initialized gate** adds their attention output
back to the original attention output residually — so at the start of training the
wrapped model behaves identically to the original.

## Mechanism

Let `attn_output` (bsz, q_len, hidden) be the wrapped module's output. The parameters are:

- `adaption_prompt`: shape (1, adapter_len, hidden), initialized with `normal_()` —
  the batch dim is 1 (shared across the batch);
- `adaption_gate`: shape (1,), zero-initialized.

### 1. Adapter K/V projection

The prompt goes through the wrapped module's k_proj / v_proj (two separate projections):

```
key   = prompt @ k_w.T   # (1, L, kv_heads*head_dim)
value = prompt @ v_w.T
```

Then the GQA replication: view to `(1, L, kv_heads, head_dim)`, **transpose(1, 2) first**
to `(1, kv_heads, L, head_dim)`, then `repeat_interleave(factor, dim=1)` along the
KV-head dim to `(1, H, L, head_dim)` so query head h reads KV group `h // factor`.
Note `repeat_interleave` yields the adjacent-group layout `[g0, g0, g1, g1]`;
`torch.repeat` would interleave to `[g0, g1, g0, g1]` — a different broadcast.

### 2. Gated attention branch

For the (already rotary-applied) queries `q` (bsz, H, q_len, head_dim):

```
scores = q @ adapter_k^T / sqrt(head_dim)          # (bsz, H, q_len, L)
probs  = softmax(scores, dim=-1, dtype=float32)     # explicit fp32 upcast
scores = gate * probs                               # gate multiplies AFTER softmax
adapter_output = (scores @ adapter_v).transpose(1,2).reshape(bsz, q_len, -1)
adapter_output = adapter_output @ o_w.T             # o_proj
```

Three judge-sensitive points:

- **The gate multiplies after the softmax**: `g·softmax(s)` differs from
  `softmax(g·s)` (the official code does the former);
- **Explicit fp32 softmax upcast**: even for float64 inputs the softmax runs in fp32
  and is cast back — numerically distinct from a pure fp64 softmax;
- **Scaling before the softmax**: `/sqrt(head_dim)` applies to scores, not probs.

### 3. Residual combine

```
output = attn_output + adapter_output
```

The zero-init gate makes `adapter_output` exactly zero at step 0, so the combine is an
identity at training start — the core "zero-init attention" trick of LLaMA-Adapter v2.

### 4. Parameter budget

Trainable = `adapter_len × hidden + 1` (prompt + scalar gate). Compared to a full
per-layer prefix that stores K and V separately (`2 × adapter_len × hidden`),
AdaptionPrompt stores a single shared prompt — half the size.

## Judging domain

Single entry `solve(mode, **kw)`, five modes:

| mode | inputs | outputs |
|---|---|---|
| `init_params` | adapter_len, hidden, seed, dtype | `{"prompt": (1,L,H), "gate": (1,)}` (explicit generator RNG domain) |
| `adapter_kv` | prompt (1,L,H), k_w, v_w, num_heads, head_dim, factor | `{"adapter_k", "adapter_v"}` (1,H,L,hd) |
| `adapter_attn` | q (bsz,H,q_len,hd), adapter_k/v, gate, head_dim, o_w | `{"probs", "adapter_output"}` |
| `combine` | attn_output, adapter_output | elementwise sum |
| `param_count` | adapter_len, hidden | `{"trainable", "full_prefix", "saving"}` |

Numeric anchors live in the float64 domain; the fp32 softmax upcast is official
behavior and must be preserved.
