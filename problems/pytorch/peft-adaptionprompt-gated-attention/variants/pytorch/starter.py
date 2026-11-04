"""LLaMA-Adapter v2 / AdaptionPrompt — starter template.

Implement `solve(mode, **kw)` reproducing the trainable adaption-prompt attention
branch from HuggingFace PEFT's adaption_prompt tuner (peft/tuners/adaption_prompt/layer.py):

- init_params: prompt (1, L, H) standard normal via normal_(generator=), gate (1,) zeros.
- adapter_kv: project the prompt with k_w/v_w, view (1, L, kv, hd), transpose(1, 2),
  then repeat_interleave(factor, dim=1) along the KV-head dim -> (1, H, L, hd).
- adapter_attn: scores = q @ k^T / sqrt(head_dim), softmax over the adapter dim
  upcast to float32, multiply the (scalar) gate AFTER the softmax, @ v,
  transpose(1, 2).reshape(bsz, q_len, -1), then o_proj.
- combine: original attention output + adapter branch output (residual add).
- param_count: trainable = L*H + 1 vs full per-layer prefix 2*L*H.

See description.md for the full contract.
"""

MODES = ["init_params", "adapter_kv", "adapter_attn", "combine", "param_count"]


def solve(mode, **kw):
    raise NotImplementedError
