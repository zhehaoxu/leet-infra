"""LLaMA-Adapter v2 / AdaptionPrompt (HuggingFace PEFT, adaption_prompt tuner).

Reproduces the trainable adaption-prompt attention branch exactly as implemented in
peft/src/peft/tuners/adaption_prompt/layer.py (AdaptedAttention for Llama-style GQA):

- adaption_prompt: nn.Parameter (1, adapter_len, hidden) initialized with normal_() —
  shared across the batch (view(1, L, kv, hd) then repeat to bsz).
- adaption_gate: nn.Parameter (1,) zero-initialized ("zero-init" keeps the wrapped
  model's outputs unchanged at the start of training).
- adapter_k/adapter_v: prompt tokens projected by the wrapped module's k_proj/v_proj,
  split into (kv_heads, head_dim), then repeat_interleave(factor, dim=1) to broadcast
  the key-value heads across the full query-head count (grouped-query attention).
- attention branch: scores = q @ adapter_k^T / sqrt(head_dim), softmax over the
  adapter-length dim upcast to float32, multiplied by the (scalar) gate, then
  scores @ adapter_v -> transpose(1,2).reshape(bsz, q_len, -1) -> o_proj.
- final output: original attention output + adapter branch output (residual add).

The judging domain keeps everything deterministic: q/k/v weights, prompts and gates are
function inputs; RNG only enters through init_params (explicit generator domain).
"""

import math

import torch
import torch.nn.functional as F


def solve(mode, **kw):
    if mode == "init_params":
        adapter_len, hidden, seed = kw["adapter_len"], kw["hidden"], kw["seed"]
        dtype = kw.get("dtype", torch.float64)
        g = torch.Generator().manual_seed(seed)
        # Official: torch.empty(1, L, H).normal_() — standard normal, shared batch dim.
        prompt = torch.empty(1, adapter_len, hidden, dtype=dtype).normal_(generator=g)
        gate = torch.zeros(1, dtype=dtype)
        return {"prompt": prompt, "gate": gate}

    if mode == "adapter_kv":
        # prompt: (1, L, H); k_w: (kv_heads*head_dim, H); v_w likewise.
        prompt, k_w, v_w = kw["prompt"], kw["k_w"], kw["v_w"]
        num_heads, head_dim = kw["num_heads"], kw["head_dim"]
        hidden = prompt.shape[-1]
        kv_heads = num_heads // kw["factor"]
        assert k_w.shape == (kv_heads * head_dim, hidden), k_w.shape
        assert v_w.shape == (kv_heads * head_dim, hidden), v_w.shape
        # k_proj_layer == v_proj_layer is not assumed: two separate projections.
        key = F.linear(prompt, k_w)    # (1, L, kv*hd)
        value = F.linear(prompt, v_w)  # (1, L, kv*hd)
        # Official order: view(1, L, kv, hd) -> transpose(1, 2) to (1, kv, L, hd),
        # then repeat_interleave(factor, dim=1) along the KV-head dim.
        adapter_k = key.view(1, prompt.shape[1], kv_heads, head_dim).transpose(1, 2)
        adapter_v = value.view(1, prompt.shape[1], kv_heads, head_dim).transpose(1, 2)
        # Official: repeat_interleave(repeats=factor, dim=1) — grouped replication so
        # that query head h reads KV group h // factor. (torch.repeat would interleave
        # the groups instead and is a different, wrong broadcasting.)
        adapter_k = torch.repeat_interleave(adapter_k, repeats=kw["factor"], dim=1)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=kw["factor"], dim=1)
        return {"adapter_k": adapter_k, "adapter_v": adapter_v}

    if mode == "adapter_attn":
        # q: (bsz, H, q_len, hd) already rotary-applied; adapter_k/v: (1, H, L, hd).
        q, adapter_k, adapter_v = kw["q"], kw["adapter_k"], kw["adapter_v"]
        gate, head_dim = kw["gate"], kw["head_dim"]
        o_w = kw["o_w"]  # (hidden, hidden)
        bsz, _, q_len, _ = q.shape
        # Official: scores = q @ k^T / sqrt(head_dim); softmax upcast to fp32;
        # gate multiplies AFTER the softmax (scalar broadcast).
        scores = torch.matmul(q, adapter_k.transpose(2, 3).to(q.dtype)) / math.sqrt(head_dim)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        scores = gate * probs
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        # Official: adapter_output = o_proj(adapter_output) when o_proj is present.
        adapter_output = F.linear(adapter_output, o_w)
        return {"probs": probs, "adapter_output": adapter_output}

    if mode == "combine":
        # Official: output = original attention output + adapter branch output.
        attn_output, adapter_output = kw["attn_output"], kw["adapter_output"]
        return attn_output + adapter_output

    if mode == "param_count":
        # Trainable budget: adaption_prompt (L x H) + gate (1). For comparison the
        # "full prefix" alternative stores per-layer L x H keys AND values:
        # 2 x L x H. AdaptionPrompt stores a single shared L x H prompt.
        adapter_len, hidden = kw["adapter_len"], kw["hidden"]
        trainable = adapter_len * hidden + 1
        full_prefix = 2 * adapter_len * hidden
        return {"trainable": trainable, "full_prefix": full_prefix,
                "saving": full_prefix - trainable}

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_init_params",
    "check_adapter_kv",
    "check_adapter_attn",
    "check_combine",
    "check_param_count",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
