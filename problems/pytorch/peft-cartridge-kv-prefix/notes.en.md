# Notes: Cartridge's Directly Parameterized KV-Prefix Adapter

## Source

Cartridges (arXiv:2506.06266, Stanford Hazy Research, blog 2025-06-08): train a compressed KV cache (cartridge) on a **synthetic self-study corpus**, injected as a prefix at inference so the model "carries the memory" without putting the whole corpus in context. PEFT PR #2953 (2026-01) landed the **adapter-side** ledger as `tuners/cartridge/` (config.py + model.py + utils.py, 455 lines total) — the base implementation is equivalent to prefix tuning, but it **stores the KV directly instead of reparameterizing via an MLP**, and adds a frozen first token (attention sink) plus KV-initialization utilities.

## Line-by-line translation of the official implementation

**Packing (`utils.py::prompt_embeddings_from_past_key_values`)**:

```python
legacy = _to_legacy_past_key_values(past_key_values)   # excluded from the judging domain; input is [(k,v),...]
if len(legacy) == 0: raise ValueError(...)
num_layers = len(legacy); key0, _ = legacy[0]
# rank 4 / batch 1 / seq_len >= nvt validation (all ValueError)
packed = torch.empty(num_virtual_tokens, num_layers*2, num_heads, head_dim, ...)
for layer_idx, (k, v) in enumerate(legacy):
    # per-layer head check k/v.shape[:2] == (1, num_heads), length k/v.shape[2] >= nvt,
    # head-dim check k/v.shape[3] == head_dim
    packed[:, 2*layer_idx]     = k[0, :, :nvt, :].transpose(0, 1).contiguous()
    packed[:, 2*layer_idx + 1] = v[0, :, :nvt, :].transpose(0, 1).contiguous()
return packed.reshape(num_virtual_tokens, -1)
```

Element ledger: `pe[t, (2l)*H*D + h*D + d] == k_l[0, h, t, d]` and `pe[t, (2l+1)*H*D + h*D + d] == v_l[0, h, t, d]`. **K/V layer interleaving + heads first + taking the FIRST NVT tokens** are the three spots easiest to get backwards.

**Encoder (`model.py::CartridgeEncoder`)**:

```python
num_frozen_tokens = int(config.num_frozen_tokens)
if num_frozen_tokens < 0 or num_frozen_tokens > num_virtual_tokens: raise ValueError(...)
self.num_trainable_tokens = num_virtual_tokens - num_frozen_tokens
if self.num_frozen_tokens:
    frozen = torch.empty(nft, hidden)
    self.frozen_embedding = torch.nn.Parameter(frozen, requires_grad=False)
trainable = torch.empty(ntr, hidden)
self.trainable_embedding = torch.nn.Parameter(trainable, requires_grad=not config.inference_mode)
self.reset_parameters()          # Embedding-style: frozen normal_ FIRST, trainable SECOND (std=1)

weight: frozen_embedding is None ? trainable_embedding : cat([frozen, trainable], dim=0)
load_prompt_embeddings: rows != (nft+ntr, hidden) raises; [0:nft] -> frozen, [nft:] -> trainable
forward(prefix_tokens): ignores ids; weight.unsqueeze(0).expand(batch, -1, -1)
```

The config layer (`CartridgeConfig.__post_init__`) adds another `num_frozen_tokens < 0` / `> num_virtual_tokens` check (`num_frozen_tokens` defaults to 1; the paper recommends freezing the first token as an attention sink) — merged into the encoder constructor in the judging domain.

**Composition (`utils.py::compose_cartridge_adapters`)**: `len < 2` raises; each config must have `peft_type == CARTRIDGE`; the five compat attributes (task_type / token_dim / num_layers / num_attention_heads / num_transformer_submodules) must match the first, and `num_frozen_tokens` comes **only from the first** (comment verbatim: "matches a single attention-sink prefix"); `torch.cat(prompt_embeddings, dim=0)`.

**Initializers (`initialize_kv_prefix_from_past_key_values` / `from_text`)**: usable for both CARTRIDGE and PREFIX_TUNING (`prefix_projection=True` rejected); `from_text` runs a real model prefill — **excluded under the real-forward red line**; `load_prompt_embeddings` is their landing spot (included).

## Exclusions and red lines in the judging domain

- `_to_legacy_past_key_values` (transformers Cache compatibility branches): no transformers in `.venv-eval`; the input is a fixed legacy list.
- `initialize_kv_prefix_from_text` (tokenizer + real prefill) and the `model.disable_adapter()` context: real-forward red line.
- safetensors/torch.save serialization (save_file/load_peft_weights) and disk-path bookkeeping: IO red line.
- `PeftConfig.from_pretrained` / config serialization: the judging domain passes plain dicts.
- GQA `token_dim` adjustment (`_prepare_prompt_learning_config`): a global PEFT mechanism; the judging domain takes token_dim explicitly.

## Boundary with existing problems

- `peft-prefix-tuning` (batch 765): that one tests the **attention forward** (prefix-first KV concatenation) plus the two-form PrefixEncoder parameter-count ledger (MLP reparameterization); this one tests the **KV cache → flat parameter packing layout** + frozen/trainable segmentation + composition, with no MLP and no attention forward.
- The kvcache collection (snapkv/h2o/quest etc.): all **eviction/quantization/merging** ledgers; this one is a **training-side parameter layout** ledger, no overlap.
- `kv-streaming-llm-sink`: its sink is an **eviction-protected position**; this problem's frozen first token is a **frozen parameter segment** — different mechanisms.

## Mutation testing (35/36, 1 equivalent)

- Packing: no transpose / K-V swap / column-slot overwrite / transpose(0,2) / batch flip / rank flip / relaxed head check / wrong head-dim index / relaxed length check / empty check / wrong head source — all CAUGHT.
- Encoder: trainable frozen / RNG order / std / cat dim / dropped frozen / reversed concat / wrong load segment / wrong frozen segment / ndim check / rows check / expand shape / unsqueeze axis / trainable count / hidden formula / always-true frozen branch / frozen upper bound / negative check — all CAUGHT.
- Composition: len gate / peft_type / cat dim / reversed concat / frozen from the tail / flipped attribute check / truncated attribute list — all CAUGHT.
- **m-pk-seq excluded as truly equivalent**: the leading `seq_len < nvt` check (first layer) is fully redundant with the per-layer `k/v.shape[2] < nvt` check inside the loop for the judging domain — any input triggering the former is rejected with a ValueError by the in-loop check anyway (tests only assert the ValueError type), so the strictness of the leading check is unobservable.

## RNG anchoring

`torch.nn.init.normal_` and `torch.randn` are **not the same RNG stream** (same lesson as the batch-804 uniform_ note); after `torch.manual_seed(42)`, following the official order `empty(nft, h).normal_()` then `empty(ntr, h).normal_()` reproduces both segments bit-exactly. float32 and float64 streams also differ; anchors are pinned to float32.
