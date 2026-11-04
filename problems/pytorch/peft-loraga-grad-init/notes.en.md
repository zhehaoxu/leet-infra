# Notes: LoRA-GA's Gradient-Approximation Initialization

## Source

- Paper: **LoRA-GA: Low-Rank Adaptation with Gradient Approximation** (arXiv:2407.05000, Wang, Yu, Li).
- Official implementation: `Outsider565/LoRA-GA` (a custom peft fork; the core lives in the
  update_layer branch of `tuners/lora/layer.py` plus `utils/lora_ga_utils/`).
- Key difference (vs the PiSSA/LoRA-DORA family): after initialization LoRA-GA **absorbs**
  \(W \leftarrow W-\eta\,BA\) into the base weight, so training starts exactly at the pretrained
  model; PiSSA keeps \(W\) untouched and moves the principal components into the adapter.
  Combined with first-step gradient alignment, the update direction equals full fine-tuning.

## Line-by-line translation of the official implementation

**grad_svd** (layer.py, verbatim):

```python
grad = self.kwargs["grad"].to(device).to(torch.float32)
U, S, V = torch.svd_lowrank(grad.float(), q=min(4 * lora_r, min(grad.shape)), niter=4)
V = V.T
```

```python
# Neither svd_lowrank call in the official layer sets a manual_seed (both rely on the global
# RNG state); the judging domain makes the seed explicit (unlike 808 LoRA-SB, where the official code calls manual_seed(42)).
```

**direction (4 modes)** (layer.py, verbatim):

```python
if init_config.direction == "ArBr":
    B = U[:, 0: 2 * lora_r: 2]      # even columns
    A = V[1: 2 * lora_r: 2, :]      # odd rows
elif init_config.direction == "A2rBr":
    B = U[:, :lora_r];  A = V[lora_r: 2 * lora_r, :]
elif init_config.direction == "ArB2r":
    B = U[:, lora_r: 2 * lora_r];  A = V[:lora_r, :]
elif init_config.direction == "random":
    random_list = random.sample(range(2 * lora_r), 2 * lora_r)   # Python random!
    indexes_A = random_list[0:lora_r]      # A rows drawn first
    indexes_B = random_list[lora_r:2 * lora_r]
    B = U[:, indexes_B];  A = V[indexes_A, :]
```

**scale (4 modes)** (layer.py, verbatim):

```python
scaling_factor = self.scaling["default"]
if init_config.scale == "gd":
    A = A / scaling_factor;  B = B / scaling_factor
elif init_config.scale == "unit":
    pass                                    # A,B orthogonal, no rescale
elif init_config.scale == "stable":
    m, n = grad.shape                        # m: feature_out
    gamma = init_config.stable_gamma
    B = B * m ** 0.25 / gamma ** 0.5
    A = A * m ** 0.25 / gamma ** 0.5
elif init_config.scale == "weightS":
    _, S, _ = torch.svd_lowrank(weight.data.float(), q=4 * lora_r, niter=4)
    S = S / self.scaling["default"]
    avg_s = torch.sqrt(S[:lora_r]).mean()
    B = B * avg_s;  A = A * avg_s
```

**offset absorption** (layer.py, verbatim):

```python
offset = B @ A
...
offset *= scaling_factor
if hasattr(init_config, "norm_clip") and init_config.norm_clip:
    ratio = torch.max(torch.abs(weight.data)) / torch.max(torch.abs(offset))
    if ratio < 1:
        offset *= ratio
        A *= ratio ** 0.5
        B *= ratio ** 0.5
weight.data -= offset
self.lora_A[adapter_name].weight.data = A.contiguous()
self.lora_B[adapter_name].weight.data = B.contiguous()
```

```python
# norm_clip invariant: (B*sqrt(ratio)) @ (A*sqrt(ratio)) * eta == offset * ratio
# — scaling A and B each by sqrt(ratio) inflates BA by exactly ratio, in sync with offset.
# At ratio == 1 both branches multiply by 1 (m-nc-lt removed as a true equivalent).
```

## Exclusions and red lines in the judging domain

- **Excluded**: quantized-weight forward (get_float_weight/bitsandbytes), the bf16/fp32 dtype
  branches, OffloadContext/Accelerator/distributed, and the weight svd_lowrank of weightS
  (replaced by an explicit input).
- **Kept**: the full grad_svd flow, all four direction modes (including the Python random
  account), all four scale modes, and the offset absorption with the norm_clip invariant.
- **Anchoring**: S/U/Vt fixtures are anchored bit-for-bit; an niter=0 vs 4 difference case is
  included.

## Key quirks (verified line-by-line against the source)

- **No seed in the official layer**: both svd_lowrank calls rely on the global RNG; the
  judging domain makes the seed explicit (the official manual_seed(42) belongs to LoRA-SB's
  svd_utils).
- **ArBr is interleaved, not halves**: `0:2r:2` truncates at 2r — it is NOT `0::2` to the end
  (the slice difference was caught in the sandbox).
- **random uses Python random** (not the torch RNG) and indexes_A is drawn first — mutation
  m-di-ab swapping the order is caught.
- **stable's m is out_features** (B.size(0), official comment "m: feature_out"); not r and not
  in_features.
- **weightS divides by eta BEFORE the sqrt**: `S /= eta` then `sqrt(S[:r]).mean()`.
- **gd is not an equivariant rescale**: with both A and B divided by eta, eta*BA = UVᵀ/eta —
  the official semantics are simply a linear division, no deeper interpretation needed.
- **norm_clip's ratio == 1 is unobservable**: both branches multiply by 1 (true equivalent).

## Mutation testing (23/23, 1 equivalent exclusion)

- m-nc-lt (ratio<1 -> ratio<=1): at ratio == 1 both branches multiply by 1; removed as a true
  equivalent.
- 23 mutations cover: q/niter/seed/V.T/float32/rank validation, the four direction slices /
  random order / enum, the four scale factors / m source / S division / sqrt / enum, and the
  offset eta / BA form / add-subtract sign / norm_clip's A exponent.

## Boundary with existing problems

- **LoRA-SB (808)**: sign gradient estimate + the asymmetric U·s/Vᵀ split + an r×r latent;
  LoRA-GA: a raw gradient SVD + an **interleaved direction bank** + four scales + **offset
  absorption into the base weight** (LoRA-SB never touches the base).
- **TinyLoRA (795) / PVeRA (804)**: completely different SVD sources and splits (see their
  notes).
- No existing problem has an "offset absorption / weight subtraction" account (grep-verified).
