# PEANuT: generating task-adaptive updates with weight-aware neural tweakers

LoRA approximates `ΔW` from scratch with two trainable low-rank matrices — a **weight-agnostic** linear approximation: the adapter structure is identical no matter what the frozen weights look like. PEANuT (arXiv:2410.01870) instead conditions on the frozen weights themselves: a compact neural network (officially a *tweaker*) reads `Wᵀ` and generates the task-adaptive update, with provably no less expressivity than any linear PEFT method and strictly fewer parameters than LoRA on layers where `in_features > out_features` (e.g. the MLP `down_proj`).

## Official structure (HF PEFT `src/peft/tuners/peanut/layer.py`)

With base `W: (out, in)`, rank `r`, depth `depth`:

- `peanut_A: Linear(out_features → r)` — note the input dim is the base's **out_features** (not in_features); this is where the parameter saving comes from;
- `depth` pairs of `encoder/decoder`, each `Linear(r → r)`;
- `peanut_B: Linear(r → out_features)`.

Forward delta construction (`_compute_delta_weight`):

```python
d = act(W.T @ A.T)          # (in, r) — the weight-conditioning entry
res = []
for enc in encoders:        # save residual first, then encode
    res.append(d)
    d = act(enc(d))
for i, dec in enumerate(decoders):
    d = act(dec(d)) + res[depth - 1 - i]   # residuals added in reverse
delta = (B(d) * scaling).T  # (out, in)
```

Three non-obvious details:

1. **Reverse-order residuals**: the encoder stage saves `depth` residuals, and the decoder stage adds them back as `res[depth-1-i]` — the pairing is `(decoder_i, residual_{depth-1-i})`, not stack-symmetric;
2. **Activation at every step**: the entry projection, every encoder, and every decoder all apply `act_fn` (default relu); residuals are added **after** the decoder activation;
3. **Zero-init invariant**: with `init_weights=True`, B is all zeros → the merged delta is identically zero (training starts from a zero update, same convention as LoRA).

## Parameter count

```
LoRA   = in·r + r·out
PEANuT = out·r + r·out + 2·depth·r²
saving = r·(in − out) − 2·depth·r²
```

For a `down_proj` (1024→256, r=8, depth=1): LoRA 10240, PEANuT 4224, saving 6016. The price of depth-for-expressivity is `2r²` per encoder/decoder pair.

## Initialization

`reset_peanut_parameters` consumes RNG in a fixed order: `A` → `encoders[0..depth-1]` → `decoders[0..depth-1]`, each `kaiming_uniform_(a=√5)` (i.e. `uniform(±1/√fan_in)`, where `peanut_A`'s fan_in is **out_features**), then B zero-init or kaiming. Anchor replay must follow this order.
