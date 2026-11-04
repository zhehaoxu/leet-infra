# Notes: PVeRA Probabilistic Vector-based Random Matrix Adaptation

## Source

PVeRA (Probabilistic Vector-based Random Matrix Adaptation, arXiv:2512.07703, Fillioux et al.) landed in HF PEFT v0.20.0 (`src/peft/tuners/pvera/`). VeRA showed that "cross-layer shared frozen random \(A,B\) + two tiny vectors per layer" approaches LoRA at 1/10 the parameters; PVeRA upgrades each layer's latent adaptation from a fixed \(\lambda_d \odot (xA^{\top})\) to a distribution — \(\lambda_d\) doubles to length \(2r\) (front half scales the μ path, back half the logvar path), the forward samples via the reparameterization \(z=\mu+\epsilon\,e^{\frac12\log\sigma^2}\), and after training one can sample calibrated confidence intervals (beats VeRA on VTAB-1k).

## Line-by-line translation of the official implementation

**Shared banks (`model.py::_init_pvera_A_pvera_B`)**:

```python
generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
pvera_A = torch.nn.init.kaiming_uniform_(torch.empty(config.r * 2, linear_in_dim), generator=generator)
pvera_B = torch.nn.init.kaiming_uniform_(torch.empty(linear_out_dim, config.r), generator=generator)
```

Note: `kaiming_uniform_` defaults to `a=0` (not the \(a=\sqrt5\) of `nn.Linear.reset_parameters`) → bound \(=\sqrt{6/\text{fan\_in}}\). **\(A\) has \(2r\) rows** (the double μ/logvar bank), \(B\) only \(r\) cols. Multi-layer shapes take the max (`_find_dim`); each layer slices at forward time.

**Per-layer (`layer.py`)**:

```python
self.pvera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features))   # ones — then…
self.pvera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r * 2))              # reset fills d_initial
# reset_pvera_parameters:
nn.init.zeros_(self.pvera_lambda_d[...]).fill_(d_initial)
nn.init.zeros_(self.pvera_lambda_b[...])                                           # clears the ones!
```

Close reading: in `reset_pvera_parameters`, `zeros_(lambda_d).fill_(d_initial)` only touches lambda_d; `zeros_(lambda_b)` **clears lambda_b**. So with init_weights=True the `torch.ones` creation is immediately overwritten — the training-start delta is 0 (same zero-start semantics as VeRA). With init_weights=False the reset is skipped: lambda_b stays all ones and lambda_d keeps its `torch.randn` draw (undefined, unanchorable). This problem's `layer_setup` pins the **post-reset account** (λ_b zeros, λ_d all d_initial) and explicitly returns `lambda_b_uninit` (the all-ones branch). `d_initial` only enters lambda_d.

**Forward (`layer.py::forward` + `_reparametrize`)**:

```python
sliced_A = pvera_A[:, :in]; sliced_B = pvera_B[:out]
mu, logvar = (lambda_d * F.linear(dropout(x), sliced_A)).chunk(2, dim=-1)
result = result + lambda_b * F.linear(self._reparametrize(mu, logvar, sample_at_inference), sliced_B)
# _reparametrize: training or sample_at_inference -> mu + randn*exp(.5*logvar); else mu
```

**Merge (`get_delta_weight`)**:

```python
lambda_d = lambda_d[:lambda_d.size(0)//2]     # mu half
sliced_A = sliced_A[:sliced_A.size(0)//2]
output = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), fan_in_fan_out)
```

## Pitfalls

1. **λ_b's "ones creation, reset clearing"**: `update_layer` creates with `torch.ones`, then `reset_pvera_parameters`'s `zeros_` overwrites it — with init_weights=True the training start is **zero**. This is easy to misread in the official source (the adjacent zeros_ + fill_ pair affects only lambda_d's value; `zeros_(lambda_b)` separately clears λ_b). This problem anchors the post-reset semantics; `layer_setup` distinguishes both branches (True→λ_b zeros, False→λ_b ones).
2. **μ/logvar share one linear map**: the `chunk` splits the feature dim of `(λ_d ⊙ xA^{\top})` — \(A\)'s top rows produce μ, its bottom rows logvar, and λ_d's halves each scale their own half. Not two independent Linears; λ_d multiplies the **joint 2r-dim output** elementwise before chunking.
3. **No scaling term**: VeRA and LoRA both have an \(\alpha/r\) or \(1/\sqrt r\) scale; PVeRA's forward and merge have none (the config has no lora_alpha field).
4. **merge takes the μ half only**: the logvar half is invisible to both merge and the eval forward — the tests pin "mutating the logvar half leaves eval/merge unchanged".
5. **eval vs sampling**: `_reparametrize` samples when `self.training or sample_at_inference`, else returns μ. The judge anchors the sampling path with eps as an explicit input (z=μ+eps·exp(½logvar)); the eval path is deterministic.
6. **kaiming default a=0**: bound \(=\sqrt{6/fan\_in}\); float32 RNG anchored bit-for-bit (float64 vs float32 uniform RNG sequences differ — they cannot be cross-checked across dtypes; sandbox lesson).

## Cross-check conventions

Anchors dumped from the sandbox: F1 shared-bank RNG account (float32 bit-for-bit) + two-layer slicing + eval/sampling dual paths + merge cross-check \(x@dW^{\top}\); F2 r=1 minimum rank + d_initial=0.25; F3 uniform λ_d and zero μ-half zero-start; setup pins both init_weights branches (True→λ_b zeros, False→λ_b ones) with d_initial only entering λ_d. Cross-checks: eval forward ≡ \(x\,\Delta W^{\top}\), logvar-half mutations leave eval/merge unchanged, and the sampling path matches a manual reparameterization bit-for-bit.

## Complexity

Each layer's forward is \(O(BN(in\cdot 2r + r\cdot out))\); trainable parameters are \(2r + out\) per layer (\(A,B\) shared and frozen across layers) — the same order as VeRA (lambda_d doubled), with the distribution semantics for free.
