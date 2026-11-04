# Notes: Astra's Activation-Space Tail-Eigenvector Initialization

## Source

- Paper: **Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models** (arXiv:2602.19111, ACL 2026 Findings, Ping An Technology).
- Official implementation: merged into **HF PEFT** (PR #3725, commit 73f9a1a, `init_lora_weights="astra"`; `tuners/lora/astra.py` + `layer.py` `astra_init`; the LyoAI/Astra repo hosts the entry script). The PEFT source comment states "This implementation builds on `corda.py`".
- Key difference (vs the CorDA family): CorDA orients an SVD of \(W\cdot C\) by the **input** covariance and multiplies \(C^{-1}\) back; Astra runs `eigh` on the **output** covariance (no inverse, no damp), and the README stresses "no weight SVD or covariance inverse".

## Line-by-line translation of the official implementation

**Covariance accumulation** (astra.py hook, verbatim):

```python
output = output[0] if isinstance(output, (tuple, list)) else output
output = output.detach().reshape(-1, output.size(-1)).data   # (num_tokens, dim)
output = output.float()                       # unless use_float16_for_covariance
output = output / torch.max(output).abs()     # abs of the SIGNED max!
if torch.isnan(output).any() or torch.isinf(output).any():
    raise ValueError("Invalid value found in output...")
covariance = output.t().matmul(output)
module.sample_count += 1
module.covariance_matrix += covariance
# after the run
module.covariance_matrix /= module.sample_count
```

```python
# torch.max(out) is the signed maximum (2.83), then .abs() — different from out.abs().max() (3.15);
# with negative-dominant activations the divisor shrinks and the covariance blows up — official behavior, the judging domain follows the code.
```

**Eigendecomposition** (collect_eigens_for_layer, verbatim):

```python
covariance_matrix = linear.covariance_matrix.float()
S, V = torch.linalg.eigh(covariance_matrix)   # eigh ascending: S[0] smallest
# shape checks S (dim,), V (dim, dim)
module.eigens = AstraEigens(S=S.cpu(), V=V.cpu())
```

**Crop** (crop_astra_eigens, verbatim):

```python
if module.rank > weight_out_dim:
    raise ValueError(f"Astra requires `r` <= out_features ...")
module.eigens.S = module.eigens.S.clone()               # S kept full length
module.eigens.V = module.eigens.V[:, -module.rank:].clone().to(device)
```

**astra_init** (layer.py, verbatim):

```python
if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
    raise TypeError("Please initialize Astra under float32, float16, or bfloat16.")
weight = weight.to(torch.float32)
# r > out_dim / V (out_dim, r) / nan-inf validations
lora_A = (V.t() @ linear_weight).contiguous()     # (r, in)
lora_B = V.contiguous()                            # (out, r)
delta = self.scaling[adapter_name] * lora_B @ lora_A
weight = weight.data - delta                       # residualization (zero delta)
```

## Exclusions and red lines in the judging domain

- **Excluded**: model-level preprocess traversal / cache_file / covariance_file, the Conv1D branch, the hook registration machinery (activation matrices are explicit inputs), use_float16_for_covariance, CPU offload.
- **Kept**: the hook account (flatten / float / normalize / accumulate / count), eigh + crop (S full length, V last r columns), astra_init (A/B/delta/residualization), the dtype gate and all validations.
- **Anchors**: S/V_r/A/W_new fixture anchors (float32 pipeline); the zero-delta identity.

## Key quirks (verified line-by-line against the source)

- The normalization divisor is the abs of the SIGNED max: `torch.max(out).abs()` != `out.abs().max()` (sandbox: 2.83 vs 3.15) — caught by m-ca-absmax; the difference is largest on an all-negative batch (smaller divisor, inflated covariance).
- eigh ascending vs SVD descending: `V[:, -r:]` under eigh is the **top-r eigenspace** (the paper's "tail" wording points the other way — the judging domain follows the code); mistakenly using SVD descending and taking the tail yields the smallest directions (m-ae-head / m-ae-eigh catch it).
- S is kept full length, only V is cropped (m-ae-cropS catches cropping S).
- delta carries the scaling (the lora_alpha/r business factor): caught by m-ai-scaling.
- B@A = V(VᵀW) is exactly the projection: the mathematical root of the zero-delta identity.
- The dtype gate: float64 weights raise TypeError directly (the judging domain defaults to float64, so the test builds float32 explicitly).

## Mutation testing (18/18, 0 equivalent exclusions)

- 18 mutations cover: the normalization divisor (absmax / mean subtraction), flatten, accumulation, the sample division, the float32 cast (caught by a dtype-contract assertion), eigh→svd, head-vs-tail crop, cropping S, 2 r validations, the dtype gate, A's transpose, B's transpose, scaling, the residual sign, the delta product order, and the V NaN check.

## Boundary with existing problems

- CorDA (810): input covariance + SVD of W@fix + \(C^{-1}\) multiplied back + damp loop + KPA/IPA modes; Astra: output covariance + eigh + no inverse + a single top-r subspace.
- LoRA-GA (809) / LoRA-SB (808): their SVD source is the gradient / sign gradient; Astra's source is the second-order statistics of the output activations.
- EVA (807): explained-variance-driven **rank budget allocation** across layers; Astra is a per-layer initialization with no allocation account.
- No output-activation-covariance / eigh account exists in the problem set (grep-verified).
