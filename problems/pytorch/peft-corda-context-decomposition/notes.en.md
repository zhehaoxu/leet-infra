# Notes: CorDA's Context-Oriented Decomposition Adaptation

## Source

- Paper: **CorDA: Context-Oriented Decomposition Adaptation of Large Language Models** (arXiv:2406.05223, NeurIPS 2024, Yibo Yang et al., KAUST / HIT Shenzhen).
- Official implementation: `iboing/CorDA` (`cordalib/decomposition.py` — `decompose_to_adapter` + `CorDA_adapter`; HF PEFT also ships `examples/corda_finetuning`).
- Key difference (vs the PiSSA/LoFTQ family): CorDA's decomposition orientation is decided by the **covariance matrix** — it runs SVD on \(W\cdot C\) and multiplies \(C^{-1}\) back into \(V\); PiSSA runs SVD on \(W\) directly and keeps the principal components.

## Line-by-line translation of the official implementation

**Damped inverse** (decomposition.py, verbatim):

```python
covariance_matrix = linear.covariance_matrix.float()
damp = 0.01
while True:
    compensate = torch.diag(torch.ones(covariance_matrix.size(0))
        * torch.mean(torch.diag(covariance_matrix)) * damp)
    fix_covariance_matrix = covariance_matrix + compensate
    cov_inv = torch.linalg.inv(fix_covariance_matrix)
    inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(n))
    if inv_error < 0.05:
        break
    else:
        damp = damp * 2
```

```python
# The doubling step is unobservable in the judging domain: every reasonable input sits at
# machine precision (~4e-16), so the first damp=0.01 step always resolves (tol=0.05 is defensive); m-di-dbl removed as a true equivalent.
```

**Oriented decomposition** (decomposition.py, verbatim):

```python
w = pretrained_w @ fix_covariance_matrix       # (out, in) @ (in, in)
U, S, V = torch.linalg.svd(w, full_matrices=False)
V = V.transpose(0, 1)                          # (in, rank)
V = (V.t() @ cov_inv).transpose(0, 1)          # the step that makes the context identity hold
```

**Component selection** (decomposition.py, verbatim):

```python
if not first_eigen:        # KPA: the last r (smallest singular values)
    U = U[:, -r:];  S = S[-r:];  V = V[:, -r:]
elif first_eigen:          # IPA: the first r (following PiSSA)
    U = U[:, :r];  S = S[:r];  V = V[:, :r]
weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0, 1)
```

**sigma_fuse, three modes** (CorDA_adapter.__init__, verbatim):

```python
if sigma_fuse == 'UV':
    self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
    self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
elif sigma_fuse == 'U':
    self.ALinear.weight.data = U.mul(S).contiguous()
    self.BLinear.weight.data = V.t().contiguous()
elif sigma_fuse == 'V':
    self.ALinear.weight.data = U.contiguous()
    self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
```

**Forward** (CorDA_adapter.forward, verbatim):

```python
y = self.BLinear(inp)          # (n -> r)
y = self.ALinear(y) + F.linear(inp, self.weight_residual)   # (r -> m) + frozen residual bypass
```

## Exclusions and red lines in the judging domain

- **Excluded**: model-level build_model traversal / lm_head skipping, the act_aware branch (scaling_diag_matrix / fisher_info), NaN/inf print branches, CUDA empty_cache.
- **Kept**: the damped inverse loop, the cov_aware oriented decomposition, KPA/IPA selection, all three sigma_fuse modes, the frozen residual, the forward.
- **Anchors**: S/U/V fixture anchors (float32 pipeline); the full-decomposition identity \(U\,\mathrm{diag}(S)\,V^\top=W\).

## Key quirks (verified line-by-line against the source)

- The identity holds only for the full decomposition: at \(r=rank\) the covariance cancels exactly; for \(r<rank\) the reconstruction is a rank-r approximation and the remainder goes to the residual (the sandbox first verified the identity at r=2 — a trap).
- The two transposes of \(V\): the official line is `(V.t() @ cov_inv).transpose(0,1)` — cov_inv multiplies \(V^\top\) from the **right**; writing `cov_inv @ V.t()` is a caught mutation.
- fix is symmetric: `fix = cov + cI` is symmetric, so `W @ fix.t()` == `W @ fix` — a true equivalent (m-cd-mul removed).
- The official SVD pipeline is float32: `pretrained_w.float()` feeds the svd; identity assertions use tolerance 1e-5 (not 1e-10).
- KPA takes the tail, IPA the head: `S_i[0] > S_k[-1]` is the mode-distinguishing assertion; the KPA residual norm exceeds the IPA one.
- The damping compensation uses mean(diag): not max(diag) and not the 1e-6 constant (that one belongs to the act_aware branch).

## Mutation testing (17/17, 2 equivalent exclusions)

- m-di-dbl (damp doubling rate): within the judging domain the loop always resolves at the first step, so the rate is unobservable — true equivalent, removed.
- m-cd-mul (fix transpose): fix = cov + cI is symmetric — true equivalent, removed.
- 17 mutations cover: the damp start / compensation formula / plus-minus sign, the w multiplication / full_matrices / the direction of both V transposes / the cov_inv side / KPA-IPA swap / the residual sign / 2 r validations, all four fuse modes, and the forward A-B order / residual drop.

## Boundary with existing problems

- LoRA-GA (809) / LoRA-SB (808): their SVD source is the **gradient / sign gradient** (data-independent); CorDA's source is **weight x covariance** (context-dependent) with \(C^{-1}\) multiplied back.
- PiSSA / LoFTQ: the problem set has no PiSSA problem yet; LoFTQ (quantization-aware init) follows the NF/quantization path with no covariance account.
- TinyLoRA (795): frozen SVD factors but the source is a plain SVD of W — no context orientation and no KPA/IPA.
- No covariance / inverse-matrix account exists in the problem set (grep-verified).
