# Notes: GraLoRA Granular Sub-block Forward and Block Merge

## Source

Official repo `SqueezeBits/GraLoRA` (arXiv:2505.20355, NeurIPS 2025 Spotlight). `model.py`/`config.py` are standard PEFT scaffolding (merge explicitly `raise NotImplementedError`); the deterministic core is only two places in `peft/src/peft/tuners/gralora/layer.py`: the layout/init account in `update_layer` and the double-einsum chain in `Linear.forward`. The implementation is now part of HuggingFace PEFT as `tuners/gralora`. Official training configs (`scripts/train/llm.llama-3_1-8b+data.code.sh`): Llama-3.1-8B, r in {16,32,64,128}, gralora_k=2 for r<=32 else 4, hybrid_r=8 only for r=16 and 0 otherwise — the default configuration is pure granular with no bypass.

## Line-by-line translation of the official einsum chain

```python
h = einsum("blni,nir->blnr", x.view(B,L,N,in//N), A)          # A: (N, in//N, rank)
h = h.view(B,L,N,N,sub).permute(0,1,3,2,4).reshape(B,L,N,N*sub)
o = einsum("bljr,jro->bljo", h, B)                            # B: (N, rank, out//N)
result += scaling * o.reshape(B,L,-1)
```

- First einsum: input segment \(n\) goes through \(A_n\), output \((B,L,N,r_\text{gra})\). Note A is stored after a transpose(1,2): each block is \((d_\text{in}/k, r_\text{gra})\) — **each segment's A spans the full \(r_\text{gra}\) rank**.
- view/permute/reshape: the rank axis is cut into \((N, \text{sub})\) (row-major, `rank_idx = seg*sub + off`), the permute moves the "segment" axis forward, and the "input-block" and "offset" axes are compound-reshaped into a new rank axis. The result is `mid[bl, seg j, n*sub+off] = x_n·A[n][:, j*sub+off]`.
- Second einsum: `y[j,o] = Σ_r mid[j,r]·B[j,r,o]`, r summed (mid's compound index ↔ B's rank rows), o free (\(d_\text{out}/k\)). Each B block is \((r_\text{gra}, d_\text{out}/k)\).

## Equivalent block formula (the structure made visible)

\[\Delta W^{(j,n)} = B_j[n\cdot s{:}(n{+}1)s,:]^{\top} A_n[:,\, j\cdot s{:}(j{+}1)s]\]

Block \((j,n)\) has rank \(=s=r_\text{gra}/k\). B's \(n\)-th **row segment** pairs with A's \(j\)-th **column segment** — A is shared along the input-block axis, B along the output-segment axis. Parameter count \(N\cdot r_\text{gra}\cdot\frac{d_\text{in}}{N}+N\cdot r_\text{gra}\cdot\frac{d_\text{out}}{N}=r_\text{gra}(d_\text{in}+d_\text{out})\), identical to a same-rank plain LoRA. That is the paper's selling point: under the same parameter budget the bottleneck narrows from \(r_\text{gra}\) to \(\text{sub}\), and each block is thinner and independent.

## Pitfalls

1. **scaling uses the total rank \(r\), not \(r_\text{gra}\)**. Officially `self.scaling[adapter_name] = gralora_alpha / r` — when hybrid_r>0 (the official r=16, hybrid=8, alpha=32 setup) scaling=2, not 4. This keeps the \(\alpha/r\) convention of plain LoRA.
2. **The permute swaps "segment" and "input-block"**. Dropping it makes the reshape merge \((n,\text{off})\) in the original order, so mid's compound index becomes \(n\cdot\text{sub}+\text{off}\) and no longer pairs with B's row-segment semantics — every value is wrong (pinned by mutants m-permute/m-noview).
3. **B's zero initialization**. B and \(B_g\) are all zeros ⇒ \(\Delta W=0\) at the start of training (A is kaiming, nonzero). The `zero_init` set in `param_layout` pins this as a testable account.
4. **The two validations are `r > 0` and `r_\text{gra} % k == 0`** (the latter an assert officially). Note it is **gralora_rank** (\(r-r_\text{hyb}\)) that must be divisible by k, not \(r\) itself — the hybrid bypass rank does not participate in the split.
5. **The hybrid bypass shares the same scaling**. Officially `result += scaling * Bg(Ag(dropout(x)))`, using the same scaling as the gralora branch.

## Verification anchors

Anchors were dumped after a three-track sandbox cross-check (numpy float64):

- **Track A**: literal transcription of the official einsum chain (np.einsum and torch.einsum share subscript semantics).
- **Track C**: an explicit quadruple loop expanding `y[j,o] = Σ_{n,b} (x_n·A[n][:,j·sub+b])·B[j][n·sub+b, o]` — no einsum engine, pure subscript semantics.
- **Track D**: the block formula `ΔW[j,n] = B[j][n·sub:(n+1)·sub,:].T @ A[n][:, j·sub:(j+1)·sub]` assembled into a dense \(\Delta W\), then \(x\Delta W^\top\).

After all three tracks agreed to 1e-14 on four size configurations (including hybrid>0, in≠out, k=3/4), Track A's output was frozen into EXPECTED. `check_block_delta` adds an **in-test cross-check** (x@ΔWᵀ·scaling ≈ the forward anchor), `check_forward_delta` pins the einsum-chain values, `check_full_forward` pins the hybrid branch and its two ValueError guards, `check_param_layout` pins the shapes and the zero_init set, `check_split_dims` pins the 5-tuple and both validations.

## Complexity

forward_delta performs the same two low-rank contractions as the official code: \(O(BL\cdot d_\text{in}\cdot r_\text{gra} + BL\cdot r_\text{gra}\cdot d_\text{out})\), same order as a plain LoRA forward; block_delta writes a dense \(\Delta W\) at \(O(d_\text{out}d_\text{in})\) (only needed for merge/analysis — the official merge is unimplemented and training always runs the einsum chain).
