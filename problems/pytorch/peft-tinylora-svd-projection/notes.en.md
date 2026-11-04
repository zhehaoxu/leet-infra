# Notes: TinyLoRA Frozen SVD Factors and Random Projection Composition

## Source

Paper: Learning to Reason in 13 Parameters (arXiv:2602.04118, Meta; Morris/Mireshghallah/Ibrahim/Mahloujifar). The finding: RL (GRPO) fine-tuning is fundamentally **activation**, not learning — Qwen2.5-8B reaches 91% on GSM8K with 13 bf16 parameters (26 bytes), while SFT needs 100–1000× larger updates. Lineage: LoRA-XS (arXiv:2405.17604, ΔW=B·R·A with a trainable r×r R) → TinyLoRA further factors R into a linear combination of fixed random projections. The official reference implementation is ANRGUSC/TinyLoRA (2026-02 public research pack); HF PEFT PR #3024 (merged 2026-02, `tuners/tinylora`) is the source of truth for this problem.

## Line-by-line translation of the official implementation

**SVD split (`_init_svd`)**:

```python
U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
actual_r = min(r, min(weight.shape))
sqrt_S = torch.sqrt(S[:actual_r])
tinylora_A = (sqrt_S.unsqueeze(1) * Vh[:actual_r, :]).contiguous()   # (r, in)
tinylora_B = (U[:, :actual_r] * sqrt_S.unsqueeze(0)).contiguous()    # (out, r)
```

√S is split equally to both sides (the same convention as the LoRA-XS paper): both factors are √S-scaled, and the norm of the i-th factor pair is exactly \(\sqrt{S_i}\) — avoiding a norm imbalance on one side. The SVD runs in float32 (numerical stability); anchors were dumped along the same float32-then-float64 path.

**Projection init (`_init_projection`)**:

```python
seed = base_seed + self._layer_idx          # deterministic index in injection order
gen = torch.Generator().manual_seed(seed)
P = torch.normal(0.0, 1.0 / r**0.5, size=(u, r, r), generator=gen)
```

P lives in a `BufferDict` (frozen, not in the optimizer); with `save_projection=False` checkpoints omit P and it is replayed from the seed on load. The N(0, 1/r) distribution is the Johnson–Lindenstrauss convention (per the code comment).

**R composition and forward**:

```python
R = torch.einsum("i,ijk->jk", v, P)         # (r, r)
h = F.linear(x, A); h = F.linear(h, R); delta = F.linear(h, B)
```

i.e. delta \(=x A^\top R^\top B^\top\), exactly \(x\Delta W^\top\) with \(\Delta W=BRA\) (cross-checked in the tests). v lives in a model-level `nn.ModuleDict` (layers in a group share the same Parameter object; PyTorch does not double-register), and each layer stores only a `_tinylora_v_key` string.

**weight_tying groups (model.py `_create_and_replace`)**:

```python
num_groups = max(1, round(num_target_layers * (1.0 - weight_tying)))
group_size = max(1, num_target_layers // num_groups)
group_idx = min(layer_idx // group_size, num_groups - 1)
```

Each clamp has its own job: `max(1,·)` keeps one group at wt=1 (`round(0)=0` would mean zero groups); `max(1, n//groups)` keeps group_size ≥ 1; `min(·, groups-1)` keeps group_idx in range when n does not divide the group count (the tail group absorbs the remainder). Anchor spot checks: n=5, wt=0.4 → round(3.0)=3 groups, size=5//3=1, indices=[0,1,2,2,2]; n=7, wt=0.5 → round(3.5)=4 (Python banker's rounding: 3.5→4), size=1, indices=[0,1,2,3,3,3,3].

## Pitfalls

1. **√S goes to both sides, not one**. Giving it only to B (or only to A) breaks the reconstruction \(B@A\neq U_r\mathrm{diag}(S_r)V_r\) and fails the norm spot check immediately (pinned by both m-split-side-a/b).
2. **The rank clamp is on the SVD short side**: \(r\) may exceed \(\min(out,in)\) (F2 gives r=8 on a 5×4 weight); `actual_r=min(r, min(shape))` clamps it — dropping this slices S out of range or gives wrong A/B shapes.
3. **P's distribution is N(0, 1/r)** (std=1/√r, not 1/r), and the numeric path is **float32 draw → cast to v's dtype** — anchors follow the official path; drawing directly in float64 gives bit-different Gaussian samples.
4. **R composes via einsum "i,ijk->jk"**: v[i] weights P[i], summed over i; swapping the output subscripts (j,k) keeps the (r,r) shape but transposes the values (pinned by m-r-einsum).
5. **The three clamps in the group ledger**: `round` cannot become `int` (round(3.5)=4 vs 3, pinned by m-tie-round); dropping `max(1,·)` gives zero groups and division by zero at wt=1 (m-tie-max); dropping `min(·, groups-1)` runs the tail group out of range (m-tie-clip).
6. **The forward chain order is A→R→B**, each step `@·.T` (F.linear semantics); dropping R's transpose breaks values as soon as R is non-symmetric (pinned by m-fwd-trans).

## Verification anchors

Anchors were dumped in a sandbox (.venv-eval torch float64): F1 is a 6×8 weight at r=4 (full decomposition) plus a u=6/seed=7 projection and an explicit v; F2 is a 5×4 weight at r=8 (clamped). Two cross-identities: `x@dW.T == forward` (the LoRA-XS chain vs the merge form) and `B@A == U_r diag(S_r) V_r` (lossless split). Five tie configurations (wt∈{0,0.4,0.5,1}, n∈{5,6,7}) cover the clamp edges. The tests also do: same-seed projection replay with `torch.equal`, B column / A row norm = √S_i spot checks, and an explicit weighted-sum comparison for R.

## Complexity

compute_R is an \(O(u\cdot r^2)\) weighted sum; the delta \(B@R@A\) costs \(O(out\cdot r^2+r^2\cdot in)\); the forward is three linears, the same order as LoRA \(O(BL(in\cdot r+r^2+r\cdot out))\). The trainable parameter count is exactly groups×u — the paper's budget=13 comes from tuning u and the tie grouping.
