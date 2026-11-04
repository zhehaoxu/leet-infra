# Notes: MiSS Shard-Sharing Expand, Forward and Merge

## Source

Paper: MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure (arXiv:2409.15371 v10, ICLR 2026; the Bone → DiSHA → MiSS lineage). Official repo `Joluck/MiSS` (same as JL-er/bone), merged into HF PEFT in 2024.11 (`tuners/miss`). The repo's `flashop/miss.py` is a TileLang GPU kernel (not problem material), but its `__main__` PyTorch reference semantics (`a.reshape(…, D//R, R).sum(dim=-2) @ b`) match this problem's core account. The README efficiency table: MiSS space \(O(dr)\), forward \(O(bldk)\), both strictly smaller than a same-rank LoRA's \(O((d+k)r)\) — **fewer parameters and faster compute**, the core selling point versus the VeRA/DoRA family.

## Line-by-line translation of the official implementation

**Forward (MissLinear.forward, non-bat)**:

```python
miss = self.miss_block[adapter]            # (r, out), zero-initialized
if self.miss_fn[adapter] == "mini":
    miss = miss.repeat(1, out_features // self.miss_mini_r[adapter])
r = miss.size(0)
if x.size(-1) % r != 0:
    x = F.pad(x, (0, (r - x.size(-1) % r) % r))
result += torch.sum(dropout(x).reshape(*x.shape[:-1], x.size(-1)//r, r), dim=-2) @ miss
```

The key is `sum(dim=-2)` after the reshape: in \((\dots,n_\text{blocks},r)\), dim=-2 is the **shard index** and dim=-1 the **in-shard offset**; summing removes the shard index → an \(r\)-dim output (cross-shard sum aligned by in-shard offset). This is exactly isomorphic to the flashop kernel's `sum_A[s,j] = Σ_i A[s, i*R+j]`.

**Merge (get_delta_weight_miss, non-bat)**:

```python
W_blocks = W.reshape(-1, W.size(1)//r, r).permute(1, 2, 0)   # (nb, r, out)
output = (W_blocks + sign * miss_B).permute(2, 0, 1).reshape(W.shape)
```

`miss_B (r, out)` broadcasts onto every block — equivalently \(\Delta W[:,\, i\cdot r+j] = B[j,:]\) for every block \(i\), i.e. \(\Delta W=\text{tile}(B^\top,(1,nb))\). The two forms can be verified directly: \(\Delta W\cdot x = B\cdot\sum_i x_i\) (with \(x_i\) the \(i\)-th shard) equals the forward's `shard_sum(x) @ B`.

**bat (get_delta_weight)**:

```python
W = orig_weight.reshape(out//r, r, in//r, r).permute(2, 0, 1, 3)  # (nb_in, nb_out, r, r)
result = W @ miss_B + miss_B          # miss_B[j]: (r, r), per output block
dW = result.permute(1, 2, 0, 3).reshape(out, in)
```

unmerge uses \((W-B)(I+B)^{-1}\) (an exact inverse); not taken into this problem.

## Pitfalls

1. **The aggregation axis**: `sum(dim=-2)` removes the shard index and keeps the in-shard offset dim \(r\). Writing `sum(dim=-1)` yields \((\dots,n_\text{blocks})\) — it either crashes against miss_B \((r,out)\) or, if the shapes accidentally fit, gives entirely wrong values (pinned by both m-sum-axis and m-reshape-swap).
2. **The pad account**: the pad is right zero-padding to `n_blocks*r`; zeros do not change the cross-shard aggregate (only the shape), and `pad_size=(r-in\%r)\%r` is 0 when divisible.
3. **mini's repeat direction**: repeat along the **output axis** (`(r, mini_r) -> (r, out)`), with count `out//mini_r`; getting `+1` wrong shows up in the dims or values (pinned by m-mini-repeat).
4. **bat's additive term**: the second \(+B_j\) in \(W_\text{block}B_j+B_j\) is MiSS's structural fix over Bone (breaking collinearity); dropping it shifts every \(\Delta W\) block (pinned by m-bat-noadd). The bat reshape order \((nb_\text{out},r,nb_\text{in},r)\) must pair with the post-permute block assignment — under a square fixture (in=out) the reshape-order mutant is unobservable, so the fixture must be non-square (this problem's main anchor F5 uses \(in=12,out=8\)).
5. **Zero initialization**: all three forms initialize `miss_block` to zeros (`zero_init=True`) — \(\Delta W=0\) at the start of training; unlike LoRA's "B zero, A random", MiSS does not even have an A.

## Verification anchors

Anchors were dumped after a three-track sandbox cross-check (numpy float64): Track A = literal PEFT forward (pad + reshape + sum + matmul), Track B = the merge ΔW tile expansion via \(x\Delta W^\top\), Track C = the explicit loop \(dW[b,l,o]=\sum_{n,j}x[b,l,n\cdot r+j]\cdot B[j,o]\); the bat block account was additionally checked against the \((j{=}0,i{=}0)\) block. After all tracks agreed to 1e-14, EXPECTED was frozen. The tests add an **in-test form cross-check** (F1/F2/F4: `x @ dW.T ≈ forward`). Five fixtures pin: F1 divisible, F2 k=3, F3 triggers pad, F4 mini, F5 bat (non-square). Validation paths cover four raises (r≤0 / r>in / bat indivisible / mini indivisible) plus the two mini-missing-out_features raises.

## Complexity

The forward aggregate costs \(O(in)\) and the matmul \(O(r\cdot out)\) per position — \(O(BL\cdot(in+r\cdot out))\) total, strictly below a plain LoRA's \(O(BL\cdot(in+out)\cdot r)\); expanding \(\Delta W\) is an \(O(out\cdot in)\) tile write (only for merge/analysis, training runs the forward delta). The bat account does \(nb_\text{in}\cdot nb_\text{out}\) per-block \(r\times r\) matmuls.
