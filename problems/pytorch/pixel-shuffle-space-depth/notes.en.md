# In Depth: pixel_shuffle and space-to-depth/depth-to-space

## Source

- pixel_shuffle: Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network* (ESPCN), CVPR 2016, **arXiv:1609.05158**. The paper proposes sub-pixel convolution: first compute \(C\cdot r^2\) channels at low resolution, then rearrange them to high resolution by "periodic shuffling" — the official `torch.nn.PixelShuffle` documentation cites this paper directly. Note the provenance mapping: ESPCN (1609.05158). Community material occasionally conflates it with EDSR; this problem does not adopt that citation (the number does not correspond to this operator's original paper — to be re-sourced if ever needed).
- space-to-depth / depth-to-space: standard TensorFlow operators (`tf.nn.space_to_depth` / `tf.nn.depth_to_space`), no paper number. Channel-block ordering is not uniform across frameworks — ONNX's `DepthToSpace`/`SpaceToDepth` defaults to the DCR order and offers a CRD mode to absorb the difference (torch's pixel_shuffle family takes the CRD side); this problem's s2d/d2s commit to the exact channel order of `F.pixel_unshuffle`/`F.pixel_shuffle`, \(c\cdot r^2+i\cdot r+j\), with the equivalence pinned by `torch.equal` in the tests.
- `F.pixel_unshuffle`: the official torch documentation — "Reverses the PixelShuffle operation", with semantics \(out[n,c\cdot r^2+i\cdot r+j,h,w]=x[n,c,h\cdot r+i,w\cdot r+j]\); this problem's unshuffle/s2d match it bit-for-bit.

## Full derivation and index convention

**One bijection table, four directions**. The core identity (shuffle/d2s view):

\[out[n,\,c,\,h\cdot r+i,\,w\cdot r+j]=x[n,\,c\cdot r^2+i\cdot r+j,\,h,\,w]\qquad(0\le i,j<r)\]

Reading the channel index \(c\cdot r^2+i\cdot r+j\): \(c\) is the logical output channel (outer), \((i,j)\) are the row/column offsets of that channel inside the \(r\times r\) block (inner, row before column). Reading the spatial indices \(h\cdot r+i\), \(w\cdot r+j\): \((h,w)\) are block coordinates (slow), \((i,j)\) are in-block offsets (fast). unshuffle/s2d read the same table backwards: divide the output channel by \(r^2\) to get \(c\), take the remainder and split out \((i,j)\); divide spatial coordinates by r to get block coordinates, take the remainder for offsets. All four modes share this one table, pairwise inverted.

**The reshape/permute path** (d2s as the example): \((N,C\cdot r^2,H,W)\) → view \((N,C,r,r,H,W)\) (split the channel into in-block \((i,j)\) axes) → permute \((0,1,4,2,5,3)\) into \((N,C,H,i,W,j)\) → reshape \((N,C,H\cdot r,W\cdot r)\). The purpose of the permute is to insert \(i\) after \(h\) and \(j\) after \(w\) — in row-major memory, reshape merges the **last two dims**, so only adjacent \((h,i)\) and \((w,j)\) can merge into \(h\cdot r+i\) and \(w\cdot r+j\). s2d is the same path reversed: \((N,C,H,W)\) → \((N,C,H/r,i,W/r,j)\) → permute \((0,1,3,5,2,4)\) → \((N,C,i,j,H/r,W/r)\) → reshape \((N,C\cdot r^2,H/r,W/r)\).

**Why s2d ≡ pixel_unshuffle**: torch's `pixel_unshuffle` official semantics are precisely the reversed table above — in the channels-first layout this is the same bijection as TF's space_to_depth "blocks packed into channels", **provided the channel block is ordered as \((c\) outer, \(i\) row, \(j\) column\()\)**. Framework differences live only in the in-block ordering convention (TF/ONNX default DCR puts depth innermost; only ONNX's CRD mode aligns with torch's order); the implementation must commit to \(c\cdot r^2+i\cdot r+j\), pinned by `torch.equal` over 6 configurations.

**r=1 degeneration**: the block shrinks to \(1\times1\), \((i,j)\equiv(0,0)\), and the permutation table is the identity — all four modes must return the input with `torch.equal`. The judge checks values only, so any value-equivalent path is acceptable.

**Side-specific divisibility**: shuffle/d2s require \(C\bmod r^2=0\) (the channels hide \(r^2\) sub-channels) while spatial dims may grow freely; unshuffle/s2d require \(H,W\bmod r=0\) (space hides \(r\times r\) blocks) while channels are free. Checks on the wrong side either miss or false-alarm on the other mode. The flip side of "spatial only grows" (shuffle) / "spatial only divides" (unshuffle) is that no output-size formula is needed — in contrast with unfold/fold's \(H_{out}\) formula.

**Non-contiguous inputs**: a view from transpose has a memory layout inconsistent with its logical shape; either call `.contiguous()` first (or let `reshape` copy automatically), or use addressing that is correct for arbitrary strides — the judge requires `torch.equal` against the same-valued contiguous input.

## Key points and common mistakes

1. **The permute axis order**: d2s's \((0,1,4,2,5,3)\) and s2d's \((0,1,3,5,2,4)\) point in opposite directions; mixing them up yields an "in-block transposed" wrong image — the 16-value handwritten case catches it instantly.
2. **Channel order is \(c\cdot r^2+i\cdot r+j\)** (c outer): writing \(i\cdot r^2+c\cdot r+j\) or similar will not match the native op.
3. **Split the axes before reshaping**: reshaping directly to \((N,C,r^2,H,W)\) and permuting interleaves h with the in-block offsets wrongly; view out independent \(i,j\) axes first.
4. **r's type check**: `isinstance(r, int)` excluding bool (Python's bool is an int subclass, `True` would masquerade as 1); reject \(r<1\).
5. **Side-specific divisibility**: shuffle/d2s check C, unshuffle/s2d check H and W — do not cross-wire the two sides.
6. **Bijection self-check**: the output is a permutation of the input (equal sorted flattened values) — a quick way to catch "lost/duplicated element" bugs.
7. **No F.pixel_\* calls**: that is an implementation constraint (the point of the problem); native may appear only on the test/cross-check side.
8. **dtype and input protection**: a pure rearrangement changes neither dtype nor the input; the reshape/permute path satisfies this by construction — do not sneak a `to()` into an explicit-indexing path.

## Differences from neighboring problems

- **`concat-stack-split`**: stacking/splitting along existing dims copies elements in blocks and adds dim lengths — not a bijection; here the volume is strictly invariant and every element appears exactly once, merely at a new address. Completely different machinery.
- **`unfold-fold-roll`**: unfold/fold is "windows ↔ columns"; windows may overlap, fold accumulates, and the map is generally not invertible. Here the rearrangement never overlaps and is always invertible (round-trip `torch.equal`). roll is a one-dimensional circular shift — a whole dim spinning — not a channel↔space exchange.
- **`im2col`**: flattens into a column matrix serving GEMM, with an extra receptive-field dimension and a volume inflated by \(k_hk_w\); here the volume is strictly invariant.
- One-line memory aid: concat = copied stacking, unfold/fold = windows↔canvas (overlap is many-to-one), the pixel_shuffle family = strict permutation.

## Cross-check convention and complexity

- **Main criterion**: `torch.equal` against `F.pixel_shuffle` / `F.pixel_unshuffle` (a pure rearrangement has no floating-point error, so no allclose), over 6 \((n,c,h,w,r)\) configurations with all four modes checked on the same configurations; the s2d≡unshuffle and d2s≡shuffle equivalence asserts are also `torch.equal`.
- **Independent oracle**: explicit quadruple loops computing the source index per target coordinate (d2s: source channel \(c\cdot r^2+(ho\bmod r)\cdot r+wo\bmod r\); s2d: split the output channel into \(c,i,j\) and read \(x[\cdot,\,c,\,hi\cdot r+i,\,wi\cdot r+j]\)), fully separate from the vectorized reshape/permute path; plus the bijection assert (equal sorted flattened values), round-trip identities, the r=1 identity, and non-contiguous views.
- **Deterministic data**: golden-ratio-hash pseudo-random tensors (no RNG), seeds derived from the shape parameters.
- **Complexity**: time \(O(N\cdot C\cdot H\cdot W\cdot r^2)\) — i.e. \(O(\text{numel})\), one addressing step per element (input and output element counts are equal; a rearrangement adds no volume); space \(O(\text{numel})\) (a new tensor; the reshape/permute path includes one forced copy).
