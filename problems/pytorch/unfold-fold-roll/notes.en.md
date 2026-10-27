# In Depth: unfold/fold/roll: im2col, col2im Overlap Accumulation and Circular Shift

## Source

- `F.unfold`: the official torch documentation (`torch.nn.Unfold` / `F.unfold`) — "Extracts sliding local blocks from a batched input tensor", output \((N,\,C\times\prod(\text{kernel\_size}),\,L)\) with the exact divisibility formula for \(L\); the docs state that for two input spatial dimensions "this operation is sometimes called `im2col`". The im2col lineage traces back to Caffe's same-named preprocessing that turns convolution into GEMM.
- `F.fold`: the official torch documentation (`torch.nn.Fold` / `F.fold`) — "Combine an array of sliding local blocks into a large containing tensor", the mirror spec of `Unfold`'s output, "by summing the overlapping values"; the docs give the same \(L\)-vs-output_size matching condition, and for two output spatial dimensions call it "sometimes `col2im`" (Caffe lineage).
- `torch.roll`: the official torch documentation — "Elements that are shifted beyond the last position are re-introduced at the first position. If `dims` is `None`, the tensor will be flattened before rolling and then restored to the original shape." (the second half is exactly the native behavior this problem's convention deliberately departs from).
- This problem's own conventions (starter docstring): stride=None → kernel_size default, roll with dims=None does not flatten, int shifts + tuple dims rolls only dims[0]. All three differ from native defaults; the starter is the judging standard.

## Full derivation and index convention

**Index convention**: input \(x\ (N,C,H,W)\), kernel \((k_h,k_w)\), strides \((s_h,s_w)\), dilations \((d_h,d_w)\), paddings \((p_h,p_w)\); \(H_{out}=(H+2p_h-d_h(k_h-1)-1)//s_h+1\), \(W_{out}\) likewise, \(L=H_{out}W_{out}\).

**unfold's element account**:

\[cols[n,\; c\cdot k_hk_w+i\cdot k_w+j,\; hi\cdot W_{out}+wi] = xp[n,c,\; hi\cdot s_h+i\cdot d_h,\; wi\cdot s_w+j\cdot d_w]\]

where \(xp\) is the zero-padded image (canvas coordinates starting at \(-p_h,-p_w\)); out of bounds means 0. The row order flattens \((c,i,j)\) into \(c\cdot k_hk_w+i\cdot k_w+j\) — **channels outermost, within the kernel row before column** — aligned with the flattening order of a \((C_{out},C_{in},k_h,k_w)\) weight; this is exactly the permutation that makes the im2col matrix multiplication work in one shot (cf. the `im2col` problem). The column order \(hi\cdot W_{out}+wi\) keeps horizontally adjacent windows adjacent in columns.

**fold's scatter account** (the same account reversed):

\[out[n,c,\; hi\cdot s_h+i\cdot d_h-p_h,\; wi\cdot s_w+j\cdot d_w-p_w] \;\mathrel{+}=\; cols[n,\; c\cdot k_hk_w+i\cdot k_w+j,\; hi\cdot W_{out}+wi]\]

accumulate only when the landing spot is inside the canvas; the canvas includes the padding band and is cropped at the end. **Where the summation comes from**: an output pixel \((h,w)\) accumulates once per \((i,j,hi,wi)\) satisfying both equations — with stride<kernel that is necessarily multiple hits; the all-ones cols, k=2, s=1 account (corner 1, edge 2, center 4) can be verified by hand cell by cell.

**Invertibility of fold∘unfold**: with stride=kernel and padding=0 (and divisible spatial dims) every input position belongs to exactly one window, the permutation is a bijection and `torch.equal` identity holds; with padding>0, as long as the padded size stays divisible by the kernel, identity also holds after cropping. With stride<kernel windows overlap and several input positions map to one output position, so fold can only "recover by summation" and the map is no longer invertible — exactly why the native docs stress summing overlapping values and that unfold cannot be inverted exactly.

**roll's index account**: per-dim independently \(out[pos]=x[(pos-s)\bmod n]\) (equivalently: element \(i\)'s new position is \((i+s)\bmod n\)). Negative and over-long shifts fall into the same \(\bmod\,n\) equivalence class: on length 5, shift 7 ≡ 2 and shift −2 ≡ 3. A zero-length dim has nothing to move and is returned as-is (guard \(n>0\) first, otherwise mod 0 divides by zero). Negative dims are normalized by \(+ndim\); out of range raises `IndexError`.

**Two deliberate differences from native `torch.roll`**: (1) with dims=None this problem rolls the **last dim**, while native **flattens** first, rolls, and restores — on a \((2,3)\) tensor with shifts=1, ours gives \([[3,1,2],[6,4,5]]\), native \([[6,1,2],[3,4,5]]\); (2) int shifts + tuple dims rolls only dims[0], a combination native rejects. The tests assert the native results too, pinning these as conventions, not bugs.

**Padding semantics of unfold**: the top-left window starts at \((-p_h,-p_w)\) — the first window hangs half outside the canvas; sampling consults the zero-padded image rather than skipping out-of-bounds elements. Dilation appears as the extra offset \(i\cdot d_h\) in the slice.

## Key points and common mistakes

1. **The double ordering**: \(row=c\cdot k_hk_w+i\cdot k_w+j\), \(col=hi\cdot W_{out}+wi\). Writing "windows outermost" or \(j\) before \(i\) fails the 2-channel/multi-kernel cases immediately.
2. **fold must accumulate with `+=`**: an overwrite assignment leaves the overlap holding only the last window's contribution.
3. **fold's L check is computed from output_size**: an output_size too large or too small both give \(L\ne H_{out}W_{out}\) → `RuntimeError`; never reverse-engineer output_size from cols' L.
4. **stride=None → kernel_size** (both unfold and fold), unlike native's default stride=1 — pass stride explicitly when comparing against native.
5. **roll conventions**: dims=None does not flatten; int+tuple rolls only dims[0]. Do not just call `torch.roll` as the implementation.
6. **Error taxonomy**: non-4-D/non-3-D inputs, unknown mode, missing output_size, missing shifts or length-mismatched dims → `ValueError`; too-small windows (unfold) and fold shape mismatch → `RuntimeError`; dims out of range → `IndexError`.
7. **Modulo and empty dims**: cast to `int()` before the modulo; skip zero-length dims.
8. **Return a new tensor**: unfold/fold write into fresh canvases and roll uses `index_select`, so inputs are untouched by construction; do not "optimize" with in-place ops.

## Differences from neighboring problems

- **`im2col`**: that problem is "convolution = one GEMM"; unfold is merely its tool (with the layout \((N,\,oh\cdot ow,\,C\cdot k_hk_w)\): rows = output positions, columns = receptive fields — a transposed relationship to this problem's cols \((N,\,C\cdot k_hk_w,\,L)\)); col2im is out of its scope. This problem implements the im2col/col2im pair and pins the official row/column ordering.
- **`unfold-dilated-conv`**: dilated patch extraction only, output keeps the window dims \((B,C,oh,ow,k_H,k_W)\), no padding/stride and no fold side; this problem is full-parameter unfold + fold + roll.
- **`concat-stack-split`**: concat/split stacks along existing dims, copying elements in blocks with dim lengths adding up; this problem's three modes are pure permutations (without overlap, unfold/fold and roll are all bijections) and the element count never changes.
- **Native operators** can serve as oracles, but the three convention differences (stride default, dims=None, int+tuple) must be implemented per this problem's spec, or the discriminating cases will catch it.

## Cross-check convention and complexity

- **Dual oracles**: native `F.unfold`/`F.fold` (7 parameter combinations × 2 shapes, `allclose(1e-6)`; the stride=None convention gets a dedicated group) and `torch.roll` (13 shifts/dims cases, `torch.equal`); plus independent pure-Python multi-loop oracles (unfold writes per window per kernel offset with boundary checks, fold accumulates per window with `+=`, roll re-indexes per element via \((pos-s)\bmod n\)) on a separate path from the vectorized implementation.
- **Discriminating criteria**: the 2-channel window-ordering handwritten case, zero-padded windows, dilated skipping, the ones-account overlap summation, both round-trip identities (stride=kernel, and padding with divisible padded size), the two roll conventions (including a contrast assert against the native flattening), and the full error taxonomy.
- **Complexity**: unfold/fold outputs contain \(N\cdot C\cdot k_hk_w\cdot L\) elements, each written/accumulated once — time and space \(O(N\cdot C\cdot k_hk_w\cdot L)\) (\(L=H_{out}W_{out}\); with stride=kernel this is on the order of the output volume itself); roll is an \(O(\prod \text{shape})\) index rearrangement. The kernel size \(k_hk_w\) is a small constant factor; the dominant cost scales with the output volume.
