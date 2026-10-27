# unfold/fold/roll: im2col, col2im Overlap Accumulation and Circular Shift

## Problem

Three image-rearrangement workhorses behind one entry point: `unfold` flattens each sliding-window receptive field into a column — i.e. **im2col**, the classic trick that turns convolution into a single GEMM (the backbone of Caffe's convolution, and the exact semantics of torch's `F.unfold`); `fold` is its reverse **col2im** — scattering the columns back onto an image canvas while **summing overlapping contributions** (torch `F.fold`: combine local blocks "by summing the overlapping values"); `roll` performs a circular shift whose out-of-range elements wrap back to the head. Every difficulty here is a bookkeeping detail of the behavior spec: the row/column ordering of unfold's windows (row-major with channels outermost), fold's overlap-accumulation account, the modulo semantics of roll — plus two **deliberate convention differences from native `torch.roll`** (`dims=None` does not flatten; int shifts + tuple dims rolls only the first dim). CPU, fully deterministic, all three modes return a new tensor.

## Rules

- The full signature, parameter meanings and defaults live only in the starter's docstring; that is the single judging standard. `mode` takes only `"unfold"` / `"fold"` / `"roll"`; any other value raises `ValueError`.
- Common parameters: kernel_size / stride / dilation / padding each accept an int or an \((h,w)\) pair; **stride=None means kernel_size** (this problem's convention, for both unfold and fold; note native F.unfold/F.fold default to stride=1). The window count is \(H_{out}=(H+2p_h-d_h(k_h-1)-1)\,//\,s_h+1\) (\(W_{out}\) likewise), \(L=H_{out}\cdot W_{out}\).
- **unfold (im2col)**: x must be 4-D \((N,C,H,W)\) (otherwise `ValueError`); zero-pad first, then slide, returning cols \((N,\,C\cdot k_h k_w,\,L)\). **Rows follow the \(c\to k_h\to k_w\) row-major order**: \(row=c\cdot(k_h k_w)+i\cdot k_w+j\) (channels outermost, within the window row before column); **columns follow \(hi\to wi\)**: \(l=hi\cdot W_{out}+wi\) — matching `F.unfold` as measured. Window elements outside the input boundary are provided by the zero padding. \(H_{out}\) or \(W_{out}\le 0\) (kernel larger than the padded size) raises `RuntimeError` (native-consistent).
- **fold (col2im)**: x is now cols \((N,\,C\cdot k_h k_w,\,L)\) 3-D (otherwise `ValueError`); output_size=\((H,W)\) is required (missing → `ValueError`); returns \((N,C,H,W)\). **Overlapping positions accumulate by summation**; \(C=\text{shape}[1]/(k_h k_w)\) (must divide exactly). If shape[1] is not divisible by \(k_h k_w\), or L does not exactly equal \(H_{out}W_{out}\) computed from output_size (too large or too small both count as a mismatch), raise `RuntimeError` (native-consistent).
- **roll**: circular shift, per-dim \((i+shift)\bmod n\) semantics (modulo, negative shifts supported); kernel_size / stride / dilation / padding / output_size are all ignored. shifts is required (None → `ValueError`):
  - int shifts, dims None: roll the **last dim** (this problem's convention; native `torch.roll` flattens first — here we do **not** flatten);
  - int shifts, int dims: roll that dim;
  - int shifts, tuple dims: roll **only dims[0]** (this problem's convention; native `torch.roll` rejects this combination);
  - tuple shifts: dims must be an equal-length tuple, each dim rolled by its own shift; dims None or a length mismatch raises `ValueError`;
  - dims out of range raises `IndexError` (negative dims normalized by \(+ndim\) first).
- All three modes return a **new tensor** and never modify the input; beyond the `ValueError` / `IndexError` / `RuntimeError` cases listed above, no extra validation is performed.

## Example

Main example (values from the back-filled `EXAMPLE_*` constants in test_cases): single-channel \(2\times3\) image \(x=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\), kernel=2, stride=1, mode=unfold:

\[cols=\begin{bmatrix}1&2\\2&3\\4&5\\5&6\end{bmatrix}\]

One column per window; within a column the 4 rows follow the kernel offsets \((i,j)=(0,0),(0,1),(1,0),(1,1)\) (top-left, top-right, bottom-left, bottom-right). On the same spirit, roll: \([1,2,3,4,5]\) with shifts=2 (dims omitted → last dim) \(\to[4,5,1,2,3]\).

Window ordering and zero padding (2-channel handwritten case that pins the row order): \(x=\mathrm{arange}(24)\) reshaped \((1,2,3,4)\), k=2, s=1 → cols \((1,8,6)\); the window-\((0,0)\) column is \([0,1,4,5\,|\,12,13,16,17]\) — c0's \(\begin{bmatrix}0&1\\4&5\end{bmatrix}\) occupies the first 4 rows, c1's \(\begin{bmatrix}12&13\\16&17\end{bmatrix}\) the last 4. Zero-padded window: \(x=1..12\) reshaped \((1,1,3,4)\), k=3, s=1, pad=1; the window-\((0,0)\) column is \([0,0,0,0,1,2,0,5,6]\) (the \(3\times3\) kernel flattened row-major, out-of-bounds filled with 0). Dilated skipping: same image with k=2, d=2 → cols \((1,4,2)=\begin{bmatrix}1&2\\3&4\\9&10\\11&12\end{bmatrix}\).

The fold overlap account: all-ones cols \((1,4,6)\), k=2, s=1, output_size=\((3,4)\) →

\[\begin{bmatrix}1&2&2&1\\2&4&4&2\\1&2&2&1\end{bmatrix}\]

corners are covered by 1 window, edges by 2, the center by 4 — overlap **sums**, it does not overwrite.

Roll conventions vs native: \(\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\) with shifts=1, dims omitted → \(\begin{bmatrix}3&1&2\\6&4&5\end{bmatrix}\) (per-row last-dim roll); native `torch.roll` **flattens first**, giving \(\begin{bmatrix}6&1&2\\3&4&5\end{bmatrix}\) — deliberately different, and the tests assert both results to pin the convention. shifts=1, dims=\((0,1)\) → roll dim 0 only → \(\begin{bmatrix}4&5&6\\1&2&3\end{bmatrix}\).

Round-trip identity: with stride=kernel and padding=0 (and divisible spatial dims), fold∘unfold restores the input exactly (`torch.equal`); with padding>0, as long as the padded size stays divisible by the kernel, identity also holds after the border is cropped back.

## Constraints

- Numeric cross-check: unfold/fold against native `F.unfold`/`F.fold` over 7 \((k,s,d,p)\) combinations (including pairs, asymmetric padding/dilation) × 2 input shapes with `allclose(1e-6)`; the stride=None → kernel_size convention gets a dedicated check (native needs an explicit stride to align). Roll against `torch.roll` over 13 shifts/dims cases (negative dims, over-long shifts) with `torch.equal`.
- Independent multi-loop oracles on top: unfold writes each column element per window per kernel offset (with boundary checks), fold accumulates per window with `+=`, roll re-indexes per element via \((pos-s)\bmod n\) — a separate code path from the vectorized implementation.
- Discriminating criteria the judge asserts directly: window row/column ordering (the 2-channel handwritten case), overlap summation (the ones account), stride=kernel round-trip identity, the two roll convention differences (contrasted against the native flattening behavior), and the three-way error taxonomy (`ValueError` / `RuntimeError` / `IndexError`).
- The returned tensor shares no memory with the input (a new tensor); no extra validation beyond what the docstring specifies.
