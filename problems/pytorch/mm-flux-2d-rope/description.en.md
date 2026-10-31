# Multimodal: FLUX.1 2-D RoPE Frequency Tables and Adjacent-Pair Rotation

## Problem description

FLUX.1's transformer packs text tokens and image tokens into **one joint sequence** and applies **one and the same** RoPE table to both $q$ and $k$. Its head dimension $128$ is split into three axes $(16,56,56)$: axis 0 is the frame (image frames are always $0$, and text ids are $0$ on all three axes), axis 1 is the latent row index, axis 2 is the latent column index. This problem builds the joint-sequence ids and the two float32 frequency tables exactly as `huggingface/diffusers` does for FLUX.1, and rotates `query` through **adjacent pairs**.

Parameters, shapes, return order, dtypes and error contracts live **only in the bilingual starter docstring**; this statement defines the computation rules.

Scope: only FLUX.1's **position-id and frequency-table construction** is reproduced. The per-attention-head replication and the $[B,S,H,D]$ layout are folded away (one table serves every head, so the head dimension is removed and only $[B,S,D]$ remains); no diffusion model weights, no latent packing/unpacking, no attention score computation is involved.

## Grid ids

The image part builds a dense $(h,w,3)$ grid first, then flattens it in row-major order to $(h\cdot w,3)$:

```text
image_ids = zeros(h, w, 3)
image_ids[..., 1] += arange(h)[:, None]      # row index broadcast over columns
image_ids[..., 2] += arange(w)[None, :]      # column index broadcast over rows
image_ids = image_ids.reshape(h * w, 3)      # row-major: k = row * w + col
text_ids  = zeros(text_len, 3)               # FLUX.1 sets every text id to 0
ids       = cat([text_ids, image_ids], 0)    # text first, the joint sequence shares one table
```

Axis 0 is always $0$ (single-frame image), so the first block of angles is $0$ **for every row**.

## Each axis uses its own frequency

Let $d_a$ be the dimension of axis $a$ (here $d=(16,56,56)$ with $\sum d_a=\text{head\_dim}$). The axis owns $d_a/2$ pairs with inverse frequencies and angles

$$
\text{inv\_freq}_a[j]=\theta^{-2j/d_a},\qquad
\text{angles}_a=\operatorname{outer}\!\bigl(\text{pos}[:,a],\ \text{text{inv\_freq}}_a\bigr),\qquad j=0,\dots,d_a/2-1 .
$$

Two details are easiest to get wrong: **the exponent denominator is that axis's own full dim $d_a$, not head_dim, and it is not pre-halved**; and positions come from `ids.float()` (float32) while `inv_freq` and `angles` are evaluated in float64. Concatenating the three `outer` results along the last dim gives $\sum d_a/2=\text{head\_dim}/2$ angles per token.

## Tables and rotation

For each axis, $\cos/\sin$ are evaluated in float64, then `repeat_interleave(2)` along the last dim duplicates every value into two adjacent columns, then `.float()` casts to float32, and finally the three axes are concatenated along the last dim into two tables of shape $(S,\text{head\_dim})$. Because of the duplication the tables satisfy

$$
\cos[2m]=\cos[2m+1],\qquad \sin[2m]=\sin[2m+1],
$$

so the rotation is the classic **adjacent-pair** rotation, each pair $m$ using its own angle:

$$
\begin{pmatrix} q'_{2m}\\ q'_{2m+1}\end{pmatrix}=
\begin{pmatrix} c[m]&-\;s[m]\\ s[m]&c[m]\end{pmatrix}
\begin{pmatrix} q_{2m}\\ q_{2m+1}\end{pmatrix},\qquad
c[m]=\text{freqs\_cos}[\dots,2m],\ s[m]=\text{freqs\_sin}[\dots,2m].
$$

Both tables are always float32, independent of the `query` dtype; the rotation runs in the `query`'s own dtype and `rotated` keeps that dtype.

## Checkable properties

- The frame axis is always $0$ and $d_0=16$: the **first 16 columns** of every table row hold $\cos=1,\sin=0$ **for every row**, so those 16 components are never rotated.
- Text rows have all-zero ids: the whole row is $\cos=1,\sin=0$, i.e. **FLUX.1 does not rotate text tokens at all**; the image cell $(0,0)$ behaves the same way. This is real library behaviour, not an invention of this problem.
- At $\theta=1$ every $\text{inv\_freq}=1$, so the angles degenerate to the positions themselves; rows whose positions are $0$ stay exactly the identity.

## Example 1: small axes $(2,4,4)$

Displayed values keep seven decimals and all come from an actual run of the reference implementation.

```text
axes_dim = (2, 4, 4), theta = 10000.0, text_len = 1, image_grid = (1, 2)
S = 1 + 1*2 = 3, head_dim = 10, angles per token = 5
query = [[[1,0, 1,0, 1,0, 1,0, 1,0],     token 0: text
          [1,0, 1,0, 1,0, 1,0, 1,0],     token 1: image row 0, col 0
          [1,0, 1,0, 1,0, 1,0, 1,0]]]    token 2: image row 0, col 1

ids       = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]                       # float64 (3, 3)
inv_freqs = [1.0, 1.0, 0.01, 1.0, 0.01]
            axis0 (dim 2): [1.0] | axis1 (dim 4): [1.0, 0.01] | axis2 (dim 4): [1.0, 0.01]
angles    = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0.01]]  # float64 (3, 5)
freqs_cos = float32 (3, 10), row 2 = [1, 1, 1, 1, 1, 1, 0.5403023, 0.5403023, 0.99995, 0.99995]
freqs_sin = float32 (3, 10), row 2 = [0, 0, 0, 0, 0, 0, 0.8414710, 0.8414710, 0.0099998, 0.0099998]
rotated   = float32 (1, 3, 10)
            token 0 and token 1 are bit-exact copies of query
            token 2 = [1, 0, 1, 0, 1, 0, 0.5403023, 0.8414710, 0.9999500, 0.0099998]
```

## Example 2: FLUX.1 defaults $(16,56,56)$

Every `query` row is $[1,0]$ repeated $64$ times, so position $2m$ of the rotated row is that pair's $\cos$ and position $2m+1$ is its $\sin$.

```text
axes_dim = (16, 56, 56), theta = 10000.0, text_len = 2, image_grid = (2, 3)
S = 2 + 2*3 = 8, head_dim = 128, angles per token = 64

ids (8, 3) float64:
[0, 0, 0]  token 0 (text)      [0, 0, 2]  token 4 (row 0, col 2)
[0, 0, 0]  token 1 (text)      [0, 1, 0]  token 5 (row 1, col 0)
[0, 0, 0]  token 2 (row 0, col 0)          [0, 1, 1]  token 6 (row 1, col 1)
[0, 0, 1]  token 3 (row 0, col 1)          [0, 1, 2]  token 7 (row 1, col 2)

inv_freqs (64,) float64, denominators are the axis dims:
  axis0 indices  0.. 7 (dim 16): 1.0, 0.3162278, 0.1, 0.0316228, 0.01, ...
  axis1 indices  8..35 (dim 56): 1.0, 0.7196857, 0.5179475, 0.3727594, ...
  axis2 indices 36..63 (dim 56): 1.0, 0.7196857, 0.5179475, 0.3727594, ...

angles (8, 64) float64:
  tokens 0..2 are all zeros
  token 4: angles[36] = 2.0                 # col 2 * inv_freqs[36]
  token 7: angles[ 8] = 1.0, angles[ 9] = 0.7196857, angles[36] = 2.0

freqs_cos / freqs_sin (8, 128) float32:
  every row, cols  0..15 -> cos 1.0, sin 0.0            # frame axis is always 0
  token 7, cols 16..21   -> cos 0.5403023, 0.5403023, 0.7520130, 0.7520130, 0.8688372, 0.8688372
  token 7, cols 16..21   -> sin 0.8414710, 0.8414710, 0.6591483, 0.6591483, 0.4950979, 0.4950979

rotated (1, 8, 128) float32:
  tokens 0, 1 and 2 are bit-exact copies of the input
  token 7, cols  0.. 3 -> 1.0, 0.0, 1.0, 0.0
  token 7, cols 16..21 -> 0.5403023, 0.8414710, 0.7520130, 0.6591483, 0.8688372, 0.4950979
  token 7, cols 36..39 -> 0.9993053, 0.0372673, 0.9996402, 0.0268237
```

## Constraints

- `query` is a CPU float32 or float64 tensor of shape $(B,S,\text{head\_dim})$ with $B\in[1,3]$ and $S=\text{text\_len}+h\cdot w$; it may be non-contiguous and must not be modified.
- $h,w\ge 1$, $\text{text\_len}\ge 0$, `axes_dim` holds three positive even integers, $\theta\ge 1.0$; out-of-range or non-integral arguments raise `ValueError` as listed in the starter contract.
- Per-output grading tolerance: `atol=rtol=4e-5` for float32 outputs, `atol=rtol=2e-12` for float64 outputs, and `ids` must match bit for bit.
- No backward pass; vectorized `torch` operators are freely allowed, this problem bans no tensor API.
