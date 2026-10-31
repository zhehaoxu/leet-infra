# Solution Notes: FLUX.1 2-D RoPE Frequency Tables and Adjacent-Pair Rotation

## Actual source and differences in scope

The application is **FLUX.1's two-dimensional RoPE position encoding**, taken from the `main` branch of `huggingface/diffusers` (fetched and checked on 2026-10-22; diffusers is not installed in the local evaluation environment, so the line numbers refer to the upstream files):

- `src/diffusers/models/transformers/transformer_flux.py`, lines 500–526: `FluxPosEmbed`, whose file header notes it was "modified from black-forest-labs/flux .../src/flux/modules/layers.py#L11". It loops over `ids.shape[-1]` axes calling `get_1d_rotary_pos_embed(..., theta=self.theta, repeat_interleave_real=True, use_real=True, freqs_dtype=float64)` and concatenates each axis's cos/sin along the last dimension.
- Same file, lines 749–750: `ids = torch.cat((txt_ids, img_ids), dim=0)` followed by `image_rotary_emb = self.pos_embed(ids)` — **one table for the whole joint sequence**; lines 118–119 call `apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)`, so q and k receive the identical table with no img/text split.
- `src/diffusers/models/embeddings.py`: `get_1d_rotary_pos_embed` (the flux / hunyuan-dit / cogvideox branch) and `apply_rotary_emb`.
- `src/diffusers/pipelines/flux/pipeline_flux.py`, lines 508–519: `_prepare_latent_image_ids` builds `latent_image_ids = torch.zeros(height, width, 3)`, adds two `arange` broadcasts for row and column, then `reshape(height * width, 3)`; the text side is `text_ids = torch.zeros(prompt_embeds.shape[1], 3)`.
- Model defaults: `axes_dims_rope=(16, 56, 56)`, `theta=10000`, `attention_head_dim=128 = sum(axes_dims_rope)`.

**Only the FLUX.1 construction is reproduced, and some implementation detail is deliberately folded away:**

- Upstream works on $[B,S,H,D]$ ($H$ attention heads sharing one table). This problem removes the head dimension and keeps only $[B,S,D]$, because one table serves every head; that is a deliberate simplification, not a replica of the upstream tensor layout.
- No diffusion model weights, no `_pack_latents` / `_unpack_latents` latent packing, and no attention computation of any kind.
- Upstream `FluxPosEmbed` adjusts `freqs_dtype` per device (float64 on CPU); this problem fixes the CPU float64 convention. Multi-frame/video, NTK scaling (`ntk_factor`) and linear scaling (`linear_factor`) are out of scope.

## How this differs from rope-embedding and mm-mrope-position-ids

This problem is **not** `rope-embedding` (order 34) and **not** `mm-mrope-position-ids` (order 228); the three differences below are the core of the grading contract:

1. `rope-embedding` has a **single axis**: the whole head dimension shares one set of $\theta_i=\text{base}^{-2i/d}$ where $d$ is head_dim, and the angle comes from a one-dimensional token index. Here head_dim is **split across three axes** $(16,56,56)$ and each axis's exponent uses **that axis's own dim** as denominator (`arange(0, dim, 2) / dim`, with `dim` the FULL axis dim, not pre-halved), driven by three different position components $(t,h,w)$; changing the axis split (e.g. to $(4,4,4)$) changes the frequencies, whereas a single-axis implementation only depends on head_dim.
2. `mm-mrope-position-ids` only constructs **integer $(t,h,w)$ position ids**, with no trigonometry and no rotation at all. Here the ids follow FLUX's `_prepare_latent_image_ids` layout (frame axis always 0, text ids all zero on every axis, row-major flatten) and must then be turned into frequency tables, angle tables and an adjacent-pair rotation.
3. The tables here are **computed in float64, duplicated with `repeat_interleave(2)`, then cast down to float32 with `.float()`**; the single-axis RoPE problem computes everything in the input dtype and has no "each pair angle occupies two columns" layer.

## Upstream numeric chain and this problem's precision

```text
# get_1d_rotary_pos_embed, the flux / hunyuan-dit / cogvideox branch
freqs     = 1.0 / (theta ** (arange(0, dim, 2, dtype=freqs_dtype) / dim))   # dim = FULL axis dim
freqs     = outer(pos, freqs)                                               # (S, dim/2), float64 on CPU
freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()                 # (S, dim), float32
freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
# apply_rotary_emb with use_real=True and sequence_dim=1, x is [B, S, H, D]
x_real, x_imag = reshape(x, ..., -1, 2).unbind(-1)                          # adjacent pairs
x_rotated = stack([-x_imag, x_real], dim=-1).flatten(last two dims)
out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
```

`repeat_interleave(2)` duplicates every value into two adjacent columns, so $\cos[2m]=\cos[2m+1]$: this "two-dimensional" table is really the classic adjacent-pair rotation with $\sum d_a/2=64$ angles per token.

The precision policy of this problem (written into the starter contract and graded accordingly):

- **Tables are always float32**: `angles` and the remaining three return values are float64, while `freqs_cos` / `freqs_sin` are float32 regardless of the `query` dtype, because upstream casts the tables with `.float()`.
- **The rotation runs in the query dtype**: float32 inputs stay float32 throughout; float64 inputs promote the float32 tables to float64 before multiplying, so the result stays float64. This **intentionally differs** from upstream `apply_rotary_emb`'s `x.float()`: upstream would downcast a float64 $x$ to float32, while this problem keeps the query's own precision so a float64 scalar oracle can cross-check at `2e-12`. `rotated` always has exactly the `query` dtype.
- Positions follow upstream's `pos = ids.float()` (one pass through float32), but the returned `ids` are float64. The small integer positions used by the tests are exact in float32, so grading is unaffected.

## Derivation and implementation

1. Validate against the starter contract first: `axes_dim` is three positive even ints (bools and floats rejected, length exactly 3); `theta` is a finite real $\ge 1.0$; both `image_grid` entries are ints $\ge 1$; `text_len` is a non-negative Python int (bool rejected); `query` has exactly three dims with last dim `sum(axes_dim)`; and $S=\text{text\_len}+h\cdot w$. Any violation raises `ValueError`.
2. Build `ids`: `zeros(h, w, 3)`, add `arange(h)[:, None]` on axis 1 and `arange(w)[None, :]` on axis 2, then flatten row-major with `reshape(h*w, 3)`; text is `zeros(text_len, 3)`; `cat([txt, img], dim=0)` with text first.
3. Per-axis frequencies: `inv_freq = theta ** (-arange(0, d_a, 2, dtype=float64) / d_a)` and `angles = outer(pos[:, a], inv_freq)` evaluated in float64, giving $(S, d_a/2)$.
4. Per axis take cos/sin in float64, duplicate with `repeat_interleave(2, dim=-1)`, cast with `.float()`, then concatenate the three axes along the last dim into two $(S,\text{head\_dim})$ tables.
5. Rotate adjacent pairs: use `freqs_cos.reshape(S, -1, 2)[..., 0]` as $c[m]$ (equivalent to taking `[..., 2m]` since both columns are equal), then `out[2m]=q[2m]c[m]-q[2m+1]s[m]` and `out[2m+1]=q[2m+1]c[m]+q[2m]s[m]`, finishing with `stack(...).reshape(...)`; keep the `query` dtype and accept non-contiguous inputs.
6. The return order is fixed as `(rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs)`.

FLUX.1 therefore has two real properties that are often misreported: all-zero text ids mean **text tokens are not rotated at all**; and because the frame axis is constantly zero with $d_0=16$, **the first 16 columns of both tables are $\cos=1,\sin=0$ for every row**. At $\theta=1$ every $\text{inv\_freq}=1$, so the angles degenerate to the positions themselves (this is *not* the identity); only rows whose positions are zero stay exactly unchanged.

## Independent oracle and recorded execution

The grading `_oracle` is a pure-Python scalar implementation: nested `for` loops generate the grid ids by $(row,col)$, each `inv_freq` is recomputed from scratch with `math.pow`, every trig value comes from `math.cos` / `math.sin`, the duplication is done by hand with `extend((v, v))` instead of `repeat_interleave`, and each $2\times2$ rotation is scalar arithmetic; it never calls `torch.outer`, `repeat_interleave`, or tensor `cos/sin`. `math` and torch CPU float64 share the same libm, and `struct.pack/unpack` reproduces the `.float()` rounding, so float64 cases hold at `2e-12` while float32 cases use the sibling convention `4e-5`.

On local PyTorch **2.14.1**, `add_problem.py --selftest mm-flux-2d-rope --mode submit` reported **15/15 ACCEPTED** and `--mode run` reported **2/2 ACCEPTED** (the two example checks carry `example = True`). Coverage: both recorded examples, several hand-written configurations against the oracle, a 10-configuration seeded random sweep (varying $B,h,w,\text{text\_len},axes\_dim,\theta$ and dtype), the first-16-columns identity, the fully-identity text rows, the $\theta=1$ degeneracy, a $1\times1$ grid, `text_len=0`, the per-axis denominators for $(4,4,4)$ and $(2,4,6)$, the `cos[2m]==cos[2m+1]` duplication, the float32/float64 table policy, non-contiguous inputs, immutability, and every `ValueError` branch.

Every number in the statement's two examples comes from an actual run of the reference implementation:

```text
axes_dim=(2, 4, 4), theta=10000.0, text_len=1, image_grid=(1, 2), query rows = [1,0] x 5
ids[2]      = [0.0, 0.0, 1.0]
angles[2]   = [0.0, 0.0, 0.0, 1.0, 0.01]
freqs_cos[2]= [1, 1, 1, 1, 1, 1, 0.5403022766113281, 0.5403022766113281, 0.9999499917030334, 0.9999499917030334]
rotated[0,2]= [1, 0, 1, 0, 1, 0, 0.5403022766113281, 0.8414709568023682, 0.9999499917030334, 0.009999833069741726]

axes_dim=(16, 56, 56), theta=10000.0, text_len=2, image_grid=(2, 3), query rows = [1,0] x 64
ids[7]       = [0.0, 1.0, 2.0]
inv_freqs[1] = 0.31622776601683794        # axis0, 10000**(-2/16)
inv_freqs[9] = 0.719685673001152          # axis1, 10000**(-2/56)
angles[7][8] = 1.0, angles[7][36] = 2.0
rotated[0,7][16:22] = [0.5403022766113281, 0.8414709568023682, 0.7520129680633545,
                       0.6591483354568481, 0.8688372373580933, 0.4950978755950928]
rotated[0,0] and rotated[0,2] equal the input bit for bit (text rows and the (0,0) cell are unrotated)
```

These values are execution records of this problem's reference implementation, not outputs taken from FLUX.1 weights or a full pipeline forward pass.

## Complexity

Building the tables costs $O(S\cdot \text{head\_dim})$ (of which the trigonometry is $O(S\cdot \text{head\_dim}/2)$) and the rotation costs $O(B\cdot S\cdot \text{head\_dim})$; extra space is $O(S\cdot \text{head\_dim})$ for `ids`, `angles` and the two tables, plus the output tensor.
