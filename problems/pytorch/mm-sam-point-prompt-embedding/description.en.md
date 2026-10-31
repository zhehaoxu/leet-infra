# Multimodal: SAM Sparse Prompt Embedding and Dense Positional Grid

## Task

Convert sparse prompts (foreground/background points and boxes) given in **original image pixel coordinates** into the embedding vectors that the Segment Anything prompt encoder actually consumes, together with the geometry it uses to normalise those coordinates: the preprocess size, the per-axis rescaled coordinates, the raw positional encoding before label write-back, and an optional dense positional grid.

This is the **weight-free** part of the SAM inference chain: no image resizing, no mask decoder, no learnable parameters. The gaussian positional-encoding matrix and the four embedding vectors are supplied by the caller, because upstream initialises them with an unseeded `torch.randn`.

The bilingual starter contract is the single reference for parameters, shapes, numeric limits, and exceptions. The rules below define the computation.

## Rules

1. Longest-side scale: with the original size `H x W`, `scale = image_size / max(H, W)`, then `new_h = int(H * scale + 0.5)` and `new_w = int(W * scale + 0.5)`. Add `0.5` first and then truncate: this is neither floor nor banker's rounding.
2. Per-axis factors: x is multiplied only by `new_w / W`, y only by `new_h / H`. The factor of the longest axis equals `scale`, but the shortest axis, because of the rounding above, usually gives a factor that is **not** `scale`. The two factors are not interchangeable, and a single shared `scale` is wrong.
3. Pixel-centre offset: points and box corners both receive `+ 0.5` after scaling. The padding point is created after that offset, so its coordinate is exactly `(0, 0)`.
4. Padding row: it is appended only when there are no boxes (the upstream `pad = boxes is None` semantics) and carries label `-1`. Input point labels are restricted to `0` (background) and `1` (foreground); `-1` never appears as input. The padding row sits after all input points and before all box corners.
5. Normalisation: both x and y are divided by `image_size` (the prompt encoder input image is square), so the result lies in `[0, 1]`.
6. Positional encoding: for the normalised coordinate `c`, compute `2 * c - 1`, multiply by the `(2, P)` matrix (x uses row 0, y uses row 1), multiply by `2 * pi`, and emit **P sine entries followed by P cosine entries**, width `2P`.
7. Label write-back: for a `-1` row the positional encoding is computed and only then wiped to zero before `not_a_point_embed` is written, so its sparse output is **independent** of that row's raw encoding; a `0` row adds `point_embeddings[0]` and a `1` row adds `point_embeddings[1]`.
8. Boxes: each `xyxy` box splits into a top-left and a bottom-right corner, which add `point_embeddings[2]` and `point_embeddings[3]` respectively and are appended after all point rows in "box index ascending, top-left first" order.
9. Dense grid: `x_embed` varies along columns and `y_embed` along rows; both are `cumsum - 0.5` divided by the length of their own axis, i.e. column `j` takes `(j + 0.5) / w` and row `i` takes `(i + 0.5) / h`. Stacking `(x_embed, y_embed)` into a last dimension of 2 gives the coordinates, which go through the same encoding function, then the axes are swapped to obtain `(2P, h, w)`. It shares the conventions of steps 5 and 6, the only difference being that these coordinates already live in `[0, 1]`.

Return a five-tuple: sparse embeddings `(M, 2P)`, the coordinates actually fed to the positional encoding `(M, 2)`, the preprocess size as an integer `(2,)` pair `(new_h, new_w)`, the raw encodings before label write-back `(M, 2P)`, and the dense grid `(2P, h, w)` (an empty `(0,)` tensor when it is not requested). Here `M = N + (1 when there are no boxes, else 0) + 2B`.

## Example 1: Point prompts only, with a dense grid

All floating tensors are float64, `boxes` is an empty tensor of shape `(0, 4)`, `image_size = 1024`, and `dense_size = (2, 3)`.

```text
points = [[1.0, 2.0], [3.0, 5.0]]
labels = [1, 0]
grid_size_in = (48, 64)
pe_matrix = [[0.5, -1.0, 0.25], [1.0, 0.5, -0.75]]
point_embeddings = [[0.5, -0.5, 1.0, 0.0, -1.0, 0.25],
                    [-0.5, 1.0, 0.5, -1.0, 0.0, 0.25],
                    [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
                    [0.0, -1.0, 1.0, 0.5, 0.5, -0.5]]
not_a_point_embed = [2.0, -2.0, 0.5, -0.5, 1.5, -1.5]

sparse_embeddings = [
                    [-0.979494, 1.003068, 0.745955, -1.877545, -0.999995, -0.719281],
                    [-0.459572, -0.398930, 1.558119, -0.281465, -1.994879, -0.579761],
                    [2.000000, -2.000000, 0.500000, -0.500000, 1.500000, -1.500000]]
prompt_coords = [
                [16.500000, 32.500000],
                [48.500000, 80.500000],
                [0.000000, 0.000000]]
preprocess_size = [768, 1024]
positional_raw = [
                 [-0.479494, 0.003068, 0.245955, -0.877545, -0.999995, -0.969281],
                 [-0.959572, 0.101070, 0.558119, -0.281465, -0.994879, -0.829761],
                 [-0.000000, 0.000000, 0.000000, -1.000000, -1.000000, -1.000000]]
dense_pe[0] = [[0.866025, -0.000000, -0.866025], [0.866025, 0.000000, -0.866025]]
dense_pe[1] = [[0.500000, -1.000000, 0.500000], [-0.500000, 1.000000, -0.500000]]
dense_pe[2] = [[0.965926, 0.707107, -0.258819], [0.258819, -0.707107, -0.965926]]
dense_pe[3] = [[0.500000, -1.000000, 0.500000], [0.500000, -1.000000, 0.500000]]
dense_pe[4] = [[-0.866025, 0.000000, 0.866025], [0.866025, 0.000000, -0.866025]]
dense_pe[5] = [[0.258819, -0.707107, -0.965926], [-0.965926, -0.707107, 0.258819]]
```

The long side 64 gives factor 16, and the short side 48 happens to give factor 16 as well, so the point `(1, 2)` scales and shifts to `(16.5, 32.5)`. The last row, index 2, is the padding row: its raw encoding is nonzero, yet its sparse output is exactly `not_a_point_embed`. The first three planes of the dense grid are the sine block and the last three are the cosine block.

## Example 2: Points plus a box, with unequal axis factors

All floating tensors are float64, `image_size` and `dense_size` keep their defaults `1024` and `None`; `pe_matrix` takes the first `P = 2` columns of the matrix from example 1, and `point_embeddings` and `not_a_point_embed` keep only their first `2P = 4` components.

```text
points = [[1.0, 1.0]]
labels = [1]
boxes = [[2.0, 3.0, 6.0, 9.0]]
grid_size_in = (30, 50)

sparse_embeddings = [
                    [-0.876515, 1.128458, -0.426411, -1.991715],
                    [0.151036, 1.128620, -0.528450, -0.991694],
                    [-0.117659, -0.628341, 1.993054, -0.428369]]
prompt_coords = [
                [20.980000, 20.966667],
                [41.460000, 61.900000],
                [123.380000, 184.700000]]
preprocess_size = [614, 1024]
positional_raw = [
                 [-0.376515, 0.128458, -0.926411, -0.991715],
                 [-0.848964, 0.128620, -0.528450, -0.991694],
                 [-0.117659, 0.371659, 0.993054, -0.928369]]
dense_pe = empty tensor with shape (0,)
```

The long side is 50, so `scale = 20.48` and `new_h = int(614.4 + 0.5) = 614`, giving an x factor of `1024 / 50 = 20.48` and a y factor of `614 / 30 = 20.466667`: the two components of the same point scale by different amounts. With a box present there is no padding row, so there are `1 + 2 = 3` rows in total; rows 1 and 2 are the box's top-left and bottom-right corners, and the row at index 2 adds `point_embeddings[3]`.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; values are rounded to six decimal places.
