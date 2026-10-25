# Solution Notes · SnapKV Pooling

## The essence of the three steps

The structure of kvpress's official `SnapKVPress.score`:

```python
scores = attn_weights.mean(dim=-2)               # (B,H,P)
scores = F.avg_pool1d(scores.unsqueeze(1),
                      kernel_size=self.kernel_size,
                      padding=self.kernel_size // 2,
                      stride=1).squeeze(1)       # (B,H,P)
# omitted: per kv-group averaging (num_kv_groups=1 in this problem)
scores = F.pad(scores, (0, W), value=scores.max().item() + 1)
```

`attn_weights` already comes from upstream as `[..., -W:, :-W]`, i.e. the part covering the queries inside the window against earlier keys—so the problem hands it to you directly.

## Why avg_pool1d needs unsqueeze

`F.avg_pool1d` expects a 3D input `(N, C, L)`. `scores` is `(H, P)`; adding a batch dimension of 1 turns it into `(1, H, P)`—each head is treated as one channel. Pooling only slides along the last dimension P, and after pooling you squeeze back to `(H, P)`.

## Padding semantics

With `avg_pool1d`, zero padding plus `count_include_pad=False` is what makes the denominator count only valid positions; SnapKV instead uses the **default `count_include_pad=True`**—the denominator is always `kernel_size`. So the pooled result at position 0 is `(0 + base[0] + base[1]) / K`.

## Differences from TOVA

TOVA uses only the **last query**, whereas SnapKV uses **the average over the last W queries plus pooling**. Pooling "smooths" the scores of adjacent keys, reducing the problem of isolated high-score points being preferentially kept.

## max+1 for the trailing window

The `window` is the most recent W tokens—they must be kept during inference (otherwise generation loses its memory). Filling the tail with a constant "one higher than the global maximum" guarantees that `topk(n_kept)` always picks them first.
