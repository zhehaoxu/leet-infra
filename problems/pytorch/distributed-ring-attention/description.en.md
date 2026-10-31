# Ring Attention: Rotating the KV Blocks, Online Softmax, and Whole-Block Causal Skipping

## Problem

Sequence / context parallelism cuts a long sequence into C segments living on C GPUs, each holding
only its own Q, K and V. Attention, however, needs every query to see the keys of the whole
sequence, so Ring Attention (Liu et al., 2023) sends the KV blocks around a ring: in step s, rank
r consumes the block handed over by its upstream neighbour and then forwards its own block
downstream. After C steps every rank has seen all keys **without ever materialising the full
L x L score matrix**.

Two engineering consequences follow, and this problem asks for both of them cell by cell:

- **Online softmax.** Each step only brings one block of keys, so the running maximum m and the
  denominator l must be corrected in place when a new block arrives, instead of collecting all
  logits and normalising once at the end.
- **Whole-block causal skipping.** Under a causal mask the queries of rank r only see keys at
  positions <= their own. The blocks that arrive late around the ring come from segments further
  down the sequence, so they are entirely invisible: the block can be skipped with not a single
  exp evaluated. The block is still received and still forwarded, though: what is saved is
  arithmetic, not bandwidth.

## Model (follow this definition exactly)

### Layout and steps

- `q_shards`, `k_shards` and `v_shards` hold C >= 1 shards each; a shard has S >= 1 rows and each
  row has D >= 1 numbers.
- The global sequence is the concatenation of the shards **in rank order**: N = C*S, and row i of
  rank r is global row r*S+i.
- In step s (0 <= s < C) rank r receives the KV shard
  $$ \text{src} = (r - s) \bmod C $$
  so over C steps every rank sees each shard exactly once and step 0 is its own shard: the source
  order of rank r is r, r-1, ..., 0, C-1, ..., r+1.

### Whole-block skipping

When `causal` is true and src > r, the keys of that block sit at global positions
[src*S, src*S + S - 1], all of them > r*S + S - 1 (the position of the rank's last query row), so
the block is **entirely invisible**: skip it, evaluate no exp, change no state.

As a position comparison the test is src*S > r*S + S - 1; as an index comparison it is src > r.
When `causal` is false nothing is ever skipped.

### Online softmax

The logit is a = scale * (q_row . k_row); the causal mask keeps only keys whose global position is
<= r*S+i (so the rank's own block is a triangle and every earlier block is kept in full). Each row
keeps a running maximum m, a denominator l and a numerator acc (D dims). Processing one block:

$$ \begin{aligned}
b &= \max_{\text{kept } j} a_j & &\text{(at least one key survives the mask)}\\
m' &= \max(m, b)\\
\text{corr} &= \exp(m - m') & &(0.0 \text{ naturally when } m = -\infty)\\
l &\leftarrow l\cdot\text{corr} + \sum_j \exp(a_j - m')\\
\text{acc}_d &\leftarrow \text{acc}_d\cdot\text{corr} + \sum_j \exp(a_j - m')\,v_{jd}\\
m &\leftarrow m'
\end{aligned} $$

After C steps, out[r][i][d] = acc_d / l.

## Rules

Implement `solve(q_shards, k_shards, v_shards, causal, scale, dtype_bytes)` and return the 5-tuple
`(out, traces, sources, skipped, comm_bytes)`:

- `out`: C shards of S rows and D floats, i.e. acc / l.
- `traces`: traces[r][s][i] is the pair `(m, l)` of row i **after** step s. A step that was skipped
  as a whole block copies the previous step verbatim (step 0 is never skipped, since it is always
  the rank's own block).
- `sources`: sources[r][s] = src, an int.
- `skipped`: a bool per step, marking whether the whole block was swallowed by the causal mask.
- `comm_bytes`: comm_bytes[r][s] = 2 * S * D * dtype_bytes when s < C-1, else 0. The 2 stands for
  one block received plus one block sent; after the last step there is no downstream hop left, so
  every rank pays C-1 times. **A skipped block is still charged.**

Any illegal argument raises `ValueError`: an empty shard list, disagreeing rank counts across the
three arguments, disagreeing shard lengths or row widths, an entry that is not a finite int/float
(`bool`, `None`, complex numbers, strings, `inf` and `nan` all count as illegal), a `causal` that is
not a `bool` (0/1 does not count), a `scale` that is not strictly positive, or a `dtype_bytes` that
is not a positive int.

## Examples

**Example 1**: C=2, S=2, D=1, `causal=True`, `scale=1.0`, `dtype_bytes=4`. q is all zeros and both
k and v are `[[[1.0], [2.0]], [[3.0], [4.0]]]` (the global keys are 1, 2, 3, 4 in order). Every
logit is 0, so the output is simply the mean of the visible keys: row 0 only sees 1, row 3 sees all
four.

```
out     = [[[1.0], [1.5]], [[2.0], [2.5]]]
traces  = [[[(0.0, 1.0), (0.0, 2.0)], [(0.0, 1.0), (0.0, 2.0)]],
           [[(0.0, 1.0), (0.0, 2.0)], [(0.0, 3.0), (0.0, 4.0)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, True], [False, False]]
comm    = [[16, 0], [16, 0]]
```

Step 1 of rank 0 receives src=1, a block entirely past all of its queries, so it is skipped:
`traces[0][1]` is identical to `traces[0][0]`, yet `comm[0][0] = 2*2*1*4 = 16` is still paid.

**Example 2**: C=2, S=1, D=1, `causal=True`, `scale=1.0`, `dtype_bytes=4`; q is all ones and both
k and v are `[[[1.0]], [[2.0]]]`. The two logits of rank 1 are 2 and 1:

```
out     = [[[1.0]], [[1.7310585786300049]]]
traces  = [[[(1.0, 1.0)], [(1.0, 1.0)]],
           [[(2.0, 1.0)], [(2.0, 1.3678794411714423)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, True], [False, False]]
comm    = [[8, 0], [8, 0]]
```

Rank 1: first m=2, l=1; in the second step corr = exp(2-2) = 1,
l = 1 + exp(1-2) = 1.3678794411714423, acc = 2 + exp(-1)*1, and
out = 2.3678794411714423 / 1.3678794411714423 = 1.7310585786300049.

**Example 3**: example 1 with `causal=False`. Nothing is skipped and every query sees all four
keys:

```
out     = [[[2.5], [2.5]], [[2.5], [2.5]]]
traces  = [[[(0.0, 2.0), (0.0, 2.0)], [(0.0, 4.0), (0.0, 4.0)]],
           [[(0.0, 2.0), (0.0, 2.0)], [(0.0, 4.0), (0.0, 4.0)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, False], [False, False]]
comm    = [[16, 0], [16, 0]]
```

**Example 4**: C=3, S=1, D=1, `causal=True`, `scale=1.0`, `dtype_bytes=2`; q is three zeros and
both k and v are `[[[1.0]], [[2.0]], [[4.0]]]`. The ring has to walk two steps backwards to come
home:

```
out     = [[[1.0]], [[1.5]], [[2.3333333333333335]]]
traces  = [[[(0.0, 1.0)], [(0.0, 1.0)], [(0.0, 1.0)]],
           [[(0.0, 1.0)], [(0.0, 2.0)], [(0.0, 2.0)]],
           [[(0.0, 1.0)], [(0.0, 2.0)], [(0.0, 3.0)]]]
sources = [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
skipped = [[False, True, True], [False, False, True], [False, False, False]]
comm    = [[4, 4, 0], [4, 4, 0], [4, 4, 0]]
```

Both late steps of rank 0 are skipped (src 2 and 1 both lie behind it), so its `traces` stay at
`(0.0, 1.0)` for all three steps; rank 2 never skips anything. With C=1 the problem degenerates to
reading only one's own block: `sources = [[0]]`, `skipped = [[False]]`, `comm = [[0]]`.

## Constraints

- 1 <= C <= 32, 1 <= S <= 64, 1 <= D <= 64; `dtype_bytes` is one of 1, 2, 4, 8
- Entries and `scale` guarantee that every logit is finite, so `inf` / `nan` need not be handled
- `out` and `traces` floats are compared with a relative tolerance of 1e-12; `sources`, `skipped`
  and `comm_bytes` are compared exactly
- Complexity: one call is O(C^2 S^2 D), about 1e7 multiply-adds at the largest size
