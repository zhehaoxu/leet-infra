# Distributed Data Sharding: Rank Slices, Padding and a Coverage Ledger

## Task

Implement `solve`. Given a globally ordered sequence of sample indices, cut it for the current rank the way PyTorch's `DistributedSampler` does, and report how many times each sample is read across the whole world. Return the rank's slice, the per-rank step count, the global slot count and the coverage ledger.

The part of DDP training that people get wrong is not the communication but "the dataset size does not divide evenly over the ranks". Production code splits into two opposite camps: `drop_last=True` truncates (one fewer step everywhere, no sample read twice), `drop_last=False` pads from the head (every sample is read at least once, at the price of duplicates). The padded slots also **change which sample lands on which rank**, so "duplicate the first few entries" is not an arbitrary choice. This problem turns the three pieces — the step formula, the padding algebra, the strided slice — into a pure function and additionally asks for the coverage ledger, which is the number you actually look at when a rank runs one step short or an evaluation set turns up double-counted. The shuffle itself (`torch.Generator().manual_seed(seed + epoch)` feeding `randperm`) is out of scope: `order` arrives as an explicit argument, so the judge never touches a random number.

## Input

- `order`: CPU int64 `[N]`, $N\ge 1$, holding a **permutation** of $0..N-1$ (each index exactly once).
- `num_replicas`: Python `int` (not `bool`), $\ge 1$.
- `rank`: Python `int` (not `bool`), $0\le \mathit{rank}<\mathit{num\_replicas}$.
- `drop_last`: Python `bool`.

Inputs are never mutated; invalid input raises `ValueError`.

## Exact Rules

Let $R=\mathit{num\_replicas}$ and $N=|\mathit{order}|$.

**Steps**:

$$
\mathit{num\_samples}=
\begin{cases}
\lfloor N/R\rfloor,& \mathit{drop\_last}\ \text{true and}\ N\bmod R\neq 0,\\
\lceil N/R\rceil,& \text{otherwise},
\end{cases}
\qquad \mathit{total\_size}=\mathit{num\_samples}\cdot R.
$$

The branches differ only in the one indivisible case; when $N\bmod R=0$ the two formulas coincide and $\mathit{total\_size}=N$. With $N<R$ and $\mathit{drop\_last}$ true, $\mathit{num\_samples}=0$ and that rank runs no step at all.

**The effective sequence** (pad or truncate, one of the two):

$$
\mathit{padding}=\mathit{total\_size}-N,\qquad
\mathit{effective}=
\begin{cases}
\mathit{order}[:\mathit{total\_size}],& \mathit{drop\_last},\\
\mathit{order}\,+\,\mathit{order}[:\mathit{padding}],& \mathit{drop\_last}=\mathit{False},\ \mathit{padding}\le N,\\
\mathit{order}\,+\,(\mathit{order}\cdot\lceil \mathit{padding}/N\rceil)[:\mathit{padding}],& \mathit{drop\_last}=\mathit{False},\ \mathit{padding}>N.
\end{cases}
$$

The third case only shows up when $N<R$ without truncation — the padding is longer than the dataset, so a single sample gets copied several times. $\mathit{effective}$ always has length $\mathit{total\_size}$.

**Slicing**: the rank receives the strided slice

$$
\mathit{shard}=\mathit{effective}[\mathit{rank}:\mathit{total\_size}:R],
$$

left-inclusive, starting at $\mathit{rank}$, step $R$, of length exactly $\mathit{num\_samples}$.

**Coverage ledger**: $\mathit{coverage}[i]$ counts how often sample $i$ occurs in $\mathit{effective}$ — that is, summed over **all** ranks — and has length $N$. It satisfies $\sum_i\mathit{coverage}[i]=\mathit{total\_size}$; every entry is $\ge 1$ without truncation, and indices that never made it into $\mathit{effective}$ read 0 when truncating.

## Returns

`(shard, num_samples, total_size, coverage)`: `shard` is int64 `[num_samples]`, `num_samples` and `total_size` are Python `int`s, `coverage` is int64 `[N]`. Everything compared bit-for-bit.

## Example

$N=7$, $R=3$, `order = [6, 2, 0, 4, 1, 3, 5]`:

```python
# drop_last = False
# num_samples = ceil(7/3) = 3, total_size = 9, padding = 2
# effective = [6, 2, 0, 4, 1, 3, 5, 6, 2]      (= order + order[:2])
# rank 0 -> [6, 4, 5]   rank 1 -> [2, 1, 6]   rank 2 -> [0, 3, 2]
# coverage = [1, 1, 2, 1, 1, 1, 2]             (sum = 9)

# drop_last = True
# num_samples = floor(7/3) = 2, total_size = 6
# effective = [6, 2, 0, 4, 1, 3]               (= order[:6])
# rank 0 -> [6, 4]      rank 1 -> [2, 1]       rank 2 -> [0, 3]
# coverage = [1, 1, 1, 1, 1, 0, 1]             (sample 5 is dropped, sum = 6)
```

Read it against the rules: sample 6 sits at positions 0 and 7 of `effective`; $0\bmod 3=0$ and $7\bmod 3=1$, so it appears in **both** rank 0's and rank 1's shard (the trailing 6 of `rank 1 -> [2, 1, 6]`); likewise sample 2 occupies positions 1 and 8, landing on rank 1 and rank 2. Interleaving the three shards as "rank 0 element 0, rank 1 element 0, rank 2 element 0, rank 0 element 1, ..." reconstructs `effective` exactly — that is the defining property of the strided slice and the most useful invariant in the judge.
