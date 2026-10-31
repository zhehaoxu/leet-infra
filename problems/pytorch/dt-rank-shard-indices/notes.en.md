# Reference Approach: Fix the Step Count, Build effective, Then Slice by Stride

## Implementation

The three steps must not compensate for each other.

1. **Compute $\mathit{num\_samples}$ and $\mathit{total\_size}$ first.** The branch tests "`drop_last` is true **and** $N\bmod R\neq 0$", not merely "`drop_last` is true". When $R$ divides $N$ the two formulas coincide, so `if drop_last and size % R: floor else: ceil` is case-by-case identical to upstream's `math.ceil((len(dataset) - num_replicas) / num_replicas)`; writing it with integer division keeps floats out entirely, so no large-$N$ rounding can leak in through `ceil`.
2. **Then build `effective`.** Without truncation you append the **first** `padding` entries of `order`; when `padding > N` (only possible for $N<R$) you repeat the whole `order` $\lceil \mathit{padding}/N\rceil$ times and cut. The whole block is equivalent to indexing by remainder: $\mathit{effective}[k]=\mathit{order}[k\bmod N]$. The reference solution spells it as explicit list concatenation while the oracle uses the remainder rule, so two structurally unrelated paths have to collide.
3. **Finally the strided slice** $\mathit{effective}[\mathit{rank}:\mathit{total\_size}:R]$. Its length is already $\mathit{num\_samples}$; no second cut is needed.

$\mathit{coverage}$ is a global ledger counted over `effective`, not over this rank's shard, so it must be accumulated **before** the slice, over the whole padded or truncated sequence.

The same upstream logic, verbatim except for the type comments:

```text
if self.drop_last and len(self.dataset) % self.num_replicas != 0:
    self.num_samples = math.ceil(
        (len(self.dataset) - self.num_replicas) / self.num_replicas
    )
else:
    self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
self.total_size = self.num_samples * self.num_replicas

padding_size = self.total_size - len(indices)
if padding_size <= len(indices):
    indices += indices[:padding_size]
else:
    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
# drop_last branch instead does: indices = indices[: self.total_size]

indices = indices[self.rank : self.total_size : self.num_replicas]
```

Cell-by-cell walk of $\mathit{order}=[6,2,0,4,1,3,5]$ with $R=3$ and `drop_last=False`:

```python
# num_samples = ceil(7/3) = 3, total_size = 9, padding = 2
# effective  = [6, 2, 0, 4, 1, 3, 5, 6, 2]
# positions   0  1  2  3  4  5  6  7  8
# rank 0 takes positions 0,3,6 -> [6, 4, 5]
# rank 1 takes positions 1,4,7 -> [2, 1, 6]
# rank 2 takes positions 2,5,8 -> [0, 3, 2]
# coverage by value: 0->1 1->1 2->2 3->1 4->1 5->1 6->2   sum = 9
```

## Correctness checks

`order` is an explicit argument and must be a permutation of $0..N-1$, so no random number ever reaches the comparison; the `torch.randperm` inside `_case` only **produces** permutations, with hardcoded seeds. Eight `check_*` functions:

- `check_example`: both branches of the worked example compared bit-for-bit as 4-tuples, all three ranks checked.
- `check_matches_oracle`: 25 $(N,R,\mathit{rank},\mathit{drop\_last})$ combinations × 3 permutations, covering $N=1$, $N<R$, $N=R$, $R=1$ and every $N\bmod R$ bucket.
- `check_shards_interleave_to_effective`: the shards of **all** ranks, interleaved as "each step take rank 0's $k$-th, rank 1's $k$-th, ...", must reproduce `effective` exactly; `coverage` is then re-derived independently with `merged.count(value)`. This one pins down both "the slice is strided" and "coverage is global" — a contiguous-block answer interleaves into something the remainder rule cannot produce.
- `check_padding_and_truncation_edges`: with $\mathit{padding}=0$ the two branches must return identical 4-tuples; $\mathit{padding}=N$ ($N=5,R=10$, an exact full copy); the $N<R$ padded case with $\mathit{padding}<N$ ($N=3,R=5$, `coverage=[2,2,1]`, `rank 0 -> [1]`, `rank 4 -> [0]`); the repeat-the-whole-list branch with $\mathit{padding}>N$ is reached by $N=1,R=4$ (`coverage=[4]`) and by the $N=2,R=5$ row of `check_matches_oracle`; truncating with $N<R$ gives `num_samples=0` and an empty int64 tensor rather than `None`; plus the $N=1$ and $R=1$ extremes.
- `check_duplicated_samples_are_the_head`: uses `coverage` to prove that exactly the first `padding` entries of `order` are read twice, and that when truncating exactly the tail samples read 0 while the rest read 1, summing to `total_size`.
- `check_identity_order_is_strided`: with $\mathit{order}=\mathit{arange}$ the difference between a strided slice and a contiguous block is loudest — at $N=12,R=4$ every rank gets $\mathit{range}(\mathit{rank},12,4)$; at $N=10,R=4$ the shards are `[[0,4,8],[1,5,9],[2,6,0],[3,7,1]]`, so the two extra duplicated slots land on ranks 2 and 1 rather than "the last rank takes one more".
- `check_invalid`: 17 rejected inputs (non-tensor, 2-D, empty, int32, float, not a permutation, index out of range, negative index, $R\le 0$, `float`/`bool` typed scalars, `rank == R`, non-bool `drop_last`) must raise `ValueError`; 4 legal boundary calls must go through.
- `check_maximum_shape`: $512\times 7$ and $257\times 64$, two ranks and both `drop_last` settings.

Mutation testing ran 17 mutants: contiguous-block slice, `effective[rank:size:R]`, start shifted by one, stride $\max(1,N-\mathit{rank})$, padding taken from the tail instead of the head, no repetition when `padding > N`, unconditional `floor`, ceil-steps combined with truncation, `coverage` counted over the shard, `coverage` hardcoded to 1, `coverage` counted over `order`, `total_size` reported as $N$, empty shard returned as `None`, `coverage` cast to int32, plus three loosened validators (skip the permutation check, accept an int `drop_last`, allow `rank == R`) — **16 caught**. The single survivor is "always `floor` under `drop_last`, else `ceil`": when $N\bmod R=0$ floor and ceil agree, so it is bit-identical to the reference on every input. That is a genuinely equivalent mutant and the judge neither can nor should separate it.

## Scope and sources

The logic comes from the torch installed in the local `.venv-eval`, file `torch/utils/data/distributed.py` (`DistributedSampler.__init__` for the step count, `__iter__` for padding, truncation and subsampling, including its two `AssertionError` self-checks); the fence above is a verbatim excerpt. The following points are **defined by this problem** and have no upstream counterpart, so no attribution is claimed:

- The `coverage` return value (upstream yields only the `indices` sequence, never a per-sample ledger).
- Externalising the shuffle as an explicit `order` argument that must be a permutation. Upstream uses `g.manual_seed(self.seed + self.epoch)` with `torch.randperm`; that part is out of scope here and no seed-level reproduction is asserted.
- Raising `ValueError` for every invalid input (upstream raises `ValueError` for an out-of-range rank and `AssertionError` for a length mismatch).
- No `IterableDataset` branch, no `set_epoch`, and the result is tensors rather than an iterator.

Division of labour against the already-registered `fw-distributed-sampler` (order 253): that problem never implements anything, it audits the **real** `DistributedSampler` as a black box (fixed $N=7$, `seed=0`, testing epoch-permutation reproducibility and replay consistency). This one is the mirror image: whether you can write the index allocation as a pure function for arbitrary $(N,R,\mathit{rank},\mathit{drop\_last})$ without the library, and produce the coverage ledger the library never hands back.
