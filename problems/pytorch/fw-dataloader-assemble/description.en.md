# DataLoader Single-Process Assembly Ledger

## Problem

The DataLoader index-assembly chain (single process, `num_workers=0`) has three layers: Sequential/RandomSampler emits the index stream, BatchSampler slices batches, then collate. Hand-write the whole chain (no `torch.utils.data`), with fixed N=7 and batch_size=3, emitting sequential / random / with-replacement / len ledgers; the oracle uses the real official classes.

## Solution

Official semantics (`torch/utils/data/sampler.py` + `dataloader.py` v2.14.1):

- **Sequential**: `range(N)`; BatchSampler `drop_last=True` drops the tail batch, `False` keeps it (any non-empty tail is emitted);
- **RandomSampler** (explicit generator, state carried across iterations): `replacement=False` → `num_samples//n` full `randperm(n)` calls, then **unconditionally** one more `randperm` sliced to `num_samples%n` — when the remainder is 0 this produces nothing but **still consumes the generator** (quirk, shifting later permutations); `replacement=True` → 32-block `randint` plus a remainder block (pure code structure; the RNG sequence equals a single call); non-int or ≤0 `num_samples` → `ValueError` (bool is an int subclass >0, officially **allowed**, True acts as 1); non-bool `replacement` → `TypeError` (different class from BatchSampler's `ValueError`);
- **DataLoader per iteration** (a fresh `_BaseDataLoaderIter` per epoch): first draws an int64 `_base_seed` from `loader.generator` (`torch.empty((), dtype=torch.int64).random_`), **then** RandomSampler's `randperm` runs — so the DataLoader epoch0 permutation ≠ bare `randperm(seed)`; iterating the same loader across epochs continues the generator state;
- **len**: `len(DataLoader) = len(BatchSampler)`; `drop_last → N//bs`, else `(N+bs-1)//bs`; batch_size > N gives 0 batches with drop_last=True and 1 without;
- **BatchSampler validation**: non-int/bool or ≤0 `batch_size`, non-bool `drop_last` → `ValueError`.

## Constraints

- 17 ledger keys: 2 sequential batch ledgers, 5 random permutation/batch ledgers (two epochs on one generator, replay, seed1, drop_last slicing), 2 with-replacement (num_samples=7 remainder block, 100 as 3+1 blocks), 2 `num_samples=2n` quirk ledgers (concatenated permutations + fourth-permutation probe after the empty draw), 4 len ledgers.
- Oracle calls the real `DataLoader` / `RandomSampler` directly, comparing batch elements as ints.
