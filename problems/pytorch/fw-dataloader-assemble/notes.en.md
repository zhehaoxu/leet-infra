# notes — fw-dataloader-assemble (order 711)

## Provenance

- Source: the judge environment's own `.venv-eval` `torch/utils/data/sampler.py` and `dataloader.py` (v2.14.1). Key implementations: `RandomSampler.__iter__` (32-block randint / randperm plus unconditional remainder slice), `BatchSampler.__iter__` (drop_last=True via zip of one iterator referenced batch_size times vs False via islice), `_BaseDataLoaderIter.__init__` (`_sampler_iter` created first (lazy), then `_base_seed` drawn).
- Boundary with `fw-distributed-sampler` (253): this is single-process assembly (the Sequential/Random/Batch three layers plus the DataLoader iterator seed draw); 253 is DistributedSampler's multi-replica pad/slice; boundary with `dataset-dataloader` (intro): this pins the official RNG-consumption ledger.

## Judge surface

- solve() takes no arguments (fixed scenario); the oracle `_py` uses the real `DataLoader` / `RandomSampler` directly (the official classes themselves as anchor), comparing batch elements as ints (DataLoader collates an int dataset into tensors).
- Validation is pinned to the official exception classes: BatchSampler's batch_size/drop_last → ValueError; RandomSampler's num_samples → ValueError but replacement → **TypeError**.

## Anchors (all verified against the real DataLoader)

- **DataLoader epoch chain (the hardest anchor)**: each epoch's fresh `_BaseDataLoaderIter` first draws `_base_seed` (int64 random_) from the generator, then RandomSampler's `randperm` runs; and RandomSampler ends every epoch by **unconditionally** calling `randperm` for `num_samples%n` (an empty draw when the remainder is 0, still consuming state). For N=7, seed=0: epoch0 = `random_ + randperm + randperm(empty)`, epoch1 likewise. Hence the DataLoader's e0 ≠ bare `randperm(seed0)` (bare = the replay scenario).
- **num_samples=2n quirk**: `num_samples//n=2` producing randperms + a third empty draw → the fourth-permutation probe differs from "the third without the empty draw" (measured [4,0,3,6,2,5,1] vs [1,4,5,3,0,2,6]).
- **bool num_samples allowed**: RandomSampler lacks BatchSampler's explicit bool check; True is an int subclass >0 → treated as 1, yielding exactly one sample.
- len: drop_last=N//bs=2, ceil=3; bs=10>N → 0/1.
- replacement 32 blocks: 3 full blocks + 4 remainder produces exactly the same as a single `size=(100,)` call (see mutation notes).

## Mutation log (10/12 caught)

seq-shift, drop-reverse, tail-drop-keep, gen-reset-per-epoch, replay-same-seed1, repl-no-block32, repl-single-call, quirk-no-consume, len-floor-always, epoch-no-baseseed, epoch-no-quirk-consume, bs-gt-n-len-swap — 10 caught.

**Two equivalent mutations recorded (important provenance finding)**: `repl-no-block32` (`size=(N%32,)`→`size=(7,)`; N%32=7 is identity) and `repl-single-call` (32 blocks → one `size=(100,)` call) — measured: CPU `torch.randint` consumes the RNG per sample, so blocked calls and a single call produce **element-identical** output; the official 32-block structure is yield style, not RNG semantics (unlike randperm's empty-draw quirk, which really consumes state). Two test bugs were also fixed: an unsorted comparison target in `sorted(x) == list(range(7))*2`, and the replay-vs-e0 `_base_seed` draw difference.

## Red lines

The solution may not use torch.utils.data (hand-written chain; RNG primitives limited to randperm/randint with an explicit Generator); no multi-process / persistent_workers / prefetch (num_workers=0 ledger only); no inner collate (index ledger only); no overlap with 253's DistributedSampler replica ledger.
