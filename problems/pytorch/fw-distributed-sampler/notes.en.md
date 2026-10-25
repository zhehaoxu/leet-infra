# Explained: DistributedSampler Index Accounting

## Origin

`torch.utils.data.distributed.DistributedSampler` (PyTorch docs and source `torch/utils/data/distributed.py`): the shuffle branch creates a fresh `torch.Generator()` per `__iter__` and calls `manual_seed(self.seed + self.epoch)`, so the epoch is the only variable of the shuffle round (the docs strongly recommend calling `set_epoch` at epoch boundaries, otherwise every round gets the same permutation); `num_samples = ceil(N/R)` (drop_last=False) or `floor(N/R)` (True); padding **repeats the leading entries of the permutation** (source: `indices += indices[:padding_size]`, switching to full copies if the padding exceeds N), not random draws; slicing is `indices[rank : total_size : num_replicas]`. With `num_replicas`/`rank` passed explicitly no process group is required, so it can be simulated in a single process.

## Key points and pitfalls

1. **No `set_epoch` → no reshuffle**: the permutation seed is solely `seed+epoch`, default epoch=0 — epoch0 matching epoch0_again bitwise while differing completely from epoch1 is the single most important ledger of this problem.
2. **Which ranks hold the duplicates**: with \(N=7, R=3\) two entries are padded (the permutation's leading 4 and 0). After stride slicing the padded positions (indices 7, 8) land on ranks 1 and 2 — so rank1's `[0,2,4]` secretly contains the duplicated 4. Every sample appears \(\ge 1\) time and the total is exactly 9.
3. **`drop_last=True` truncates instead of padding**: `total_size = floor(N/R)*R = 6`, and the permutation's tail entry (sample 1) is seen by no rank all epoch — the origin of "distributed training actually sees slightly fewer samples per epoch".
4. **Epoch is inert when shuffle=False**: the sequence is always `range(7)` padded; even `set_epoch(3)` yields \([0,3,6]/[1,4,0]/[2,5,1]\).
5. **Each rank's length is exactly `num_samples`**: padding guarantees an even split; the source itself `assert`s a mismatch.

## Cross-check basis

The solution goes through `DistributedSampler.__iter__`; the judge's `_oracle` never instantiates a sampler: it borrows only `torch.Generator + torch.randperm` to reproduce the same permutation stream (torch's permutation algorithm cannot be reproduced with numpy — this is the single shared framework primitive needed), while padding/truncation/stride slicing are all rewritten as pure-Python list operations. Additional assertions: epoch-replay consistency, drop_last truncation reconstruction, and the static no_shuffle ledger.

## Complexity

\(O(\text{epochs}\times R\times N)\) of index shuffling — microsecond level.
