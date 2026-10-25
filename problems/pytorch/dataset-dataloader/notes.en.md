# Dataset and DataLoader in Depth

## The map-style dataset protocol

PyTorch's `torch.utils.data.Dataset` only needs two methods implemented to be consumed by a DataLoader:

- `__len__(self)` → the total number of samples (fixed at 100 here).
- `__getitem__(self, i)` → the $i$-th sample (returns `(x, y)` here).

There is also the iter-style variant (implementing `__iter__`, e.g. for streaming corpora); this problem is map-style.

## Data specification for this problem

`__getitem__(i)` returns:

$$ x=\texttt{float32}\,[i,\,2i,\,3i,\,4i]\ \text{shape }(4,),\qquad y=\texttt{int64}\,[\,i\bmod 2\,] $$

The dtypes must be strict: features `float32`, labels `int64` (the standard for classification labels; feeding `cross_entropy` requires int64). `solve()` returns the dataset instance itself (not wrapped in a DataLoader).

## Negative indices must be rejected (key pitfall)

Python lists allow `x[-1]` to fetch the last element, but **the dataset protocol requires illegal indices to raise `IndexError`** (supporting negative-index semantics would break DataLoader's sampler logic and silently turn out-of-range accesses into valid samples). The reference solution explicitly does:

```python
if i < 0 or i >= len(self):
    raise IndexError(...)
```

DataLoader's default `SequentialSampler`/`RandomSampler` only emit legal indices in $[0,\text{len})$, so normal training never triggers this; but grading probes directly with out-of-range/negative indices, and you must raise `IndexError` rather than returning data or raising `ValueError`.

## Working with DataLoader

DataLoader builds batching on top of this protocol (`collate_fn` stacks several `(x,y)` into `(B,4)`/`(B,)`), along with shuffling and multi-process prefetching. `__len__` determines the number of steps and `__getitem__` determines what each sample looks like — any ambiguity in either will crash downstream shapes/types.

## Related

optimizer-training-loop is the training loop that consumes a DataLoader; tensor-shape-indexing explains the shape operations on the tensors returned by `__getitem__`; nn-module-block provides the model that receives the data.
