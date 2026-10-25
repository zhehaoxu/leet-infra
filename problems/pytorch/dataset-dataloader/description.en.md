# Dataset and DataLoader

## Problem

PyTorch uses the map-style `Dataset` convention to define the data access interface: implement the two methods `__len__` and `__getitem__`, and `DataLoader` can fetch samples by index and automatically collate them into batches. If the interface contract is written incorrectly (wrong return types, no error on out-of-range indices), DataLoader will blow up far away with a very hard-to-understand stack trace.

## Rules

Implement `solve() -> torch.utils.data.Dataset` and return a map-style dataset instance:

1. `len(dataset) == 100`.
2. `dataset[i]` (`0 <= i < 100`) returns the tuple `(x, y)`:
   - `x`: a float32 tensor `[i, 2i, 3i, 4i]` of shape `(4,)`
   - `y`: an int64 scalar tensor `i % 2` (odd/even label)
3. For negative `i` or `i >= 100`, raise `IndexError` per the standard convention (Python negative-index semantics must not be allowed through either; `dataset[-1]` must raise `IndexError` as well).

Do not implement batching yourself — grading uses `DataLoader(dataset, batch_size=8, shuffle=False)` to check that your dataset can be batched correctly.

## Examples

```
ds = solve()
len(ds)      -> 100
ds[7]        -> (tensor([ 7., 14., 21., 28.]), tensor(1))
ds[0][0]     -> tensor([0., 0., 0., 0.])
```

## Constraints

- `x` must be `dtype=torch.float32`; `y` must be `dtype=torch.int64` with `dim()==0`
- Sample values may simply be generated from the formula; there is no need to hold 100 pre-stored records
