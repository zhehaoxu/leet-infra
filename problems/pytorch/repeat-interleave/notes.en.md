# Explained: repeat_interleave (Element-wise Repetition via index_select)

## Origin

`torch.repeat_interleave` has the same semantics as numpy's `np.repeat`, while the whole-block tiling behaviour belongs to `np.tile` / `torch.repeat`. It shows up constantly in inference code: expanding $n_{kv}$ KV heads into $n_h$ for GQA (`kv.repeat_interleave(n_h // n_kv, dim=1)`), growing a per-layer scalar threshold into a per-head mask, aligning a batch index vector with tokens, and on the `data-tokenization` side repeating a document once per token. "What is the difference between `repeat` and `repeat_interleave`?" is a classic interview question — easy to answer verbally, but writing it out exposes how well one builds index mappings.

## The Index Vector Is the Whole Point

The two semantics differ only in the order of indices. With $L$ elements along the axis, counts $r_i$ and total length $T=\sum_i r_i$:

$$\text{interleave}:\; \underbrace{0,\dots,0}_{r_0}, \underbrace{1,\dots,1}_{r_1}, \dots \qquad\text{tile}:\; 0,1,\dots,L-1,\;0,1,\dots,L-1,\;\dots$$

Once the interleave indices exist, everything reduces to `index_select`:

```
idx = [i for i, k in enumerate(r) for _ in range(k)]   # length T, zero counts allowed
out = y.index_select(axis, torch.tensor(idx, dtype=torch.long))
```

`dim=None` just means `y = x.reshape(-1)` and `axis = 0`, the rest of the path is identical. `index_select` has **copy** semantics (fresh memory), which is exactly what the "mutating the output must not touch the input" assertion needs. The tempting `expand` + `reshape` shortcut looks fine because `reshape` implicitly copies when contiguity is required, but an expanded zero-length axis cannot express per-position counts at all — with a tensor `repeats` that approach simply cannot be written.

The loop-free equivalent locates each output position by prefix sums: with $c_i = \sum_{j\le i} r_j$, the source index of output position $t$ is $\min\{i : c_i > t\}$, i.e.

$$\text{idx} = \texttt{searchsorted}(c,\ \texttt{arange}(T),\ \texttt{right}=\texttt{True})$$

This is also the standard way to avoid a Python loop in variable-length expansion problems (dropping the per-element append overhead); when `repeats` is all zeros, $T=0$, `arange(0)` is empty and the result is precisely an empty tensor with the axes preserved, so no special case is needed.

## Core Points and Pitfalls

1. **Length check on a tensor `repeats`**: with `dim` given its length must equal `x.shape[dim]`, with `dim=None` it must equal `x.numel()`. Skip this and the "length-2 repeats applied to a 4-row tensor" probe slips through.
2. **Integer tensor required**: a float `repeats` such as `torch.rand(4)` must raise ValueError rather than being silently truncated by `.long()`.
3. **One dimension only**: a 2-D `repeats` (`[[1,2],[3,4]]`) is an error in torch and must raise ValueError here too.
4. **A count of 0 is not an error**: the output length may be 0, with shape `(0, ...)` — both the number of axes and the dtype must be preserved; returning `torch.tensor([])` (1-D) for a 2-D input along `dim=0` is wrong.
5. **Unify int and tensor**: an int count equals a length-$L$ vector filled with that value, so the reference funnels both through one `_counts()` entry point instead of duplicating the later branches.

## How It Differs from Neighbouring Problems

- `gather-scatter` / `tensor-shape-indexing`: arbitrary indices for taking and placing; here the indices are fully determined by `repeats`.
- `concat-stack-split`: `cat` over slices can produce the same result, but without the unified index-vector view a tensor `repeats` turns into a long hand-rolled loop.
- `sequence-pack-pad`: packing variable-length sequences into fixed blocks (padding and truncation); this problem only repeats and never trims.

## Cross-Checking and Complexity

- The oracle is `torch.repeat_interleave` compared with `torch.equal` — an **exact** assertion, since there is pure indexing and no arithmetic. Coverage: None/positive/negative axes × int/random/all-zero/int32 counts, plus the interleave≠tile negative assertion, per-position count verification, copy semantics and seven illegal-argument probes.
- Complexity $O(T\cdot\text{other dims})$ with $T=\sum r_i$ the output length along the axis; building the index vector is the only non-vectorised part and is negligible at judging sizes.
