# Cumulative Sum Scan in Depth

## From reduction to scan: keeping intermediate results

`sum` gives only a single number, "the total over the whole axis", whereas `cumsum` keeps every partial sum:
$$ \text{out}_i = \sum_{j\le i} x_j,\qquad \text{out}_i = \text{out}_{i-1} + x_i $$
It is a prefix scan over an **associative operator** $(\mathcal S, +)$. Replacing $+$ with `max/prod/min` gives the corresponding
scan (e.g. `cummax` is used for the running max in numerically stable softmax).

## The parallel view: scan need not be serial

The naive recursion is $O(n)$ serial, but prefix sums are associative, so they can be parallelized with **Blelloch / work-stealing** in logarithmic depth
— exactly why modern GPUs, `torch.cumsum`, and Mamba's "selective state-space scan" can parallelize efficiently.
The serial implementation aids understanding of the semantics; the engineering implementation pursues parallelism.

## Typical uses

- **top-p / nucleus sampling**: run `cumsum` over the sorted probabilities and find the first position exceeding `p` as the cutoff
  (directly echoing the "top-p sampling" problem in this collection).
- **Causal/cumulative masks**: build a lower-triangular mask in attention by comparing prefixes of position indices.
- **Integration/normalization**: turn a histogram into a distribution function (CDF); Gumbel / inverse-transform sampling relies on `cumsum` to obtain the CDF.

## Boundary reminders

- `axis` may be negative; remember to normalize it to a positive axis before slicing.
- The **last element of the cumulative sum = the sum over the whole axis** (see `check_last_is_total`), usable as a self-checking invariant.
- Numerically, accumulating over a long axis drifts; use `allclose` for floating-point tolerance.
