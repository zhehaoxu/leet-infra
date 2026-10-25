# Top-K Selection In Depth

## Selection vs sorting

A full sort costs $O(n\log n)$, whereas when only the top-k are needed, **partial selection** is faster:

- This solution uses the "repeated argmax + masking" selection method, $O(k\,n)$, which pays off and stays
  intuitive when `k` is much smaller than `n`.
- Theoretically optimal is quickselect at $O(n)$ on average, or maintaining a min-heap of size `k` at
  $O(n\log k)$.
- On GPU, `torch.topk` usually goes through a bitonic/partial sort kernel; `sorted=True` guarantees
  descending order.

## Masking trick: set the selected position to -inf

Once a maximum has been taken, it must not be selected again. The cleanest way is to write
`float('-inf')` into that position with `scatter_`, so the next `argmax` naturally skips it. Symmetrically,
to get the top-k **smallest**, write `+inf` into the selected positions and pair it with `argmin`.

## Ties and determinism

When values repeat, `argmax` returns the **smallest index**, whereas the tie-breaking order of
`torch.topk` is not guaranteed to match. The judge therefore uses continuous random values (which are
essentially never exactly equal), making the top-k unique; in production, if ties must be handled, an
explicit convention is needed (e.g. equal values ordered by ascending index) that both sides follow.

## Applications

- **MoE routing**: score the experts for each token and activate the top-k experts (sparse computation).
- **top-k sampling**: in language models, truncate to the k most probable tokens and then normalize and
  sample (complements "top-p sampling" in this collection, and the two are often stacked).
- **Pre-stage of retrieval / detection NMS**: take the top-k candidates by score, then apply IoU suppression.
