# One-Hot Encoding in Depth

## Definition

Map a class index $i\in\{0,\dots,K-1\}$ to a vector of length $K$ whose $i$-th position is 1 and the rest are 0:

$$ \operatorname{onehot}(i)_k=\mathbb 1[k=i] $$

Shape $(\dots)\to(\dots,K)$: a class dimension is **appended at the end**, and each position of the index tensor is expanded into its own 0/1 vector.

## Why one-hot

- Classes are **unordered nominal quantities**; using the integers 1/2/3 directly introduces spurious magnitude and distance relations. One-hot turns each class into an orthogonal unit vector, and with an embedding lookup table it can learn an arbitrary representation.
- It is also the sparse form of cross-entropy labels (`F.cross_entropy` takes integer labels and internally picks the corresponding logit at the one position of the one-hot vector).

## Implementation & Numerical Notes

- Reference solution: `torch.zeros(*indices.shape, K, dtype=torch.long)` then `out.scatter_(-1, indices.unsqueeze(-1).long(), 1)`. `unsqueeze(-1)` aligns the indices with the newly added last dim, and `scatter_` writes a 1 into column $i$ of each all-zero row.
- **Class count $K$ convention**: with `num_classes=None` use $\max+1$ (0 included, so the minimum is $K\ge1$); when given explicitly use the given value — if $K\le\max$ indexing is out of bounds and raises, if $K>\max$ there are extra all-zero columns.
- For an empty tensor with `numel()==0`, $K=0$ and the result is empty.
- The output is integer (`long`), consistent with `F.one_hot`; for a float one-hot, call `.float()` yourself.

## Related

embedding-lookup is the equivalent efficient form of one-hot × weight matrix; gather-scatter is the underlying primitive of this implementation; the integer labels of cross-entropy can be viewed as an "implicit one-hot".
