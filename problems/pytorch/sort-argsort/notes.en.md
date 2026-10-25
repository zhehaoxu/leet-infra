# Sort & Argsort Explained

## The Division of Labor between values and indices

`sort` returns the sorted **values**, while `argsort` returns only the **position permutation**. The latter is what is truly flexible: once you have the
`indices`, you can use `gather` to reorder **any associated tensor of the same shape** in the same order — for example, when sorting boxes by confidence in
detection you move the corresponding coordinates and classes along with them; in NLP, sorting by sequence length enables packing (bucketing).

$$ \text{values} = x[\text{indices}],\quad \text{i.e. } \text{values}_i = x_{\,\text{indices}_i} $$

## Selection-Sort Implementation

This solution uses the selection method: each round takes `argmin` over the unsorted region, writes out the value and the original index, then sets it to `+inf` to remove it from the candidates —
$O(n^2)$ but semantically transparent. Internally `torch.sort` is a parallel sorting network (bitonic/merge) of depth $O(\log^2 n)$.

## Stability and Ties

A "stable sort" guarantees that equal elements keep their original relative order. `torch.sort(stable=True)` and `argsort` can control this;
`argmin` naturally takes the **smallest index**, so this implementation gives a deterministic "earlier index first" result on ties. The judge uses continuous
random numbers to avoid tie ambiguity, aligning exactly with `torch.sort`.

## Relations to Other Operators

- **top-k** (another problem in this collection) = take only the first k after sorting, with no full sort needed.
- **The inverse of argsort** (rank): `rank[argsort[i]] = i`, usable to map a "position" back to the "original location".
- Double argsort: `argsort(argsort(x))` yields each element's **rank**, a common trick for non-parametric de-ranking /
  quantile transforms.
