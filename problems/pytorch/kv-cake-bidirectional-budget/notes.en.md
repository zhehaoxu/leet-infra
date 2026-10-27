# Solution Notes · CAKE

## Why not a fixed pyramid

`kv-pyramid-layer-budget` slices the budget with one fixed formula (more shallow, less deep). CAKE starts from the observation that **layer preference is model-dependent**: a formula that happens to fit model A will starve the middle layers of model B—exactly the layers that carry the retrieval load. So step one is to "measure": use each layer's total accumulated attention mass as the preference proxy, then slice by share.

Total mass rather than "largest column" or "entropy" is used because it directly reflects how busy that layer's cache is and is naturally additive: the global budget is exactly the sum of per-layer budgets, so slicing needs no extra normalization.

## floor plus largest remainder

Budgets are integers. Take the `floor` as a lower bound, then hand out the remaining slots by **descending fractional part**—the classical **largest remainder method** (Hamilton apportionment). It guarantees \(\sum b_l=\min(B, L_cL)\) and gives the first extra slot to whoever was rounded down the most.

`round` is not acceptable: it does not preserve the total and its behaviour at `.5` depends on the rounding mode. First-come-first-served is not acceptable either: it systematically favours small layer indices and destroys proportionality.

Ties go to the **smaller layer index**, consistent with the rest of the library, so the result never depends on the backend.

## Why cascading means "add, in edge order"

CAKE's cascading is essentially "layers must not decide in isolation". We model dependencies as directed edges, where `(src, dst)` means dst's selection should reference src's evidence. The implementation is simply `w[dst] += w[src]`:

- **Add, don't overwrite**: when several edges point at one layer, the evidence of all sources stacks;
- **Follow edge order**: with the chain `(0,1),(1,2)`, `w[1]` already contains `w[0]` when the second edge is processed, so layer 2 inherits layer 0 indirectly—that is the cascade;
- The price is potential double counting (layer 0 evidence reaches layer 2 via layer 1 and is effectively amplified), which in a real system encodes the assumption that skip-layer dependencies matter more. It is a design trade-off, not a defect.

## dst only

Propagation modifies the destination only. If the source were modified too, every edge would contaminate both ends, the result would no longer depend on direction, and forward and reverse edges would degenerate into the same operation—destroying exactly what this problem tests.

## Division of labour with top-k

The difficulty here is **budgeting** and **propagation**, not scoring. Scores are given as precomputed accumulated importance and the final selection is still the standard "descending, ties to the smaller index" top-k. Separating the two keeps the judge focused on integer apportionment and graph traversal.

## Complexity

Apportionment is \(O(L_c\log L_c)\) (one sort for the remainder order), propagation is \(O(|E|L)\), selection is \(O(L_cL\log L)\). Space is \(O(L_cL)\).

## Related problems

- `kv-pyramid-layer-budget`: fixed pyramid formula plus SnapKV scoring, no adaptive apportionment.
- `kv-chunk-independent-topk`: top-k per sequence chunk, no cross-layer budget.
- `kv-fastgen-head-profiles`: unequal budgets across **heads**; this problem works across **layers**, so they are orthogonal.
