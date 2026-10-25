# In Depth: Selective Activation Recomputation

## Source
Korthikanti et al. 2022, *Reducing Activation Recomputation in Large Transformer Models* (arXiv:2205.05198). The paper proposes sequence parallelism and **selective activation recomputation**: recompute only the element-wise operators that "occupy little memory and are cheap to recompute" (Dropout / LayerNorm, etc.) while keeping the outputs of expensive GEMMs such as $SQK$ and $S^2V$, thereby recomputing less at the same memory. This problem abstracts it into a knapsack model of "minimum recomputation cost under a memory budget and a minimum keep count" — the model itself is a combinatorial optimization defined by the statement; the source provides only the motivation and the core idea "selective = keep some, recompute the rest".

## Key Points and Pitfalls
1. **Translate into a knapsack**: minimizing "recomputation cost" = maximizing "kept cost", because $\sum_{i\notin K}c_i = \sum_i c_i - \sum_{i\in K}c_i$ is a constant difference. It therefore becomes: maximize $\sum_{i\in K}\texttt{cost}_i$ subject to $\sum_{i\in K}\texttt{mem}_i\le B$ and $|K|\ge$ `min_keep`.
2. **Two-constraint DP**: memory must stay within budget (the weight dimension) and the keep count has a lower bound (the count dimension). So `dp[j][m]` expands over both "number of kept blocks $j$" and "memory used $m$"; take the optimum over $j\ge$`min_keep`. An ordinary knapsack over memory alone would miss `min_keep`.
3. **`min_keep` genuinely binds**: when the highest-`cost` block has a large `mem` and eats most of the budget, greedily "keeping only the expensive ones" leaves $|K|<\texttt{min\_keep}$ and forces cheap small blocks in — `check_min_keep_binds` tests exactly this.
4. **0/1 update order**: each block can be kept at most once, so both $j$ and $m$ must be iterated in **reverse** to avoid counting the same block twice in one round.
5. **Infeasibility detection**: when the `min_keep` smallest `mem` already sum to $>B$ (or `min_keep>N`) there is no solution and `ValueError` is raised — detect it via a DP table in which all results are still at the initial negative value; remember `min_keep=0` is always feasible (the empty set).

## Cross-check Approach
The reference solution is an $(N\times N\times \texttt{budget})$ knapsack DP; the grader's `_oracle` **brute-forces all combinations** for cases with $N\le 12$ (`itertools.combinations` layered by keep count), cross-validating the minimum recomputation cost completely independently. Additional assertions cover "budget large enough → recompute 0", "`min_keep` crowding out", and "`min_keep>N`/budget too small → ValueError".

## Complexity
DP \(O(N^2\cdot \texttt{budget})\) ($N\le50$, budget$\le2000$); the brute-force oracle is \(O(2^N)\), used only for small-$N$ cross-checks.
