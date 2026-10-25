# In Depth: GRPO Group-Normalized Advantage

## Source
Shao et al., *DeepSeekMath* (arXiv:2402.03300) proposed GRPO: for a group of \(G\) outputs from the same prompt, the advantage \(\hat A_i=(r_i-\text{mean})/\text{std}\) is obtained by normalizing the within-group reward mean and standard deviation, replacing PPO's critic. This problem implements only this "reward → advantage" step.

## Core idea and pitfalls
1. **Normalize per group, not globally**: after reshaping to \((N/G, G)\), compute mean/std along the group axis (dim=1); each prompt is one group, independent of the others. Global normalization would be wrong.
2. **Population standard deviation, ddof=0**: `std` divides by \(G\) (`unbiased=False`), consistent with most GRPO implementations; using \(G-1\) gives different values.
3. **Consecutive grouping**: in the flat rewards, the first \(G\) entries are group 0 and the next \(G\) are group 1; it is not interleaved or random.
4. **Zero-variance degeneration**: when all rewards in a group are equal, std=0 (GRPO often adds a tiny ε to avoid division by zero); the examples given here have std>0, the judge uses the strict \((r-\text{mean})/\text{std}\), and the tests also guarantee nonzero-variance groups.

## Cross-check methodology
The reference solution reshapes with `view(-1,G)` and then vectorizes `(x-mean)/std(unbiased=False)`. The judging `_oracle` computes, group by group in pure Python, the mean, the square root of the population variance, and the element-wise subtract-then-divide (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. There are also assertions for "each group's mean is zeroed", "each group's sample variance ≈1", and "groups are independent".

## Complexity
\(O(N)\).
