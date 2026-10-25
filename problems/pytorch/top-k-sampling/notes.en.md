# Solution Notes · Top-k Sampling

## Core idea

In the full vocabulary distribution produced by softmax, the overwhelming majority of tokens have
extremely low probability (the long tail). Sampling directly from the full distribution lets the
accumulated long-tail mass make the model occasionally emit meaningless words. Top-k applies a
**hard cutoff**: keep only the `k` most likely tokens, discard the rest, then normalize those `k`
and sample from them.

## Why selecting by logit == selecting by probability

softmax is monotonic: $p_i > p_j \iff \text{logit}_i > \text{logit}_j$. So "the k largest logits"
and "the k largest probabilities" are the same set. Taking the k-th largest logit as a threshold
`thr`, `logits >= thr` is the keep mask. (In practice the two are fully equivalent when there are no
ties; with ties, the threshold approach here keeps all tied entries, and the test data uses random
float32 values that essentially never tie.)

## Inverse-transform sampling (inverse-CDF)

To map "a continuous uniform random number u" onto "draw one discrete token according to the filtered
distribution": compute the cumulative sum CDF of `filtered` and take the first position i with
$\text{CDF}[i] \ge u$. Because the CDF climbs monotonically from 0 to 1, every u falling in
$[\text{CDF}[i-1],\text{CDF}[i])$ selects i, with probability exactly $\text{filtered}[i]$ — precisely
the distribution we want. Passing u as an explicit argument lets the judge reproduce results bit for
bit and avoids flakiness from a true RNG.

## Comparison with top-p / min-p

| Method | Truncation criterion | Relation to confidence |
|--------|----------------------|------------------------|
| top-k | fixed count k | independent (always keeps k) |
| top-p | smallest prefix whose cumulative mass reaches p | dependent (nucleus is small when peaked) |
| min-p | floor at p·p_max relative to the top probability | dependent |

top-k is simple but takes a "one-size-fits-all" cut: when the model is very confident it still forces k
noisy candidates, and when it is confused it may cut words that should have been kept. That is why the
community treats it as legacy and recommends top-p/min-p combinations (llama.cpp defaults top-k to 40).
