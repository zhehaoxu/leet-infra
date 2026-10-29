# Reference Approach: Prefix Partitions and Backward Sampling

## Why local softmax is wrong

An ending token is not weighted solely by its own score. Its preceding prefix may have many segmentations of different total weights, or may be unreachable. Summing all prefix paths ending in token $i$ gives:

$$
W_t(i)=\exp(L[s]+\alpha\,\mathrm{log\_scores}[i])
$$

Compute prefix log partitions first, then sample backward with probability $W_t(i)/\exp(L[t])$. An already selected suffix contributes the same multiplicative factor to every remaining candidate path, so that factor cancels when conditioning on the suffix.

Multiplying step probabilities along the selected path makes the partition terms telescope:

$$
\prod_j \exp\bigl(L[s_j]+\alpha\,\mathrm{log\_scores}[i_j]-L[t_j]\bigr)
=\exp\bigl(\alpha S(\pi)-L[n]\bigr)
$$

This is the full-path posterior, not best-path selection or a local greedy rule. For example, `aaa` with vocabulary `["a", "aa"]` has three paths when $\alpha=0$. The last token is `a` with probability $2/3$, not $1/2$.

## Implementation and numerical details

- Validate vocabulary first, including for empty text. Duplicate or empty items must not bypass validation.
- Set `L[0]=0`; unreachable prefixes have negative-infinity partitions. Enumerate ending vocabulary IDs and retain matching tokens with reachable prefixes.
- Use maximum-shifted log-sum-exp and `math.fsum` for exponential terms and the final path score. Raw weights can underflow: a length-12 path of one-character tokens with score -100 and alpha=4 has log weight -4800.
- Accumulate the CDF in log space too: combine `log_cumulative` with the next conditional log probability using LSE and compare `log_cumulative > log(u)`. For `u=0`, use `log(u)=-inf`. A positive probability below the smallest representable float then remains eligible at zero.
- For `text="ab"`, vocabulary `["b","a","ab"]`, scores `[-100,-100,0]`, and alpha=4, the final `b` has probability approximately $\exp(-800)>0$. At `u=0` it must be selected, yielding `[1,0]`, not the whole-word path `[2]`.
- Compare candidates in ascending vocabulary ID order, using strict greater-than. Even a forced step consumes one uniform. The final candidate only absorbs cumulative tail error; reverse IDs at the end.
- `log_probability` is the normalized probability of the complete path in log space, not a raw score or one step's local probability.

For $n$ text code points, vocabulary size $V$, and maximum token length $K$, including vocabulary validation and Python substring comparisons, time is $O(VK+nV(K+1)+n)$ and space is $O(V+nV+n)$. Backward sampling takes at most $n$ steps with at most $V$ candidates each.

## Independent oracle and tests

The oracle enumerates complete paths left to right without prefix DP. Unique nonempty vocabulary entries give at most one path per boundary subset, hence at most $2^{11}=2048$ paths for length 12. It stores full-path exponential weights using the standard-library `decimal` module with 80-digit local precision, so tiny positive probabilities do not also underflow in the oracle.

Filter full paths by the selected suffix, group their original weights by ending token ID, choose through a high-precision conditional CDF, and strip that ending token. This uses neither the reference's prefix recurrence nor its log CDF, and is not a single whole-path draw.

The 16 `check_*` functions cover empty/invalid inputs, global path uniformity, score effects, extreme negative scores, tiny probabilities at zero and the smallest positive uniform, ID order, CDF equality, forced-step consumption, Unicode, random enumeration, unreachable prefixes, maximum paths, immutability, and forbidden global RNG use. Submit selftest and direct eval execution both pass 16/16. Description examples were rerun; the tiny-probability example returns `([1,0], 0.0, -800.0, 2)`.

## Sources and reproduction scope

The [SentencePiece 0.2.2 source distribution](https://files.pythonhosted.org/packages/cc/33/ea3cb3839607eb175da835244a798f797f478c5ddf0e8ecdf57ea85a4c70/sentencepiece-0.2.2.tar.gz) was read in memory to inspect `sentencepiece/src/unigram_model.cc`:

- Lines 217–233, `Lattice::ForwardAlgorithm(inv_theta)`: log-sum-exp over predecessor scores and forward masses.
- Lines 540–570, `Lattice::Sample(inv_theta)`: sample predecessors backward from EOS using conditional prefix mass times token weight, then reverse the path. `inv_theta` corresponds to this problem's inverse temperature `alpha`.

Both use forward accumulation and backward conditional sampling. Native code uses `float`, internal node order, C++ `discrete_distribution`, internal RNG, and BOS sampling; this exercise uses Python floats, a log CDF, ID ordering, and explicit uniforms. The native lattice also includes unknown-token fallback. Training, normalization, whitespace metasymbols, unknown tokens, and special vocabulary entries are excluded here. API, rounding, and seed-level parity are not promised.
