# Unigram Posterior Path Sampling

## Task

Implement `solve` with arguments `text`, `vocab`, `log_scores`, `alpha`, and `uniforms`. Sample a segmentation from the posterior over all complete paths, and return its normalization constant, log probability, and uniform-consumption count. Only the Python standard library is needed.

This is an independent exercise with an **explicit mathematical model and random-number consumption protocol**. It is neither Viterbi nor a local softmax over matching tokens at each position. It does not claim to be a verified implementation of SentencePiece `Sample` or reproduce its seeded outputs.

## Inputs, constraints, and errors

- `text` is a Unicode string of length 0..12. Positions count Python Unicode **code points**, not UTF-8 bytes or grapheme clusters.
- `vocab` is a list of 0..20 strings. A token's ID is its list index. Valid items have length 1..12, are nonempty, and are unique. **An empty or duplicate item raises `ValueError`; validate vocabulary even when `text` is empty.**
- `log_scores` is an equally sized list of finite Python `float` values in [-100, 0]. They need not define normalized probabilities.
- `alpha` is finite and in [0, 4].
- `uniforms` is a list of **exactly `len(text)`** Python floats, each satisfying $0\le u<1$. Do not consult the global random generator.
- If nonempty `text` has no complete segmentation, raise `ValueError`. Apart from these specified error cases, inputs are guaranteed valid.
- Do not mutate any input, normalize strings, insert whitespace markers, or use an unknown-token fallback. Match original strings exactly. A token ID may be used repeatedly.

## Distribution over full paths

Let $\Pi$ contain all complete segmentation paths. A path $\pi=(i_1,\ldots,i_k)$ is valid exactly when concatenating its vocabulary strings produces `text`. Define:

$$
S(\pi)=\sum_{j=1}^{k}\mathrm{log\_scores}[i_j],\qquad
w(\pi)=\exp(\alpha S(\pi))
$$

$$
Z=\sum_{\pi\in\Pi}w(\pi),\qquad
\log P(\pi)=\alpha S(\pi)-\log Z
$$

When $\alpha=0$, every **complete path** has equal weight, not every local token choice. Compute the log partition stably rather than directly summing full-path weights that may underflow to zero.

## Prefix recurrence

Let $n=\mathrm{len}(text)$ and let $L[t]$ be the log partition for `text[:t]`, with $L[0]=0$. After shorter prefixes have been computed, define:

$$
C_t=\{(i,s):s=t-\mathrm{len}(\mathrm{vocab}[i])\ge0,
\ text[s:t]=\mathrm{vocab}[i],\ L[s]>-\infty\}
$$

$$
L[t]=\operatorname{LSE}_{(i,s)\in C_t}
\bigl(L[s]+\alpha\,\mathrm{log\_scores}[i]\bigr)
$$

An empty candidate set gives $L[t]=-\infty$. For a nonempty collection of finite values, stable log-sum-exp is:

$$
\operatorname{LSE}(x_1,\ldots,x_r)
=m+\log\sum_{j=1}^{r}\exp(x_j-m),\qquad m=\max_j x_j
$$

If $L[n]=-\infty$, there is no full path. Otherwise, `log_partition` is $L[n]$.

## Exact sampling protocol

1. Start at the character endpoint $t=n$ with `uniforms_used=0`.
2. Keep only candidates in $C_t$: each must match the current ending and have a reachable preceding prefix. Order them by **ascending vocabulary ID**, not length or lexicographic order.
3. Consume `uniforms[uniforms_used]` and increment the count. Consume one value even when there is only one candidate.
4. Candidate $(i,s)$ has conditional probability:

$$
p_t(i,s)=\exp\bigl(L[s]+\alpha\,\mathrm{log\_scores}[i]-L[t]\bigr)
$$

5. Accumulate probabilities in candidate order. Choose the first candidate whose cumulative probability is **strictly greater** than the current $u$. Equality belongs to the next candidate. The last candidate may absorb floating-point cumulative tail error. Do not reweight, reorder, or replace this distribution with a local softmax.
6. Record the selected ID and move to its starting position $s$. Repeat until $t=0$, then reverse the collected IDs into left-to-right order.

Ignore unused uniforms at the end of the list. Uniforms are consumed during **stepwise right-to-left sampling**, not by a single whole-path CDF draw using only the first value.

Every reachable candidate has strictly positive mathematical probability; exponential underflow must not exclude it. At `u=0`, select the first reachable candidate. Accumulate the CDF in log space with `log(0)=-inf` if needed. For `text="ab"`, `vocab=["b","a","ab"]`, `log_scores=[-100,-100,0]`, `alpha=4`, and `uniforms=[0,0]`, return `([1,0], 0.0, -800.0, 2)`.

## Return value

Return the four-item `tuple` `(ids, log_partition, log_probability, uniforms_used)`.

- `ids`: a list of Python `int`, in left-to-right order.
- `log_partition`: a Python `float` equal to $\log Z$.
- `log_probability`: a Python `float` equal to $\alpha S(\pi)-\log Z$ for the sampled path.
- `uniforms_used`: a Python `int`, equal to the number of sampling steps and to `len(ids)`.

After vocabulary validation, empty `text` returns `([], 0.0, 0.0, 0)`, representing the unique empty path. Floating-point outputs are compared with numerical tolerance; IDs, their order, and consumption counts must follow the protocol exactly.

## Example 1: Uniform over complete paths

Input:

```python
text = "aaa"
vocab = ["a", "aa"]
log_scores = [-90.0, 0.0]
alpha = 0.0
uniforms = [0.6, 0.75, 0.0]
```

Output:

```python
([1, 0], 1.0986122886681096, -1.0986122886681096, 2)
```

There are three paths: `a+a+a`, `a+aa`, and `aa+a`. At the final position, ID 0 has probability $2/3$ and ID 1 has probability $1/3$, so `0.6` chooses the final `a`. The two possibilities for the remaining `aa` are equally likely; `0.75` chooses `aa`. Reversing the selected IDs gives `[1, 0]`. The last uniform is unused.

## Example 2: Exact CDF boundary

Input:

```python
text = "aa"
vocab = ["a", "aa"]
log_scores = [0.0, 0.0]
alpha = 0.0
uniforms = [0.5, 0.0]
```

Output:

```python
([1], 0.6931471805599453, -0.6931471805599453, 1)
```

The first candidate's cumulative probability is exactly `0.5`, which is not strictly greater than `u`. The second candidate, `aa`, is selected.

## Example 3: Scores change the posterior

Input:

```python
text = "aa"
vocab = ["a", "aa"]
log_scores = [-3.0, -0.1]
alpha = 1.0
uniforms = [0.5, 0.9]
```

Output:

```python
([1], -0.09726430062146395, -0.002735699378536058, 1)
```

The full paths have weights $\exp(-6)$ and $\exp(-0.1)$. A log partition can be negative; it is not itself a path probability.

## Example 4: Empty text

Input `text=""`, `vocab=[]`, `log_scores=[]`, `alpha=1.0`, `uniforms=[]` returns `([], 0.0, 0.0, 0)`. Replacing the vocabulary with `[""]` and supplying its score must instead raise `ValueError`; the empty result must not bypass validation.
