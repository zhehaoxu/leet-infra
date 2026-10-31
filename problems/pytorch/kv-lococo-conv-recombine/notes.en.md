# Solution Notes · LoCoCo Convolutional Recomputation Merging

## In one sentence

H2O does the "hard keep" of heavy hitters; everything else is not thrown away but convolved along the token axis, softmaxed, and soft-merged into `mem_compress` summary tokens, then mixed with the one-hot choice of representatives through `normalizer`.

## Why convolve along the token axis

There are \(n=S-\textit{keep}\) non-heavy-hitter tokens to squeeze into \(m<n\) summaries. Averaging K/V position by position loses "who is next to whom", while local attention suggests that near neighbours matter most. A kernel of 21 with `padding=10` makes every output position cover exactly its local window:

\[
c_{o,t}=\sum_{d,j}w_{o,d,j}\,x_{d,t+j-10}
\]

The first \(D\) channels of `x` are K and the last \(D\) are V, so one output channel reads keys and values at once; the `m` output channels are the `m` summary tokens. Softmax along the token axis then turns each row into a convex combination over all non-heavy-hitter tokens, which is what earns the name "merge" instead of "select again".

## Pipeline (by shape)

1. `hh_ids`: take `keep-local_len` top scores on the prefix \([0,S-\texttt{local\_len})\), union the trailing `local_len`, sort ascending by token index;
2. `res_ids`: walk the complement in token order, take the top-`m` scores, sort ascending — that order is the row order of `residual_weight`;
3. non-heavy-hitter K/V are concatenated in ascending token order into `(H, 2D, n)`, same-convolved into `(H, m, n)`, ReLU'd, softmaxed on the token axis → `conv_prob`;
4. `residual_weight` is one-hot with a single 1 per row, and `blend = residual*(1-ν) + conv_prob*ν`;
5. `merged = [heavy_hitter; blend @ non_hh]`, and the two blocks together have exactly `mem_size` rows.

## Easy to get wrong

- **Truncation, not rounding**: \(m=\lfloor \texttt{mem\_size}(1-\texttt{rate})\rfloor\) uses `int()`, so `10*0.25=2.5` becomes 2; `keep` must be recovered as `mem_size-m`, otherwise the two blocks do not add up.
- **Mask order**: upstream uses `masked_select` everywhere, so results come out in ascending token order, not score order. Emitting the heavy block in score order already disagrees with upstream on the first `keep` rows.
- **Residual row order**: `make_residual_weight` sorts before scattering, so row \(r\) of the one-hot is the \(r\)-th smallest representative token, not the \(r\)-th highest score.
- **The local tail is always kept**: `topk` runs on the prefix only with `keep-local_len` slots, and the tail is set to True directly. A high-scoring tail never takes a prefix slot — exactly the "recent window cannot be evicted" rule.
- **ReLU after the convolution**: the default is `hidden_act='relu'`, and the activation sits right after the conv inside `nn.Sequential`; skip it and negative conv outputs make the softmax weights meaningless.
- **Bias and training**: upstream `Conv1d` has a bias and `normalizer` is an `nn.Parameter`. Here the bias is treated as zero, `normalizer` is an input, and nothing is trained — the problem judges the merge convention, not the training loop.

## Complexity

Each head costs \(O(m\cdot n\cdot K\cdot 2D)\) for the convolution plus \(O(m\cdot n)\) for the softmax, then \(O(m\cdot n\cdot D)\) for the matmul; `m` is typically a fraction of `mem_size`, far below the original \(n\).

## Source

`VITA-Group/LoCoCo` (arXiv 2406.05317), `llama/memory_compressor.py`:

- L56-58 `mem_compress = int(...)`, `keep_hh_size = mem_size - mem_compress`, `local_len`;
- L69-70 `nn.Conv1d(in_size, hidden_size, kernel_size, padding=(kernel_size-1)//2)`;
- L76 `normalizer = Parameter(ones(1)*normalizer_init)`;
- L88-100 `make_residual_weight`: L94 topk, L95 sort, L96-98 scatter the one-hot;
- L110-117 the prefix topk takes `keep_hh_size-local_len`; L116-117 sets the tail to True;
- L121-132 `masked_select` collects heavy and non-heavy tokens (ascending);
- L147-152 `cat([k,v])` → rearrange → conv → softmax → `residual*(1-normalizer)+weight*normalizer`;
- L155-159 `matmul` back into K/V and `cat` with the heavy block.

Defaults live in `train.py` L87-98: `mem_size=128`, `local_len=0`, `hh_keep_rate=0.5`, `n_convlayer=1`, `kernel_size=21`, `hidden_act='relu'`, `normalizer_init=0.5`.

**Additions of this problem**: ties go to the smaller token index; `n_convlayer=1`, `expand=1` and bias=0 are fixed; fewer than `m` residuals raises an explicit error instead of letting `topk` throw a low-level one.

## Related problems

- `kv-h2o-heavy-hitters`: hard eviction only, no recomputation; this problem soft-merges the evicted part back.
- `kv-wkvd-least-squares-merge`: solves for merge coefficients by least squares; here a convolution learns the weights from a neighbourhood.
- `kv-snapkv-pooling`: pooling also compresses, but its window is chosen by observed attention and there is no learnable blend.
