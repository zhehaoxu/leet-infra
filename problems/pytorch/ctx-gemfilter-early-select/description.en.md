# Context Compression: GemFilter Early-Layer Attention Input Selection

## Problem

GemFilter's idea (2409.17422): instead of computing full attention at every layer, use the QK inner products of **layer 13** alone to decide which input tokens are relevant to the current query, drop everything else, and let all later layers attend to the surviving KV only. The scoring function `standard_dis_index` has four steps: the QK inner product after GQA expansion, looking at **only the last** query position, **summing** scores over heads (not averaging), and a 1-D mean-pooling smoothing before the top-k. Each step has a natural "wrong" alternative that produces a different top-k. Given the selection layer's K, Q (post-RoPE) and the GQA group count, reproduce the scoring and the selection.

## Task

Implement `solve(key_states, query_states, num_key_value_groups, topk=1024, kernel_size=5, norm=1.0)`:

- `key_states`: nested numeric `[kv_heads][n][d]`, the selection layer's K (post-RoPE); `n >= 1`, `d >= 1`;
- `query_states`: nested numeric `[num_heads][q_len][d]`, the same layer's Q; **only the last query position takes part in scoring**;
- `num_key_value_groups`: GQA group count $g \ge 1$, with `num_heads == kv_heads * g`; K is first copied to every head via `repeat_kv` (**each KV head occupies g consecutive query heads**, i.e. `repeat_interleave` semantics, not `repeat` tiling);
- `topk`: an `int >= 1`; the effective count is `k_eff = min(topk, n)`;
- `kernel_size`: a positive odd number (upstream default 5); mean pooling with `padding = kernel_size // 2`, `stride = 1`;
- `norm`: a positive real (upstream default 1); when `norm != 1`, `distances` are additionally divided by `norm` while `indices` stay unchanged.

Return a `dict` with keys `indices`, `distances`, `pooled`.

## Rules

1. **GQA expansion**: head $h = c \cdot g + t$ ($0 \le t < g$) uses `key_states[c]` — KV head $c$ occupies $g$ consecutive query heads.
2. **Last query**: `q_last[h] = query_states[h][-1]`, shape `[num_heads][d]`.
3. **Inner product and head sum**: $s_j = \sum_{h=0}^{H-1} \langle q_{last}[h],\ K_{expanded}[h][j] \rangle$ — a length-$n$ score sequence (**summed**, not averaged; the head count changes the scale).
4. **Mean pooling**: `pooled[j] = (Σ_{t=j-r}^{j+r} s_t) / kernel_size`, out-of-range indices contribute **zeros** yet the **denominator stays kernel_size** (`avg_pool1d`'s default `count_include_pad=True`); $r = \lfloor kernel\_size / 2 \rfloor$.
5. **Top-k**: take the `k_eff` largest `pooled` values in descending order; **ties broken by ascending index**. `indices` follows that order; `distances` are the matching `pooled` values (divided by `norm` when `norm != 1`); `pooled` returns the full sequence unchanged.

## Example

```text
2 KV heads (g=2, 4 query heads in total), n=6, d=2:
  K0 = [[1,0],[0,1],[1,1],[2,0],[0,2],[1,-1]]
  K1 = [[0,1],[1,1],[2,0],[0,2],[1,0],[1,1]]
  Last query position of Q (4 heads) = [(1,1), (2,1), (0,1), (1,0)]

interleave expansion: heads 0,1 use K0 (paired with (1,1), (2,1)); heads 2,3 use K1 (with (0,1), (1,0))
  per-head inner products:
    head0 (K0,(1,1)): [1, 1, 2, 2, 2, 0]
    head1 (K0,(2,1)): [2, 1, 3, 4, 2, 1]
    head2 (K1,(0,1)): [1, 1, 0, 2, 0, 1]
    head3 (K1,(1,0)): [0, 1, 2, 0, 1, 1]
  head sum sums = [4, 4, 7, 8, 5, 3]
mean pooling with kernel_size=3 (zero padding):
  pooled = [8/3, 5.0, 19/3, 20/3, 16/3, 8/3]
topk=2 → indices = [3, 2], distances = [20/3, 19/3]
topk=10 → k_eff=6 takes all, indices = [3, 2, 4, 1, 0, 5] (j0 and j5 tie at 8/3, j0 first)
norm=2 → indices unchanged, distances = [10/3, 19/6]
```

Writing `repeat` instead of interleave (heads 0,2 on K0 and heads 1,3 on K1) gives `sums = [3,5,9,8,7,3]` and a completely different top-k — this is the core trap of the problem.

## Constraints

- Every value is a finite real; all heads share the token count, all vectors share the dimension, and Q and K share `d`.
- `num_heads == kv_heads * num_key_value_groups`.
- `topk >= 1`; `kernel_size` a positive odd number; `norm > 0`.
- Inputs must not be modified; all invalid inputs raise `ValueError`.

## Boundaries with Neighboring Problems

- `ctx-llmlingua2-word-compress` / `ctx-longllmlingua-doc-budget` compress **words and documents of the prompt** (classification probabilities / document PPL); this problem drops **prefill input tokens** (early-layer attention scores), with the score computed once at the selection layer and reused everywhere.
- `mm-fastv-visual-token-prune` prunes visual tokens **layer by layer from the last layer**; this problem selects text input tokens **once at an early layer**.
- `kv-snapkv-pooling` pools the **KV cache** (decoding time, over an observation window); this problem pools **input token scores** (prefill time, no cache).
