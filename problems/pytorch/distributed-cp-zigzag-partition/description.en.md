# Context Parallelism: Zigzag Token Sharding and Causal Load

## Task

Simulate context-parallel (CP) token sharding in a single process: pad a sequence to a multiple of $2C$, cut it into $2C$ chunks, hand them to $C$ ranks under one of two layouts, and compute each rank's causal-attention load.

- **contiguous**: rank $r$ takes chunks $2r$ and $2r+1$ (equivalently, one long contiguous block).
- **zigzag**: rank $r$ takes chunk $r$ together with chunk $2C-1-r$ (front chunk then back chunk).

Under a causal mask, later tokens attend to more keys, so contiguous sharding is inherently unbalanced; zigzag pairs the back half symmetrically onto the early ranks and, with no padding, makes every rank's query-key pair count **exactly equal**. This problem only covers the layout and the count, not the attention values.

## Inputs

- `seq_len`: Python `int`, `1 <= seq_len <= 65536`.
- `cp_degree`: Python `int`, `1 <= cp_degree <= 256`.
- `layout`: `"contiguous"` or `"zigzag"`, case-sensitive.

A `bool` posing as `int`, a float, an out-of-range value, a misspelled or non-string `layout` — all raise `ValueError`.

## Exact rules

Let $C=\texttt{cp\_degree}$, alignment $A=2C$, padded length

$$
P=\left\lceil \frac{N}{A}\right\rceil A,
$$

chunk length $K=P/(2C)$, and local length $2K$ per rank. The padded sequence is cut into $2C$ chunks; chunk $c$ covers original indices $[cK,\,(c+1)K)$, and positions $\ge N$ are padding.

**Ownership.**

- `contiguous`: rank $r$'s first chunk is $2r$, second is $2r+1$.
- `zigzag`: rank $r$'s first chunk is $r$, second is $2C-1-r$.

The local sequence is the first chunk followed by the second. Local entries carry **original indices** (0-based); padding is `-1`.

**Causal load.**

For every real token a rank owns (original index $i$), causal attention lets it see all keys at original positions $\le i$, i.e. $i+1$ of them. Hence

$$
\text{causal\_pairs}[r]=\sum_{i\in \text{rank }r,\ i<N}(i+1).
$$

Padding is not counted.

**Imbalance ratio.**

$$
\text{imbalance}=\frac{\max_r \text{causal\_pairs}[r]}{\min_r \text{causal\_pairs}[r]}.
$$

If $\min=0$: return `1.0` when every rank is 0, otherwise `float("inf")` (with $C>N$ some rank owns no real token).

## Returns

Return a 4-tuple:

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `local_indices` | `list[list[int]]`, $C\times 2K$ | Local token indices per rank, `-1` for padding |
| `local_mask` | `list[list[bool]]`, $C\times 2K$ | `True` for a real token |
| `causal_pairs` | `list[int]`, length $C$ | Causal query-key pair count per rank |
| `imbalance` | `float` | `max/min`, with the zero-minimum convention above |

## Example

```text
N = 8, C = 2, layout = "zigzag"
  P = 8, K = 2, local length 4
  rank 0: chunk 0 + chunk 3 = [0, 1, 6, 7]   load 1+2+7+8 = 18
  rank 1: chunk 1 + chunk 2 = [2, 3, 4, 5]   load 3+4+5+6 = 18
  imbalance = 1.0

N = 8, C = 2, layout = "contiguous"
  rank 0: [0, 1, 2, 3] -> 10
  rank 1: [4, 5, 6, 7] -> 26
  imbalance = 2.6

N = 6, C = 2, layout = "zigzag"
  P = 8, K = 2, chunk 3 is all padding
  rank 0: [0, 1, -1, -1] -> 3
  rank 1: [2, 3, 4, 5]   -> 18
  imbalance = 6.0
```

The first example shows zigzag's balance: rank 0 gets "first chunk + last chunk", rank 1 gets the two middle chunks, and the front-chunk load gap is exactly cancelled by the back chunk. The third shows padding breaking the balance — the tail padding lands in zigzag's back chunk and only lowers the rank that owns it.

## Boundaries against neighbouring problems

- `distributed-ring-attention` computes online softmax and whole-block skipping after KV rotation; this problem never touches attention values and only produces the static token layout and work counts.
- `distributed-sequence-parallel` is a closed-form activation-memory and communication-byte ledger; this problem has no bytes.
- `distributed-tp-split-rearrange` is Ulysses-style head-dimension splitting with all-to-all; this problem splits the token dimension for ring CP, with heads replicated on every rank.
- `quant-duquant-zigzag-permute` is a quantization zigzag permutation unrelated to CP chunking.
