# Context Parallelism: Zigzag Layout Plan for Packed Variable-Length Sequences

## Task

Real training often **packs** several variable-length sequences into one token stream (THD layout, with `cu_seqlens` holding the segment boundaries). Context parallelism must spread this stream over $C$ ranks while satisfying two constraints: zigzag front/back pairing inside every segment, and an equal local length on every rank.

This problem produces that layout plan: pad each segment to a multiple of $2C$, cut and pair the chunks, and finally emit **which original token index belongs in each rank-local slot** (`-1` for padding), together with the padded segment boundaries.

## Inputs

- `cu_seqlens`: a `list`/`tuple` of length $B+1\ge 2$ whose entries are non-`bool` Python `int`s; `cu_seqlens[0] == 0`, strictly increasing, last entry $\le 65536$ (so every segment has length $\ge 1$).
- `cp_degree`: Python `int`, `1 <= cp_degree <= 256`.

A nonzero start, non-strict increase, negatives, wrong types, out-of-range values, or an invalid `cp_degree` all raise `ValueError`.

## Exact rules

Let $C=\texttt{cp\_degree}$, segment $b$'s length $n_b=\texttt{cu\_seqlens}[b+1]-\texttt{cu\_seqlens}[b]$, and alignment $A=2C$.

**Per-segment padding.**

$$
p_b=\left\lceil\frac{n_b}{A}\right\rceil A,\qquad P=\sum_b p_b,\qquad L=\frac{P}{C}.
$$

$P$ is always a multiple of $2C$, so $L$ is an integer. The padded boundaries are the prefix sums $q_b=\sum_{j<b}p_j$, i.e. the returned `cu_seqlens_padded` (starting at 0).

**Chunking and pairing.**

Segment $b$ is cut into $2C$ equal chunks of length $k_b=p_b/(2C)$; chunk $c$ covers offsets $[c k_b,\,(c+1)k_b)$ inside the segment. Rank $r$ takes chunk $r$ and chunk $2C-1-r$.

**Local positions.**

Segment $b$'s local base is $q_b/C$ (each segment occupies $2k_b=p_b/C$ slots on every rank). The front chunk occupies

$$
[q_b/C,\ q_b/C+k_b),
$$

and the back chunk occupies

$$
[q_b/C+k_b,\ q_b/C+2k_b).
$$

Slots hold the token's **original index** in the whole packed stream (continuous across segments, 0-based); tail padding inside a segment is `-1`.

**Padding ledger.**

- `pad_total` $=P-N$ with $N=\texttt{cu\_seqlens}[-1]$.
- `pad_between_seqs`: `True` as soon as any $p_b\ne n_b$. Padding sits at each segment's tail, so it can end up between two sequences — hence the name.

## Returns

Return a 4-tuple:

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `cu_seqlens_padded` | `list[int]`, length $B+1$ | Padded boundaries, starting at 0 |
| `rank_order` | `list[list[int]]`, $C\times L$ | Original token index per rank slot, `-1` for padding |
| `pad_total` | `int` | $P-N$ |
| `pad_between_seqs` | `bool` | Whether any segment was padded |

## Example

```text
cu_seqlens = [0, 3, 5], C = 2
  n = [3, 2] -> p = [4, 4], P = 8, L = 4, chunk = 1
  segment 0, base 0: rank0 = chunk0 + chunk3 = [0, -1]; rank1 = chunk1 + chunk2 = [1, 2]
  segment 1, base 2: rank0 = chunk0 + chunk3 = [3, -1]; rank1 = chunk1 + chunk2 = [4, -1]
  cu_seqlens_padded = [0, 4, 8]
  rank_order = [[0, -1, 3, -1],
                [1,  2, 4, -1]]
  pad_total = 3, pad_between_seqs = True

cu_seqlens = [0, 4], C = 2
  n = [4] -> p = [4], L = 2
  rank_order = [[0, 3],
                [1, 2]]
  cu_seqlens_padded = [0, 4], pad_total = 0, pad_between_seqs = False
```

In the first example segment 0 is padded by 1 and segment 1 by 2, both at their tails; rank 0's local stream ends up pairing the first chunk of each segment with the last. The second example divides evenly and degenerates to single-sequence zigzag sharding.

## Boundaries against neighbouring problems

- `distributed-cp-zigzag-partition` shards a **single fixed-length sequence** and counts causal load; this problem is a **multi-segment packed** layout plan whose core difficulty is the cross-segment slot mapping after per-segment padding.
- `distributed-ring-attention` performs KV rotation and online softmax; this problem only emits a static index plan.
- `distributed-sequence-parallel` is a closed-form activation/communication byte ledger; this problem has no bytes.
- `distributed-tp-split-rearrange` is Ulysses head-dimension splitting; this one stays on the token dimension.
