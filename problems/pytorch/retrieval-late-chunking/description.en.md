# Retrieval: Late Chunking and Parent-Child Expansion

## Problem

**Late chunking** (Günther et al., 2024) first lets a long-context encoder embed the whole document into contextualized token vectors and only **then** pools per chunk. Every chunk vector therefore carries cross-chunk context, mitigating the semantic break caused by "split first, encode each chunk independently". This task does not run the encoder: the inputs are already-encoded token vectors, an attention mask and child-chunk boundaries. It pools them inside the mask to obtain chunk vectors, then implements retrieval-side **parent-child expansion** — retrieval hits fine-grained child chunks, but the output deduplicates and returns the parent block that each hit belongs to (a run of `parent_size` adjacent children), so the generation stage receives fuller context.

Two conventions decide correctness: the denominator of mean pooling is the **number of valid tokens**, not the interval length; and a parent vector must be **pooled again from tokens** over the merged interval, not averaged from child vectors (the two differ when child lengths differ). When child windows overlap, adjacent parent intervals may overlap as well.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- **Inputs**: `token_emb` is an \((L,D)\) floating tensor; `attention_mask` is \((L,)\) with non-zero meaning valid; `chunk_bounds` is \((C,2)\) integer, each row \([s,e)\) a child chunk, given in non-decreasing \(s\), allowing overlaps, gaps, zero-length chunks (\(s=e\)) and short tail chunks, but requiring \(0\le s\le e\le L\); `hit_chunks` is a \((H,)\) integer tensor of hit indices, duplicates and emptiness allowed; `parent_size` is an integer at least 1; `normalize` defaults to True.
- **Chunk pooling**: the vector of chunk \(c\) is
  \[ v_c=\frac{\sum_{t=s_c}^{e_c-1} m_t\,h_t}{\sum_{t=s_c}^{e_c-1} m_t}, \]
  where \(h_t\) is token vector \(t\) and \(m_t\in\{0,1\}\) is the mask. When the denominator is 0 (a zero-length chunk or one fully masked out), \(v_c\) is all-zero and no error is raised. With `normalize=True` each \(v_c\) is then L2-normalized row by row, with zero vectors staying 0; with `normalize=False` the raw mean is returned.
- **Parents**: child \(c\) belongs to parent \(p=c\,//\,\text{parent\_size}\), i.e. every `parent_size` adjacent children form one group. A parent interval runs from the **first child's start to the last child's end** (half-open). Its vector is **recomputed from tokens** over that interval with the same mask-mean convention (not an average of child vectors), and obeys `normalize` as well.
- **Hit deduplication**: only hit parents are emitted, in ascending \(p\) (i.e. original child order); repeated hits inside one parent occupy a single row. The function also returns a mapping from every hit child to its parent row, preserving the **input order and duplicates** of `hit_chunks`.
- **Overlap**: child boundaries may overlap (sliding windows), so adjacent parent intervals may overlap too; each parent is pooled independently — do **not** union or re-merge overlapping parent intervals.
- **Empty hits**: with \(H=0\) the parent bounds have shape \((0,2)\), parent vectors \((0,D)\) and the mapping is empty; `chunk_vecs` is still returned in full.
- **Error contract**: `token_emb` not 2-D or not floating, a wrong `attention_mask` length, `chunk_bounds` not shaped \((C,2)\) or not integral, an interval outside \([0,L]\) or with \(s>e\), unsorted `chunk_bounds` starts, `hit_chunks` not 1-D/integral/inside \([0,C)\), and `parent_size<1` all raise `ValueError`.
- **dtypes**: chunk and parent vectors follow the floating dtype of `token_emb`; bounds and mapping are always int64. Fully deterministic, no RNG calls.

## Example

`token_emb` (7 tokens, 2 dimensions), the mask and 4 child chunks:

```
t:      0        1        2        3        4        5        6
emb:  [1,0]   [0,2]   [1,1]   [2,0]   [0,1]   [1,3]   [0,0]
mask:   1        1        0        1        1        1        0
bounds: [0,3) [2,5) [4,6) [5,7)     parent_size = 2
hits:   [2, 0, 3, 2]
```

Chunk pooling (only tokens with a non-zero mask, then L2 normalization):

| chunk | interval | valid tokens | mean | normalized chunk vector |
|---|---|---|---|---|
| 0 | [0,3) | 0, 1 | [0.5, 1] | [0.4472, 0.8944] |
| 1 | [2,5) | 3, 4 | [1, 0.5] | [0.8944, 0.4472] |
| 2 | [4,6) | 4, 5 | [0.5, 2] | [0.2425, 0.9701] |
| 3 | [5,7) | 5 | [1, 3] | [0.3162, 0.9487] |

The hits `[2,0,3,2]` map to parent ids \(2//2=1,\ 0//2=0,\ 3//2=1,\ 2//2=1\), so after deduplication and ascending order only parents 0 and 1 remain:

- parent 0 covers children 0 and 1, interval \([0,5)\), valid tokens 0, 1, 3, 4, mean \([0.75,0.75]\), normalized \([0.7071,0.7071]\);
- parent 1 covers children 2 and 3, interval \([4,7)\), valid tokens 4, 5, mean \([0.5,2]\), normalized \([0.2425,0.9701]\).

Hence `parent_bounds = [[0,5],[4,7]]` (the two intervals overlap at token 4) and `hit_to_parent = [1,0,1,1]`. Four hits produce only two parents, the repeated child 2 is counted once, and mapping entries 0, 2 and 3 all point at parent row 1.

## Constraints

- Return a four-tuple: all chunk vectors, token intervals of the hit parents, corresponding parent vectors, and the hit-child to parent-row mapping.
- Bounds and mapping are judged with `torch.equal`; floating values use tolerance \(10^{-5}\) for float32 and \(10^{-12}\) for float64.
- \(0\le s\le e\le L\), `parent_size` \(\ge 1\); \(L=0\), \(C=0\) and empty hits are legal.
- CPU, deterministic, no RNG; the mean denominator must be the valid-token count, and fully masked or zero-length chunks must emit an explicit 0.
