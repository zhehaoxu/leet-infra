# In Depth: Late Chunking and Parent-Child Expansion

## Source

- Late chunking: Günther, Mohr, Williams, Wang, Xiao, *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, arXiv:2409.04701, 2024. The method encodes the whole document with a long-context encoder first, then pools token vectors per chunk so that chunk vectors carry cross-chunk context. This task's input is exactly the already-encoded contextual token vectors; it does not run the encoder.
- Parent-child (small-to-big / parent document retrieval) is a RAG engineering pattern: retrieve with small child chunks and return the parent block to the generation stage. LangChain's `ParentDocumentRetriever` is a common implementation, but there is no single canonical academic paper, so the precise lineage is **to be sourced**. This task judges the mechanism defined in the starter docstring.
- Neighbors: `retrieval-rag-chunk` only emits sliding-window token index matrices; `retrieval-mmr` reranks; `retrieval-colbert-maxsim` scores query tokens against document tokens with MaxSim; `retrieval-pq-codes` compresses vectors. None performs mask-aware chunk pooling or parent deduplication.

## Full derivation and index convention

**Mask-mean pooling**: chunk \(c\) covers \([s_c,e_c)\) and its vector is
\[
v_c=\frac{\sum_{t=s_c}^{e_c-1} m_t h_t}{\sum_{t=s_c}^{e_c-1} m_t},
\]
where the denominator is the **number of valid tokens** \(\sum m_t\), not the interval length \(e_c-s_c\). Measured: with token vectors \([1,0],[3,0],[5,0],[7,0]\) and mask \([1,0,1,0]\), the segment mean is \((1+5)/2=3\), not \((1+3+5+7)/4=4\). Counting masked tokens in the denominator is the most common mistake. A zero denominator (zero-length chunk or fully masked) yields an all-zero vector.

**Pool first, normalize afterwards**: `normalize=True` applies L2 normalization to the **chunk vector**, not to each token before averaging. The two generally differ: tokens \([3,0]\) and \([0,4]\) normalized individually then averaged give \([0.5,0.5]\), normalized to \([0.7071,0.7071]\); averaged raw then normalized gives \([1.5,2]\to[0.6,0.8]\). Late chunking means "the mean of contextualized tokens", so normalization must come after the mean. A zero vector stays zero after normalization.

**Parent grouping**: child \(c\)'s parent id is the integer division \(p=c\,//\,\text{parent\_size}\), so every `parent_size` adjacent children form one group and the last group may be short. The parent interval runs from the **first child's start** to the **last child's end**, i.e. \([s_{p\cdot ps},\ e_{\min((p+1)ps,C)-1})\). It is not the hit child's own interval: even if only the last child of a group is hit, the parent starts at the group's first child.

**Parent vectors must be recomputed from tokens**: a parent vector is another mask mean over the whole merged interval, not an average of child vectors. The difference is large when child lengths differ. Measured: children \([0,1)\) with token \([2,0]\) and \([1,4)\) with tokens \([0,6],[0,2],[0,2]\) have raw child means \([2,0]\) and \([0,3.3333]\), whose average is \([1,1.6667]\); the token mean over parent \([0,4)\) is \([0.5,2.5]\). Averaging child vectors gives a short chunk the same weight as a long one and is not equivalent to pooling the whole span.

**Deduplication and mapping**: every hit is first converted to a parent id \(p=h//ps\); duplicate ids are removed and parents are emitted in ascending \(p\), i.e. original child order rather than hit order. `hit_to_parent` instead follows the input order of `hit_chunks` item by item, with duplicates mapped to the same row. The example hits \([2,0,3,2]\) give parent ids \([1,0,1,1]\), deduplicated ascending to \([0,1]\), with mapping \([1,0,1,1]\): four hits produce only two parents.

**Overlap**: child boundaries may overlap as sliding windows, and parent intervals run from the group's first start to the last end, so adjacent parents may overlap too. In the example the parent intervals are \([0,5)\) and \([4,7)\), overlapping at token 4; each parent is pooled independently, so token 4 appears in both parent vectors. Do not union or re-merge parent intervals — that would change `parent_bounds` and the mapping.

**Boundary conventions**: chunks may be zero-length (\(s=e\)) and then emit all zeros; a chunk may start at token 1 (earlier tokens belong to no chunk); the last chunk may end at \(L\); `chunk_bounds` starts must be non-decreasing, while ends are arbitrary (an earlier chunk may be longer than a later one). Intervals outside \([0,L]\), \(s>e\) and unsorted starts are `ValueError`.

**Empty hits and empty documents**: with \(H=0\) the parent outputs have shapes \((0,2)\), \((0,D)\) and \((0,)\), while `chunk_vecs` is still returned; with \(C=0\), `chunk_vecs` has shape \((0,D)\) and any hit is out of range; with \(L=0\) every chunk can only be \([0,0)\) or absent. None of these may produce NaN.

## Key points and common mistakes

1. **The denominator is the valid-token count**: masked-out tokens enter neither numerator nor denominator; using the interval length spreads padding into the mean.
2. **Mean first, normalize afterwards**: normalize the chunk vector, not each token before averaging; a zero vector stays zero.
3. **Fully masked / zero-length chunks emit an explicit 0**: do not skip the row (shape would shrink) and do not divide by zero (NaN).
4. **Parent vectors are recomputed from tokens**: not an average of child vectors; with unequal child lengths the two differ substantially (measured \([1,1.6667]\) vs \([0.5,2.5]\)).
5. **A parent interval spans the whole group**: from the first child's start to the last child's end, not the hit child's own interval.
6. **Deduplicate by parent id and emit ascending**: `parent_bounds` does not follow hit order; `hit_to_parent` is what aligns with the hits.
7. **The mapping keeps input order and duplicates**: repeated hits inside one parent map to the same row; do not deduplicate `hit_to_parent`.
8. **Do not merge overlapping parents**: with sliding windows parent intervals may overlap, and each is pooled independently; a union changes the row count and intervals.
9. **Boundaries and errors**: zero-length chunks, a first chunk not starting at 0 and a chunk touching the tail are all legal; out-of-range, \(s>e\), unsorted starts, out-of-range hits and `parent_size<1` are `ValueError`.
10. **dtype**: chunk/parent vectors follow `token_emb` (float64 judging uses \(10^{-12}\)); bounds and mapping are always int64.

## Differences from neighboring problems

- **vs `retrieval-rag-chunk`**: that task only outputs the **token index matrix** of overlapping sliding windows, with no vectors, mask, pooling or parents. Here the input is already token vectors and the output is chunk vectors plus the parent structure.
- **vs `retrieval-mmr`**: MMR **reranks** a candidate document set; this task does not order documents, it expands hit children to parents.
- **vs `retrieval-colbert-maxsim` (same batch)**: that task scores query tokens against document tokens with MaxSim and contrasts repeated counting with count-once. This task pools document tokens into chunk vectors and performs hierarchical (child to parent) expansion, with no query and no cosine.
- **vs `retrieval-pq-codes`**: PQ compresses vectors into codeword indices; this task keeps floating chunk vectors and does no quantization.
- **vs `global-pooling` / `adaptive-avgpool`**: those pool over regular grids without an attention mask, variable-length spans or a parent hierarchy; here the spans come from external boundaries and the denominator changes with the mask.
- **vs `powernorm-attention-pool`**: that is pooling with learnable/attention weights; this is a deterministic 0/1 mask mean.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle` is pure Python, accumulating chunk by chunk and token by token (`while t < e`, only mask-nonzero tokens) and normalizing independently; parents are grouped by `h // parent_size` and **recomputed** through the same per-token path, with the mapping built from a dict. It is structurally different from the reference's tensor slicing plus `unsqueeze` multiplication. Coverage: the example, overlapping windows, gaps between chunks, zero-length chunks, fully masked chunks, a first chunk not at 0, tail-touching chunks, `parent_size` larger than the chunk count, `normalize=False`, a bool mask, \(L=0\), float64.
- **Explicit discriminators**: mask participation in the mean (3 rather than 4), fully masked chunks giving 0 without NaN, the three empty shapes for empty hits, overlapping parent intervals \([0,6)\) and \([5,10)\) with five hits producing only two parents, repeated hits in one parent occupying one row, ascending reordering for `parent_size=1`, a single parent spanning first start to last end, a float64 comparison at \(10^{-12}\), and several `ValueError` branches.
- **Complexity**: let \(S_c\) be the sum of child interval lengths and \(S_p\) the sum of hit-parent interval lengths; time is \(O((S_c+S_p)D + C + H)\) and space \(O((C+P)D)\). Sliding overlap can make \(S_c,S_p\) exceed \(L\), but judging sizes are tiny. Any implementation path is fine as long as the denominator, pooling order and parent recomputation conventions match.
