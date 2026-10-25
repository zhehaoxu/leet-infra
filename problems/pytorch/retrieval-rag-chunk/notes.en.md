# Walkthrough: RAG Overlapping Sliding-Window Chunking

## Source
LangChain `text_splitter` (the `chunk_size`+`chunk_overlap` of `CharacterTextSplitter`/`RecursiveCharacterTextSplitter`) and LlamaIndex's window/sentence sliding-window splitters. Standard preprocessing when building a RAG index: keep `overlap` overlapping tokens between adjacent chunks to mitigate the recall loss of "the key sentence got cut in half".

## Core and pitfalls
1. **The stride is `chunk_size - overlap`, not `chunk_size`**: the larger the overlap, the smaller the stride, the more chunks, the higher the redundancy. With `overlap=0` it degenerates into a hard split without overlap.
2. **A chunk is emitted when "start < num_tokens", not when "start+chunk_size ≤ num_tokens"**: the tail produces one chunk that is mostly out of range and padded with `-1` — a commonly missed cut (using `while start+cs<=n` would drop that trailing chunk).
3. **Fixed width `chunk_size`**: every row has the same width, out-of-range positions get the `-1` sentinel; you cannot return a ragged list.
4. `step≥1` is guaranteed by `overlap<chunk_size`; otherwise the start would never advance and the loop would hang.
5. The index matrix only describes "which tokens go into which chunk"; it is unrelated to the actual text.

## Comparison setup
The reference solution uses `starts=arange(0,num_tokens,step)`, broadcasts `starts[:,None]+arange(cs)[None,:]`, then `where(idx<num_tokens, idx, -1)`. The judge's `_oracle` is pure Python: `s=0; while s<num_tokens: row=[s+j if s+j<num_tokens else -1 for j in range(cs)]; s+=step` (different structure, elementwise equivalent). Additional assertions: "with overlap=0 the split is hard, no overlap", "the valid indices cover every position of 0..N-1 at least once (overlap loses no token)", "shape=(number of chunks, chunk_size) and values ∈ {-1}∪[0,N)".

## Complexity
\(O(\lceil \text{num\_tokens}/\text{step}\rceil \cdot \text{chunk\_size})\).
