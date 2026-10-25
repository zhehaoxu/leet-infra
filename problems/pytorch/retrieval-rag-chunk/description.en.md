# Retrieval: RAG Overlapping Sliding-Window Chunking

## Problem

Before building the index, RAG (retrieval-augmented generation) has to split a long document into embeddable pieces (chunks). Cutting too long dilutes the semantics, cutting too short loses context, so the mainstream approach is **overlapping sliding-window chunking** (the `chunk_overlap` of LangChain's `CharacterTextSplitter`/`RecursiveCharacterTextSplitter`, LlamaIndex's window chunking): window size `chunk_size`, neighbouring chunks share `overlap` tokens, stride `step = chunk_size - overlap`. The overlap guarantees that semantics spanning a chunk boundary are not severed mid-sequence. This problem takes a document that is "already a token sequence" and splits it with overlap, returning the **token-index matrix** covered by each chunk (short positions at the tail are filled with `-1`).

## Rules

Implement `solve(num_tokens, chunk_size, overlap)`. Integer inputs: the document has `num_tokens` tokens (indices \(0\dots \text{num\_tokens}-1\)), window `chunk_size`, overlap `overlap` (\(0\le\) `overlap` \(<\) `chunk_size`). Stride \(\text{step}=\text{chunk\_size}-\text{overlap}\). Chunk starts begin at 0, one every step, and any start \(<\) `num_tokens` forms a chunk. The token index in column \(j\) of chunk \(b\) is \(\text{start}_b + j\); if that is \(\ge\) `num_tokens`, fill the position with `-1`. Return an int64 tensor of shape \((\text{number of chunks},\ \text{chunk\_size})\).

## Example

\(num\_tokens=7, chunk\_size=3, overlap=1\) (step=2, starts 0,2,4,6):

```
chunk 0: 0 1 2
chunk 2: 2 3 4
chunk 4: 4 5 6
chunk 6: 6 -1 -1     (start 6<7 forms a chunk, but 7 and 8 are out of range -> -1)
out = [[0,1,2],[2,3,4],[4,5,6],[6,-1,-1]]   shape = (4, 3)
```

## Constraints

- \(0\le\) `overlap` \(<\) `chunk_size`, hence `step ≥ 1` (so progress is never zero).
- A start \(<\) `num_tokens` yields one chunk — even if most of that chunk is out of range (like start 6 above).
- Out-of-range positions get `-1` (a sentinel), no wrap-around, and no trimming the row to a width smaller than `chunk_size`.
- Return an int64 index matrix; pure chunking only, no text content and no embeddings involved.
