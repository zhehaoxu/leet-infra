# Chain of Agents Sequential Relay Ledger

Chain of Agents (CoA, arXiv 2406.02818) handles long text by **relaying** rather than
cramming: the source is split into chunks, and worker after worker reads its own chunk
while rewriting the previous worker's Communication Unit (CU) into a new one; finally the
manager answers from the **last CU alone**.

The paper and the public implementation (`rudrankriyam/Chain-of-Agents`) disagree on one
point: the public manager concatenates **all** worker outputs, while §3.2 formula (2)
reads `Response = LLM_M(I_M, CU_l, q)`, and footnote 1 explicitly reports that feeding
the manager **all** `CU_i` hurts performance. This problem follows the paper.

Here CoA is reduced to a pure function: LLM calls become the `worker_out` / `manager_out`
arguments, keeping only **chunking, relaying, bookkeeping**. Implement
`solve(source, query, worker_out, manager_out, k=8, max_cu=None, window=None, instr_words=0, manager_instr_words=0, order="left-to-right")`.

## Input

- `source`: the long text, split into paragraphs on blank lines (`\n\n`).
- `query`: the user question, visible to every worker and to the manager.
- `worker_out`: the raw generation of worker i, one entry per chunk (the length must
  equal the number of chunks).
- `manager_out`: the manager generation, used verbatim as `answer` (no strip).
- `k`: the chunking window, i.e. the word cap per chunk.
- `max_cu`: word cap for a single CU; `None` means no truncation.
- `window`: the agent window; when given, checks whether `I_W + CU_{i-1} + c_i + q` fits.
- `instr_words` / `manager_instr_words`: how many words `I_W` / `I_M` take.
- `order`: `"left-to-right"` (default) or `"right-to-left"`.

## Output

```
{
  "num_chunks": number of chunks,
  "chunks": [{"idx", "text", "words", "start", "end"}],
  "order": processing order (list of chunk indices),
  "cus": CU text after each hop (in processing order),
  "workers": [{"chunk_idx", "chunk_words", "cu_in_words", "raw_words",
               "cu_out_words", "input_words", "truncated", "reused"}],
  "cu_l": the last CU,
  "manager_cus": the CUs the manager actually saw,
  "manager_input_words": manager input word count,
  "answer": manager output,
}
```

`start` / `end` in `chunks` are the chunk's **word-index span** in the source
(half-open), independent of processing order.

## Chunking

Split on blank lines, then pour paragraph by paragraph into the current chunk:

- paragraphs that are pure whitespace are skipped;
- in-paragraph newlines, runs of spaces and edge whitespace are all eaten by `split()`,
  and chunk text is rejoined with single spaces;
- if adding this paragraph would exceed `k` words, flush the pending tail first, then
  handle the paragraph;
- a paragraph longer than `k` words is hard-cut into full-window chunks, and the leftover
  tail becomes the start of the next chunk;
- at the end, flush whatever remains; **no empty chunk is ever produced**.

## Relay

`CU_0` is the empty string, then chunks are processed in `order`:

1. Compute this hop's input word count `I_W + CU_{i-1} + c_i + q`; if `window` was given
   and the total exceeds it, raise.
2. Take the chunk's raw generation: zero words means "nothing relevant in this chunk", so
   **keep the previous CU** (in §3.1, W3 writes the answer into a new CU while preserving
   the old information — it never blanks it out).
3. Otherwise, if `max_cu` was given and is exceeded, truncate by words (truncation goes
   through tokenization and normalizes whitespace).
4. Otherwise use the raw text as the new CU — **no whitespace normalization at all**.

## Manager and answer

`manager_cus` holds exactly one entry: `[CU_l]`. `manager_input_words = I_M + |CU_l| + |q|`,
independent of the source length, the chunk count and the intermediate CUs. If `window`
was given and the total exceeds it, raise. `answer` is `manager_out` **verbatim** (no
stripping of surrounding whitespace).

## Example

```
source = "The cat sat.\n\nThe dog ran fast.\n\nA bird flew away."
query  = "who ran", k = 4
worker_out = ["cat", "dog ran", "bird flew"]
returns: num_chunks=3, cus=["cat", "dog ran", "bird flew"],
         cu_l="bird flew", manager_cus=["bird flew"], answer="the dog"
```

## Constraints

- Pure function: no network, no file I/O, no mutation of the arguments.
- `k < 1`, a non-positive `max_cu`, an unknown `order`, a `worker_out` length that does
  not match the chunk count, and a source that yields no chunk (empty or all whitespace)
  all raise `ValueError`.
- With `window` given, a worker or manager input that exceeds the window also raises
  `ValueError`.
