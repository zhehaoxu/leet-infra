# notes (ag-coa-relay-ledger)

## Sources

- Paper: Chain of Agents (arXiv 2406.02818) §3.1 formula (1)
  `CU_i = LLM_{W_i}(I_W, CU_{i-1}, c_i, q)`, §3.2 formula (2)
  `Response = LLM_M(I_M, CU_l, q)`, the Algorithm 1 pseudo-code, and the worker / manager prompt
  templates in Appendix B. This is the only authority for the **protocol**.
- The paper has **no** official code repository (neither the arXiv abs page nor the body links
  one). Chunking is only described as "each chunk shorter than k, ⌈n/k⌉ chunks in total"; token vs
  word, overlap, and boundary alignment are all unspecified, so the chunking rules come from the
  public implementation `rudrankriyam/Chain-of-Agents` main, `utils.py::split_into_chunks`
  (lines 43-88).
- Only structure and bookkeeping are ported: LLM calls become the `worker_out` / `manager_out`
  arguments, and `k` / `max_cu` / `window` are the three cap parameters.

## Paper semantics (verified one by one)

1. **The manager consumes only the last CU**: §3.2 says "the manager agent consumes accumulated
   knowledge from last worker `CU_l`", and the Appendix B manager prompt reads "the source text is
   too long and has been summarized". Footnote 1 goes further: feeding the manager **all** `CU_i`
   hurts because the CUs conflict. Hence `manager_cus` is always a single-element list and
   `manager_input_words` is independent of the chunk count and the source length.
2. **`CU_0` is the empty string**: Algorithm 1's `Initialize CU_0 ← empty string`. So the first
   worker's `cu_in_words` is necessarily 0, which distinguishes "start from zero" from "start from
   the first generation".
3. **Only the previous CU is passed**: worker i receives `I_W + CU_{i-1} + c_i + q`, never the full
   history. A long intermediate CU affects only the next hop and never accumulates into the manager.
4. **No relevant information means rewrite-and-keep, not blank out**: §3.1 describes W3, finding
   nothing in its chunk, as "directly rewrites `CU_2` by putting the correct answer as the first
   token of `CU_3` without adding any unrelated information" — the old information survives. The
   problem encodes this as "empty generation → keep the previous CU".
5. **Sequential, left-to-right by default**: §5.5 states left-to-right performs best (compared
   against right-to-left and permutation). Reversing changes only the worker processing order; the
   chunk table keeps source order, and `cu_l` becomes the CU of the source's **first** chunk.
6. **`k` is both the chunk cap and the agent window**: Algorithm 1 requires each chunk to be
   shorter than `k`, and a worker's input `CU_{i-1} + c_i + q + I_W` must also fit in `k`. The
   problem splits these into `k` (chunking) and `window` (optional check, off by default), because
   the paper says nothing at all about what to do when a CU is too long, and a mandatory check
   would make ordinary cases raise immediately.

## Semantics supplied by the public implementation

7. **Paragraphs first**: split on `\n\n`, then pour word counts into the current chunk; paragraphs
   that are pure whitespace are skipped. Chunk boundaries therefore tend to land on paragraph
   boundaries instead of cutting sentences arbitrarily.
8. **In-paragraph whitespace is normalized**: chunk text is `" ".join(words)`, so in-paragraph
   newlines, runs of spaces and edge whitespace all disappear, and the `\n\n` separators between
   paragraphs are not preserved either — the original layout is lost at this step.
9. **Over-long paragraphs are hard-cut into full-window chunks, the tail kept as a new start**:
   `while len(words) > k` emits `words[:k]` and the rest becomes `pending`. The condition is
   **strictly greater**, so the remainder always has length `1..k` and is never zero.
10. **Full blocks linger**: a flush happens only when `len(pending) + len(words) > k`, so a block
    that has just reached `k` words is not emitted right away — it waits for the next paragraph (or
    for the loop to end). The block contents and ordering therefore differ from eager flushing,
    though the chunk **count** still matches greedy filling.
11. **Over-long CUs are only truncated**: the paper gives generation settings ("Maximum generation
    token is set to 2048 for gemini-ultra and 1024 for the rest") but no CU cap and no policy for
    exceeding one. The problem's `max_cu` is an explicit engineering convention: truncate by words.
    Truncation necessarily goes through tokenization and therefore normalizes whitespace, whereas
    **no truncation returns the text verbatim** — that distinction leaves "always normalize"
    implementations nowhere to hide.
12. **`answer` is returned verbatim**: the public implementation returns
    `response.choices[0].message.content` directly, with no strip.

## Where the public implementation diverges

- `ManagerAgent.synthesize` takes **all** `worker_outputs` and joins them as
  `"Worker {i+1}: {output}"`, which contradicts formula (2) and footnote 1. This problem follows the
  paper and makes "the manager sees only `CU_l`" an explicit ledger field plus a checkpoint — a
  solution copied from the public implementation fails immediately.
- The public implementation has no CU cap, no window check and no reverse option; all three are
  conventions declared by the problem statement, not paper or code semantics, and are listed under
  "Input".

## Equivalent mutants

1. **Not skipping pure-whitespace paragraphs** (`if not para.strip(): continue` → removed): a blank
   paragraph `split()`s to `[]`, and `len(pending) + 0 > k` holds only when `pending > k`, which is
   impossible since `pending` is always ≤ `k`; the `extend([])` branch does nothing. Zero
   differences over 20,000 random sources.
2. **Hard-cut condition `>` → `>=`**: `>=` merely cuts the "exactly k words" case one step earlier
   and leaves `pending` empty; the emitted full-window chunk is identical in content and position,
   and later paragraphs group identically. Zero differences over 20,000 random sources.
3. **`num_chunks` from `len(walk)` instead of `len(chunks)`**: `walk` is `range(len(chunks))` or its
   reverse, so the lengths are always equal and the difference is unobservable.

## Mutation statistics

- 39 substantive mutants, all CAUGHT, covering: chunking (`>` / `>=`, hard cut, tail drop, chunk
  text joining, `split()` semantics), word spans (`end` offset, `pos` step), processing order (no
  reverse, rotation), the CU chain (blanking, `== 0` / `<= 1`, truncation threshold, truncating by
  characters, always normalizing, assignment order, recording the raw generation), input counts
  (missing query, missing instruction, counting characters), the manager (feeding all CUs, counting
  the source, taking the first CU as `cu_l`, missing instruction words, dropping the window check),
  validation (`k` / `max_cu` bounds, unknown order, arity, empty source, window `>=`), and ledger
  fields (truncated/reused flags, `chunk_idx`, `words` by characters, `chunk_words` hard-wired to
  `k`, `cu_out_words` from the raw generation, `cu_in_words` after the update, `answer` stripped).
- 3 equivalents (above): two backed by 20,000 randomized cases plus a structural argument, one a
  length identity.

## Engineering lessons

- **"The paper doesn't say" must be declared, not disguised as official semantics**: three items
  here (`max_cu` truncation, the `window` check, the reverse option) are absent from the paper.
  Listing them under "Input" and naming them in the notes is far more honest than slipping them
  into the quirks list, where a reviewer would assume they were read out of the code.
- **When the public implementation contradicts the paper, the conflict is the best checkpoint**:
  "feed the manager all CUs" is easy to get wrong if you only *know* it; turning it into a
  `manager_cus` field plus the assertion "longer source, same CU_l → same manager input" makes a
  copy-pasted solution fail hard.
- **Don't reuse the chunking window as the agent window**: the first version had only `k` for both,
  so a chunk of exactly `k` words plus a one-word query always overflowed and ordinary cases all
  turned into errors. Splitting into `k` plus an optional `window` made it usable.
- **Prove equivalent mutants by exhaustion, not by reading**: "skip blank paragraphs" and "hard cut
  with `>=`" both look behaviour-changing at first glance; only after 20,000 randomized comparisons
  with zero differences were they recorded as equivalent. Conversely, "never emit an empty chunk"
  was originally judged equivalent too, and a single "empty source with one generation" case turned
  it into CAUGHT — the difference is whether an **earlier check masks it**.
- **`_bad` only accepts `ValueError`**: changing the `k` check to raise `TypeError` is still caught
  by `_bad` (via the `except Exception` branch reporting "wrong error"), but deleting the check
  outright sends `while len(words) > 0` into an infinite loop and kills the mutation process. Before
  writing a mutant, ask whether removing that check leaves the control flow able to terminate.
