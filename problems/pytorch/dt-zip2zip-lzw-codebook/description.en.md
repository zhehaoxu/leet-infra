# Context Compression: zip2zip LZW Dynamic Codebook Encoding

## Problem

zip2zip (2610.13605) observes that LLM input id sequences contain many repeated phrases, but the vocabulary is frozen after training, so new phrases cannot be added. zip2zip's fix is to attach an **LZW codebook per input**: when "current buffer + next token" forms a phrase not in the codebook, it is inserted into a **dynamic codebook** as a hyper id, and the known prefix of the buffer is emitted. The codebook is not persisted — it travels with the input (the Rust codec encodes/decodes on the fly at inference). Repeated patterns then cost one id each after their first occurrence.

Three details trip up codec implementations. First, **when the buffer is emitted vs. extended** — after an unknown phrase is inserted, you must pop the last token, emit the rest, and restart the buffer with that token. Second, **three independent caps** — a full codebook (`next_id` hitting `initial_vocab_size + max_codebook_size`) stops growing and falls back to token-by-token emission, phrases reaching `max_subtokens` are force-emitted, and `truncation` at `max_length` stops consuming input entirely (so the codebook only grows from consumed input). Third, **disabled-id interruption** — a disabled token joins no phrase and acts like a separator. Given one id sequence and all cap parameters, reproduce this encoding and return the compressed ids, attention mask, and codebook.

## Task

Implement `solve(ids, initial_vocab_size, max_codebook_size, max_subtokens, pad_token_id, disabled_ids=None, truncation=False, max_length=None, padding="do_not_pad")`:

- `ids`: non-empty `list[int]`, each element a non-negative integer **strictly less than `initial_vocab_size`** (only base-vocab ids are encoded);
- `initial_vocab_size`: integer $\ge 1$; hyper ids are numbered starting from it;
- `max_codebook_size`: integer $\ge 1$; the codebook holds at most this many phrases; `next_id` is capped at `initial_vocab_size + max_codebook_size`;
- `max_subtokens`: integer $\ge 2$, the maximum number of base tokens per phrase;
- `pad_token_id`: integer, **must be $\ge$ `initial_vocab_size + max_codebook_size`** (must not collide with any emittable id);
- `disabled_ids`: `list` or `None`, unique ids within the base vocab that never join a phrase;
- `truncation`: bool; if `True`, `max_length` must be provided;
- `max_length`: integer $\ge 1$ or `None`;
- `padding`: only `"do_not_pad"` or `"max_length"`; the latter requires `max_length`. `"longest"` is batch-level semantics and out of scope.

Return a `dict` with keys `compressed_ids`, `attention_mask`, `codebook` (hyper id → phrase, dict sorted by hyper id ascending).

## Exact rules

Maintain `buffer` (phrase being built), `codebook` (phrase tuple → hyper id), `next_id` (initialized to `initial_vocab_size`), `limit = initial_vocab_size + max_codebook_size`, and output `compressed`. Consume `ids` token by token:

1. **Disabled token**: emit and clear the pending buffer (if any), then emit this token **alone** and clear the buffer — it never enters a phrase.
2. **Regular token**: append to the buffer. Compute `known = len(buffer) == 1 or tuple(buffer) in codebook` (single tokens are known by definition).
3. **Unknown phrase**: if `next_id < limit`, insert `tuple(buffer)` with id `next_id` and increment; then in either case **pop the last token, emit the remaining buffer, clear it, and restart the buffer with that token**.
4. **Cap emission**: if the buffer length reaches `max_subtokens`, emit the whole buffer and clear it (the buffer is guaranteed to be a known phrase at that point).
5. **Flush**: after input is consumed, emit a non-empty remaining buffer.
6. **Truncation early-exit**: check `truncation and len(compressed) >= max_length` before every `emit` (skip the emission) and at the top of the loop (**break immediately, stop consuming input**) — so the codebook only grows from consumed input.
7. **Emitted value**: `buffer[0]` if `len(buffer) == 1`, else `codebook[tuple(buffer)]`.
8. **Padding**: with `"max_length"`, left-pad `pad_token_id` if shorter than `max_length`; raise `ValueError` if longer (without truncation); **the same `max_length` is both the truncation cap and the padding target**. `attention_mask` is 0 exactly where `id == pad_token_id`, else 1.

## Example

```text
ids = [0,1,0,1,0,1,2], initial_vocab_size=4, max_codebook_size=6, max_subtokens=3

i=0: buffer=[0], single token is known
i=1: buffer=[0,1] unknown -> insert 4:[0,1], emit [0], buffer=[1]
i=2: buffer=[1,0] unknown -> insert 5:[1,0], emit [1], buffer=[0]
i=3: buffer=[0,1] known
i=4: buffer=[0,1,0] unknown -> insert 6:[0,1,0], emit [0,1] (id 4), buffer=[0]
i=5: buffer=[0,1] known
i=6: buffer=[0,1,2] unknown -> insert 7:[0,1,2], emit [0,1] (id 4), buffer=[2]
flush: emit [2]

compressed_ids = [0,1,4,4,2], attention_mask all 1
codebook = {4:[0,1], 5:[1,0], 6:[0,1,0], 7:[0,1,2]}
```

Variants (same input unless noted):

- `ids=[0,1,0,1,2,2,3,3]`, `max_codebook_size=2`: the codebook freezes at `{4:[0,1], 5:[1,0]}`; later phrases are never inserted and tokens are emitted one by one → `[0,1,4,2,2,3,3]`. **A full codebook is detected by `next_id < limit` failing, not by "phrase absent"** — known `[0,1]` still emits its hyper id; only *new* phrases (like `[2,2]`) stop being absorbed.
- `truncation=True, max_length=3`: emission stops at `[0,1,4]`, the loop breaks early, and the codebook only reaches `6:[0,1,0]` (`7:[0,1,2]` is never inserted — the input was not consumed that far).
- `padding="max_length", max_length=8`: `[100,100,100,0,1,4,4,2]` with mask `[0,0,0,1,1,1,1,1]`.
- `disabled_ids=[1]`: every 1 is emitted alone and cuts merging, so `[0,1,0,1]` never yields the phrase `[0,1]`.

## Constraints

- Do not modify the input; all invalid inputs raise `ValueError` (types, ranges, `pad_token_id` collision, `disabled_ids` out of range or duplicated, truncation/padding + `max_length` combination constraints, over-long without truncation).

## Boundaries with adjacent problems

- `bpe-training` / `bpe-encoding` learn a **frozen training-time merge table**; zip2zip attaches an **input-level codebook at inference**, which never enters the model and lives and dies with one sequence.
- `data-collator` / `sequence-pack-pad` pad for batch collation; here padding is a **single-sequence tail product**, and `max_length` also governs truncation.
- `dt-minhash-lsh-dedup` compresses the **number of corpus documents** (dedup); this problem compresses the **length of one sequence** (token count).
