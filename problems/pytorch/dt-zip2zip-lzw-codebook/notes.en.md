# Reference: LZW growth phase, three caps, and disabled interruption

## Implementation

1. Validate first, **scalars before sequences**: `initial_vocab_size/max_codebook_size/max_subtokens/pad_token_id` one by one as `int` (reject bool) with lower bounds, `pad_token_id >= initial + max_codebook`; then per-element checks on `ids` (type and range — **the upper bound `< initial_vocab_size` is easy to miss**; checking only non-negativity lets `[0, 4]` slip through), `disabled_ids` within the base vocab and unique, `truncation` a bool, `padding` whitelisted, truncation/max_length combination constraints. One more combination check: `padding="max_length"` with an over-long result raises `ValueError`.
2. **Five states in the main loop**: disabled interruption (flush + solo emit + clear), buffer push, known check (single tokens are known by definition), unknown → insert + **pop-emit-restart**, `max_subtokens` full → emit + clear. After inserting you must `buffer.pop(); emit(); clear(); append(token)` — emitting the whole buffer would also emit the not-yet-collected new phrase, and one off-by-one poisons everything downstream.
3. **Three independent caps**:
   - Full codebook: insert only when `next_id < limit`; when full, no error is raised and phrases are **emitted token by token** (known phrases still emit their hyper id).
   - `max_subtokens`: the buffer is force-emitted at the cap — guaranteeing output phrase length $\le$ `max_subtokens`.
   - `truncation`: checked both in `emit` and at the loop top via `len(compressed) >= max_length`; hitting it at the loop top **breaks and stops consuming input**, so the codebook only grows from consumed input — the crux of the truncation cases.
4. **Padding and truncation share one `max_length`**: it is the truncation cap and also the left-pad target; "truncating to exactly max_length makes padding a no-op". Over-long without truncation raises in the padding stage.
5. Output: `attention_mask` is 0 at pads; `codebook` is reordered ascending by hyper id as `{hyper: list(phrase)}`.

## Verification

7 `check_*` functions cover: the problem example (hand-computed full LZW growth), 5 fixed-data parities (full codebook, disabled interruption, compact `max_subtokens=2` merging, long repeated block hitting the cap, all-disabled) plus a **roundtrip check**, two truncation shapes (including `max_length=1` stopping after one emission), four padding shapes (left pad, truncation+padding sharing max_length, capped exactly, exactly equal length), codebook freeze, the `max_subtokens` cap (`[0]*6` → `[0,4,4,0]`), and 18 invalid inputs.

The **roundtrip check** is the core of the oracle: `test_cases.py::_oracle_decode` independently ports the upstream `decode_fn` (KwKwK self-reference — on an unknown hyper id, `buffer.append(buffer[0])` then insert **using the incoming id**, force-merge when the buffer is full, `buffer.clear()` freeze after the codebook fills, while-merge update phase), asserting `decode(encode(ids)) == ids` on cases without truncation or padding. Any emit/insert ordering mistake on the encoding side blows up the roundtrip — more convincing than parities on compressed sequences alone. `_verify` adds structural assertions: mask consistent with pads, hyper ids contiguous from `initial_vocab_size`, phrase lengths and ranges legal, input unmodified.

Mutation testing (8/8 caught): `next_id` not incremented, insert/emit order swapped, no emission at `max_subtokens`, missing flush before disabled, left pad changed to right pad, truncation without early exit (codebook grows one extra level), no error when over-long, single tokens not treated as known. All caught by hand-computed cases or the roundtrip.

## Scope and sources

Upstream epfl-dlab/zip2zip-compression (2610.13605):

- `src/codec.rs::encode_fn` (L213-303): validation (non-empty ids, `pad_token_id >= initial + max_codebook_size`, disabled within base vocab and unique, `max_subtokens >= 2`, truncation/padding require max_length), the main loop (disabled flush+solo emit, unknown insert then pop-emit-reset, `buffer.len() == max_subtokens` force emission, truncation early-exit at the loop top), left padding, mask zeroed at pads.
- `src/codec.rs::decode_fn` (L310-385): KwKwK self-reference (on an unknown id, `buffer.push(buffer[0])` then `insert_buffer(id)` **inserting with the incoming id**), force-merge, `buffer.clear()` freeze when the codebook is full, while-merge update. Oracle only in this problem — not the tested operation.
- `src/config.rs`: defaults and constraints of the `CodebookConfig` cap fields.

Deliberate deviations:

- **No tokenizer, no model**: the input is a base-vocab id sequence and hyper ids are numbered from `initial_vocab_size` — matching upstream's "hyper tokens appended after the tokenizer vocab" layout, minus the tokenizer wiring.
- **solve only encodes**; decoding exists as an independent oracle for roundtrip verification (upstream decode's KwKwK insertion quirk would make a good separate problem).
- `padding="longest"` (batch-level, pad to the batch max) is out of scope; `attention_mask` is a per-sequence product, no batch stacking.
- The codebook dict key order is pinned to hyper id ascending (Rust HashMap is unordered; Python side adds the determinism contract).
