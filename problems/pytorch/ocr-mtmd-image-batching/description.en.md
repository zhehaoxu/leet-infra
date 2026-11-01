# llama.cpp mtmd Image Token Batching Ledger

## Problem

llama.cpp's multimodal library mtmd splits multimodal input into a chunk sequence (text token spans / image or audio placeholder spans), then decodes chunk by chunk under the context batch limit `n_batch`. Image chunks are special: the whole span is first encoded into embeddings by mmproj, then decoded in batch slices, with positions advanced only once. This problem transcribes the scheduling ledger of `mtmd_helper_eval_chunks / eval_chunk_single / decode_image_chunk`: given the chunk sequence, batch limit and logits switch, output the batch sequence, position advancement, and logits landing point.

## Solution

The ledger is pinned by the spec (faithful to upstream):

1. **Empty chunk sequence**: return immediately, no batches, positions untouched (official "no chunks to eval");
2. **text chunk**: a while loop slices by `n_batch` (each batch ≤ n_batch), positions advance **per token** — batch k's pos interval is `[n_past+sent, +batch_size)`; logits are set on the batch's last token only when "this chunk is the last chunk AND `logits_last` AND this batch is the chunk's last batch";
3. **image chunk**: `n_img_batches = ceil(n_tokens / n_batch)`; batch k's pos interval is `[n_past + k*n_batch, +min(n_batch, remaining))` — the official `set_position_normal(pos_0 + i)` sets positions for the whole chunk at once and the batch loop takes views by offset; **after decoding, positions advance once by `n_tokens`** (`n_pos == n_tokens` without M-RoPE); **the image branch never handles `logits_last`** — if the last chunk is an image there is no logits output (official quirk: logits normally come from a text tail after the image);
4. `logits_pos` = the position of the last logits mark (None if never marked).

Token ids and embedding values are out of scope — only the scheduling and position ledger.

## Constraints

- 0 ≤ chunk count ≤ 10^3, 0 ≤ n_tokens per chunk ≤ 10^6, 1 ≤ n_batch ≤ 10^6.
- chunks not a list, chunk not a dict, type outside {text, image}, n_tokens not a non-negative int, n_batch non-positive or non-int, logits_last not bool, n_past negative → `ValueError`.
