# GLM-4-Voice interleaved stream parsing ledger (notes)

Source: zai-org/GLM-4-Voice `web_demo.py` (front half of the main loop).

- **Classification**: `token_id >= audio_offset` means audio (value = id - offset);
  officially there is no upper bound — any special-token id beyond `<|audio_0|>` is
  treated as audio. The `>=`->`>` and offset-subtraction mutations are both anchored.
- **Flush vs classification order**: the official loop checks the flush condition
  **before** classifying the current token — a full 25-frame chunk is only emitted when
  the 26th audio arrives. Moving the check after classification is caught by an anchor.
- **The end token has three effects**: sets is_finalize -> immediately emits a finalize
  chunk if audio is non-empty -> the end token itself and everything after it never
  enter complete_tokens (classification is wrapped in `if not is_finalize`). End with
  empty audio does not flush — also official (anchor 4).
- **Block ladder**: `block_size_list = [25,50,100,150,200]`, upgraded after each flush,
  capped at the last; an empty ladder IndexErrors officially — the judging convention
  allows assert/IndexError.
- **No final flush at stream end**: leftover audio_tokens are silently dropped — an
  official quirk surfaced explicitly as `dropped_tail`.
- Mutations 14/14 caught; no equivalent mutations. The interleaving cadence in the
  system prompt ("13 text token followed by 26 audio tokens") is a prompt constraint,
  not parsing, and is out of scope.
