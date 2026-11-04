# SenseVoice Rich-Token Multi-Task Prefix Ledger

## Background

SenseVoiceSmall (FunAudioLLM/SenseVoice, `model.py`) produces four kinds of outputs from a single CTC head: speech content tokens, language ids, event/emotion tokens, and text-normalization marks. The trick is to **prepend 4 frames of rich-token queries to the CTC input**: predictions on the first 4 frames are trained with a rich CE, and real speech CTC starts at frame 5. This problem rebuilds the decidable parts of that design.

## Mechanism breakdown

### 1. Prefix query construction (encode_prefix)

The 16-row small embedding table and official constants:

```
embed rows: 0=auto, 1=event, 2=emo, 3/4/7/11/12/13=zh/en/yue/ja/ko/nospeech, 14/15=withitn/woitn
LID_INT_DICT = {24884:3, 24885:4, 24888:7, 24892:11, 24896:12, 24992:13}
TEXTNORM_INT_DICT = {25016:14, 25017:15}
EMO_DICT = {unk:25009, happy:25001, sad:25002, angry:25003, neutral:25004}
```

Inference concatenation order (reading the two nested `torch.cat` calls in the official code): first `speech = cat(textnorm, speech)`, then `speech = cat((language, [event, emo]), speech)` — **the final prefix is `[lid, event, emo, textnorm]`**, and `speech_lengths += 4`. Training encode uses the same order (textnorm reads `text[:, 3]`, lid reads `text[:, 0]`, and during training lid falls back to auto with 20% probability — that RNG branch is excluded from the judge).

### 2. Rich CE (rich_ce)

`_calc_rich_ce_loss`: the first 4 encoder frames are projected to vocab via `ctc.ctc_lo`, the labels are the 4 rich token ids, and the loss is a LabelSmoothingLoss (Kaldi style: `(1-ε)·nll + ε·(-mean logp)`, with ignore_id positions excluded from both numerator and denominator; normalization over non-ignored positions). Accuracy is `th_accuracy` (ignored positions not counted in the denominator).

### 3. CTC greedy (ctc_greedy)

`log_softmax → per-frame argmax → torch.unique_consecutive (collapse consecutive equal ids) → drop blank`. frames keeps the pre-collapse argmax sequence. Non-adjacent equal ids are not collapsed (CTC semantics, not global dedup).

### 4. ban_emo_unk (ban_emo)

`ctc_logits[:, :, emo_unk] = -inf`. After masking, that position can never be selected by argmax; the `changed` mask equals exactly `before == emo_unk` (for frames where before≠emo_unk, masking touches no other column and the argmax is unchanged — which makes "changed = before != after" equal to it).

### 5. CTC offset window (ctc_shift)

The content CTC uses `encoder_out[:, 4:]` with `encoder_out_lens - 4`, i.e. frames `[4, lens)` of the original sequence (lens-4 content frames); empty window when lens<=4.

## Judging domain

- embed weights, ctc_lo, logits are all explicit arguments, fp64, no RNG (the training-time 20% lid fallback is out of scope);
- Modes: encode_prefix, rich_ce, ctc_greedy, ban_emo, ctc_shift;
- Excluded: encoder forward, Specaug/CMVN frontend, ctc_forced_align timestamps (funasr tooling), rich_transcription_postprocess text cleanup (no source on the python side of the official repo; it lives inside the funasr package).
