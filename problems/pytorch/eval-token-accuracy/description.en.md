# Evaluation: Masked Token Accuracy

## Problem

The single most-quoted number in instruction-tuning and dialogue reports is token-level accuracy: what fraction of the predicted tokens land on the positions that actually need supervision. **The denominator is the whole trick** — prompt, system template and right-hand padding are all marked `ignore_index` and not one of them may enter the denominator; and "compute each sample's accuracy first, then average over samples" (macro) versus "dump every valid token into one pool" (micro) are **two different numbers** whenever sample lengths differ. Given one set of predictions you must be able to produce whichever convention is being reported.

## Rules

- `pred_ids`, `labels`: int64, both of shape \((B, L)\), **already position-aligned** (`pred_ids[b, i]` is the prediction judged against `labels[b, i]`; no extra shift in this problem).
- Positions with `labels[b, i] == ignore_index` (default \(-100\)) are prompt/padding: they enter neither numerator nor denominator, and whatever `pred_ids` holds there is irrelevant — even if it also happens to be \(-100\).
- A hit means \(\text{pred\_ids}[b,i] = \text{labels}[b,i]\) at a valid position.
- Return the pair `(micro, macro)` of Python floats:
  - \(\text{micro} = \dfrac{\sum_{b,i}\text{hit}}{\sum_{b,i}\text{valid}}\) — sum across the whole batch first, then divide.
  - \(\text{macro} = \dfrac{1}{|\mathcal B|}\sum_{b\in\mathcal B}\dfrac{\text{hit}_b}{\text{valid}_b}\), where \(\mathcal B\) is the set of samples with **more than zero** valid positions; prompt-only/pad-only samples do not even increase the sample count.
- If the batch has no valid position at all, return \((0.0, 0.0)\) by convention (the HuggingFace `evaluate` convention for an empty data set).
- Ready-made evaluation libraries are forbidden.

## Example

```
pred_ids = [[7, 7, 1, 2, 9],      labels = [[-100, -100, 1, 2, 3],
            [4, 6, 1, 1, 1]]                [4, 5, -100, -100, -100]]

row 0: 3 valid positions (ids 1,2,3); predictions 1,2,9 -> 2 hits -> row acc = 2/3
row 1: 2 valid positions (ids 4,5);   predictions 4,6   -> 1 hit   -> row acc = 1/2

micro = (2 + 1) / (3 + 2)              = 0.6
macro = (2/3 + 1/2) / 2 = 0.5833333333
```

The two numbers differ: the long sentence (row 0) carries more weight in micro, while macro gives both rows exactly half of the weight. If every sample has the same number of valid positions the two coincide.

## Constraints

- CPU int64 inputs; \(B, L \ge 1\).
- Both returned values lie in \([0, 1]\).
- **Degenerate properties**: all `ignore_index` \(\Rightarrow (0.0, 0.0)\); `pred_ids` equal to `labels` \(\Rightarrow (1.0, 1.0)\); equal valid counts per sample \(\Rightarrow\) micro \(=\) macro; only one sample containing valid positions \(\Rightarrow\) the two also coincide.
- The denominator is the **valid count**, not \(B\cdot L\): counting ignored positions as mistakes systematically deflates the metric, counting them as correct (both sides hold -100, so the comparison happens to succeed) inflates it.
