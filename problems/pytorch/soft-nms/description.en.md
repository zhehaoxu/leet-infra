# Soft-NMS: Non-Maximum Suppression with Continuous Confidence Decay

## Problem

Hard NMS **deletes** every box whose IoU with the winner exceeds the threshold, so in crowded or occluded scenes the second genuine object is routinely eaten. Soft-NMS (Bodla et al., ICCV 2017, arXiv:1704.04503) instead only **lowers the confidence**: each round still takes the current top-scoring box, but overlapping boxes are not removed — they get multiplied by a decay factor driven by IoU, and only fall out once their score drops below `score_thr`. This problem implements both decay kernels — `"linear"` (multiply by \(1-\text{IoU}\) only when IoU strictly exceeds the threshold) and `"gaussian"` (unconditionally multiply by \(e^{-\text{IoU}^2/\sigma}\)) — and returns the kept boxes' **original indices** together with the score they had **at the moment they were selected**. CPU, fully deterministic, no random number calls.

## Rules

- The complete signature, argument meanings and defaults live only in the starter docstring; that is the sole judging contract. boxes is (N,4) with \([x_1,y_1,x_2,y_2]\) valid boxes (not validated), scores is (N,); N=0 is allowed.
- **Four steps per round** (order is not interchangeable):
  1. take the highest current score among the remaining boxes, **ties resolved by the smallest original index**, append it to the kept order, and record the score it has **at that moment** (i.e. after all previous rounds' decay);
  2. decay every remaining box by its IoU with the box just selected;
  3. delete remaining boxes whose **post-decay** score is `< score_thr` (**strictly less deletes; exactly equal is kept**);
  4. stop when nothing remains.
- **The two decay kernels** (method accepts only these two values, anything else raises ValueError):
  \[\text{"linear"}:\ \text{if}\ \text{IoU}>\text{iou\_thr},\ s\leftarrow s\,(1-\text{IoU}),\ \text{else unchanged};\qquad
    \text{"gaussian"}:\ s\leftarrow s\,\exp\!\big(-\text{IoU}^2/\sigma\big)\ \text{(unconditional, ignores iou\_thr entirely).}\]
  `sigma` is used only by gaussian and must be > 0 (otherwise ValueError); when `method="linear"`, any sigma value — however absurd — must be **ignored completely** (no validation). `iou_thr` is used only by linear.
- **IoU uses exactly the `iou-compute` convention**: intersection width/height `clamp_min(0)`, inter = iw·ih, union = sum of areas minus inter, IoU = 0 when the union is 0.
- Decay is **cumulatively multiplicative across rounds**: one box can be pushed down by several already-selected boxes in turn, so its recorded score is generally below its input score.
- Return a 2-tuple `(keep_indices, keep_scores)`: long (M,) original indices in selection order, and float (M,) scores at selection time. N=0 returns two (0,) empty tensors; with N≥1 the first-round top box is always kept (M≥1), while all the others can be wiped out by `score_thr`.

## Example

boxes \(=[[0,0,10,10],[1,1,11,11],[50,50,60,60]]\), scores \(=[0.9,0.8,0.7]\), all defaults (linear, iou_thr=0.3, score_thr=0.001).

- Round 1: select box 0 (0.9). Its intersection with box 1 is \(9\times9=81\), union \(=100+100-81=119\) → IoU \(=81/119=0.680672>0.3\) → \(s_1=0.8\times(1-0.680672)=0.2554622\); box 2 does not overlap → stays 0.7.
- Round 2: remaining \(\{0.2554622,\,0.7\}\), the top is **box 2** (0.7) → kept; its decay pass leaves box 1 untouched (IoU=0).
- Round 3: box 1 is kept **at 0.2554622** — its suppressed score, not the original 0.8.

Return (backfilled EXAMPLE_* constants):

```
keep_indices = [0, 2, 1]
keep_scores  = [0.8999999762, 0.6999999881, 0.2554621696]
```

Contrast with hard NMS: IoU=0.68 > 0.3 would delete box 1 outright, leaving `[0, 2]`.

## Constraints

- method ∈ {`"linear"`, `"gaussian"`}, anything else ValueError; gaussian with sigma ≤ 0 ValueError; under linear sigma is ignored and must not raise; N=0 and N=1 are both legal
- Ties for the top score take the **smallest original index**; the `score_thr` test is "strictly less after decay deletes", equality must be kept
- The linear threshold comparison is **strictly greater**: IoU exactly equal to iou_thr does not decay
- Under gaussian, IoU=0 gives a factor of exactly \(e^0=1\), so disjoint boxes' scores must be bit-for-bit unchanged
- Index tensor dtype is long, score tensor dtype matches the input float; indices compared exactly (torch.equal / tolist), scores with \(10^{-5}\) tolerance (\(10^{-6}\) for the example)
