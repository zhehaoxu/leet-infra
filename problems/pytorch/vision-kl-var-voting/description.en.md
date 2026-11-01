# KL-Loss Variance Voting Fusion Ledger

## Problem

KL-Loss (CVPR'19) makes the detection head predict a per-coordinate
localization variance sigma alongside each coordinate. At inference these
sigmas power **variance voting**: after NMS decay, each surviving box is
fused with its neighbors, weighted by "smaller sigma, louder voice". The
official implementation hides in the `stdiou` branch of `py_cpu_nms.soft`,
wrapped in a soft-NMS greedy loop. Given the detection list, per-coordinate
sigmas and parameters, reproduce the official ledger.

Given:
- `dets`: N x 5 `[x1, y1, x2, y2, score]` (float)
- `confs`: N x 4 per-coordinate sigmas (one per x1/y1/x2/y2, >0; smaller
  means more certain)
- `thresh`: STD_TH (clamped to 0.01 when >0.1 under stdiou)
- `iou_sigma`: the vote Gaussian kernel width sigma_IoU

## Processing chain

1. **Pairwise IoU (legacy VOC +1 convention)**:
   `inter w = max(0, min(x2)−max(x1)+1)`, areas likewise +1;
   `iou = inter / (area_i + area_j − inter)`. Precomputed in full,
   afterwards rows/cols are **only swapped, never recomputed**.
2. **Greedy loop (slot view)**: among remaining slots `[i, n)` pick the
   **first max in slot order** (numpy argmax semantics) and swap it into
   slot i (dets/confs/original index/IoU rows+cols all swapped together).
   ⚠️ The pick swap rewrites the relative order of active boxes — on ties
   the pick depends on slot order; simulating by original-index order
   answers ties wrong.
3. **Variance voting (stdiou branch)**:
   - voters: `iou(pick, k) > thresh` (strict; clamp thresh to 0.01 first
     if >0.1; self always included, iou(i,i)=1);
   - per-coordinate fusion (4 independent averages):
     `coord_j = Σ_k p_k·v_kj/σ_kj² / Σ_k p_k/σ_kj²`,
     `p_k = exp(−(1−iou_k)²/iou_sigma)`;
   - the weight **excludes score** (score acts only through pick order
     and decay);
   - fused coords are **never clipped**, IoU **never recomputed** (picked
     boxes never vote again, unpicked boxes keep original coords, so
     stale IoU × original coords stays consistent).
4. **Soft decay**: for every later-slot box with `iou(pick, k) > 0`
   (**any positive overlap**, independent of thresh):
   `score_k ×= exp(−iou²/0.5)` (sigma 0.5 hardcoded).
5. **Floor removal**: `score < 0.001` → swap with slot n−1 and shrink n
   (tail-swap; the swapped-in box is re-checked at the same position).
6. Processing order = output order: each round emits the picked box's
   fused coords with its **then-current** (decayed) score.

## Output ledger

```json
{
  "ious":    [[...]],            // initial N x N IoU matrix
  "steps":   [{"idx": <orig>,    // this round's picked original index
               "box": [x1,y1,x2,y2],           // fused coords
               "voters": [...],                // voter original indices (sorted ledger)
               "weights": [[w_x1,w_y1,w_x2,w_y2], ...]}],  // p/sigma^2 per coord
  "final":   [[x1,y1,x2,y2,score], ...],  // processing order, decayed scores
  "removed": []                  // original indices tail-swapped out below 0.001 (time order)
}
```

## Input constraints

- `confs` same length as `dets`, exactly 4 **positive** finite sigmas per row;
- `dets` rows have exactly 5 finite values;
- `iou_sigma > 0`; finite `thresh`; violations raise `ValueError`;
- empty input is legal (empty ledger).

## Example

`dets=[[0,0,10,10,0.9],[2,2,12,12,0.7]]`, `confs=[[1,1,1,1],[2,2,2,2]]`,
`thresh=0.3, iou_sigma=0.5`:
IoU=81/161≈0.503 (+1 convention); pick box 0, voters=[0,1],
p=exp(−(1−81/161)²/0.5); fused x1=(0·1+2·p/4)/(1+p/4),
x2=(10+12·p/4)/(1+p/4) (for a single pair p half-cancels → coords lean
toward the smaller sigma); box 1 decays `0.7·exp(−(81/161)²/0.5)`.
On round 2 picking box 1, voters=[1] only (box 0 already retired) →
coords unchanged.
