# DIoU-NMS Suppression Criterion Ledger

## Background

Distance-IoU Loss (1911.08287, AAAI 2021) observes: NMS with only IoU wrongly suppresses boxes with low IoU but distant centers, and keeps boxes with low IoU but close centers. DIoU-NMS swaps the criterion to **DIoU = IoU − ρ²/c²**: the NMS flow is untouched, only the formula. The official repo (Zzhiter/DIoU-NMS) is 404; this problem follows the paper formula plus the community-standard transcription.

## Exact Ledger

1. **DIoU = IoU − ρ²/c²**:
   - ρ² = squared Euclidean distance between box centers;
   - c² = w² + h² of the **enclosing box** (squared diagonal).
2. **NMS flow unchanged**: stable descending score sort (ties to lower index), greedy keep of the current top, suppress remaining boxes with **DIoU > thresh (strict)** against kept boxes, repeat.
3. **Directionality**: disjoint boxes → IoU=0 → DIoU = −ρ²/c² ≤ 0, never suppressed by a positive threshold; identical boxes → ρ=0 → DIoU=IoU; boundary equality (DIoU == thresh) does NOT suppress.
4. Output: keep list (original indices in keep order) and suppression map {suppressed → suppressor}.

## Constraints

1. Boxes [x1,y1,x2,y2] with x2≥x1, y2≥y1; finite coords/scores; thresh ∈ [0,1].
2. c²=0 (point boxes) degenerates to plain IoU.
3. No input mutation; pairwise criterion only, no global state.

## Data Scale

- n ≤ 10³ (O(n²) criterion ledger).

## Acceptance Anchors (excerpt)

- A=[0,0,2,2], B=[1,1,3,3]: iou=1/7, ρ²=2, c²=18 → **DIoU=2/63**; thresh=2/63 exactly does NOT suppress, 0.03 does.
- Small-shift duplicate (B=[0.2,0,2.2,2]): DIoU≈0.8137 > 0.5 → suppressed.
- Disjoint ([0,0,1,1] vs [5,5,6,6]): DIoU=−32/50 < 0.5 → kept.
- **The criterion-swap selling point**: A=[0,0,10,10] vs B=[8,8,12,12], IoU≈0.0345 > 0.03 would be suppressed by IoU-NMS; DIoU≈−0.139 keeps it — the center penalty outweighs the overlap term.
