# Notes: vision-diou-nms

## Source

- Paper 1911.08287 (AAAI 2021, Zheng et al.) Eq. (3): `DIoU = IoU − ρ²(b, b_gt)/c²`; DIoU-NMS is Algorithm 1 with the criterion swapped.
- Official repo Zzhiter/DIoU-NMS is 404 (same fate as abhi2610/Ohem-caffe); the paper formula is the sole source of truth. Neither mmcv nor ultralytics ships a standalone DIoU-NMS (torchvision nms doesn't support it) — declared in notes.

## Key Judgments

- **c² is the enclosing box's squared diagonal** — not the IoU union, not either box's own diagonal. Mutations `enclose-swap` (min/max flipped) and `c-not-squared` pin this.
- **ρ² uses centers**, not corners — `rho-center-off` pins it.
- **Directionality**: disjoint boxes get DIoU ≤ 0 and are never suppressed by a positive threshold — a direct corollary of the criterion swap, pinned standalone (`check_disjoint_never_suppressed`).
- The real proposition point is `check_diou_catches_iou_miss`: the same pair suppressed under IoU-NMS but kept under DIoU-NMS (0.0345 > 0.03 vs −0.139 < 0.03) — numeric evidence that the center penalty outweighs the overlap term.
- The NMS flow is isomorphic to `nms` (66) (stable descending, greedy, strict >); mutation `keep-all` (not consuming processed entries) pins the flow.

## Boundary

- vs `nms` (66): IoU criterion; `soft-nms` (308): score decay without deletion; `vision-rotated-nms` (379): rotated boxes. This problem swaps the criterion formula only.
- No DIoU/CIoU loss (loss side could be a follow-up); no rotated DIoU.

## Test Design

- Oracle `_py` replays independently (pop-first, pairwise compare, no shared helpers).
- 9 checks: DIoU numeric anchor (2/63 exact-equality boundary), ρ=0 degeneration, disjoint keep, identical suppress, **criterion-swap contrast pair**, score-tie stability, three-box chain, 200 random, 6 invalid tiers.
- Mutations 9/9: penalty removed/sign-flipped, enclosing box flipped, c²→c, center offset, ≥, keep-all, self-suppression mapping, area formula → sum.
