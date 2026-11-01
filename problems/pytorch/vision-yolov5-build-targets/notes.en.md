# YOLOv5 build_targets Double-Expansion Assignment Ledger — Implementation Notes

## Source

- Repo: ultralytics/yolov5 tag v7.0, `utils/loss.py::build_targets`.
- No paper (engineering artifact); the code is the spec. v6.x matches v7.0 here.
- `self.anchors = m.anchors`: pixel units (relative to the input image).
  Targets are normalized [img, class, x, y, w, h]; this problem uses the
  single-image form [class, x, y, w, h].

## Line-by-line ledger

```python
gain = torch.ones(7)                       # [1,1,?,?,?,?,1]
ai = arange(na).view(na,1).repeat(1,nt)
targets = cat((targets.repeat(na,1,1), ai[...,None]), 2)   # replicate + idx
off = tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]]).float() * 0.5
for i in range(nl):
    gain[2:6] = tensor(shape)[[3,2,3,2]]   # [W,H,W,H] (note 3,2 order)
    t = targets * gain
    r = t[..., 4:6] / anchors[:, None]     # grid wh / pixel anchor
    j = torch.max(r, 1/r).max(2)[0] < anchor_t    # two-sided, strict <
    t = t[j]
    gxy = t[:, 2:4]; gxi = gain[[2,3]] - gxy      # [W,H] − gxy
    j, k = ((gxy % 1 < g) & (gxy > 1)).T
    l, m = ((gxi % 1 < g) & (gxi > 1)).T
    j = stack((ones_like(j), j, k, l, m))
    t = t.repeat((5,1,1))[j]                      # dir-major, then rows
    offsets = (zeros_like(gxy)[None] + off[:, None])[j]
    gij = (gxy - offsets).long()                  # truncation
    gi, gj = gij.T
    indices.append((b, a, gj.clamp_(0, shape[2]-1), gi.clamp_(0, shape[3]-1)))
    tbox.append(cat((gxy - gij, gwh), 1))
```

## Key subtleties

1. **gain order**: shape = (b, na, H, W, ·); `shape[[3,2,3,2]] = [W,H,W,H]` —
   x scales with W, y with H. gain[0]/[1]/[6] stay 1.
2. **Two-sided wh ratio**: `max(r, 1/r).max(2) < anchor_t` (default 4.0).
   Equality also filters (an `==anchor_t` boundary anchor is needed to catch
   the `>=`→`>` mutation). **Official mixup quirk**: r = grid-space wh /
   pixel anchor — for the same physical target the grid width halves and the
   anchor doubles per layer, so r shrinks 4x per layer and eligibility swings
   wildly across layers, contrary to the "independent per layer" intuition.
3. **Offset dual gate**: x+ direction fires iff `gx % 1 < 0.5 AND gx > 1`;
   the x− direction uses the inverse `(W−gx) % 1 < 0.5 AND > 1`. x and y are
   independent. Integer-coordinate cells fire all four directions; a
   coordinate exactly 1.0 fires none on that side (`>1` strict; the `>=1`
   mutation needs an exact-1.0 anchor); fraction exactly 0.5 never fires.
   The dual gate mathematically guarantees in-bounds expansion — the clamp is
   **defensive dead code** (the clamp-upper-bound mutation is equivalent).
4. **gij truncation**: `.long()` truncates (floor for positives), not round:
   x=14.6 with off=−0.5 → int(15.1)=15. When x is an exact integer, dir0 and
   dir3 land on the same cell (int(1.0)=int(1.5)=1) with different tbox
   residuals.
5. **Expansion order**: `t.repeat((5,1,1))[j]` is direction-major, then row
   order; the ledger follows it (a reversed-order mutation is catchable).
6. **tbox residual**: `(gxy − gij, gwh)` — the base is the **truncated cell**,
   not the offset: x=16.0 dir1 → 16−15 = 1.0 (right half of the cell).
7. **nt=0**: `t = targets[0]` (empty), offsets=0 → no positives anywhere.

## Problem spec

- Inputs: targets (nt×5 [class,x,y,w,h] normalized, class a non-negative
  int), anchors (nl×na×[w,h] pixels, positive finite), shapes (nl×[H,W],
  dims ≥1), anchor_t (>0).
- Per-layer ledger: gain, wh_keep (na·nt, anchor-major), expanded
  ([a, cls, gx, gy, gw, gh, dir], dir-major), indices ([a, gj, gi] after
  clamp), tbox ([dx, dy, gw, gh]), anch.
- Validation: row lengths/finiteness, wh>0 and ≤1, x,y∈[0,1], class int ≥0,
  anchor positive, shape ≥1, anchor_t>0; violations raise ValueError.
- Oracle: direct (anchor, dir) enumeration (no replicate-then-filter
  structure) + 200 random cross-checks (seed 707, anchor_t ∈ {1,2,4,8}).

## Equivalent-mutation notes

- **clamp upper bound W instead of W−1 is equivalent**: the dual gate keeps
  gij in bounds, so the clamp never fires (official defensive code).

## Mutation testing

13 caught + 1 equivalent: one-sided wh ratio, anchor_t boundary (exact
equality anchor), gain H/W swap (non-square [8,16] anchor), fraction gate 0.4,
edge gate >=1 (exact-1.0 anchor), inverse W−1−x, offset 1.0, gij round,
tbox residual base, reversed direction order, anchor index, gwh without gain,
y gate reusing the x gate.
