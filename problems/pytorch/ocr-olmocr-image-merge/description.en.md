# olmOCR Image-Element Union-Find Merge Ledger

Before allenai/olmocr's `olmocr/prompts/anchor.py` generates the PDF-page
anchor text for a VLM (the `pdfreport` engine), it runs
`_merge_image_elements` to fuse neighboring (small-gap) image XObjects into
one element — the same illustration is often split into several placed
pieces in a PDF, so the extractor records them one by one and must merge
them back by bbox tolerance; otherwise the anchor text shows a burst of
fragmented `[Image ...]` lines. Replicate this merge ledger rule by rule.

## Interface

```python
solve(images: list, tolerance: float = 0.5) -> list[dict]
```

- `images`: each item `[name, x0, y0, x1, y1]` (PDF page coordinates);
- `tolerance`: merge tolerance, official default `0.5`;
- return one `{"name": ..., "bbox": [x0, y0, x1, y1]}` per connected
  component; **the output order is the ascending order of each component's
  minimum original index**.

## Rules

1. **Overlap test**: the separation distances between two boxes are
   - `h_dist = max(0, max(the two x0s) - min(the two x1s))`
   - `v_dist = max(0, max(the two y0s) - min(the two y1s))`

   The pair overlaps iff **both** are `<= tolerance`. This is a closed
   interval: touching (`h_dist = 0`) and exactly `gap == tolerance` both
   count; one close axis with the other far apart does not. Containment
   gives distance 0 on both axes, so it always overlaps.
2. **Merging is a transitive closure**: double loop over indices `i < j`,
   union on overlap. If A~B and B~C while A and C are not adjacent
   themselves, all three end up in one component (the union-find
   implementation; path compression and root choice affect only the internal
   representation, never component membership).
3. **Within-component aggregation** (members in ascending index order):
   - `bbox` is the bounding rectangle of all members (per-axis min/max);
   - `name` starts from the lowest-index member's name, then appends the
     remaining names in **ascending index order** joined by `"+"` —
     regardless of the direction unions happened, the smaller index is
     always earlier.
4. Empty input returns `[]`; a single element returns itself unchanged.

## Examples

```python
solve([["a", 0, 0, 1, 1], ["b", 1.5, 0, 2, 1]])
# h_dist = max(0, 1.5 - 1) = 0.5 <= 0.5 -> merge
# -> [{"name": "a+b", "bbox": [0, 0, 2, 1]}]

solve([["z", 0.8, 0.8, 2, 2], ["a", 0, 0, 1, 1]])
# names concatenate in index order; "z" sits at index 0
# -> [{"name": "z+a", "bbox": [0, 0, 2, 2]}]

solve([["p", 0.8, 0.8, 2, 2], ["q", 0, 0, 1, 1], ["r", 10, 10, 11, 11]])
# component {0,1} has min index 0 < component {2} min index 2
# -> [{"name": "p+q", ...}, {"name": "r", ...}]
```

## Hints

- Name concatenation order, per-axis min/max bbox, and the output grouping
  order are all driven by the **original indices**, not by union-find roots.
- The `max(0, ...)` clamp in `h_dist` only matters for **negative** values
  (real overlap records distance 0) — removing the clamp is unobservable
  under this test (negatives also satisfy `<= tol`), but swapping min/max
  is immediately observable.
- At the official call site (`_linearize_pdf_report`) the merged results
  then feed the anchor-text length budget, whose over-budget path contains
  `random.shuffle` — out of scope here.
