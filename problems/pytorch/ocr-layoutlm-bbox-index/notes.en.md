# notes — ocr-layoutlm-bbox-index (order 669)

## Source

- Origin: HF `transformers` `models/layoutlm{,v2}/modeling_layoutlm*.py`, `_calc_spatial_position_embeddings` (same lineage as unilm `layoutlmft`):
  - the four corners look up `x_position_embeddings(bbox[:,:,0|2])` and `y_position_embeddings(bbox[:,:,1|3])`;
  - `h = bbox[:,:,3] - bbox[:,:,1]`, `w = bbox[:,:,2] - bbox[:,:,0]` look up shape embeddings;
  - coordinate domain 0–1000 (HF raises IndexError "should be within 0-1000 range" beyond it; the unilm pipeline normalizes + clamps before feeding — the spec pins the clamp into the ledger).
- v2's 1D position_ids: text `arange(seq_len)`; visual tokens' position_ids **restart from 0** (no text-length offset); the relative 2D bias buckets differences of `x0` and `y1` (lower) coordinates. This problem takes the observable ledger of the grid index `y * 1001 + x`, not the embedding lookup or bucketing.
- Boundary with `mm-qwen25vl-window-index`: Qwen VL is a **window/patch resolution ledger**, this is a **document bbox coordinate ledger**; boundary with #107⑦ olmOCR cross-page anchoring: this is an intra-model single-page index.

## Judging

- solve args: boxes (pixel coords), page_w, page_h. Returns norm_boxes/wh_ids/corner_grid/center_grid.
- Oracle: `test_cases._py` is an independent implementation (int(round()), divmod), 150 random cross-checks (page sizes 100–3508, random boxes).
- Anchors: page 1000×500 doubling ledger with hand-computed corner/center grids; negative and over-page clamps (exactly 1000 not clipped); banker's rounding (page=2000, x=3 → 1.5 → 2; x=1 → 0.5 → 0); full-page and degenerate boxes (w/h=0); normalized x0>x1 → ValueError.
- Invalid inputs: empty boxes / non-4-tuples / non-numerics / NaN/Inf / bool, non-positive or bool page dims → ValueError.

## Mutation log (9/9 caught)

no-clamp-lo / no-clamp-hi (both clamp ends disabled), round-to-int (banker's → truncation), norm-x-use-page-h (x normalized by page height), grid-base-1000 (grid 1001→1000), grid-transpose (row/col swapped), wh-swap (width/height swapped), center-ceil (center //2 → ceil), center-grid-corner (center grid replaced by top-left corner).

## Red lines

No real images/PIL/torchvision (coordinates all explicit inputs); no embedding lookup or relative-position bucketing (ids only); cross-checked against the `mm-*` resolution ledgers to avoid duplication.
