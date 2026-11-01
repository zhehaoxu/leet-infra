# PP-Structure Recovery Two-Column Reading-Order Ledger

PaddleOCR's PP-Structure recovery module calls `sorted_layout_boxes` in
`ppstructure/recovery/recovery_to_doc.py` to reorder regions into reading
order before exporting to docx: two-column documents are emitted as "the
left column top-to-bottom, then the right column top-to-bottom", while
full-width blocks (titles, spanning tables) act as `single` separators that
cut the column grouping. Replicate this state machine rule by rule.

## Interface

```python
solve(bboxes: list[list[float]], w: float) -> dict
```

- `bboxes`: layout boxes `[x0, y0, x1, y1]`; the list index is the original
  order;
- `w`: page width;
- return `{"order": [original indices...], "layouts": ["single"/"double"...]}`,
  where `order` is the permuted output order and `layouts` is aligned with it.

## Rules

1. **Single-box passthrough**: when `len(bboxes) == 1`, return
   `order=[0], layouts=["single"]` without sorting; empty input returns two
   empty lists.
2. **Scan order**: otherwise first stable-sort by `(y0, x0)` ascending
   (boxes with equal keys keep their original relative order) to get the
   scan sequence.
3. **Scan state machine**: maintain a left bucket and a right bucket and
   process boxes left to right:
   - **Last-box branch** (current box is the last of the scan sequence):
     - if `y0 > prev box's y1` and `x0 < w/2` and `x1 > w/2` (a vertical gap
       to the previous box and the box spans the middle line): flush the
       left bucket, then the right bucket, and append the last box as
       `single`;
     - else if `x1 > w/2`: append the last box as `double` into the right
       bucket, then flush left then right (the last box sits **inside** the
       right-bucket group);
     - else if `x0 < w/2`: append the last box as `double` into the left
       bucket, then flush left then right (the last box sits inside the
       left-bucket group);
     - otherwise (only a degenerate zero-width box with
       `x0 == x1 == w/2` falls here): **silently drop** it, and **discard
       both accumulated buckets without emitting them**.
   - **Non-last boxes**:
     - `x0 < w/4` and `x1 < 3w/4`: mark `double`, push into the left bucket;
     - else `x0 > w/4` and `x1 > w/2`: mark `double`, push into the right
       bucket;
     - else: flush the left bucket then the right bucket, and append this
       box as `single` (a separator between column groups).
4. Every comparison uses a **strict** inequality; the flush order is always
   "left bucket first, then right bucket".

## Example

```python
solve([[50,100,300,200],[600,100,900,200],
       [50,250,300,350],[600,250,900,350]], 1000)
# alternating columns: last box (3) joins the right bucket, then flush
# -> {"order":[0,2,1,3], "layouts":["double"]*4}

solve([[50,100,300,200],[600,100,900,200],
       [100,400,900,500],[50,600,300,700]], 1000)
# box 2 is full-width -> flush then single; box (3) joins the left bucket
# -> {"order":[0,1,2,3], "layouts":["double","double","single","double"]}
```

## Hints

- The two `double` branches of the last box flush at different moments: the
  spanning `single` branch is "flush then append", while the column branches
  are "push then flush" — the output order differs.
- The degenerate zero-width drop branch discards both buckets, so the
  output can be an empty list.
- The official loop only exits through the last-box branch or the empty
  sequence exit; the trailing flush after the loop is unreachable.
