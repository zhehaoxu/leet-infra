# In Detail: Qwen2-VL-Style M-RoPE Position IDs

## Source
The idea comes from the M-RoPE of Bai et al., *Qwen2-VL* (arXiv:2409.12191): position numbering is split into three components \((t,h,w)\), images are expanded along height/width, and text advances on all three axes together, so that rotary position encoding can express 1-D text and 2-D vision at the same time. This problem implements a simplified deterministic version with **images + text only** (the full algorithm includes a video time dimension, which is not covered here).

## Core points and common mistakes
1. **The three axes share a single `running` base**: text sets all three axes to `running` and then increments by 1; an image block keeps \(t\) constantly at `running` and adds the row/column offsets to \(h/w\) on top of `running`. Do not count the three axes independently.
2. **After an image block, `running += max(h,w)`** (not h·w): this way an image only advances the diagonal span max(h,w), leaving room for the 2-D expansion — this is the easiest detail to get wrong.
3. **Image tokens must form contiguous blocks**, corresponding one-to-one with the order of `grid_thw`, with block size = \(h\cdot w\).
4. The output is \((3,L)\) integer position IDs, not floating-point features, and grading uses exact equality.

## Cross-check convention
The reference solution scans with a single cursor while-loop and expands block by block. The grading `_oracle` recomputes with pure Python nested lists under the same rules but a different structure (built phase by phase), comparing element-wise with integer `==`. It also asserts "a text-only sequence degenerates to three identical axes with values 1..L", "t is constant inside an image block while h/w form a row-column grid", and "the example values".

## Complexity
\(O(L)\) single pass.
