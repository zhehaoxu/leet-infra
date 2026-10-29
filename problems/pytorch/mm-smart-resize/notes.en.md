# Smart Resize: Alignment, One-Shot Branching, and Slot Counts

## Implementation

Validate positive integers, budget ordering, and aspect ratio first. The factor is `patch_size * merge_size`, not the patch side alone; this makes both patch-grid dimensions divisible by the merge side.

Choose shrinking or enlargement only after the first rounding step. Always derive the scale from the original area. Shrinking applies `max(factor, ...)` independently to each side; enlargement rounds upward. Derive patch and visual-slot counts from the final dimensions, dividing patch count by the merge **area**, not just its side length.

Time and auxiliary space are both $O(1)$. No image tensor, interpolation, or model execution is involved.

## Why the budget is not guaranteed

For an extremely narrow image, the shorter side cannot shrink further. Clamping it to one factor can push the area above the upper bound. Rounding down can also undershoot the lower bound. There is no second repair pass, so compute `in_budget` from the final area rather than hard-coding true or clipping the result.

`H=28,W=5600,p=14,m=2,L=U=3136` produces `(28,784)`, area 21952, and 28 visual slots. It also exercises the inclusive aspect-ratio boundary of 200.

## Floating-point order is part of the contract

Python ties-to-even differs from round-half-up. Quantization after a square root is also sensitive to binary floating point: `H=442,W=884,p=5,m=2,L=1300,U=7200` produces `(50,110)` upstream, while exact-real algebra yields `(60,120)`. This task reproduces the specified upstream operation order rather than defining an integer search for an optimal area.

Do not replace consecutive divisions with a mathematically equivalent expression that rounds differently, or add epsilon before taking a floor.

## Independent verification and source

The source is the installed `transformers==5.20.2`, file `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`, function `smart_resize` at lines 63–89. Test dimensions are frozen outputs executed from that function, covering enlargement, shrinking, unchanged sizes, odd grid positions, minimum side lengths, and floating-point boundaries. The tests do not copy the reference implementation's formulas.

Additional checks use transpose symmetry, aligned identity cases, ties-to-even parity, alternative patch/merge factorizations of the same factor, and area/patch/slot conservation. Judging does not require transformers, and the problem does not claim equivalence to other versions of the standalone `qwen-vl-utils` implementation.
