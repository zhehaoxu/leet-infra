# Smart Resize Alignment and Visual Token Budget

## Scenario

A vision encoder first splits an image into pixel patches, then merges neighboring patches into visual slots for a language model. Compute the target dimensions, patch grid, and slot counts from the image size, patch and merge sizes, and pixel budget. Do not interpolate image pixels.

This problem follows the operation order of **Hugging Face transformers 5.20.2 Qwen2-VL `smart_resize`**. The standalone `qwen-vl-utils` implementation has different minimum-side handling and default budgets; do not mix the two.

## Dimension rules

Let the original size be $H,W$, patch side $p$, merge side $m$, alignment factor $f=pm$, and minimum/maximum pixel counts $L,U$.

Start with

$$
h=f\operatorname{round}(H/f),\qquad w=f\operatorname{round}(W/f).
$$

Use Python **ties-to-even** rounding: `round(2.5)=2` and `round(3.5)=4`.

- If $hw>U$, let $\beta=\sqrt{HW/U}$, then recompute
  $$
  h=\max(f,f\lfloor H/\beta/f\rfloor),\qquad
  w=\max(f,f\lfloor W/\beta/f\rfloor).
  $$
- Otherwise, if $hw<L$, let $\beta=\sqrt{L/(HW)}$, then recompute
  $$
  h=f\lceil H\beta/f\rceil,\qquad w=f\lceil W\beta/f\rceil.
  $$
- Otherwise keep the initially rounded dimensions.

Scaling uses the **original area** $HW$, not the rounded area. Select at most one branch; do not iterate to repair the budget. Follow the displayed Python `float`, `math.sqrt`, `floor`, and `ceil` operation order rather than substituting exact rational square roots.

## Grid and budget accounting

Return a dictionary containing:

- `resized_hw`: Python integer tuple $(h,w)$.
- `grid_thw`: Python integer tuple $(1,h/p,w/p)$. This is a single image, not a video temporal grouping.
- `num_patches`: Python integer $(h/p)(w/p)$.
- `num_visual_tokens`: Python integer `num_patches / m²`.
- `in_budget`: Python Boolean equal to $L\le hw\le U$.

**Do not assume `in_budget` is always true.** Alignment, minimum side lengths, and one-shot branching can violate the final budget. For example, `(28, 5600)` with $p=14,m=2,L=U=3136$ produces `(28, 784)`, outside the budget. Shrinking may also undershoot the lower bound.

## Input domain

Every argument must be a positive Python integer, excluding `bool`. Heights and widths are at most $10^6$, patch/merge sizes at most 128, and budgets at most $10^{12}$. Raise `ValueError` for a non-positive or non-integer argument, failure of $f^2\le L\le U$, or a longer-to-shorter-side ratio **greater than** 200. A ratio of exactly 200 is allowed.

## Example

Original dimensions `(560, 1120)`, `patch_size=14`, `merge_size=2`, `min_pixels=3136`, and `max_pixels=156800` produce

```text
{
  'resized_hw': (280, 560),
  'grid_thw': (1, 20, 40),
  'num_patches': 800,
  'num_visual_tokens': 200,
  'in_budget': True
}
```
