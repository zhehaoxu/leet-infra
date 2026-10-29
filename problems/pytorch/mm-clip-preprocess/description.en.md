# Multimodal: CLIP Center Crop and Pixel Normalization

Before an image enters a vision encoder, its pixel region and channel values need a consistent representation. Implement one CLIP-style preprocessing stage: **center crop → pixel rescaling → RGB normalization → CHW layout**.

This is not a complete CLIP image processor: there is no image resizing, padding, or color conversion. The editor's starter contains the full parameter and error contract.

## Rules

1. The crop starts at
   $$
   \mathrm{top}=\lfloor(H-h)/2\rfloor,\qquad
   \mathrm{left}=\lfloor(W-w)/2\rfloor.
   $$
   For an odd difference, discard the extra row or column at the bottom or right. Do not replace floor with rounding to the nearest integer.
2. Convert the cropped pixels to float32, then divide by 255. Subtract the mean and divide by the standard deviation for each RGB channel. Convert the mean and standard deviation to float32 as well; use float32 arithmetic at every stage.
3. Move channels to the front and return contiguous storage. Normalized values may be negative or greater than 1; do not clamp them.

Do not modify the input or use PIL or torchvision. Invalid crops must raise an error, not be repaired with resizing or padding.

## Example

The image has the following two rows:

```text
image = [
  [[  0, 255,   0], [255,   0, 255], [ 51, 102, 153]],
  [[255, 255, 255], [  0,   0,   0], [204, 153, 102]]
]
crop_size = (2, 2)
mean = [0.5, 0.0, 0.5]
std  = [0.5, 1.0, 0.25]
```

Output, shown as the R, G, and B planes:

```text
[
  [[-1.0, 1.0], [1.0, -1.0]],
  [[ 1.0, 0.0], [1.0,  0.0]],
  [[-2.0, 2.0], [2.0, -2.0]]
]
```

The output was recorded from the local reference implementation and checked against an independent pixel-loop oracle. Only the cropped region participates; the third column is excluded.
