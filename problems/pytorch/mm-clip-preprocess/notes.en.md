# Solution Notes: CLIP Center Crop and Pixel Normalization

## Actual source and scope

The crop and normalization conventions come from **Hugging Face Transformers' shared image transformation utilities**, not from assumptions based on the CLIP name. The following local files were inspected in this repository's `transformers==5.20.2` environment:

- `.venv-eval/lib/python3.12/site-packages/transformers/image_transforms.py`: `center_crop` starts at line 446, with offsets at lines 493–498; `normalize` starts at line 385, with dtype and channel-parameter conversion at lines 418–440.
- `rescale` in the same file, lines 90–125, defaults to float32 output but first multiplies in float64 before casting back. `normalize` preserves existing floating-point dtypes, so it uses float32 after the default rescale; it does not force every invocation to float32.
- [Upstream source entry](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py). The version and line numbers above refer to the inspected local file; upstream main may change.

**This is a scoped educational implementation, not an exact reproduction of complete CLIP/CLIPImageProcessor preprocessing.** Input is restricted to CPU uint8 HWC RGB. Only in-bounds center cropping, division by 255, channel normalization, and layout conversion are included. There is no resizing, PIL, color conversion, or padding. The HF utility may pad undersized images; this task requires ValueError instead. This task explicitly casts to float32 before dividing by 255, rather than copying HF rescale's internal float64 multiplication, and does not promise cross-backend bitwise identity.

Cropping follows HF's floor convention, not torchvision's commonly used `int(round(diff/2))`. In a local scalar run, a difference of 3 produced 1 with `3//2` and 2 with `round(3/2)`. Tests cover a difference of 3 on both axes and on each axis separately. No torchvision function was executed, and this task does not claim to reproduce its entire crop API.

## Implementation

1. Validate only the crop: two positive integers fitting within the source image. Invalid type, length, entries, or bounds raise ValueError. Other inputs are valid by the starter contract; no extra error contract is added.
2. Compute independent floor offsets for height and width and slice the HWC region.
3. Convert to independent float32 storage. Construct float32 RGB mean/std vectors and broadcast over the last axis. Divide by 255, subtract the mean, and divide by the standard deviation, in that order.
4. Apply `permute(2, 0, 1)` and then `contiguous()`. Reordering dimensions alone does not guarantee contiguous CHW storage.

Conversion from uint8 to float32 separates storage from the input, so later operations do not write back to the image. Do not subtract in uint8, move division by 255 after normalization, swap RGB and BGR, or clamp normalized values to a pixel range.

Time and output space are both $O(hw)$ for the cropped region, with a fixed channel count. Pixels outside the crop need not be copied.

## Independent validation and recorded example

The oracle neither imports the reference solution nor shares its slicing/broadcasting path. It converts input into nested Python lists, reads individual pixels in channel/row/column loops, performs Python float arithmetic, and rounds the rescale, subtraction, and division stages to float32 using `struct.pack/unpack` before constructing the expected tensor. Comparison uses `atol=rtol=1e-6`, with separate checks for dtype, shape, CPU placement, contiguity, and input/parameter immutability.

On local PyTorch **2.14.1**, `add_problem.py --selftest mm-clip-preprocess --mode submit` reported **12/12 ACCEPTED**. Coverage includes difference-3 offsets, rectangles, full crops, distinct channels, 0/255, noncontiguous and channel-strided views, single axes/pixels, float32 rounding, invalid crops, maximum sizes, and random inputs from fixed `torch.Generator` instances. `check_example.example=True`.

Recorded output for the statement's small image, in R/G/B order:

```text
R = [[-1.0, 1.0], [1.0, -1.0]]
G = [[ 1.0, 0.0], [1.0,  0.0]]
B = [[-2.0, 2.0], [2.0, -2.0]]
```

These values were recorded by executing the reference implementation and then validated by the independent oracle. The complete HF processor was not used as a testing black box, and no CLIP model was downloaded or run.
