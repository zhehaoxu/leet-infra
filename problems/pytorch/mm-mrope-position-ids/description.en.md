# Multimodal: Qwen2-VL-Style M-RoPE Position IDs

## Problem

Qwen2-VL uses **M-RoPE (Multimodal Rotary Position Embedding)** to split position numbering into three components \((t,h,w)\), so that image tokens occupy one dimension each along height/width and text tokens advance on all three axes together, thereby encoding 1-D text and 2-D visual positions in the same sequence. This problem implements a simplified, deterministic version containing **only images and text (no video)**.

Rules (starting from a "running counter" `running` and scanning the sequence left to right):
- **Text token**: all three axes take `running`, i.e. \((t,h,w)=(r,r,r)\); then `running += 1`.
- **Image tokens**: a **contiguous** run of \(h\times w\) image tokens is processed at once (\(h,w\) are given by that image's grid). The \(k\)-th one (row-major, \(r=k\,/\!/ \,w,\ c=k\,\%\,w\)) is numbered
  \[ (t,h,w)=(\text{running},\ \text{running}+r,\ \text{running}+c) \]
  i.e. \(t\) is fixed for the whole block, and within the block it is offset by row/column. After processing the entire image, `running += max(h, w)`.

## Rules (interface)

Implement `solve(mm_token_type, grid_thw)`. `mm_token_type` (int, \((L,)\), 1=image token, 0=text token; image tokens always form contiguous blocks); `grid_thw` (int, \((M,3)\), giving \((t,h,w)\) of each image in order of appearance; in this problem \(t\) is always given as 1 and only \(h,w\) are used). Return an int64 tensor \((3, L)\) whose three rows are the \(t/h/w\) position IDs in that order.

## Examples

Sequence `text text [2×3 image] text`, `mm_token_type=[0,0,1,1,1,1,1,1,0]`, `grid_thw=[[1,2,3]]`:

```
t: [0, 1, 2, 2, 2, 2, 2, 2, 5]
h: [0, 1, 2, 2, 2, 3, 3, 3, 5]
w: [0, 1, 2, 3, 4, 2, 3, 4, 5]
out shape = (3, 9)
```

## Constraints

- Image tokens form contiguous blocks corresponding one-to-one with the order of `grid_thw`, and the block size \(h\cdot w\) matches the number of 1s in `mm_token_type`.
- Pure integer position construction, no floating point.
