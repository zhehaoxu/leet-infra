# InternVL 动态分块账：宽高比匹配、tile 网格与 token 计数

## 题目描述

InternVL 系列处理任意宽高比的图像时不做粗暴缩放，而是把原图切成若干 448×448 的 tile，网格形状按宽高比挑选，另加一张整图缩略块。切几张、网格是几乘几、喂给视觉塔多少 token，全由一条确定性的账决定。本题要求把这条账逐项算清。

## 规则

实现 `solve(width, height, min_num, max_num, image_size, use_thumbnail, patch_size=14, scale_factor=0.5)`，返回账本 dict：

| 字段 | 含义 |
|---|---|
| `aspect_ratio` | `width / height` |
| `target_ratio` | 选中的网格 `[i, j]`（i 列 j 行） |
| `target_width` / `target_height` | `image_size * i` / `image_size * j` |
| `blocks` | `i * j` |
| `thumbnails` | `use_thumbnail 且 blocks != 1` 时为 1，否则 0 |
| `total_tiles` | `blocks + thumbnails` |
| `per_tile_tokens` | `(image_size // patch_size)² * scale_factor²` |
| `total_visual_tokens` | `per_tile_tokens * total_tiles` |
| `image_token_count` | `2 + total_visual_tokens`（`<img>` `</img>` 各 1） |

选择规则（与上游逐句对应）：

1. 候选网格：所有 `(i, j)` 满足 `i, j ≥ 1` 且 `min_num ≤ i*j ≤ max_num`，按 `(i*j, i, j)` 升序去重排列。
2. 遍历候选，`diff = |aspect_ratio − i/j|`：
   - `diff < best`（初始 +∞，best 初始 `(1,1)`）→ 更新
   - `diff == best` → 仅当 `width*height > 0.5 * image_size² * i * j` 时覆盖（上游面积规则：原图面积超过该网格拼合面积一半才选更大的网格）
3. `blocks != 1` 且 `use_thumbnail` 时追加一张整图缩略 tile。
4. 任一参数非法（非正整数、`min_num > max_num`）抛 `ValueError`。

## 示例

```
输入：896×448，min=1, max=6, size=448, use_thumbnail=True
r=2 精确命中 (2,1)（diff=0）；(4,2) 同 diff=0 但 area=401408 < 0.5·448²·8=802816，不覆盖
输出：target_ratio=[2,1]，blocks=2，thumbnails=1，total_tiles=3
      per_tile_tokens=256，image_token_count=2+768=770
```

## 约束

- `448/14=32` patch 每边；InternVL2 的 pixel shuffle 0.5 → 每 tile `1024*0.25=256` token
- 面积规则的并列构造必须浮点精确（如 `r=5/4` 与 `0.5`、`2.0` 的 diff 均为精确的 `0.75`）
