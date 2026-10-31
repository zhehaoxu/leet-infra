# 详解：InternVL 动态分块账

## 出处

OpenGVLab/InternVL 仓库 `internvl_chat/internvl/train/dataset.py` 的 `find_closest_aspect_ratio` 与 `dynamic_preprocess`（2026-10-31 经 GitHub contents API 逐行取得），token 账取自 transformers 5.20.2 `processing_internvl.py`（`image_seq_length = (image_size // patch_size)**2 * scale_factor**2`；`num_image_tokens = 2 + image_seq_length * num_patches`）。

## 核心与易错点

1. **候选生成是三层循环**：对 `n ∈ [min_num, max_num]`，枚举 `i, j ≤ n` 且 `i*j ∈ [min_num, max_num]`——不是所有 `i*j ≤ max_num` 的对（`i` 还受 `n` 约束，但去重后恰好等价于 `i*j ≤ max_num` 的全体，因对任意 `(i,j)` 取 `n = max_num` 即覆盖）。题面钉死按 `(i*j, i, j)` 升序去重。
2. **上游的 set 迭代序不可复刻**：官方用 `set` 收集再 `sorted(key=i*j)`，并列顺序依赖 CPython set 实现——跨版本不可判。题面钉死为 `(i*j, i, j)` 字典序，并在 notes 记录此规范化。变异测试证明 `sorted(ratios)`（无 key）在本题可测空间内**观测等价**：顺序只影响并列时的先后，而同积并列的面积条件相同、异积等距对被浮点精确性阻挡（`13/12` 类构造的两侧 diff 差 1 ulp）。
3. **面积规则的方向**：`area > 0.5 * image_size² * i * j` 时才覆盖——是「原图面积超过该网格拼合面积的一半」，不是「网格面积超过原图」。宽图 896×448 的 r=2 虽然 `(4,2)` 同 diff=0，但 `401408 < 802816` 不覆盖，保持 `(2,1)`。首版手算锚点在这里错了一次（以为会覆盖成 8 块）。
4. **浮点并列构造**：`13/12` 与 `2/3`、`3/2` 数学上等距，但浮点上 diff 相差 1 ulp，严格小于分支直接决出——构造并列必须用二进制精确值：`r=5/4` 与 `0.5`、`2.0` 的 diff 均为精确的 `0.75`。`500×400`（area 200000 < 200704 保持 `(1,2)`）与 `505×404`（area 204020 > 200704 覆盖成 `(2,1)`）两侧各一例。
5. **缩略块条件**：`use_thumbnail 且 blocks != 1`——单块图不加缩略（缩略就是它自己）；`use_thumbnail=1`（非 bool）要拒绝。
6. **token 账三层**：每 tile `(448//14)² * 0.5² = 256`；总 token `256 * total_tiles`；`<img>` 文本占位再 +2。`scale_factor` 忘了平方是高频错（变异 `no-scale2` 被 4 例抓）。

## 变异测试（9 有效全抓 + 1 等价）

`<` 改 `<=`、面积规则阈值改成 `0.5*size²`、ratio 行列互换、去缩略、恒缩略、scale 因子忘平方、`+2` 丢掉、`best_diff` 初值改 1.0、排序去 key。**等价变异**：`sorted(ratios)` 去 key（见第 2 点）。

## 复杂度

候选网格 O(max_num²)，微秒级。
