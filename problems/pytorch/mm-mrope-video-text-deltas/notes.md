# 多模态：Qwen2.5-VL 完整 M-RoPE 索引（视频时间轴与交错账）

## 背景

Qwen2-VL / Qwen2.5-VL 的 M-RoPE 把旋转位置编码拆成时间 / 高度 / 宽度三条轴：文本 token 三轴同步递增，图像 token 沿 (t, h, w) 网格展开，视频 token 的时间轴还要按「每格秒数 × 每秒 token 数」量化。库里的简化版 `mm-mrope-position-ids`（order 228）只覆盖了 Qwen2-VL 的文本 + 单个连续图像块；本题实现 transformers `Qwen2_5_VLForConditionalGeneration.get_rope_index`（v4.51.3）的完整账：视频时间轴量化、多图 / 视频交错分段、`mrope_position_deltas`，以及 padding 的处理。

## 任务

给定单条 token 序列、0/1 注意力掩码、图像与视频的 `grid_thw` 列表、每视频的 `second_per_grid_ts`、`spatial_merge_size` 与 `tokens_per_second`，输出：

- `position_ids`：`[3][L]` 的 int 嵌套列表（时间 / 高度 / 宽度三行）；
- `mrope_position_delta`：单个 int，解码期把 `cache_position` 映回 M-RoPE 坐标用的差值。

## 机制账

1. **计数**：只有紧跟在 `vision_start` 之后的 `image_token` / `video_token` 才计入段数（官方 `vision_tokens = input_ids[vision_start_indices + 1]`）。裸的视觉 token 不计数，但扫描定位用的 `index` 会找到它——两个口径不一致是官方 quirk，本题原样保留（见用例 `check_bare_image_token_quirk`）。
2. **分段扫描**：循环 `image_nums + video_nums` 次；每轮从 `st` 起找 image / video token 的首次出现，位置较小者先处理；段内先生成文本部分（三轴 `arange(text_len) + st_idx`，`st_idx` 是此前所有段三轴的全局最大值 + 1），再生成视觉部分（时间轴 `floor(i × second_per_grid_t × tokens_per_second)`，高度 / 宽度轴按 C 序展开，三轴都加 `text_len + st_idx`）。`st` 推进到 `ed + llm_grid_t·llm_grid_h·llm_grid_w`。
3. **图像时间轴恒为常数**：图像段的 `second_per_grid_t = 0`，所以即使 `grid_t = 2`，时间轴也全是同一个值（`check_image_temporal_axis_constant` 钉死）。只有视频段的时间轴会展开。
4. **取整口径**：时间步是 float 乘法后 `int()` 截断（复刻官方 `time_tensor.long()`）。`second_per_grid_t = 0.5`、`tokens_per_second = 25` 时步长 12.5 → 12、37.5 → 37；若实现用 Python `round`，37.5 会被舍成 38（half-to-even 上取整），`check_video_floor_half_odd` 专门钉这个点。
5. **padding**：`mask == 0` 的位置填 **1** 而不是 0——官方把 `position_ids` 初始化成 `torch.ones` 后只在 `mask == 1` 处写入，padding 位的 1 从不被覆盖（`check_main_path_padding` / `check_pure_text_padding_mask`）。
6. **delta 的全长 quirk**：`mrope_position_delta = 全局最大值 + 1 − L`，其中 L 是**含 padding 的全长**（官方写的是 `len(total_input_ids[i])`），不是有效长度。padding 越多 delta 越负。
7. **纯文本 fallback**：两个 grid 都为空时走官方另一条分支——三轴同为 `cumsum(mask) − 1`，`mask == 0` 处同样填 1，delta 同公式。官方入口条件是「grid 张量 is not None」，但在纯文本序列上主路径与 fallback 的结果完全一致（全 1 mask 时主路径尾部段就是 `arange`），本题按可观察行为统一成「两个 grid 都空 → fallback」。

## 与官方代码的刻意差异

- 官方 `if image_token_id in input_tokens and remain_images > 0` 用**全序列**判断、`index(..., st)` 却**从 st 起**定位，两者不一致：当 remain > 0 但视觉 token 都在 st 之前时官方会崩。本题统一为从 `st` 起判断（与 index 一致）。
- 官方对 `h // spatial_merge_size` 不做整除校验，商为 0 时会产生零长度视觉段并把 `st` 推错位。本题显式要求 h、w 能被 merge 整除且商 ≥ 1，否则 ValueError。
- 官方在段数与 grid 数不符、`vision_start` 是最后一个有效 token 时抛的是 IndexError / 形状错乱。本题全部显式化为 ValueError。
- 官方多 batch 逐行处理；本题固定单行（batch 维省略），账目逐 token 完全一致。

## 实现要点

solution 单循环扫描：`eff`（掩码过滤后的有效序列）上做 `index` 定位，段的文本 / 视觉两小节直接 append 到 `segments`，最后按 `attention_mask` 的有效位摊回 `[3][L]`。全局最大值用于两处：下一段的 `st_idx` 与 delta。h / w 轴的 C 序展开用三层推导式，与官方 `expand(...).flatten()` 语义一致。

## 独立验证

oracle 把官方单循环改写成两遍结构：先解析成 `("text", n)` / `("vis", t, gh, gw, sec)` 段列表，再逐段生成位置，段间偏移用「当前三轴全局最大值 + 1」重算。生成时用 `math.floor` 显式表达截断口径。与 solution 不共享任何中间变量；15 个 `check_*` 里 13 个结构用例逐元素对拍，外加输入不变性与 16 组非法输入。

## 覆盖清单

纯文本 fallback（全 1 掩码 / 尾 padding / 中间有洞）、单图四 token 展开、图像时间轴常数 quirk、视频时间步 floor（含 37.5 边界）、`second_per_grid_ts=None` 默认 1.0、图视频交错分段偏移链、merge 除法、裸 image token quirk、双图偏移链、主路径 padding、输入不变性、非法输入组。变异 13 个可观察变异全部 CAUGHT；1 个等价变异已分析：`merge-ceil`（在「h、w 必须整除」的前置校验下 floor 除与 ceil 除恒等），已替换为 h/w 轴交换变异。

## 复杂度

设有效长度 n、视觉 token 总数 v。扫描与分段为 O(n + v·3)；摊回输出为 O(3L)。合计 O(L + v)，除输出外只需段列表的 O(v) 中间量。

## 来源与范围

依据已回源的 transformers v4.51.3 `src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py::get_rope_index`（GitHub API contents + base64 抓取；main 分支已重构出该文件，本题钉 v4.51.3 的经典实现）。本题不含 rotary embedding 的频率计算、`mrope_section` 的三轴分拆、视觉塔内部的窗口注意力（`mm-qwen25vl-window-index` 已覆盖）、`prepare_inputs_for_generation` 的 delta 消费路径；它只是位置索引账的确定性复刻。注意 token 协议：视觉段必须以 `vision_start` 开头——这是官方 processor 拼序列的约定。

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-mrope-video-text-deltas --mode submit
```
