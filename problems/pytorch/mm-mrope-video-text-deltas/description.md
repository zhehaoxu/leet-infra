# 多模态：Qwen2.5-VL 完整 M-RoPE 索引（视频时间轴与交错账）

## 题面

Qwen2.5-VL 的 M-RoPE 把位置编码拆成时间 / 高度 / 宽度三条轴。给定：

- `tokens`：单条 token 序列（list[int]，长度 1..32）；
- `attention_mask`：0/1 列表，0 是 padding；
- `image_grid_thw` / `video_grid_thw`：各图的 (t, h, w) 网格；
- `second_per_grid_ts`：每个视频格沿时间轴的秒数（None 时默认 1.0）；
- `spatial_merge_size`、`tokens_per_second`；以及 `image_token_id` / `video_token_id` / `vision_start_token_id`。

实现 transformers `get_rope_index`（v4.51.3）的完整账，返回 `{"position_ids": [t, h, w 三行], "mrope_position_delta": int}`。

## 规则

1. **计数**：只有紧跟 `vision_start` 的 image / video token 计入段数；裸的视觉 token 不计数，但扫描定位仍会找到它（官方 quirk，见用例）。段数与 grid 数必须一致。
2. **分段**：按视觉 token 首现位置从小到大处理。每段先文本（三轴 `arange + (此前全局最大值+1)`）后视觉：
   - 时间轴 = `floor(i × second_per_grid_t × tokens_per_second)`，其中图像恒 `second_per_grid_t = 0`（时间轴为常数！），视频取参数值；
   - 高度 / 宽度轴按 (t, h', w') C 序展开（h' = h // merge，w' = w // merge）；
   - 三轴都加 `text_len + st_idx`；`st` 推进 `ed + t·h'·w'`。
3. **padding**：`mask == 0` 的 `position_ids` 位填 **1**（官方 ones 初始化后从不覆盖）。
4. **delta** = 三轴全局最大值 + 1 − **L（含 padding 的全长）**。
5. **fallback**：两个 grid 都空时，三轴 = `cumsum(mask) − 1`，padding 位同样填 1。

## 样例

```text
输入:  tokens = [7, 102, 100, 100, 100, 100, 8, 9]（102=start，100=image）
       image_grid_thw = [[1,2,2]], spatial_merge_size = 1, tokens_per_second = 25
输出:  t = [0, 1, 2, 2, 2, 2, 4, 5]
       h = [0, 1, 2, 2, 3, 3, 4, 5]
       w = [0, 1, 2, 3, 2, 3, 4, 5]
       delta = 5 + 1 - 8 = -2
（前文本两 token [0,1]；视觉 off=2：t 恒 2、h/w 网格展开；尾文本 = 全局 max(3)+1=4 起）

输入:  tokens = [102, 200, 200, 200, 9], video_grid_thw = [[4,1,1]],
       second_per_grid_ts = [0.5], tokens_per_second = 25
输出:  t = [0, 1, 13, 26, 38]
（时间步 floor(12.5)=12、floor(37.5)=37——Python round 的 half-to-even 会把 37.5 舍成 38，本题钉 int 截断）
```

## 数据范围与边界

- 1 ≤ L ≤ 32；mask 至少一个 1；三个 token id 两两不等。
- grid 的 t/h/w 为正 int，h、w 必须能被 merge 整除且商 ≥ 1。
- `second_per_grid_ts` 长度等于视频段数，元素 ≥ 0。
- 非法输入（bool 冒充 int、mask 全 0、计数与 grid 不符、start 是最后一位等）一律 ValueError。
- 禁止 torch，纯 Python；输入不被修改。

## 提示

- 偏移链的 `st_idx` 是**三轴全局**最大值 + 1，不是宽度轴单轴；图像段时间轴不展开是最容易漏的官方 quirk。
- padding 位是 1 不是 0；delta 减的是全长 L，这两点都用 `ones 初始化` / `total_input_ids` 的官方原文钉死。
