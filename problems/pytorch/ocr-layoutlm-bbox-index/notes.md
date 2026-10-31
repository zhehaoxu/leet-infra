# notes — ocr-layoutlm-bbox-index（order 669）

## 回源

- 出处：HF `transformers` `models/layoutlm{,v2}/modeling_layoutlm*.py` 的 `_calc_spatial_position_embeddings`（unilm `layoutlmft` 同源）：
  - 四角分别查 `x_position_embeddings(bbox[:,:,0|2])` 与 `y_position_embeddings(bbox[:,:,1|3])`；
  - `h = bbox[:,:,3] - bbox[:,:,1]`、`w = bbox[:,:,2] - bbox[:,:,0]` 查 shape 嵌入；
  - 坐标域 0–1000（HF 超 Range 直接 raise IndexError "should be within 0-1000 range"；unilm 管线喂入前归一化 + clamp，题面把 clamp 钉进账本）。
- v2 的 1D position_ids：文本 `arange(seq_len)`，视觉 token 的 position_ids **从 0 重新开始**（不累加文本长度）；相对 2D 偏置取 `x0` 与 `y1`（lower）坐标差分分桶。本题取网格索引 `y * 1001 + x` 的可观察账，不实现嵌入查表与分桶。
- 与 `mm-qwen25vl-window-index` 划界：Qwen VL 是**窗口/patch 分辨率账**，本题是**文档 bbox 坐标账**；与 #107⑦ olmOCR 跨页锚定划界：这是模型内单页索引。

## 判题面

- solve 入参：boxes（像素坐标）、page_w、page_h。返回 norm_boxes/wh_ids/corner_grid/center_grid。
- oracle：`test_cases._py` 独立实现（int(round())、divmod），150 组随机对拍（页尺寸 100–3508、box 随机生成）。
- 锚点：page 1000×500 的 y 翻倍账、四角/中心网格手算；负坐标与超页 clamp（恰 1000 不截断）；banker's rounding（page=2000、x=3 → 1.5 → 2；x=1 → 0.5 → 0）；全页与退化 box（w/h=0）；归一化后 x0>x1 → ValueError。
- 非法输入：boxes 空/非 4 元/非数/NaN/Inf/bool、page 尺寸非正或 bool → ValueError。

## 变异记录（9 条全抓）

no-clamp-lo / no-clamp-hi（两端 clamp 失效）、round-to-int（banker's → 截断）、norm-x-use-page-h（x 用页高）、grid-base-1000（网格 1001→1000）、grid-transpose（行列互换）、wh-swap（宽高互换）、center-ceil（中心 //2 → 向上取整）、center-grid-corner（中心网格换成左上角）。

## 红线

真图/PIL/torchvision 不出（坐标全入参）；嵌入查表与相对位置分桶不出（只考 id 账）；与 `mm-*` 分辨率账互查不重出。
