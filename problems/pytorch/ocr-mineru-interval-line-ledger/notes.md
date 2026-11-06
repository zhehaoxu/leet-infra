# 解题笔记：MinerU 检测框的区间裁除、重叠门限与 span 行聚合账

## 回源口径

- 仓库 `opendatalab/MinerU`，默认分支 `master`（不是 `main`）。WebSearch 命中的 deepwiki 描述的是 `chukonu-team/MinerU_2.5` 旧 fork，路径已变，一律以主仓源码为准。
- 通过 GitHub API `git/trees/master?recursive=1` 取树（1521 blobs / 635 个 .py，truncated=False），再用 `contents?ref=master` + base64 下载；`raw.githubusercontent.com` 通道在 833 期已证实不稳，不用。
- 核心源码：`mineru/model/ocr/geometry.py`（200 行）与 `mineru/backend/analysis/pdf/text/lines.py`（126 行），另参考 `mineru/backend/analysis/pdf/constants.py` 的阈值口径与 `mineru/types.py` 的 ContentType 枚举。
- 官方对拍脚本用 `ast.get_source_segment` 原样 exec 官方函数，对 1700 组随机输入比对，全一致。对 `group_spans_to_lines` 需注意官方内部固定 `threshold=0.6` 不透传，对拍脚本必须固定 0.6。

## 机制要点

- 重叠门限：`max(0, min(end) - max(start))` 钳位、`min(extent)` 作分母、严格 `>`、`min_extent <= 0` 不除零。默认 0.8，`lines.py::merge_spans_to_line` 显式传 0.6，`update_det_boxes` 用默认 0.8，两处必须分别覆盖。
- 区间账：`merge_intervals` 按 start 排序、严格 `<` 判不重叠（相邻闭区间合并）、`max` 扩右端；`remove_intervals` 用 ±1 离散整数点语义，遮罩 `[2,4]` 吃掉 2/3/4，前段收在 1、后段从 5 起。
- 行聚合：`interline_equation` 先降级；有效 span 才计入分母；`h/w > 2` 与 `vertical/valid > 0.8` 双严格 `>`；竖排按 `bbox[2]` 降序 + x 轴重叠分组 + 行内 `bbox[1]` 升序，横排按 `bbox[1]` 升序 + y 轴分组 + 行内 `bbox[0]` 升序；特殊类型是「自己或当前行内任意成员」双向或的强制断行。
- 检测框合并：`calculate_is_angle` 用左右两侧高度平均，`not (0.8h <= dy <= 1.2h)` 为 angle；`points_to_bbox` 取 `p1.x/p1.y/p2.x/p3.y`；宽高比严格 `>` 4 触发同行 `merge_overlapping_spans`（其内部亦用严格 `<` 判 x 边界相接）；angle 框只隔离、末尾原样追加。

## 测试设计

- 4 个 `check_*`，每个都有独立 oracle（纯 Python 重写，不 import solution）+ 网格对拍 + 手算锚点。
- 手算锚点刻意钉在所有「严格 vs 非严格」边界：ratio 恰等于阈值、y 轴完全分离负 overlap 钳位、相邻闭区间 `[0,10]`/`[10,20]`、±1 裁除、`h/w == 2`、竖排占比恰 0.8、行内 x0 与 y0 顺序相反、宽高比恰等于阈值、x 边界恰好相接。
- 三个默认值（`threshold=0.6`、`ratio_threshold=4`、`overlap_threshold=0.6`）都有「不传该参数」的独立锚点，否则改成 0.8 / 0 的变异不会被发现。

## 变异设计

- 43 个单点变异，首轮 33/43 CAUGHT、10 SURVIVED：1 个是锚点字符串重复（`gate-overlap-echo` 的 `'"overlap": overlap, "min_extent"'` 在两处 return 各出现一次），9 个是覆盖缺口。
- 补强方式：y 轴完全分离的 bbox 对、h/w 恰 2 与竖排占比恰 0.8 的 span 组、行内 x0/y0 顺序相反的 span 组、宽高比恰等于阈值的 poly、x 边界相接的双 poly，以及三组不传默认参数的调用；锚点改为 `'"overlap": overlap,'` 且 count=2 一次替换两处。最终 43/43 CAUGHT, 0 SURVIVED。
- 一处手算锚点写反（`h/w == 2` 误记成算竖排）导致正确解也失败、变异假阳性全 CAUGHT；修正为 `vertical_span_count == 0` 后 judge 双模式与 mutation 才同时成立。教训：mutation 全绿不能替代正确解自检。

## 查重划界

- 与 `ocr-layout-column-sort`（整页双栏阅读顺序状态机）、`ocr-deepseek-tile-ledger`（动态分辨率 tile 与 token 序列账）、`ocr-olmocr-image-merge`（union-find 容差合并）、`ocr-dbnet-unclip-thresh`（DB 后处理单框）赛道不同：本题只取 geometry/lines 里的区间与行聚合纯账，不碰整页排序、不碰 raster、不碰 pyclipper。
- 与 `grid-sample` / `upsample-bilinear` 等插值题、data-tokenization 的分段题亦无重叠。
- 不进判题域：真检测前向、PIL/torchvision、numpy 四点数组、标题分级 LLM 调用、LaTeX 定界符切分、表格跨页合并与 markdown 渲染。
