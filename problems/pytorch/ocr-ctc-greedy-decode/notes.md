# notes — ocr-ctc-greedy-decode（order 668）

## 回源

- 出处：`PaddlePaddle/PaddleOCR` `ppocr/postprocess/rec_postprocess.py` 的 `BaseRecLabelDecode.decode`（`CTCLabelDecode` 继承之，仅重写字典头插 `"blank"` 与 `get_ignored_tokens() → [0]`）。`__call__` 固定 `is_remove_duplicate=True`。
- 官方 selection 掩码语义：`selection[1:] = text_index[1:] != text_index[:-1]`（**原始序列相邻比较，含 blank**）再叠加 blank 过滤。两个掩码 `&=` 组合，交换顺序不影响保留集合，但「去重判定基于含 blank 的原始序列」是语义关键——`A blank A` 两个 A 都保留。
- 与 `ctc-loss`（loss 侧已占）划界：本题只考解码/后处理路径，不做前向-后向 DP。
- 增加 spans 账（官方 `get_word_info` 的 frame 区间语义简化版）：run 的 `[start, end]` 闭区间。

## 判题面

- solve 入参：logits [T, V]、blank_id（默认 0）。返回 raw_path/tokens/spans。
- oracle：`test_cases._py` 照官方 selection 掩码路径独立实现（argmax 用 `index(max())`），200 组随机对拍（T 1–20、V 2–6、60% 概率尖峰造重复/blank 路径）。
- 锚点：run 与 blank 交错手算（`[1,1,0,2,2,2] → tokens [1,2]`、spans `[[0,1],[3,5]]`）；A blank A 都保留；A A blank A A 两 run；全 blank 空输出；首尾 blank；argmax 并列取小 id；自定义 blank_id 换角色。
- 非法输入：空/行空/行不齐/非数/NaN/Inf、blank_id bool/越界/负 → ValueError。

## 变异记录（9 条全抓）

no-dedup（run 扩展条件反向）、step-instead-of-run（逐帧推进致重复 token）、keep-blank（blank 进输出）、span-end-start（end 记 run 首）、span-end-exclusive（闭改半开）、span-start-shift（start +1）、argmax-tie-max（并列取大）、blank-fixed-zero（忽略 blank_id 入参恒 0）、drop-last-frame（漏末帧 run）。

## 红线

真图/PIL/torchvision 不出；softmax/置信度不做（只取 argmax）；与 #108 `asr` 的 CTC prefix beam search 划界：本题是 **greedy 路径账**，无 beam/DP。
