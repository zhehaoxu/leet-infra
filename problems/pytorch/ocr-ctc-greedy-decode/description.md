# CTC Greedy 解码与 collapse 账

## 题意

OCR 识别头的输出是逐帧分类 logits，CTC（Connectionist Temporal Classification）greedy 解码把它变成文本：逐帧取 argmax，相邻重复帧合并（collapse），再丢弃 blank。本题转写 PP-OCR `BaseRecLabelDecode.decode` 的 strict 账本：给定 `[T, V]` logits 与 `blank_id`，输出逐帧 argmax 路径、解码出的 token 序列，以及每个 token 的帧区间。

## 求解

口径题面钉死（与官方一致）：

1. `raw_path[t] = argmax(logits[t])`，**并列取最小类 id**；
2. 相邻去重在**原始序列**上判定（含 blank）：连续相同 id 合并为一个 run。因此 `A blank A` 的两个 A 不相邻、都保留——blank 分隔语义是 CTC 的核心；
3. blank 的 run 整体丢弃；
4. `spans[i]` 为输出第 i 个 token 的 run 帧区间 `[start, end]`（**闭区间**），run 起于该 id 首帧、止于连续相同 id 的末帧。

官方实现是 selection 掩码（`selection[1:] = cur != prev` 再叠加 `cur != blank`），本题按 run 循环实现，两者逐位等价。返回 `{"raw_path", "tokens", "spans"}`。

## 数据规模

- 1 ≤ T ≤ 10^4，2 ≤ V ≤ 10^5，logits 为有限数值。
- logits 空/行空/行不等长/含非数或 NaN/Inf、blank_id 非 int/越界 → `ValueError`。
