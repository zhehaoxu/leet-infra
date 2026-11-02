# NeMo TDT token-and-duration 贪心解码账

## 题意

TDT（Token-and-Duration Transducer，arXiv 2304.06795）把 RNN-T 的单一 blank 通道拆成两路：token 头照旧含 blank，另设一个 duration 头直接预测"这次发射/停留跨几帧"。NeMo 的贪心解码器因此比 RNN-T 多一整套跳帧账。本题手写 GreedyTDTInfer.torch_impl 的标量无状态化：predictor 不建模（无状态），joint 的两路 logits 按帧显式入参，纯标量循环复现官方解码全程。

## 求解

官方口径（NVIDIA/NeMo `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py` 逐条钉死）：

- **双 argmax**：`token = argmax(token 部分)`、`duration = model_durations[argmax(duration 头)]`，并列均取首个下标（torch.argmax 语义）；
- **blank 强制 duration ≥ 1**：`durations.masked_fill_(durations == 0 & blank_mask, 1)`——blank 不允许原地停留（原地停留是 RNN-T blank 的语义，TDT 用 duration 表达）；
- **非 blank 同样按 duration 跳帧**：这是 TDT 与 RNN-T 的本质差异——发射后直接跳 duration 帧，无需 blank 收尾；duration 可把时间推过句末，发射发生在跳帧前，仍有效（官方注释 *NB: difference between RNN-T and TDT*）；
- **发射账**：token 发射于跳帧前的时间戳（`time_indices_current_labels` 在 advance 前快照）；durations 记录模型预测原值（可含 0）；
- **max_symbols 防循环**：非 blank duration=0 时会原地重复发射（label loop）。官方用 max_symbols 防护——同一时间戳连续非 blank 发射数达 max_symbols 后强制 blank，且 `time_indices += 1`（**只进 1 帧，忽略 duration**，不是 duration=1 的语义）；
- **域约束**：max_symbols is None 且出现非 blank duration=0 → ValueError（无状态 predictor 下官方会无限循环）；
- **终止**：`time >= encoder_output_length`（active_mask 全 False）；不存在 RNN-T 的"发射后需 blank 才能结束"——非 blank + duration 直接出界即结束，final_t 可 > T。

## 数据规模

- V=4（blank + 3 token）、D=4（durations [0,1,2,3]）缩微配置，argmax 恰定的 logits 构造；
- 手算锚点：basic（末 token d3 跳出句末 → final_t > T）、zero-duration-domain（ValueError）、max_symbols=1（每个非 blank 后都强制 blank）、max_symbols=2（两次连续发射后强制）、blank-zero（d0 blank 被强制为 1）、出界终止；
- 随机对拍 200 组（oracle 为 torch_impl 标量转录）。

## 评分

五键账逐项相等：tokens / timestamps / durations / final_t / forced_blanks 精确相等（全整数账）。

## 红线

- 不得调用 nemo / numba 的任何函数；纯 Python 标量循环账；
- model_durations 非负；token_logits/duration_logits 帧数一致、宽度一致（否则 ValueError）。
