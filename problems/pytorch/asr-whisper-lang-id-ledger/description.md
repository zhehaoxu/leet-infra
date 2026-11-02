# Whisper 逐窗语言 ID 账：抑制重归一化与首窗策略

## 题意

Whisper 的语言检测在主解码循环**之外**执行（避免干扰 kv-cache）：单 token `[sot]` 前向得到 SOT 位置 logits，对非语言 token 一律 −inf 抑制后 argmax + softmax。本题手写这条账——不出模型，SOT logits 显式入参，账点全在抑制、重归一化、平分与批处理语义上。

## 求解

官方口径（openai/whisper `whisper/decoding.py::detect_language` + `transcribe.py`，v20231105+）：

- **批语义**：官方对 mel `single = mel.ndim == 2`——2D 视作单条 `unsqueeze(0)` 成批、返回时解开（tokens 标量 + probs 单 dict）；批输入返回张量 + dict 列表。本题对应：logits 1D = 单条；
- **抑制**：mask 全 True、语言位置（`all_language_tokens`）置 False，`logits[:, mask] = -inf`——非语言位是 **−inf 不是 0**（softmax 后严格 0）；
- **argmax**：抑制后取，平分取**首个下标**（torch.argmax 语义）；
- **重归一化**：`language_token_probs = logits.softmax(dim=-1)` 在抑制后算——等价于在语言集合上**重新归一化**；raw softmax（对原始 logits 全词表）的语言位概率 ≠ 此值（非语言质量被剔除）；probs dict 只收语言位，键按 `all_language_codes` 顺序插入；
- **首窗策略**（transcribe 侧）：`language=None` 才检测，且只跑**首个 30s 窗**（`pad_or_trim(mel, N_FRAMES)`），后续窗沿用首窗结果（语言切换不感知）；显式 `language` 跳过检测；
- **选定规则**：`max(probs, key=probs.get)`——Python max 平分取**首个最大键**（插入序）；
- **异常**：空语言表 / 长度不齐 → `ValueError`（官方 "This model doesn't have language tokens so it can't perform lang id"）。

## 数据规模

- 缩微语言表（5 语言、词表 16/24/32），logits 随机生成（seed 显式）。
- 边界：双语言 logits 恰平（argmax 与 max 的两套平分规则）、1D/2D 双路、显式 language 跳过、重归一化 = raw/Σraw 手算核对。
- 随机对拍 200 组（oracle 按 decoding.py 官方逻辑实现）。

## 评分

账逐项与官方逻辑一致：tokens/top 精确、probs 1e-9、prob_sum = 1（重归一化验证）、raw_top_prob 为未重归一化对照（必 < 重归一化值）。

## 红线

- 不得调用 whisper / transformers 的任何函数；
- 仅 torch/numpy；logits 与语言表显式入参（不出模型、不出真音频）。
