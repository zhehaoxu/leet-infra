# notes — asr-whisper-lang-id-ledger（order 715）

## 回源

- 权威源：openai/whisper `whisper/decoding.py::detect_language` + `transcribe.py`（main，2026）。detect_language 本体极短（~40 行），账点密度高。
- **抑制后 softmax = 重归一化**：官方先 `logits[:, mask] = -np.inf`（np.inf 标量赋给张量）再 `softmax(dim=-1)`——非语言位 exp(−inf)=0，语言位概率 = raw/Σraw(语言)。raw 与重归一化的差就是本题的对照账（`raw_top_prob`）。
- **首窗策略**：transcribe 只对 `pad_or_trim(mel, N_FRAMES)`（首个 30s 窗）跑 detect_language，后续窗沿用——**语言切换不感知**（长音频中途换语言的已知局限即源于此）。`decode_options["language"] = max(probs, key=probs.get)`——dict 平分取首个插入键。
- **批语义**：`single = mel.ndim == 2`（对 mel 是 2D；对 SOT logits 是 1D）——unsqueeze(0) 成批、返回解包。批内逐条独立 softmax/argmax。
- **encoder 形状撞车 quirk**：`if mel.shape[-2:] != (n_audio_ctx, n_audio_state): mel = model.encoder(mel)`——mel 恰为 (1500, 1280)（large 档）时被当作已编码特征跳过 encoder。模型相关不出题，记 notes。
- 检测在 decode 循环外执行是为**不干扰 kv-cache**（docstring 原话）。

## 测试设计

- oracle = decoding.py 官方逻辑原样实现；缩微语言表（token id 3–11、词表 16）避免真实 5 万词表。token id 与 code 刻意非同序排列（`[7,3,11,5,9]`），argmax 平分取"下标较小"者与 max 平分取"插入序首个"者**不同**（3 vs zh=7 的对照）——两套平分规则被分开钉死。
- 随机对拍 200 组覆盖 single/batch 双路。
- `_make_logits(tie=True)` 把两个语言位 logits 置同值 2.0：断言 `tokens == min(两 token)` 且 `probs[zh] == probs[en]` 且 `top == 首个插入键`。

## 变异（12 全抓 + 2 等价）

- 全抓关键：mask 极性、−inf vs 0 抑制、softmax 前后（raw=renorm 即挂）、argmax 用 raw（可能选非语言位）、平分翻转（`reversed(list(d))` 才是真翻转——**首版变异用 `sorted(reverse=True)` 是 stable 排序、平分仍保原序，实为等价变异，写变异时要验稳定性**）、zip 键值互换、single 解包关闭、sum→max、跳过分支仍记账、detected 恒真、温度 1/2、enumerate 错位。
- 等价变异 2 条（SURVIVED）：① `batch.double().clone()` 的 clone 冗余（double() 已返回新张量）；② argmax(suppressed) ↔ argmax(softmax(suppressed))（softmax 严格单调保序，−inf→0 亦然）。

## 坑

- 缩微语言表 token id 必须 < 词表 V（首版 50259 > 512 直接 IndexError）。
- `_make_logits(n=1, single=False)` 返回 (1, V) 2D——single 路检查忘了传 single=True 会把标量断言打在列表上。
- 与 670（质量门控 fallback）划界：本题管语言检测这一步的数学与策略；670 管 decode_with_fallback 的温度递减与三门判定。与 700（seek 循环）划界：检测只在循环外的首窗，循环内无语言账。
