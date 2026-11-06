# 踩坑记录（order 830）

## 回源口径

- 仓 `QwenLM/Qwen3-ASR`，`qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`（Qwen3ASRAudioEncoder）。读源码时 grep 在该文件上返回空（文件是纯 ASCII，原因不明）——用 python 逐行 re.search 替代 grep 才定位到行号。
- **下采样长度公式是 per-chunk 的**：`leave = lens % 100` 单独三次 `(l-1)//2+1`，再加 `(lens//100)*13`。这不是对整体 mel 做三次下采样——因为每个 100 帧满 chunk 独立 conv，边界 padding 各吃一次（100 帧独立 conv 恰 13 tokens：50→25→13）。公式里 100/13 硬编码，不随 config 的 n_window 走（HF config 默认 n_window=100 → 200 帧，与权重实际的 n_window=50 不一致；以公式为准，chunk_frames 作入参默认 100）。
- chunk_plan 的**整除 quirk**：尾块 `lens % chunk_frames` 余 0 时回落满窗（`chunk_lengths[==0] = chunk_frames`）——整除样本的"尾块"就是最后一块本身，余数公式给 0 必须兜住。
- 串切打包是 **batch 内串接再统一切块**（`input_features.T.split(...)`）：chunk 坐标是全局帧坐标，跨样本混排。判题域收单样本版（batch 版同构特例），docstring 钉死语义。
- per-chunk 位置编码：位置表截前 Tmax 行逐块相加，**pad 位也加**（随后 mask 丢弃才不可见）——变异"跳过 pad 位"在展平输出上不可观测，断言必须打在 hidden 上（hidden[0,2] == padded[0,2]+pos[2]）。
- cu_windows 的窗口由 **padded 后**的块长推导（`padded_t * (infer // chunk_frames)`），不看真实尾块长——官方用同一 window_aftercnn 对所有样本分窗。
- 注意力 eager 路径的 `_prepare_attention_mask` 造 4D 块对角 mask；FA2 走 varlen cu_seqlens。两者段间隔离语义一致，判题域收"段内双向 softmax(QKᵀ/√d)V + 段间隔离"。
- MRoPE 交错布局（apply_interleaved_mrope，[THTHWHTHW...TT] 频率交错）已被 omni 合集 omni-tmrope-position 占，不入题；get_chunked_index 是 MRoPE position ids 的分块辅助，同不入题。
- conv_out 的 mel 方向输入维度公式 `((((mel+1)//2+1)//2+1)//2)` 与实际 Conv2d 推导（mel=128 → 64→32→16）对不上（公式给 17）——Whisper 沿用的历史公式，权重 in_features 才是真值。mel 方向维度是静态账，不入题。

## 测试设计踩坑

1. **手算锚点错 200**：leave=0 → 三次 (−1)//2+1 = 0，out = 0 + 2×13 = 26；首版手算写 39（那是 300 的账）。锚值必须逐层推，不要"感觉"。
2. **m-pack-cnn 幸存（第一版）**：pack 的 after_cnn 改成连续整体下采样 `(l-1)//8+1`，在 100/50 块数据上恰好同值（块长 ≤100 时 per-chunk ≡ 连续）——分叉点在块长 >100 非整百（chunk_frames=250 的 250 帧块：per-chunk 33 vs 连续 16）。用例补 chunk_frames=250 抓住。**教训：分叉点要找公式的定义域差异，不是随便换数据**。
3. **变异死于形状 ≠ 死于语义**：m-pos-shift（位置表偏移一位）在 pos_emb 只有 Tmax 行时切片越界 RuntimeError——假 CAUGHT；pos_emb 多给一行（Tmax+1）后偏移仍合法，死因变成 hidden 全错（语义）。m-attn-leak 同理：k/v 全序列泄漏首版形状不匹配 RuntimeError，改成 `q[a:b], k, v` 段内形状合法、输出被段外 token 污染，死于 allclose。**变异应尽量死于断言而非崩溃**。
4. m-plan-tailidx 死于 IndexError（tail 定位错位后 chunk_lengths[tail] 越界）——可接受：错位在测试数据规模下必然越界，语义死与崩溃死在此处不可分。
5. block_attn 的段间隔离用三重断言：oracle 对拍 + 改第二段不影响第一段 + 单窗 [0,6] 与两窗 [0,2,6] 结果必须不同（防"全序列注意力"与"块对角"在小数据上偶然同值）。
6. pack_chunks 与 chunk_plan/cnn_lengths 是组合关系（pack 内部复用两个 mode 的输出）——变异互相牵连，m-pack-cnn 变异同时验证了组合链路的可观测性。

## mut 设计（17 变异全 CAUGHT）

- cnn_lengths 族：模数偏移 / 首层 kernel 语义 / tokens 13→14 —— 逐层 steps + 手算锚点兜住；
- chunk_plan 族：ceil→floor / 尾块模数偏 / 回落值偏 / tail index pad 值 —— 尾块与整除回落双边界钉死；
- pack 族：漏 transpose / pad 值 1 / after_cnn 连续式 —— 块长 >100 分叉用例抓住组合链；
- pos_flatten 族：位置表偏移 / 不做 mask 展平 —— pad 位可观测断言（hidden 上）+ 展平长度；
- cu_windows 族：窗口丢 ×padded_t / 余 0 加空窗 —— cu_chunk_lens 精确列表断言；
- block_attn 族：丢 √d / 段间泄漏 / 丢输出投影 —— oracle 对拍 + 隔离三重。
