# Qwen3-ASR AuT 分块编码器账

## 背景

Qwen3-ASR（QwenLM/Qwen3-ASR）的音频编码器 AuT 基于 Whisper 式 Conv2d 下采样 + Transformer，但为了支持流式/离线一体与长音频显存控制，把 mel 帧切成固定窗（chunk），**每个 chunk 独立过三层 Conv2d（kernel 3 / stride 2 / padding 1，8× 下采样）**，注意力也按 cu_seqlens 分窗（段内双向、段间不往来）。分块不是实现细节而是语义本身：卷积边界 padding 每个 chunk 各吃一次，官方下采样长度公式把这一点直接写死。本题重建其中可判定的账目。

## 机制拆解

### 1. per-chunk 下采样长度（cnn_lengths）

`_get_feat_extract_output_lengths` 不是"整体三次 (l-1)//2+1"：

```
leave = lens % 100          # 硬编码 100（满 chunk 帧数 n_window*2），不随 config 走
l1 = (leave - 1) // 2 + 1
l2 = (l1 - 1) // 2 + 1
l3 = (l2 - 1) // 2 + 1
out = l3 + (lens // 100) * 13   # 每个 100 帧满 chunk 独立 conv 恰产 13 tokens（50→25→13）
```

边界：leave=0（100 的倍数）时三次 `(−1)//2+1 = 0`，长度全由满 chunk 数 ×13 顶住——与 chunk_plan 的"整除回落满窗"配套自洽（250 → 33，200 → 26）。

### 2. 分块计划（chunk_plan）

`chunk_num = ceil(lens / chunk_frames)`；chunk_lengths 初始全满窗，尾 chunk 按 `tail_chunk_index = F.pad(chunk_num, (1,0), -1).cumsum(0)[1:]` 定位，覆盖为 `lens % chunk_frames`，**余 0 回落满窗**（`chunk_lengths[==0] = chunk_frames`——整除样本的尾块恰好是第 chunk_num 块本身）。

### 3. 串切打包（pack_chunks）

**batch 内多样本的 mel 先串接成一条长序列再按 chunk_lengths 统一切块**——chunk 是跨样本的独立单元（分块以全局帧坐标计）。切分后 `pad_sequence` 对齐 (N, Tmax, C) 再 transpose 成 (N, C, Tmax)，pad 位严格 0；每块 CNN 后长度 `cnn_lengths(chunk_lengths)`，bool mask 记有效位。batch 维的串切混排与单样本版共用同一套账（单样本是其特例）。

### 4. 位置编码与展平（pos_flatten）

正弦位置表截前 Tmax 行**逐 chunk 相加**——每块的位置都从 0 开始（per-chunk 位置编码，pad 位也加、随后被 mask 丢弃）。`padded_embed[mask]` 布尔展平把各块有效位串回一条序列，次序按 (chunk, t) 字典序。

### 5. 窗口分窗（cu_windows）

`window_aftercnn = padded_t * (infer_window // chunk_frames)`——窗口长度由 **padded 后**的块长推导（块长统一按 padding 后算，不看真实尾块）。每样本按 aftercnn_len 分窗：`cnn_len // window` 个满窗 + 余数窗（余 0 无余数窗），cu_chunk_lens 全局 cumsum 成 cu_seqlens。

### 6. 块对角注意力（block_attn）

eager 路径：q/k/v 权重显式入参，`softmax(QKᵀ·scaling)V` 的 scaling=1/√d；cu_seqlens 各段内标准双向注意力，**段间完全不往来**（官方 FA2 用 varlen，eager 用 4D 块对角 mask，二者语义同：段内双向、段间隔离）。

## 判题域

- mel 特征、位置表、注意力权重全部显式入参，fp64；时间维账纯整数；无 RNG（block_attn/pos_flatten 用例内 manual_seed）；
- 6 mode：cnn_lengths、chunk_plan、pack_chunks、pos_flatten、cu_windows、block_attn；
- 不出：mel 前端（audio 合集已占信号处理）、MRoPE 交错布局（omni-tmrope-position 已占）、完整 Conv2d 前向（mel 方向维度公式歧义，官方沿用 Whisper 历史公式）、Qwen3 decoder 侧（LLM 通用账）。
