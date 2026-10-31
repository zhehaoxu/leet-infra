# RVQ 残差向量量化账

## 题意

神经音频编解码器（EnCodec / SoundStream / Mimi / DAC，也是 llama.cpp Qwen3-TTS / Pocket-TTS 的 code-to-wave 同核）用 **RVQ**（Residual Vector Quantization）把连续向量编码成逐级码本索引：第 k 级对上一级残差做最近邻量化，残差逐级递减。本题转写这条量化账：给定 `[N, D]` 待量化向量与 `[K, C, D]` 逐级码本，输出每级码索引、重建向量与逐级残差能量。

## 求解

口径题面钉死（逐样本独立）：

- `residual = latent`；对每级 `k = 0..K-1`：
  - `idx = argmin_c ||residual - codebooks[k][c]||²`（**并列取最小下标**）；
  - `codes[k][n] = idx`；`residual -= codebooks[k][idx]`（残差逐级递减）；
  - `residual_sq[k][n]` = 本级量化后残差的 L2²；
- `reconstructed[n]` = 全级被选码本条目之和（k 递增顺序累加，浮点口径固定）。

与 quantization 合集划界：那边考权重/激活标量量化与码本学习，本题考**残差级联编码结构**。返回 `{"codes", "reconstructed", "residual_sq"}`。

## 数据规模

- 1 ≤ N ≤ 10^4，1 ≤ D ≤ 512，1 ≤ K ≤ 32，1 ≤ C ≤ 4096。
- latents/codebooks 空、维度不齐（latent 间或码本条目对 D）、空码本级、非数或 NaN/Inf、bool → `ValueError`。
