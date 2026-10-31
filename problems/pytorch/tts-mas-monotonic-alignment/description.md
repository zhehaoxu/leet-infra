# Monotonic Alignment Search 单调对齐账

## 题意

TTS 的非自回归架构（Glow-TTS / FastSpeech2）需要把文本 token 对齐到 mel 帧：**Monotonic Alignment Search** 在文本-隐态对数似然矩阵上找一条**确定性单调** max 路径并提取 duration。本题转写其核心 DP 账：给定 `[T_mel, T_text]` 对数似然矩阵，输出最优路径的对数似然、硬对齐矩阵与每 token 帧数。

## 求解

口径题面钉死（无跳搜的朴素单调版）：

- 路径从 `(0,0)` 走到 `(T_mel-1, T_text-1)`：每帧恰好消耗一个 mel 帧（t 必 +1），text 指针**不动（同 token 延续）或 +1（切下一 token）**：
  `dp[t][s] = value[t][s] + max(dp[t-1][s], dp[t-1][s-1])`；
- `dp[0][0] = value[0][0]`，其余首行/首列不可达为 `-inf`；终点恰为最后一个 token ⇒ **每 token 至少 1 帧**（`T_mel ≥ T_text`，否则无解）；
- 回溯 parent 表得硬对齐 `attn[t][s] ∈ {0,1}` 与 `durations[s]`（和恒为 T_mel）。

与 `ctc-loss`（对全部对齐边缘化求和）划界：MAS 是**确定性单调 max**。返回 `{"log_prob", "attn", "durations"}`。

## 数据规模

- 1 ≤ T_text ≤ T_mel ≤ 10^4，value 为有限数值。
- value 空/行空/行不齐/含非数或 NaN/Inf、T_mel < T_text → `ValueError`。
