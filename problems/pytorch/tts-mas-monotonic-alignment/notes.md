# notes — tts-mas-monotonic-alignment（order 672）

## 回源

- 出处：Glow-TTS（2011.06840，`jaywalnut310/glow-tts` `commons.py` 的 maximum_path Cython 实现）与 FastSpeech2（`ming024/FastSpeech2`）共用的 MAS；官方版是并行 DP（按对角线扫描 + softmax 门控训练），本题取其**推理硬对齐的朴素串行 DP**：每帧必消耗、text 不动或 +1。
- 与 `ctc-loss` 划界：CTC 对全部对齐**边缘化**（loss 是 sum），MAS 是**确定性 max** 提取唯一对齐。
- 官方 MAS 要求 duration ≥ 1（每 token 至少一帧，`T_mel ≥ T_text`）；终点不可达即无解。

## 判题面

- solve 入参：value [T_mel, T_text]。返回 log_prob / attn / durations。
- oracle：**穷举全部单调切分**（`combinations(range(T_mel-1), T_text-1)` 选切换点）独立求 max，随机 200 组（T_text 1–4、T_mel ≤ 8）对拍 log_prob（1e-9）+ durations/attn 结构一致性；log_prob 与 attn 加权重算互相钉。
- 锚点：手算 2×2 与 3×1；贪心≠全局最优的构造（5/5/100 例）；T_mel == T_text 路径唯一；对称 value 并列解（log_prob 唯一、结构合法）；`T_mel < T_text` → ValueError。
- 非法输入：value 空/行不齐/非数/NaN/Inf → ValueError。

## 变异记录（9 条全抓 + 2 条无效变异记档）

no-continue-branch（丢延续分支）、no-switch-branch（丢切换分支）、switch-from-left（切换改从同行左侧转移 → 允许 mel 不消耗）、negate-value（max 变 min 路径）、parent-flip（parent 记号反转）、duration-off-by-one（首帧不计数）、attn-transpose、value-row-swap（取镜像 token 的 value）、parent-lookup-flip（回溯读反）。

**记档**：`dp[t][s] = ... if par==1 else ...` 两支相同的纯语法等价变异；终点 -inf 防御检查在 T_mel≥T_text 时不可达（该变异等价）。

## 红线

真模型/音频不出（log 似然矩阵全入参）；softmax 门控训练版与跳搜（v2）不出；与 `ctc-loss`、`asr` 的 CTC 解码族互查不重出。
