# notes — asr-whisperx-dtw-align（order 714）

## 回源

- 权威源：m-bain/whisperX `whisperx/alignment.py`（main，2026）。核心四函数 `get_trellis` / `backtrack` / `merge_repeats` / `merge_words`——即 torchaudio 强制对齐教程算法，WhisperX 加了 numba 加速分支（`_safe_njit`）；**torch fallback 分支与 numba 分支语义一致**，本题按 fallback 转录参考实现。
- **列 0 尾部 +inf**：`trellis[-num_tokens:, 0] = float("inf")`——不是笔误。作用：j==1 回溯到尾行时 `changed = trellis[t-1, 0] + e = +inf` 必 > stayed，强制收束。可证列 N 永不为 +inf（inf 从 (r0,1) 传到列 N 需 r0 ≥ T-N+2 且 N-1 步内到 T 行 → T+1 ≤ T 矛盾），故 +inf 填充不污染 argmax；效果上等价于"T ≥ N 时回溯必 ok"。
- **ok ⇔ T ≥ N**：T ≥ N 时列 N 行 ≥ N 全有限（max 传播有限值），argmax ≥ N，回溯每步 stay/change 均与 DP 一致且剩余行数足够 → 必收束；T < N 时列 N 全 −inf → argmax=0 → 空路径 → None。用例直接断言该不变量。
- **两域混用**：trellis 全 log 域（emission 原样相加），帧概率却 `exp(emission)` 回概率域；段分 = 帧均（merge_repeats），词分 = 长度加权（merge_words）——两套聚合是官方原样，不统一。
- **严格 > 平分留 blank**：回溯 `changed > stayed` 平分走 else（blank，不前进）；DP 递推 `stay if stay > change else change` 平分留 stay——两处严格性一致，变异 `>=` 即抓。

## 测试设计

- oracle = alignment.py 官方 torch fallback **原样转录**（非独立重写）——算法账的锚点就是官方代码本身；手算锚点另设（4×3 小例手核 corner/path/segments）。
- **容差 1e-6 不是 1e-9**：solution 用 float64 trellis、官方用 float32，cumsum 链上 1e-8 级漂移，1e-9 必挂（首版踩坑）。
- separator 用例：token 下标 0（= CTC blank）兼作 separator 词项——DP 上无碍（change 到 tokens[j]=0 读同一列），官方语义照旧。
- 随机对拍保证 `ok == (T >= N)`、`col0_inf == N` 两个不变量逐组成立。

## 变异（13 全抓 + 2 等价）

- 全抓关键：尾部 +inf 极性、DP max/min、回溯严格 >、帧概率取 lbl（blank 帧不冒充 token 概率）、argmax 首行、段右端 +1、帧均 vs 长度加权、空词守卫、path 反转、corner 行、col0_inf 符号。
- 等价变异 2 条（SURVIVED）：① `path.reverse()` ↔ `path = path[::-1]`；② 三元 max ↔ `max()` 内建（平分时值相等，结果同）。

## 坑

- solution 的输入校验必须在 `.shape` 访问**之前**（str 入参先 AttributeError 后 ValueError 的顺序坑）。
- `trellis[0, -N:]` 在 N=0 时会写成 `trellis[0, 0:] = -inf` 把 [0,0] 的 0 覆盖掉——官方对空 transcript 是病态的，题面守卫 N ≥ 1 不收。
- 与 670（Whisper 质量门控）/700（seek 循环）划界：本题管对齐模型的时间戳精化，那两题管转写主循环。
