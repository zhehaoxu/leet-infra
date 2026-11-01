# WhisperX 强制对齐账：trellis DP、回溯与两级合并

## 题意

WhisperX 把 Whisper 的粗时间戳升级为音素级对齐：用一个独立的 CTC 对齐模型（emission，(T, V) log 概率）对转录 token 做 Viterbi 强制对齐。核心是四段纯算法——trellis DP、回溯、`merge_repeats`（token 级段合并）、`merge_words`（词级合并）。本题手写全链，不调 torchaudio/torio/numba/whisperx。

## 求解

官方口径（m-bain/whisperX `whisperx/alignment.py`，torch fallback 分支原样）：

- **trellis** 形状 (T+1, N+1)（时间/词表各多一维，词表维代表 `<SoS>`）：`trellis[0,0]=0`；列 0 = blank 发射的 cumsum（全 blank 前缀代价）；第 0 行词表列 = −inf；**列 0 尾部 N 行 = +inf**（强制收尾机制——回溯到 j==1 的尾行时 `changed=+inf` 必胜，逼路径收束）；递推 `max(stay, change)` 全程 log 域；
- **回溯**：`t_start = argmax(trellis[:, N])`（并列取**首个**最大行）；每步 `changed > stayed` 严格大于——**平分留 blank/stay 不前进**；帧概率 `exp(emission[t-1, lbl])`（概率域，与 trellis 的 log 域混用是官方原样，blank 帧概率也计入路径）；j 未归 0 走完循环 → **None**（T < N 时列全 −inf、argmax 取 0 行、空路径必现）；
- **merge_repeats**：连续同 token 帧合并为 `[首帧, 末帧+1)`（左闭右开），score = 段内帧概率**算术平均**（blank 帧也计入分母）；
- **merge_words**：按 separator `|` 切分，词标签 = 字符段拼接，span = `[首段 start, 末段 end)`，score = **长度加权**平均 `Σ(score·len)/Σlen`——与 merge_repeats 的帧均是两套聚合；首元素即 separator / 连续 separator（`i1 == i2` 守卫）/ 尾 separator 均不产空词。

## 数据规模

- 手算锚点：T=4/V=3 小 emission 全账手核（trellis corner = −3.4、col0_inf=2、path 仅覆盖帧 1..2）。
- 边界：stay 胜 change 的 blank 帧、stay/change 恰平（严格 > 账）、T<N 返 None、separator 首中尾三态、多帧段两套聚合公式核对。
- 随机对拍 200 组（T∈[3,24]、V∈[3,8]、随机 separator 混入），对拍参考为 alignment.py 官方 torch fallback 原样转录。

## 评分

全账与官方实现逐项一致：ok/col0_inf 精确相等，path 的 (token, time) 精确、概率 1e-6，segments/words 的 span 精确、score 1e-6。

## 红线

- 不得调用 torchaudio / torio / numba / whisperx 的任何函数；
- 仅 torch 张量运算 + 纯 Python 循环；emission/tokens/transcript 显式入参（真音频 IO 不出）。
