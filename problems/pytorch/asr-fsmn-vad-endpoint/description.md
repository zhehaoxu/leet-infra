# FunASR FSMN-VAD 端点检测决策核账

## 题意

FunASR 的 FSMN-VAD 是工业界最常用的流式/离线端点检测：FSMN 网络输出每帧静音 pdf 概率，后接一套三层纯规则决策核——帧判定（概率门 + SNR 门 + 噪声 dB EMA）、WindowDetector 滑窗状态机、顶层段 FSM。本题手写决策核全链（波形缓冲管理层不建模），不出模型、不出真音频：decibels/scores 显式入参。

## 求解

官方口径（modelscope/FunASR `funasr/models/fsmn_vad_streaming/model.py` 逐条钉死）：

- **帧判定**：`exp(log(1-score)) >= exp(log(score))·speech_2_noise_ratio + speech_noise_thres` 过概率门，再过 SNR 门（`decibel - noise_avg >= snr_thres`）；**只有概率门失败分支更新噪声 EMA**（SNR 门失败、decibel 门失败均不更新）：`avg = (cur + avg·(N-1))/N`，`avg < -99.9` 首帧直赋；
- **WindowDetector**：环形缓冲 + win_sum 增量；`pre=Sil 且 win_sum >= sil2sp_th → Sil2Speech`、`pre=Speech 且 win_sum <= sp2sil_th → Speech2Sil`——两阈值同值（150/10=15）时 win_sum 恰在阈值上会随 pre_state 翻转；**环形缓冲未满时不遗忘**；
- **顶层 FSM**：start = `max(0, cur - latency)`（latency = 窗长 + lookback/frm，do_extend 时）；InSpeechSegment 中段长超 `max_single_segment_time/frm` 强切（**仅 Speech2Speech 分支置 max_time_out**）；Sil2Sil 静音计数达 `max_end_sil_frame_cnt_thresh` 收尾，end = `cur - (thresh/frm - lookahead/frm - 1)`；**thresh = max_end_silence_time − speech_to_sil_time_thres**（init_cache 相减 quirk，800−150=650）；do_extend 且非末帧且静音 ≤ lookahead 帧时尾延展；
- **NotDetected 静音**：single 模式静音超时（`> max_start_silence_time`）或末帧兜底 → **假段**（fake，不产真段）；multiple 模式段尾 ResetDetection——**不清** frm_cnt / 噪声 EMA / number_ends；single 模式终态后跳帧（DetectCommonFrames 早退）；
- **枚举值**：change_states 用官方 AudioChangeState 原值（0=speech2speech 1=speech2sil 2=sil2sil 3=sil2speech 5=invalid）。

## 数据规模

- 小参数锚点（win=5 帧、阈值 3、latency=10、end_lookback=4）全链手算核对；官方默认参数常数账（latency=40、thresh=650、end_lookback=54）。
- 边界：SNR 门失败不更新 EMA、decibel 门不更新 EMA、窗机环形遗忘时序、纯静音假段、single 静音超时、max_single 强切、双段 EMA 延续（两段噪声 dB 不同使其可观测）。
- 随机对拍 200 组（oracle 为 model.py 决策层原样转录）。

## 评分

九键账逐项与官方决策核一致：帧判定/窗机输出/段边界（ms）精确相等，noise_avg_db 1e-9。

## 红线

- 不得调用 funasr / 任何模型前向；纯 Python 标量账；
- scores 域 (0,1)（0/1 会让官方 math.log 崩溃，不入域）。
