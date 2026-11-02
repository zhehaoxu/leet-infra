# notes — asr-fsmn-vad-endpoint（order 716）

## 回源

- 权威源：modelscope/FunASR `funasr/models/fsmn_vad_streaming/model.py`（main，2026，1302 行）。决策核心三件套：`GetFrameState` / `WindowDetector` / `DetectOneFrame`（FSM）。
- **nn_eval_block_size 被块长覆盖 quirk**：`ComputeScores` 里 `self.vad_opts.nn_eval_block_size = scores.shape[1]`——DetectCommonFrames 每块处理的是"整块新帧"（frm_cnt-1-i 逆序枚举实为升序），不是固定 8 帧。离线单遍等价于逐帧升序、仅末帧 is_final=True。
- **thresh 相减 quirk**：`max_end_sil_frame_cnt_thresh = max_end_silence_time - speech_to_sil_time_thres`（init_cache，800−150=650）——文档只写 max_end_silence_time=800 的都会算错收尾时机。
- **end 回退**：lookback = thresh/frm − lookahead/frm − 1（do_extend），官方默认 65−10−1=54 帧。
- **EMA 更新面窄**：仅概率门失败的 else 分支更新；SNR 门失败 / decibel 门失败直接 Sil 且 **EMA 不动**。EMA 跨段延续（ResetDetection 不清）。
- **max_time_out 只在 Speech2Speech 分支置位**——Sil2Speech/Speech2Sil/Sil2Sil 的强切分支都不置（官方原样不对称）。
- **假段**：纯静音末帧兜底走 OnVoiceStart(0,fake)+OnVoiceEnd(0,fake)，fake 不产真段但 number_ends++ 且进入终态；single 模式还有静音超时假段。
- **动态静音调度**（inference 层）：用户未显式给 max_end_silence_time 时 offline 默认启用 DEFAULT_SILENCE_SCHEDULE 且 speech_noise_thres 强制 0.5，按 chunk（默认 60s！）粒度累计——与帧级决策耦合复杂，**本题不建模**（opts 的 max_end_silence_time 显式给即走固定阈值路径），记 notes 备查。
- **缓冲层剥离**：PopDataBufTillFrame/DropCachedFrames 只影响波形切片与 data_buf_start_frame（start = max(0, cur-latency) 的第一参数恒 0），不影响段边界判定——决策核等价。

## 测试设计

- oracle = model.py 决策层原样转录（枚举/分支顺序一致）；小参数锚点手算（win=5、阈值 3 时环形缓冲未满不遗忘——首版手算漏了这点）。
- SNR 门用例：db/score 构造"概率门过 + snr 门败"（EMA 直赋后 db=-31、snr_thres=0），断言 EMA 不变——是 ema-update-on-snr-gate 变异的抓手。
- 双段 EMA 延续用例：两段噪声 db 不同（-30/-40），断言末值 ∈ (-40, -30)——EMA 被重置则首 sil 帧直赋 -40，立即可辨。

## 变异（12 全抓 + 1 等价）

- 全抓关键：EMA 更新面（三处门分支 × 完整公式——**首版变异只写了直赋分支，实为等价**）、窗机双阈值极性、环形遗忘、起点回退、end 回退、thresh 相减、尾延展关闭、single 跳帧、假段转真、EMA 首帧直赋、跨段 EMA 重置（**变异必须同时含 nonlocal 声明与赋值——只改其一分别是 no-op/局部变量，两次假 SURVIVED**）。
- 等价变异 1 条（SURVIVED，记此）：尾延展关闭——延展只写 latest_speech_frame，而 OnVoiceEnd 的回填循环 (latest+1..end-1) 与 end=cur-lookback 已覆盖同批帧；决策核段边界不变（完整管线中它影响输出缓冲内容，故对官方非等价）。

## 坑

- 嵌套函数改外层 state 必须 nonlocal（首版 detect_one_frame 漏声明 → UnboundLocalError）。
- 手算环形窗：未满窗口不遗忘（帧 4 时旧 speech 帧仍在窗内，win_sum 不降）。
- 与 693/694（specaugment/cif）划界：本题管 VAD 端点状态机；与 670/700/714/715 共同构成 Whisper/FunASR 双线 asr 账谱。
