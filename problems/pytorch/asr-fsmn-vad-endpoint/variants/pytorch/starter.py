def solve(decibels, scores, opts=None):
    """FunASR FSMN-VAD 端点检测决策核账（帧级决策层，缓冲区管理层
    不建模）。手写三层：帧判定（score 概率门 + SNR 门 + 噪声 EMA）、
    WindowDetector 滑窗状态机、顶层段 FSM（起点回退 / 尾延展 /
    max_end_sil 收尾回退 / max_single 强切 / single 终态跳帧 /
    multiple 模式 ResetDetection）。

    官方口径要点（funasr/models/fsmn_vad_streaming/model.py）：
      - speech 门：exp(log(1-score)) >= exp(log(score)) *
        speech_2_noise_ratio + speech_noise_thres，过门后再查
        SNR（decibel - noise_avg >= snr_thres）；门失败不更新 EMA
      - 只有 else（概率门失败）分支更新噪声 EMA：
        avg = (cur + avg*(N-1))/N；avg < -99.9 首帧直赋
      - thresh = max_end_silence_time - speech_to_sil_time_thres
        （init_cache 相减 quirk）；end = cur - (thresh/frm -
        lookahead/frm - 1)；start = cur - (win + lookback/frm)
      - 窗机环形缓冲 + 双阈值（同值时恰在阈值上随 pre_state 翻转）
      - multiple 模式段尾 ResetDetection 不清 frm_cnt/EMA/number_ends；
        single 模式终态后跳帧
    opts 为官方 VADXOptions 字段子集（缺省官方默认值）。
    返回 dict：frame_states / change_states / segments / fake_segments /
    noise_avg_db / max_time_out / latency_frames / end_lookback /
    max_end_sil_thresh_ms
    红线：不得调用 funasr / 任何模型前向；纯 Python 标量账。
    """
    raise NotImplementedError
