import math


def solve(decibels, scores, opts=None):
    """FunASR FSMN-VAD 端点检测决策核账（modelscope/FunASR
    funasr/models/fsmn_vad_streaming/model.py，帧级决策层——缓冲区
    管理层（PopDataBufTillFrame/DropCachedFrames/动态静音调度）不建模，
    data_buf_start_frame 恒 0、last_drop_frames 恒 0，见 notes）。

    参数
    ----
    decibels : (T,) 每帧分贝值（list[float]）
    scores : (T,) 每帧静音 pdf 概率（list[float]，域 (0,1)——0/1 会
      让官方 math.log 崩溃，不收）
    opts : 官方 VADXOptions 字段子集（缺省用官方默认）：
      frame_in_ms=10, window_size_ms=200, sil_to_speech_time_thres=150,
      speech_to_sil_time_thres=150, speech_2_noise_ratio=1.0, do_extend=1,
      lookback_time_start_point=200, lookahead_time_end_point=100,
      max_single_segment_time=60000, snr_thres=-100.0,
      noise_frame_num_used_for_snr=100, decibel_thres=-100.0,
      speech_noise_thres=0.6, max_end_silence_time=800,
      max_start_silence_time=3000, detect_mode="multiple"|"single"

    官方口径（逐条钉死）：

      帧判定 GetFrameState（惰性逐帧、EMA 状态跨段延续）：
        t 越界 -> Sil（防御）；cur_decibel < decibel_thres -> Sil
        （不更新噪声 EMA）
        noise_prob = log(score) * speech_2_noise_ratio；
        speech_prob = log(1 - score)
        exp(speech_prob) >= exp(noise_prob) + speech_noise_thres 时：
          cur_snr = decibel - noise_avg；cur_snr >= snr_thres 且
          decibel 达标 -> Speech；否则 Sil（SNR 门失败也不更新 EMA）
        else -> Sil 且更新 EMA：avg = (cur + avg*(N-1)) / N
          （avg < -99.9 首帧直接 avg = cur）
      WindowDetector（滑窗状态机）：
        win = int(window_size_ms / frame_in_ms)；
        sil2sp_th = int(sil_to_speech_time_thres / frame_in_ms)；
        sp2sil_th 同式；环形缓冲 win_state + win_sum 增量
        pre=Sil 且 win_sum >= sil2sp_th -> Sil2Speech（pre=Speech）
        pre=Speech 且 win_sum <= sp2sil_th -> Speech2Sil（pre=Sil）
        （两阈值同值时 win_sum 恰在阈值上会随 pre_state 翻转）
        否则 Sil2Sil / Speech2Speech
      顶层 FSM（DetectOneFrame 逐帧升序；离线单遍仅末帧
        is_final_frame=True——官方 nn_eval_block_size 每块被块长覆盖）：
        Sil2Speech：NotDetected -> start = max(0, cur - latency)，
          latency = win + int(lookback_time_start_point / frame_in_ms)
          （do_extend=1 时，否则仅 win）；InSpeechSegment 中先补确认
          (latest+1..cur-1) 再查段长
        段长 cur - confirmed_start + 1 > max_single_segment_time /
          frame_in_ms -> OnVoiceEnd(cur) 强切（max_time_out）
        InSpeechSegment 中 Sil2Sil：continous += 1 后
          continous * frame_in_ms >= max_end_sil_frame_cnt_thresh ->
          OnVoiceEnd(cur - lookback)；lookback = int(thresh/frame_in_ms)，
          do_extend 再减 int(lookahead/frame_in_ms) + 1，max(0,·)；
          **thresh = max_end_silence_time - speech_to_sil_time_thres**
          （init_cache 相减 quirk）
          否则 do_extend 且非末帧且 continous <= lookahead 帧 ->
          OnVoiceDetected（尾部延展）；末帧 -> MaybeOnVoiceEndIfLast
        NotDetected 中 Sil2Sil：single 模式静音超时（continous *
          frame_in_ms > max_start_silence_time）或末帧且从未检出 ->
          假段（fake，不产真段）
        End Detected：multiple 模式 -> ResetDetection（清窗机/确认帧/
          continous；**不清** frm_cnt / noise EMA / number_ends）；
          single 模式 -> 后续帧全部跳过（DetectCommonFrames 早退）
        OnVoiceEnd 先补确认 (latest+1..end-1) 再定 end 帧
      输出段 ms = frame * frame_in_ms

    需要产出的账（键固定）：
      frame_states    [0|1|-1] 每帧判定（-1 = single 模式终态后跳过）
      change_states   [0..5] 窗机每帧输出（官方枚举：0=speech2speech
                      1=speech2sil 2=sil2sil 3=sil2speech 5=invalid）
      segments        [[start_ms, end_ms], ...] 真段按检出顺序
      fake_segments   假段（[[0, 0]]）
      noise_avg_db    最终噪声 EMA
      max_time_out    是否触发强切
      latency_frames  起点回退帧数
      end_lookback    终点回退帧数（thresh/frm，do_extend 再 -lookahead/frm-1）
      max_end_sil_thresh_ms  max_end_silence_time - speech_to_sil_time_thres

    红线：不得调用 funasr / 任何模型前向；纯 Python 标量账。
    """
    o = {
        "frame_in_ms": 10, "window_size_ms": 200,
        "sil_to_speech_time_thres": 150, "speech_to_sil_time_thres": 150,
        "speech_2_noise_ratio": 1.0, "do_extend": 1,
        "lookback_time_start_point": 200, "lookahead_time_end_point": 100,
        "max_single_segment_time": 60000, "snr_thres": -100.0,
        "noise_frame_num_used_for_snr": 100, "decibel_thres": -100.0,
        "speech_noise_thres": 0.6, "max_end_silence_time": 800,
        "max_start_silence_time": 3000, "detect_mode": "multiple",
    }
    if opts:
        for k, v in opts.items():
            if k not in o:
                raise ValueError(f"unknown opt: {k}")
            o[k] = v
    frm_ms = o["frame_in_ms"]
    T = len(scores)
    if len(decibels) != T:
        raise ValueError("decibels/scores length mismatch")

    noise_avg_db = -100.0
    max_end_sil_thresh = o["max_end_silence_time"] - o["speech_to_sil_time_thres"]
    win_size = int(o["window_size_ms"] / frm_ms)
    sil2sp_th = int(o["sil_to_speech_time_thres"] / frm_ms)
    sp2sil_th = int(o["speech_to_sil_time_thres"] / frm_ms)
    win_state = [0] * win_size
    win_sum = 0
    win_pos = 0
    pre_state = 0                      # 0=Sil 1=Speech

    latency = win_size + (int(o["lookback_time_start_point"] / frm_ms)
                          if o["do_extend"] else 0)
    lookahead_frm = int(o["lookahead_time_end_point"] / frm_ms)

    state = "not_detected"             # not_detected / in_speech / end_detected
    confirmed_start = -1
    confirmed_end = -1
    latest_speech_frame = 0
    continous_sil = 0
    number_ends = 0
    max_time_out = False

    frame_states = []
    change_states = []
    segments = []
    fake_segments = []

    def get_frame_state(t):
        nonlocal noise_avg_db
        if t >= len(decibels):
            return 0
        cur_db = decibels[t]
        if cur_db < o["decibel_thres"]:
            return 0
        sum_score = scores[t]
        noise_prob = math.log(sum_score) * o["speech_2_noise_ratio"]
        speech_prob = math.log(1.0 - sum_score)
        if math.exp(speech_prob) >= math.exp(noise_prob) + o["speech_noise_thres"]:
            cur_snr = cur_db - noise_avg_db
            if cur_snr >= o["snr_thres"] and cur_db >= o["decibel_thres"]:
                return 1
            return 0
        if noise_avg_db < -99.9:
            noise_avg_db = cur_db
        else:
            noise_avg_db = (cur_db + noise_avg_db *
                            (o["noise_frame_num_used_for_snr"] - 1)) / \
                o["noise_frame_num_used_for_snr"]
        return 0

    def on_voice_start(start_frame, fake):
        nonlocal confirmed_start
        if confirmed_start != -1:
            raise RuntimeError("not reset vad properly")
        confirmed_start = start_frame

    def on_voice_end(end_frame, fake):
        nonlocal confirmed_end, number_ends, latest_speech_frame, state
        for t in range(latest_speech_frame + 1, end_frame):
            latest_speech_frame = t
        if confirmed_end != -1:
            raise RuntimeError("not reset vad properly")
        confirmed_end = end_frame
        if not fake:
            number_ends += 1
            segments.append([confirmed_start * frm_ms,
                             confirmed_end * frm_ms])
        state = "end_detected"

    def reset_detection():
        nonlocal state, confirmed_start, confirmed_end, continous_sil
        nonlocal latest_speech_frame, win_state, win_sum, win_pos, pre_state
        continous_sil = 0
        latest_speech_frame = 0
        confirmed_start = -1
        confirmed_end = -1
        state = "not_detected"
        win_pos = 0
        win_sum = 0
        win_state = [0] * win_size
        pre_state = 0

    def detect_one_frame(cur_state, cur_idx, is_final):
        nonlocal win_sum, win_pos, pre_state, continous_sil, max_time_out
        nonlocal latest_speech_frame, state
        win_sum -= win_state[win_pos]
        win_sum += cur_state
        win_state[win_pos] = cur_state
        win_pos = (win_pos + 1) % win_size
        if pre_state == 0 and win_sum >= sil2sp_th:
            pre_state = 1
            change = 3                    # sil2speech
        elif pre_state == 1 and win_sum <= sp2sil_th:
            pre_state = 0
            change = 1                    # speech2sil
        elif pre_state == 0:
            change = 2                    # sil2sil
        else:
            change = 0                    # speech2speech
        change_states.append(change)

        def maybe_end_if_final():
            nonlocal state
            if is_final:
                on_voice_end(cur_idx, False)

        if change == 3:                   # Sil2Speech
            continous_sil = 0
            if state == "not_detected":
                start_frame = max(0, cur_idx - latency)
                on_voice_start(start_frame, False)
                state = "in_speech"
                for t in range(start_frame + 1, cur_idx + 1):
                    latest_speech_frame = t
            elif state == "in_speech":
                for t in range(latest_speech_frame + 1, cur_idx):
                    latest_speech_frame = t
                if cur_idx - confirmed_start + 1 > \
                        o["max_single_segment_time"] / frm_ms:
                    on_voice_end(cur_idx, False)
                elif not is_final:
                    latest_speech_frame = cur_idx
                else:
                    maybe_end_if_final()
        elif change == 1:                 # Speech2Sil
            continous_sil = 0
            if state == "not_detected":
                pass
            elif state == "in_speech":
                if cur_idx - confirmed_start + 1 > \
                        o["max_single_segment_time"] / frm_ms:
                    on_voice_end(cur_idx, False)
                elif not is_final:
                    latest_speech_frame = cur_idx
                else:
                    maybe_end_if_final()
        elif change == 0:                 # Speech2Speech
            continous_sil = 0
            if state == "in_speech":
                if cur_idx - confirmed_start + 1 > \
                        o["max_single_segment_time"] / frm_ms:
                    max_time_out = True
                    on_voice_end(cur_idx, False)
                elif not is_final:
                    latest_speech_frame = cur_idx
                else:
                    maybe_end_if_final()
        elif change == 2:                 # Sil2Sil
            continous_sil += 1
            if state == "not_detected":
                if (o["detect_mode"] == "single" and
                    continous_sil * frm_ms > o["max_start_silence_time"]) or \
                        (is_final and number_ends == 0):
                    fake_segments.append([0, 0])
                    on_voice_start(0, True)
                    on_voice_end(0, True)
                # else: OnSilenceDetected 仅缓冲层，不建模
            elif state == "in_speech":
                if continous_sil * frm_ms >= max_end_sil_thresh:
                    lookback = int(max_end_sil_thresh / frm_ms)
                    if o["do_extend"]:
                        lookback -= lookahead_frm
                        lookback -= 1
                        lookback = max(0, lookback)
                    on_voice_end(cur_idx - lookback, False)
                elif cur_idx - confirmed_start + 1 > \
                        o["max_single_segment_time"] / frm_ms:
                    max_time_out = True
                    on_voice_end(cur_idx, False)
                elif o["do_extend"] and not is_final:
                    if continous_sil <= lookahead_frm:
                        latest_speech_frame = cur_idx
                else:
                    maybe_end_if_final()

    for t in range(T):
        if state == "end_detected" and o["detect_mode"] != "multiple":
            frame_states.append(-1)       # single 终态后跳帧
            continue
        fs = get_frame_state(t)
        frame_states.append(fs)
        detect_one_frame(fs, t, t == T - 1)
        if state == "end_detected" and o["detect_mode"] == "multiple":
            reset_detection()

    end_lookback = int(max_end_sil_thresh / frm_ms)
    if o["do_extend"]:
        end_lookback = max(0, end_lookback - lookahead_frm - 1)
    return {
        "frame_states": frame_states,
        "change_states": change_states,
        "segments": segments,
        "fake_segments": fake_segments,
        "noise_avg_db": noise_avg_db,
        "max_time_out": max_time_out,
        "latency_frames": latency,
        "end_lookback": end_lookback,
        "max_end_sil_thresh_ms": max_end_sil_thresh,
    }
