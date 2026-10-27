import torch


def solve(wave, preemph, frame_len, hop_len, window="hann"):
    """语音模型的第一道特征流水线：预加重滤波、分帧、加窗。

    wave:(T,) 单声道波形，float32；preemph:预加重系数 a，in [0, 1)；
    frame_len:帧长（采样点）;hop_len:帧移（采样点，<= frame_len 时相邻帧重叠）；
    window:"hann"（对称型）/ "hamming"（对称型）/ "rect"（矩形窗）。

    1) 预加重：y[0] = wave[0]，y[t] = wave[t] - a * wave[t-1]（t >= 1）；
    2) 分帧：不做任何 padding，帧数 F = 1 + (T - frame_len) // hop_len，
       第 f 帧 = y[f*hop_len : f*hop_len + frame_len]；
    3) 加窗：逐帧乘以窗函数 w，w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))（hann）或
       0.54 - 0.46*cos(2*pi*n/(N-1))（hamming），N = frame_len；N == 1 时 w = [1]。

    返回 (y:(T,) float32, frames:(F, frame_len) float32)。全程 float64 计算、最后转回 float32。
    """
    # TODO
    raise NotImplementedError
