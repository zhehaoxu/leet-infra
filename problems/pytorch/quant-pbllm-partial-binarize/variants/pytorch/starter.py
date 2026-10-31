import torch


def solve(W, low_frac, high_bits):
    """PB-LLM：幅度最小的 low_frac 比例权重走二值（xnor）通路，其余走高位非对称量化。

    `W` 是 [O, C] 权重；本题固定 groupsize=-1，即整张权重为一个 low 组，
    但 low 的 mean/scale 仍按行（输出通道）统计。

    1. 掩码：`k = int(W.numel() * low_frac)`，把 `|W|` 全局升序排序后取第 k 个值为
       `thresh`，`mask = |W| <= thresh`（并列值会一起入选）。
    2. low（xnor，掩码位置置零后统计，零也计入均值）：
       `masked = W * mask`；`low_mean = masked.mean(dim=1)`；
       `low_scale = |masked - low_mean|.mean(dim=1)`。
    3. high（逐行非对称）：`xmin = min(row.min(), 0)`、`xmax = max(row.max(), 0)`，
       二者同时为 0 时置为 (-1, +1)；`maxq = 2^high_bits - 1`；
       `high_scale = (xmax - xmin) / maxq`；`high_zero = round(-xmin / high_scale)`。
    4. 重建：high 走
       `q = clamp(round(w / high_scale) + high_zero, 0, maxq)`、`(q - high_zero) * high_scale`；
       low 走 `sign(w - low_mean) * low_scale + low_mean`（`torch.sign(0) == 0`）；
       最后按 `mask` 逐元素选择。

    返回 dict：mask bool [O, C]；low_mean/low_scale/high_scale float32 [O]；
    high_zero int64 [O]；dequant float32 [O, C]。
    非法输入（high_bits 越界、low_frac 不在 [0,1)、W 非二维浮点/空/非有限）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
