import torch


def solve(k, q, v, budget=128, window_size=8, kernel_size=7,
          mix_lambda=0.07, threshold=0.5, retain_direction="last"):
    """R-KV 冗余感知打分与预算驱逐（arXiv 2505.24133，float64 内部）。
    k/v:(L,H,D)、q:(window_size,H,D) 同 dtype（float32/float64），NHD 布局，L >= budget > window_size。
    记 N=L-window_size。attn = max_pool1d(mean_w softmax(q@k[:, :N]^T / sqrt(D)))；
    redundancy = softmax(列均值(余弦>threshold 的代表位清零)) 之后切掉最后 window_size 列；
    final = mix_lambda*attn - (1-mix_lambda)*redundancy，三者均为 (H,N)。
    kept:(H,budget) int64 = 前 budget-window_size 条按 final 降序（同分取小下标）+ 末尾恒留窗口；
    kept_keys/kept_values:(H,budget,D) 按 kept 顺序 gather。返回输入 dtype。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
