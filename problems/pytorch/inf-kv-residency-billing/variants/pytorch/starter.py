import torch


def solve(prompt_len, start, out_tokens, block_size, price_out, price_block_step):
    """把「占了多少块 × 占了多少步」积分出来，并换算成一份账单。

    prompt_len:(N,) prefill 长度;start:(N,) 请求开始占块的步;
    out_tokens:(N,) 生成的 token 数;block_size:每块 token 数;
    price_out:每个 output token 的单价;price_block_step:每「块·步」的单价（均 >= 0）。

    请求 r 在全局步 t ∈ [start_r, start_r + out_tokens_r) 内持有
        blocks(r, t) = ceil((prompt_len_r + (t - start_r)) / block_size)
    个块；退出时立刻释放，不留尾巴。

    返回 (residency:(N,) int64（每请求的块·步积分）、
          charge:(N,) float32（price_out * out_tokens + price_block_step * residency）、
          peak_blocks:int64 标量（全时间轴上同时占块数的峰值）、
          total_charge:float32 标量)。
    """
    # TODO
    raise NotImplementedError
