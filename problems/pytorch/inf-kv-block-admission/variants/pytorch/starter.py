import torch


def solve(num_blocks, block_size, prompt_lens, max_new, max_steps):
    """KV 块预算下的准入控制：谁进 running、谁继续等待、谁被迫顺延。

    num_blocks:物理块总数;block_size:每块 token 数;
    prompt_lens:(N,) int64 各请求的 prefill 长度;max_new:(N,) int64 各请求要生成的 token 数;
    max_steps:最多模拟多少步。

    每一步先「解码阶段」再「准入阶段」，两阶段内都按请求 id 升序处理：
      解码阶段：对已准入且未完成的请求，尝试再生成 1 个 token。生成后占用块数
        为 ceil((prompt_len + 已生成 + 1) / block_size)；多出来的块从空闲池取，
        取不到则本步顺延（stall，不生成），不抢占已经运行中的请求。
      准入阶段：对尚未准入的请求，prefill 需要 ceil(prompt_len / block_size) 块；
        够就准入（记下准入步），不够就跳过并继续看下一个请求。

    返回 (admit_step:(N,)、finish_step:(N,)（-1 表示未完成）、
          stalls:(N,)、peak_used:int64 标量（峰值占用块数）)。
    """
    # TODO
    raise NotImplementedError
