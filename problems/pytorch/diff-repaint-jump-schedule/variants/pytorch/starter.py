import torch


def solve(num_train_timesteps, num_inference_steps, jump_length=10, jump_n_sample=10):
    """生成 RePaint 的跳步时间步表。

    num_train_timesteps:训练步数 T，正整数；
    num_inference_steps:推理步数 N，正整数；实际使用 min(T, N) 作为循环上界；
    jump_length:每次回跳向更噪方向走的步数 j（正整数）；
    jump_n_sample:每个回跳点最多回跳的次数 n（正整数）。

    先按 N 走一遍递减序列；在 j 的整数倍处若有剩余回跳配额，就向上（更噪）连走 j 步，
    然后从那一点重新递减，配额用完为止；最后把整张表乘以步长比例 T // N。
    返回一维 int64 张量。非法参数（非整数 / 布尔值 / 非正数）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
