import torch


def solve(
    logits,
    sampled_tokens,
    random_tokens,
    timestep,
    num_inference_steps,
    entropy_bound=0.1,
    t_max=0.8,
    t_min=0.4,
):
    """熵界解掩码的一步：按温度退火后的熵，从低到高接收位置并提交候选 token。

    logits:形状 (batch, block_length, vocab_size) 的浮点张量，模型对当前块的原始输出；
    sampled_tokens:形状 (batch, block_length) 的整型张量，候选 token（由调用方从分布中采好）；
    random_tokens:形状 (batch, block_length) 的整型张量，未被接收的位置要换上的均匀随机 token；
    timestep:当前步号（Python 整数），取值 [0, num_inference_steps)；
    num_inference_steps:推理步数（正整数）；
    entropy_bound:允许的联合熵上限，非负有限实数；
    t_max / t_min:温度退火的起点与终点（正有限实数）。

    温度按 temperature = t_min + (t_max - t_min) * (N - timestep) / N 退火，logits 先除以温度。
    返回 (prev_sample, accepted, token_entropy)，分别为 int64、bool、float32，形状都是 (batch, block_length)。
    非法输入抛 ValueError。
    """
    # TODO
    raise NotImplementedError
