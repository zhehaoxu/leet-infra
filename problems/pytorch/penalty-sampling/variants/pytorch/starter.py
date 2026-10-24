import torch


def solve(logits, history, penalty_repeat, penalty_freq, penalty_present, penalty_last_n):
    # TODO: 重复惩罚（返回惩罚后的 logits，不做采样）。
    # win = history if penalty_last_n < 0 else history[-penalty_last_n:]；统计每个 token 次数 count；
    # 对 count>0 的 token t：先 l = (l<=0 ? l*repeat : l/repeat)，再 l -= count*freq + present。
    # 返回 shape 同 logits 的一维张量。
    raise NotImplementedError
