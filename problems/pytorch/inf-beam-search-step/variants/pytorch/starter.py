import torch


def solve(input_ids, beam_scores, next_log_probs, num_beams):
    """Beam search 一步：候选分 = beam_scores[:,None]+next_log_probs，展平取 top-k，
    用 parent=f//V, token=f%V 回退父序列并续写。返回 (new_input_ids, new_scores)。"""
    # TODO
    raise NotImplementedError
