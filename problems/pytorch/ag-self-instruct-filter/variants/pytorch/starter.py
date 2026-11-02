import torch


def solve(response, seed_instructions, sims, finish_reason="stop",
          threshold=0.7, top_k=10, prompt_instructions=None,
          classification=False):
    # TODO: 1) 按 "\n<数字><可选空白>. " 切分响应并做空白压缩 + capitalize；
    #       2) 七道过滤门（空 / 词数 / 关键词 / Write a program / 标点 / 非 ASCII）；
    #       3) 逐条与「种子 + 已接受」池比相似度，严格大于阈值则丢弃；
    #       4) 记录 top_k 个最相似项，并按需拼装下一个提示词。
    raise NotImplementedError
