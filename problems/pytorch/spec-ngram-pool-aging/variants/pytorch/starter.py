import torch


def solve(prompt, observed, n, capacity):
    """维护一个容量受限的 n-gram 池：命中就查表预测下一个 token，未命中就现场学习。
    prompt:(L,) int64 初始语料（L >= n+1）、observed:(S,) int64 目标模型给出的真实 token、
    n 为 key 长度、capacity 为池中 key 数上限。
    池中每个 key 对应一个 {next_token: 次数} 计数表和一个最近使用时间戳；
    预测取「计数最多、并列取更小 token id」；容量超出时淘汰
    「总计数最小、再比时间戳更旧、再比 key 字典序更小」的那个 key。
    返回 (predicted:(S,) int64 未命中记 -1、hits:(S,) bool、size:(S,) int64 每步后的 key 数、
    evicted:(S,) int64 每步淘汰的 key 数)。"""
    # TODO
    raise NotImplementedError
