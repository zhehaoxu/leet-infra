import torch


def solve(a, b, dim=-1, eps=1e-8):
    # TODO: 余弦相似度 cos = (a·b) / ‖a‖ / ‖b‖，沿 dim 计算。
    # 分母模长用 clamp(min=eps) 保护零向量（对齐 F.cosine_similarity）。
    raise NotImplementedError
