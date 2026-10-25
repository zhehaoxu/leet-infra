"""参考答案：CLIP 对称 InfoNCE 对比损失。"""

import torch
import torch.nn.functional as F


def solve(image_feats, text_feats, logit_scale):
    n = image_feats.shape[0]
    li = logit_scale * (image_feats @ text_feats.T)
    lt = logit_scale * (text_feats @ image_feats.T)
    labels = torch.arange(n)
    return 0.5 * (F.cross_entropy(li, labels) + F.cross_entropy(lt, labels))
