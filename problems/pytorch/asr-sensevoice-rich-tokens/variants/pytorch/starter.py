"""SenseVoice rich-token 多任务前缀账（starter）。

实现 solve(mode, **kw)，通过全部用例。模式清单：
  encode_prefix  4 帧前缀 query 构造（[lid, event, emo, textnorm]）
  rich_ce        前 4 帧 rich CE（LabelSmoothing + ignore）与准确率
  ctc_greedy     CTC greedy：argmax → unique_consecutive → 滤 blank
  ban_emo        情绪 unk 的 -inf 屏蔽与 changed 掩码
  ctc_shift      内容 CTC 偏移窗 [4, L)（len<=4 空窗）

各 mode 的参数语义见 solution 同名分支 docstring。
官方词表常量 LID_INT_DICT / TEXTNORM_INT_DICT / EMO_DICT 供断言使用。
"""

import torch
from torch import Tensor

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def solve(mode: str, **kw):
    raise NotImplementedError
