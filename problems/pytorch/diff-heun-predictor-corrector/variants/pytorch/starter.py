import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    prediction_type="epsilon",
    clip_sample=False,
    clip_sample_range=1.0,
):
    """Heun 预测-校正半步（见题目描述）。"""
    raise NotImplementedError
